from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import time
import deepCABAC
import numpy as np
from tqdm import tqdm
import torch
import cv2
from PIL import Image
from model.dataloader import MiddleburyDataLoader
from model import losses
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

cmap = plt.cm.binary

def save_image_color(img_merge, filename):
    cv2.imwrite(filename, img_merge)

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    
    return depth.astype('uint8')   


class BlobHelper():
    """Class to help the interface with Azure Blob Storage
    """
    def __init__(self,conn_string):
        """Inits the interface by creating the BlobServiceClient from the connection string that is passed as an argument

        Args:
            conn_string (str): connection string needed to connect to blob storage
        """
        print("Initializing the blob interface...")
        # Create the BlobServiceClient object which will be used to create a container client
        self.blob_service_client = BlobServiceClient.from_connection_string(conn_string)
        # Name of the container of the models
        self.container_name = "conteinertostoremodels"
        self.container_client = self.blob_service_client.get_container_client(self.container_name)
    
    def download_model(self):
        """Downloads the model weights from Blob storage
        """
        print("Downloading model...")
        ini_time=time.time()
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob="model_best.pt")
        with open("weights/weights.pt", "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        end_time=time.time()-ini_time
        print("OK")
        return end_time

    def download_compressed_model(self):
        """Downloads the compressed model weights from Blob storage
        """
        print("Downloading compressed model...")
        ini_time=time.time()
        blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob="model_best_compressed.bin")
        with open("weights/compressed_weights.bin", "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
        end_time=time.time()-ini_time
        print("OK")
        return end_time



def decode_model_weights(model,weigths_path='weights/compressed_weights.bin'):
    """Receives a PyTorch model and outputs the same model with weights assigned and the time taken to do it

    Args:
        model (PyTorchModel): model
        weights_path (str): path to the model

    Returns:
        PyTorchModel: model with loaded weights
        float: elapsed time to decompress and load weights (s)
    """
    print("\nLoading decoder...")
    ini_time=time.time()
    decoder = deepCABAC.Decoder()
    with open(weigths_path, 'rb') as f:
        stream = f.read()
    decoder.getStream(np.frombuffer(stream, dtype=np.uint8))
    state_dict = model.state_dict()
    
    print("Decoding and assigning weights...")
    for name in tqdm(state_dict.keys()):
        if '.num_batches_tracked' in name:
            continue
        param = decoder.decodeWeights()
        state_dict[name] = torch.tensor(param)
    decoder.finish()

    model.load_state_dict(state_dict)
    end_time=time.time()-ini_time
    print("OK")
    print("Time taken to decode and load weights (s)->"+str(end_time))
    return model,end_time    
    

def save_result_row(batch_data, output, name, folder="outputs/",azure_run=None):
    """Will save a row with the different images rgb+depth+gt+output

    Args:
        batch_data ([type]): [description]
        output ([type]): [description]
    """

    #unorm_rgb = transforms.Normalize(mean=[-0.4409/0.2676, -0.4570/0.2132, -0.3751/0.2345],
    #                         std=[1/0.2676, 1/0.2132, 1/0.2345])
    #unorm_d = transforms.Normalize(mean=[-0.2674/0.1949],
    #                         std=[1/0.1949])
    #unorm_gt = transforms.Normalize(mean=[-0.3073/0.1761],
    #                         std=[1/0.1761])

    unorm_rgb=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
    unorm_d=transforms.Normalize(mean=[0],std=[1])
    unorm_gt=transforms.Normalize(mean=[0],std=[1])

    rgb=unorm_rgb(batch_data['rgb'][0, ...])
    depth=unorm_d(batch_data['d'].squeeze_(0))
    gt=unorm_gt(batch_data['gt'].squeeze_(0))
    output=unorm_d(output.squeeze_(0))
    #depth=unorm_d(batch_data['d'])

    #rgb=batch_data['rgb'][0,...]
    #depth=batch_data['d']
    #gt=batch_data['gt']



    #print("OUTPUT Size------------------->"+str(output.shape))
    img_list=[]
    
    rgb = np.squeeze(rgb.data.cpu().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))*255
    img_list.append(rgb)

    depth = depth_colorize(np.squeeze(depth.data.cpu().numpy()))
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)

    gt = depth_colorize(np.squeeze(gt.data.cpu().numpy()))
    img_list.append(gt)

    #print("OUTPUT SIZE BEFORE--->"+str(output.shape))
    output = depth_colorize(np.squeeze(output.data.cpu().numpy()))
    #output = np.moveaxis(np.squeeze(output.data.cpu().numpy()),0,2)
    #print("OUTPUT SIZE--->"+str(output.shape))
    img_list.append(output)
    
    img_merge_up = np.hstack([img_list[0], img_list[2]])
    img_merge_down = np.hstack([img_list[1], img_list[3]])
    img_merge = np.vstack([img_merge_up, img_merge_down])
    img_merge= img_merge.astype('uint8')
    if azure_run:
        imgplot = plt.figure()
        plt.imshow(img_merge)
        
        azure_run.log_image(name=name,plot=imgplot)
    else:
        save_image_color(img_merge,folder+name)
    #print("saving img to "+str(folder+name))

   
   
def test_model(model,device,tag=" compressed "):
    print("\nLoading data to eval...")
    pre_time=time.time()
    model.eval()
    criterion = losses.CombinedNew()
    #loading data to test
    dataset_test = MiddleburyDataLoader()
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        sampler=None)
    psnr_list=[]
    loss_list=[]
    #inference
    print("Inference over"+str(tag)+" ...")
    for i, batch_data in enumerate(test_dataloader):
        batch_data = {
            key: val.to(device) for key, val in batch_data.items() if val is not None
        }
        pre_time=time.time()-pre_time
        inf_time=time.time()
        output = model(batch_data)
        inf_time=time.time()-inf_time
        current_psnr = -losses.psnr_loss(output, batch_data['gt']).item()
        psnr_list.append(current_psnr)
        loss = criterion(output, batch_data['gt']).item()
        loss_list.append(loss)
        save_result_row(batch_data, output, "out_"+tag+str(i)+".png", folder="result_imgs/")
        
        
    print("Mean loss"+tag+": "+str(sum(loss_list)/len(loss_list)))
    print("Mean PSNR"+tag+ ": "+str(sum(psnr_list)/len(psnr_list)))
    print(f'Loading time ({pre_time}) and inference time ({inf_time}) (s)')


def compress_model_weights(path, params,save_route):
    model = torch.load(path,map_location=torch.device("cpu"))
    encoder = deepCABAC.Encoder()
    enc_time=time.time()
    
    for name, param in tqdm(model.items()):
        if '.num_batches_tracked' in name:
            continue
        param = param.cpu().numpy()
        if '.weight' in name:
            encoder.encodeWeightsRD(param, params['interv'], params['stepsize'], params['_lambda'])
        else:
            encoder.encodeWeightsRD(param, params['interv'], params['stepsize_other'], params['_lambda'])

    stream = encoder.finish().tobytes()

    uncompressed_size=1e-6 * os.path.getsize(path)
    compressed_size=1e-6 * len(stream)
    print("Uncompressed size: {:2f} MB".format(uncompressed_size))
    print("Compressed size: {:2f} MB".format(compressed_size))
    print("Compression ratio: "+str(uncompressed_size/compressed_size))
    enc_time=time.time()-enc_time
    print("Encoding time (s)=> "+str(enc_time))

    stream = encoder.finish()

    print("Saving encoded model to (weights.bin)")
    with open(save_route, 'wb') as f:
        f.write(stream)
