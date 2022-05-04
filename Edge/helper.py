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



def decode_model_weights(model):
    """Receives a PyTorch model and outputs the same model with weights assigned and the time taken to do it

    Args:
        model (PyTorchModel): model

    Returns:
        PyTorchModel: model with loaded weights
        float: elapsed time to decompress and load weights (s)
    """
    print("\nLoading decoder...")
    ini_time=time.time()
    decoder = deepCABAC.Decoder()
    with open('weights/compressed_weights.bin', 'rb') as f:
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
        num_workers=10,
        pin_memory=True,
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
    print("Mean loss"+tag+": "+str(sum(loss_list)/len(loss_list)))
    print("Mean PSNR"+tag+ ": "+str(sum(psnr_list)/len(psnr_list)))
    print(f'Loading time ({pre_time}) and inference time ({inf_time}) (s)')
    #calculate loss
    
    #measure time