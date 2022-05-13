import deepCABAC
import torch
import time
from tqdm import tqdm
import os
import numpy as np
from dataloaders.MiddleburyDataloaderFile import MiddleburyDataLoader
import helpers.losses as losses

def compress_model(model,weights_path,mount_point,map_location=torch.device("cpu"),compressed_saving_path="weights.bin",azure_run=None,interv=0.1,stepsize=2**(-0.5*15),stepsize_other=2**(-0.5*19),_lambda=0.):
    """Compress model using DeepCABAC to produce a binary file

    Args:
        weights_path (str): path to the uncompressed model_dict (weights)
        map_location (torch.device, optional):Device to use to compress the model, only supporting "cpu" for now. Defaults to torch.device("cpu").
        compressed_saving_path (str, optional): Path in which to store the compressed binary file. Defaults to "weights.bin".
        azure_run (Azure.Run, optional): Object to log the variables to azure Ml. Defaults to None.
        interv (float, optional): _description_. Defaults to 0.1.
        stepsize (_type_, optional): _description_. Defaults to 2**(-0.5*15).
        stepsize_other (_type_, optional): _description_. Defaults to 2**(-0.5*19).
        _lambda (int, optional): _description_. Defaults to 0.
    """
    print("Compressing model using DeepCABAC...")
    weights=torch.load(weights_path,map_location=map_location)    
    encoder=deepCABAC.Encoder()
    
    enc_time=time.time()
    
    
    for name, param in tqdm(weights.items()):
        if '.num_batches_tracked' in name:
            continue
        param = param.cpu().numpy()
        if '.weight' in name:
            encoder.encodeWeightsRD(param, interv, stepsize, _lambda)
        else:
            encoder.encodeWeightsRD(param, interv, stepsize_other, _lambda)

    stream = encoder.finish().tobytes()

    enc_time=time.time()-enc_time
    
    uncompressed_size=1e-6 * os.path.getsize(weights_path)
    compressed_size=1e-6 * len(stream)
    ratio=uncompressed_size/compressed_size
        
    print("Uncompressed size: {:2f} MB".format(uncompressed_size))
    print("Compressed size: {:2f} MB".format(compressed_size))
    print("Compression ratio: "+str(ratio))
    
    print("Encoding time (s)=> "+str(enc_time))

    stream = encoder.finish()

    if azure_run:
        azure_run.log("Encoding time DeepCABAC (s)",enc_time)
        azure_run.log("Uncompressed weights size (Mb)",uncompressed_size)
        azure_run.log("Compressed weights size (Mb)",compressed_size)
        azure_run.log("Compression ratio achieved",ratio)
    print("Saving encoded model to (weights.bin)")
    with open(compressed_saving_path, 'wb') as f:
        f.write(stream)
        
    model_compressed,_=decode_model_weights(model)    
    
    test_model_compressed(model_compressed,azure_run, mount_point)



def test_model_compressed(model,azure_run,mount_point):
    
    print("\nLoading data to eval...")
    pre_time=time.time()
    model.eval()
    criterion = losses.CombinedNew()
    #loading data to test
    dataset_test = MiddleburyDataLoader("test",mount=mount_point)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        sampler=None)
    psnr_list=[]
    loss_list=[]
    #inference
    
    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        #cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
          
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
        #save_result_row(batch_data, output, "out_"+tag+str(i)+".png", folder="result_imgs/")
        
    azure_run.log('LAST Loss Model Compressed',  sum(loss_list)/len(loss_list))
    azure_run.log('LAST PSNR Model Compressed',  sum(psnr_list)/len(psnr_list))
       
    #print("Mean loss"+tag+": "+str(sum(loss_list)/len(loss_list)))
    #print("Mean PSNR"+tag+ ": "+str(sum(psnr_list)/len(psnr_list)))
    #print(f'Loading time ({pre_time}) and inference time ({inf_time}) (s)')
    
    
def decode_model_weights(model,weigths_path='weights.bin'):
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