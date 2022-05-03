import deepCABAC
import torch
import time
from tqdm import tqdm
import os


def compress_model(weights_path,map_location=torch.device("cpu"),compressed_saving_path="weights.bin",azure_run=None,interv=0.1,stepsize=2**(-0.5*15),stepsize_other=2**(-0.5*19),_lambda=0.):
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

