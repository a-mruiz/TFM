#TODO list->
#Descargar pesos----------------|||||-----> Check
#Cargar modelo------------------|||||-----> Check
#descomprimir pesos-------------|||||-----> Check
#Cargar pesos al modelo---------|||||-----> Check
#inferencia---------------------|||||-----> Check


import json
import helper
from model import models 
import time
import torch



cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))




def download_weights():
    """
    Download the weights of the model and retrieves time statistics and compression ratio
    """
    with open('config_blob.json') as json_file:
        config_blob = json.load(json_file)

    blobHelper=helper.BlobHelper(config_blob["AZURE_STORAGE_CONNECTION_STRING"])


    time_to_download_model=blobHelper.download_model()
    time_to_download_compressed_model=blobHelper.download_compressed_model()
    ratio=time_to_download_model/time_to_download_compressed_model

    print("Time to download full model (s)->"+str(time_to_download_model))
    print("Time to download compressed model (s)->"+str(time_to_download_compressed_model))
    print("Ratio (times faster)->"+str(ratio))
    return time_to_download_model,time_to_download_compressed_model,ratio


def main():
    #Download the weights for the model
    download_weights()
    #Decompress the weights
    model_1=models.SelfAttentionCBAM().to(device)
    model_2=models.SelfAttentionCBAM().to(device)
    
    #load weights in normal way
    load_state_dict_time=time.time()
    model_1.load_state_dict(torch.load("weights/weights.pt", map_location=device))
    load_state_dict_time=time.time()-load_state_dict_time
    print("\nTime loading model uncompressed (s)->"+str(load_state_dict_time))
    #decompress and load compressed weights
    model_compressed,_=helper.decode_model_weights(model_2)    
    
    
    helper.test_model(model_1,device, " normal ")

    helper.test_model(model_compressed,device)
    

if __name__=="__main__":
    main()