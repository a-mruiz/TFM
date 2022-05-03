#Descargar pesos----------------|||||-----> Check
#Cargar modelo------------------|||||-----> 
#descomprimir pesos-------------|||||----->
#Cargar pesos al modelo---------|||||----->
#inferencia---------------------|||||----->


import json
from json.tool import main
import helper

def download_weights():
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
    
    
    
    
    
    
    

if __name__==main:
    main()