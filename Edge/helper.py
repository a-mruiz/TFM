from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import time
import deepCABAC
import numpy as np
from tqdm import tqdm
import torch

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
    print("Loading decoder...")
    ini_time=time.time()
    decoder = deepCABAC.Decoder()
    with open('compressed_weights.bin', 'rb') as f:
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
    return model,end_time    
    