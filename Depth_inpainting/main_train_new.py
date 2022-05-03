"""
This file contains the main logic to train the model
Author:Alejandro
"""

import copy
import torch
from dataloaders.MiddleburyDataloaderFile import MiddleburyDataLoader
from helpers.helper import LRScheduler, Logger
import helpers.losses as losses
import model.models as models
from dataloaders.BoschDataloaderFile import BoschDataLoader
import time
from tqdm import tqdm
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
#from torch.profiler import profile, record_function, ProfilerActivity
from train_test_routines.train_routine import train_model, test_model
from model.external import SelfSup_FangChang_2018, PENet_2021, TWISE_2021
import sys
import os
import loss_landscapes
import loss_landscapes.metrics
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle as pl
from helpers import compression
# Import Azure SKD for Python packages
from azureml.core import Run
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__


sys.path.append("/media/beegfs/home/t588/t588188/.local/lib/python3.9/site-packages") 

training_script_loading_time = time.time()


cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))


try:
    # The mount point can be retrieved from argument values
    # This value has to be passed to the dataloaders in order to get the correct path of the images
    import sys
    mount_point = sys.argv[1]
    use_data_aug = sys.argv[2]
    blob_conn_string = sys.argv[3]
except: #Not training with Azure
    use_data_aug=None
    blob_conn_string=None
    mount_point=None
    
try:
    # Get the experiment run context
    run = Run.get_context()
    train_azure=True
except: #not training with Azure
    train_azure=False


def main():
    global training_script_loading_time
    global train_azure
    try:
        global use_data_aug
        global blob_conn_string
    except:#Not training with Azure
        pass
        
    full_loading_time = time.time()
    training_script_loading_time = time.time() - training_script_loading_time
    train_or_test = "train"
    deactivate_logger=True

    
    print("===> Starting framework...")
    """Some parameters for the model"""
    lr = 0.0001
    weight_decay = 1e-07
    #weight_decay = 0
    epochs = 30
    params = {"mode": train_or_test, "lr": lr,
              "weight_decay": weight_decay, "epochs": epochs,
              "bs":1}
    
    """#1. Load the model"""
    print("===> Loading model...")
    model_loading_time = time.time()
    model = models.SelfAttentionCBAM()
    model_option = "SelfAttentionCBAM"
    params['model']=model_option
    
    writer = SummaryWriter(log_dir="runs/FAILED/",comment=f'')
    
    h=1024
    w=1024
  
    model.to(device)
    
    model_loading_time = time.time() - model_loading_time
    
    """#2. Dataloaders"""
    print("===> Configuring Dataloaders...")
    
    dataloaders_loading_time = time.time()
    
    if use_data_aug==None:
        use_data_aug=True
    
    dataset = MiddleburyDataLoader('train', augment=use_data_aug, preprocess_depth=False,h=h,w=w,mount=mount_point)
    dataset_test = MiddleburyDataLoader('test', augment=False, preprocess_depth=False,h=h,w=w,mount=mount_point)
    dataset_option = "Middlebury"

    params['dataset_option'] = dataset_option
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['bs'],
        shuffle=True,
        num_workers=6,
        pin_memory=True,
        sampler=None)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
        sampler=None)

    dataloaders_loading_time = time.time() - dataloaders_loading_time
    
    # If needed to run in multiple GPUs, use--->DistributedDataParallel

    """#3. Create optimizer and criterion"""
    print("===> Creating optimizer and criterion...")
    model_named_params = [
        p for _, p in model.named_parameters() if p.requires_grad
    ]
    optimizer = torch.optim.Adam(
        model_named_params, lr=lr, weight_decay=weight_decay, betas=(0.7, 0.99))

    criterion = losses.CombinedNew()

    """#4. If no errors-> Create Logger object to backup code and log training"""
    logger = Logger(params,deactivate=deactivate_logger)

    
    lr_scheduler=LRScheduler(optimizer)
    
    
    full_loading_time = time.time() - full_loading_time
    
    if train_azure:
        """
        Logging all the parameters to Azure ML 
        """
        run.log('Batch Size',  params['bs'])
        run.log('Number of epochs',  epochs)
        run.log('Weight decay', params["weight_decay"])
        run.log('Model',model_option)
        run.log('Dataset Img Size Height',h)
        run.log('Dataset Img Size Width',w)
        run.log('Time loading arguments (s)',training_script_loading_time)
        run.log('Time mounting data and preparing dataloaders (s)',dataloaders_loading_time)
        run.log('Time loading model (s)',model_loading_time)
        run.log('Elapsed time until training starts (s)',full_loading_time)
        
    """#5. Split execution between train and test"""
    if train_or_test == "train":
        model_original=copy.deepcopy(model)
        model=train_model(model, epochs, params, optimizer,
                    logger, train_dataloader,test_dataloader, criterion, device,lr_scheduler,writer,azure_run=run)
        
        if not train_azure:            
            logger.generateTrainingGraphs()
            model_final=copy.deepcopy(model)
            print("\nComputing loss landscapes...(takes long time...)")
            loss_landscapes_processing_time = time.time()
            #For computing the loss-landscapes
            batch = iter(train_dataloader).__next__()
            batch = {
                key: val.to(device) for key, val in batch.items() if val is not None
            }
            criterion = losses.CombinedNewForLossLandscape()
            metric = loss_landscapes.metrics.Loss(criterion, batch, batch)
            STEPS=15
            # compute loss data
            loss_data = loss_landscapes.linear_interpolation(model_original, model_final, metric, STEPS, deepcopy_model=True)
            
            
            plt.plot([1/STEPS * i for i in range(STEPS)], loss_data)
            plt.title('Linear Interpolation of Loss')
            plt.xlabel('Interpolation Coefficient')
            plt.ylabel('Loss')
            axes = plt.gca()
            # axes.set_ylim([2.300,2.325])
            plt.savefig('Linear Interpolation of Loss.png')
            
            
            #For doing the 3d version
            
            print("\nComputing 3D loss ")

            loss_data_fin = loss_landscapes.random_plane(model_final, metric, 0.01, STEPS, normalization='filter', deepcopy_model=True)
            plt.contour(loss_data_fin, levels=60)
            plt.title('Loss Contours around Trained Model')
            plt.savefig('Loss Contours around Trained Model.png')
            
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            X = np.array([[j for j in range(STEPS)] for i in range(STEPS)])
            Y = np.array([[i for _ in range(STEPS)] for i in range(STEPS)])
            ax.plot_surface(X, Y, loss_data_fin, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
            ax.set_title('Surface Plot of Loss Landscape')
            plt.savefig('Surface Plot of Loss Landscape.png')

            # Save figure handle to disk
            with open('3dLossLandscape.fig.pickle', 'wb') as file:
                pl.dump(fig,file)
            """
            For loading the figure later->

            import pickle
            with open('3dLossLandscape.fig.pickle', 'rb') as file:

                figx = pickle.load(file)
                figx._constrained = False
                plt.show() # Show the figure, edit it, etc.!

            """
        #loss_landscapes_processing_time = time.time() - loss_landscapes_processing_time
        #run.log("Time processing loss landscapes graphs (s)",loss_landscapes_processing_time)
        torch.save(model.state_dict(), "model_best.pt")
        if train_azure:
            run.register_model(model_path='model_best.pt', model_name="best_model", model_framework='PyTorch', model_framework_version=torch.__version__)
            
            print("Saving model and the compressed version to blob storage-> container=containertostoremodels")

            blob_config_and_load_time = time.time() 
            
            blob_conn_string = blob_conn_string.replace("\\", "")
            
            #print("Connecting to->"+str(blob_conn_string))
            
            # Create the BlobServiceClient object which will be used to create a container client
            blob_service_client = BlobServiceClient.from_connection_string(blob_conn_string)
            container_name = "conteinertostoremodels"
            container_client = blob_service_client.get_container_client(container_name)
            
            blob_client = blob_service_client.get_blob_client(container=container_name, blob="model_best.pt")

            # Upload the created file
            with open("model_best.pt", "rb") as data:
                blob_client.upload_blob(data,overwrite=True)        
            
            blob_config_and_load_time = time.time() -blob_config_and_load_time
            
            run.log("Time to connect to blob and upload model uncompressed (s)",blob_config_and_load_time)
                        
            compression.compress_model("model_best.pt",azure_run=run)
            
            blob_client = blob_service_client.get_blob_client(container=container_name, blob="model_best_compressed.bin")

            # Upload the created file
            with open("weights.bin", "rb") as data:
                blob_client.upload_blob(data,overwrite=True)      
            
            run.complete()

    elif train_or_test == "test":
        avg_loss = test_model(model, test_dataloader,
                              criterion, logger, device)

        print("Test avg loss--->"+str(avg_loss))
    

if __name__ == '__main__':
    main()
