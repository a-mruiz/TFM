"""
This file contains the main logic to train the model
Author:Alejandro
"""

from distutils.command.config import config
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
from torch.profiler import profile, record_function, ProfilerActivity
from train_test_routines.train_routine import train_model, test_model
from model.external import SelfSup_FangChang_2018, PENet_2021, TWISE_2021
import sys
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from filelock import FileLock
import os
from functools import partial
from ray.util.accelerators import NVIDIA_TESLA_A100
import ray

os.environ["SLURM_JOB_NAME"] = "bash"

sys.path.append("/media/beegfs/home/t588/t588188/.local/lib/python3.9/site-packages") 
cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))



def load_data():
    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/.data.lock")):
        dataset = MiddleburyDataLoader('train', augment=True, preprocess_depth=False,root_path="/home/t588/t588188/Alejandro/final_halimade/")
        dataset_test = MiddleburyDataLoader('test', augment=False, preprocess_depth=False,root_path="/home/t588/t588188/Alejandro/final_halimade/")
    return dataset, dataset_test


#@ray.remote(num_gpus=1, accelerator_type=NVIDIA_TESLA_A100)
def train_model_raytune(config):
    #dataset = MiddleburyDataLoader('train', augment=True, preprocess_depth=False)
    #dataset_test = MiddleburyDataLoader('test', augment=False, preprocess_depth=False)
    dataset,dataset_test=load_data()
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config['bs'],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        sampler=None)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        sampler=None)
    
    model=models.TFGmodel().cuda()
    
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config['wd'], betas=(0.9, 0.99))

    criterion = losses.CombinedNew(alpha1=config['alpha1'],alpha2=config['alpha2'])
    
    for epoch in range(config['epochs']):
        
        ################################################
        #                    TRAIN                     #
        ################################################
        model.train()
        running_loss = 0.0
        counter = 0
        train_losses = []
        train_psnrs = []
        train_mses = []
        val_losses=[]
        val_psnrs=[]
        val_mses=[]
        mse_calc=losses.MaskedMSELoss()
        for i, batch_data in enumerate(train_dataloader):
            batch_data = {
                key: val.to(device) for key, val in batch_data.items() if val is not None
            }
            # zero the parameter gradients
            optimizer.zero_grad()        
            # forward + backward + optimize
            output = model(batch_data)              
            
            loss = criterion(output, batch_data['gt'])
            loss.backward()  
            optimizer.step()

            #exp_lr_scheduler.step()
            current_loss = loss.item()
            train_losses.append(current_loss)
            
            current_psnr = losses.psnr_loss(output, batch_data['gt']).item()
            train_psnrs.append(current_psnr)

            running_loss += current_loss
            counter += 1
            
            current_mse=mse_calc(output, batch_data['gt']).item()
            train_mses.append(current_mse)
           
            ################################################
            #                     EVAL                     #
            ################################################
            with torch.no_grad():
                model.eval()
                val_losses=[]
                val_psnrs=[]
                val_mses=[]
                for i,batch_data in enumerate(test_dataloader):
                    batch_data = {
                        key: val.to(device) for key, val in batch_data.items() if val is not None
                    }
                    output=model(batch_data)
                    val_current_loss = criterion(output, batch_data['gt']).item()
                    val_current_psnr = losses.psnr_loss(output, batch_data['gt']).item()
                    val_current_MSE =mse_calc(output, batch_data['gt']).item()
                    val_mses.append(val_current_MSE)
                    val_losses.append(val_current_loss)
                    val_psnrs.append(val_current_psnr)

        #Report all the intermediate values to raytune
        tune.report(train_psnr=-sum(train_psnrs)/len(train_psnrs),
                    val_psnr=-sum(val_psnrs)/len(val_psnrs),
                    train_loss=sum(train_losses)/len(train_losses),
                    val_loss=sum(val_losses)/len(val_losses),
                    train_mse=sum(train_mses)/len(train_mses),
                    val_mse=sum(val_mses)/len(val_mses))



def main():

    
    print("===> Starting framework...")    
    
    raytune_searchspace={
        "alpha1":tune.loguniform(0.01, 0.99),
        "alpha2":tune.loguniform(0.01, 0.99),
        "bs":1,
        "lr":0.001,
        "wd":1e-6,
        "epochs":30
    }
    
    analysis=tune.run(partial(train_model_raytune),num_samples=3,config=raytune_searchspace,resources_per_trial={"cpu": 8, "gpu": 1})
    dfs = analysis.trial_dataframes
    [d.val_psnr.plot() for d in dfs.values()]
if __name__ == '__main__':
    main()