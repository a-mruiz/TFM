"""
This file contains the main logic to train the model
Author:Alejandro
"""

import torch
from helpers.helper import Logger
import helpers.losses as losses
import model.models as models
from dataloaders.BoschDataloaderFile import BoschDataLoader
from torch.utils.data import DataLoader,ConcatDataset
import time
from tqdm import tqdm
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, record_function, ProfilerActivity
from train_test_routines.train_routine import train_model, test_model
from model.external import SelfSup_FangChang_2018, PENet_2021, TWISE_2021
import sys
from model.pretrained_1 import AutoEncoderPretrained
import torch.nn as nn
import numpy as np
from sklearn.model_selection import KFold


sys.path.append("/media/beegfs/home/t588/t588188/.local/lib/python3.9/site-packages") 
cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))


def reset_weights(m):
    '''
    Try resetting model weights to avoid
    weight leakage.
    '''
    for layer in m.children():
        if hasattr(layer, 'reset_parameters'):
            print(f'Reset trainable parameters of layer = {layer}')
            layer.reset_parameters()


def main():
    
    k_folds=10
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    epochs=10
    
    
    train_or_test = "train"

    print("===> Starting framework...")
    """Some parameters for the model"""
    lr = 0.005
    #weight_decay = 1e-06
    weight_decay = 0
    epochs = 15
    params = {"mode": train_or_test, "lr": lr,
              "weight_decay": weight_decay, "epochs": epochs}

    """#1. Load the model"""

    #info = nvmlDeviceGetMemoryInfo(h)
    #print(f'total    : {info.total// 1024 ** 2}')
    #print(f'free     : {info.free// 1024 ** 2}')
    #print(f'used     : {info.used// 1024 ** 2}')


    if torch.cuda.is_available():
        model.cuda()
    """#2. Dataloaders"""
    print("===> Configuring Dataloaders for cross validation...")
    
    dataset_train = BoschDataLoader('train', augment=True, preprocess_depth=False, applyMask=True)
    dataset_test = BoschDataLoader('test', augment=False, preprocess_depth=False, applyMask=True)
    
    dataset_full=ConcatDataset([dataset_train, dataset_test])
    
    for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset_full)):
        print('\n------------fold no---------{}----------------------'.format(fold))
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(
                            dataset_train, 
                            batch_size=1, sampler=train_subsampler)
        testloader = torch.utils.data.DataLoader(
                            dataset_test,
                            batch_size=1, sampler=test_subsampler)

        print("===> Loading model...")
    
        model=models.BasicModelUltraLight()
        model_option = "BasicModelUltraLight"
        params['model_option']=model_option
        model=nn.DataParallel(model)
        model.apply(reset_weights)

        optimizer = torch.optim.Adam(
        model.parameters(), lr=lr,eps=1e-07, betas=(0.9, 0.99))
        criterion = losses.CombinedLoss()

        for epoch in range(0, epochs):
            # Print epoch
            print("----------- TRAINING -----------")
            print(f'Starting epoch {epoch+1}')
        
            # Set current loss value
            running_losses = []
            running_psnr = []
            for i,batch_data in enumerate(trainloader,0):
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
                running_losses.append(loss.item())
                current_psnr = losses.psnr_loss(output, batch_data['gt']).item()
                running_psnr.append(current_psnr)
            print("Mean loss->"+str(running_losses/len(running_losses)))
            print("Mean PSNR->"+str(running_psnr/len(running_psnr)))
        val_losses = []
        val_psnr = []
        with torch.no_grad():
            for i,batch_data in enumerate(testloader,0):
                batch_data = {
                    key: val.to(device) for key, val in batch_data.items() if val is not None
                }
                output = model(batch_data)
                loss = criterion(output, batch_data['gt'])
                val_losses.append(loss.item())
                current_psnr = losses.psnr_loss(output, batch_data['gt']).item()
                val_psnr.append(current_psnr)
            print("----------- VALIDATION -----------")
            print("Mean Validation loss->"+str(val_losses/len(val_losses)))
            print("Mean Validation PSNR->"+str(val_psnr/len(val_psnr)))
                