"""
This file contains the main logic to train the model
Author:Alejandro
"""

from re import M
import torch
from dataloaders.MiddleburyDataloaderFile import MiddleburyDataLoader
from helpers.helper import Logger, save_result_row
from helpers.helper import LRScheduler
import helpers.helper as helper 
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
from model.pretrained_1 import AutoEncoderPretrained
import torch.nn as nn
import numpy as np
from itertools import product
import os


sys.path.append("/media/beegfs/home/t588/t588188/.local/lib/python3.9/site-packages") 



def main():
    
    cuda = torch.cuda.is_available()
    if cuda:
        import torch.backends.cudnn as cudnn
        #cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("===> Using '{}' for computation.".format(device))
    
    
    
    
    
    train_or_test = "train"
    
    print("===> Starting framework...")
    """Some parameters for the model"""
    weight_decay = 1e-07
    epochs = 15
    
    #parameters_to_test = dict(
    #    lr = [0.01,0.005,0.001],
    #    batch_size = [16,32],
    #    weight_decay = [0,1e-06,1e-04]
    #)
    
    parameters_to_test = dict(
        lr = [0.001,0.0005,0.0001],
        loss_fun=["CombinedNew","CombinedNewBorder"],
        models=["InceptionAndAttentionModel"],
        selfAttentionLayers=[1,2,4],
        deconvLayers=[1,2,3],
        attentionChannels=[32,64,128]
    )
    
    param_values = [v for v in parameters_to_test.values()]
    
    for run_id, (lr,loss_fun,m,attLayers,deconvLayers,attChannels) in enumerate(product(*param_values)):
        if m=="TwoBranch_newModel":
            model=models.TwoBranch_newModel()
        if m=="TwoBranch_newModel_in":
            model=models.TwoBranch_newModel_in()
        if m=="TwoBranch_newModel_bn":
            model=models.TwoBranch_newModel_bn()
        if m=="SelfAttentionModel":
            model=models.SelfAttentionModel(attentionChannels=attChannels,attLayers=attLayers,deconvLayers=deconvLayers)
            
            
        model=nn.DataParallel(model)
        model.to(device)
            
            
        pre_depth=True
        aug=True        
        
        dataset = MiddleburyDataLoader('train', augment=aug, preprocess_depth=pre_depth)
        dataset_test = MiddleburyDataLoader('test', augment=False, preprocess_depth=False)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            num_workers=14,
            pin_memory=True,
            sampler=None)
        test_dataloader = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=1,
            shuffle=False,
            num_workers=14,
            pin_memory=True,
            sampler=None)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr,eps=1e-07, betas=(0.7, 0.99),weight_decay=weight_decay)
        criterion = loss_fun
        if loss_fun=="CombinedLoss":
            criterion=losses.CombinedLoss()
        if loss_fun=="L1":
            criterion=losses.MaskedL1Loss()
        if loss_fun=="BerHuLoss":
            criterion=losses.BerHuLoss()
        if loss_fun=="L2":
            criterion=losses.MaskedMSELoss()
        if loss_fun=="CombinedNew":
            criterion=losses.CombinedNew()
            
        comment = f'model={m} lr = {lr} Loss_fun = {loss_fun} AttLayers = {attLayers} AttChannels = {attChannels} DeconvLayers = {deconvLayers}'
        writer = SummaryWriter(log_dir="runs/ATT_MODEL",comment=comment)
        print(comment)
        
        total_loss_val=0
        total_loss_mse=0
        for epoch in range(epochs):
            #print("------------EPOCH ("+str(epoch+1) +") of ("+str(epochs)+")------------")
            
            model.train()
            running_loss = 0.0
            counter = 0
            loss_batch = []
            psnr_batch = []
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
                loss_batch.append(current_loss)
                
                current_psnr = losses.psnr_loss(output, batch_data['gt']).item()
                psnr_batch.append(current_psnr)

                running_loss += current_loss
                counter += 1
                
                writer.add_scalar("Train Loss- LOOP", current_loss, epochs+epoch+i)
                writer.add_scalar("Train PSNR(dB)- LOOP", -current_psnr, epochs+epoch+i)
            

            """log all the training!!!"""
            writer.add_scalar("Train Loss- EPOCH", sum(loss_batch)/len(loss_batch), epoch)
            writer.add_scalar("Train PSNR(dB)- EPOCH", -sum(psnr_batch)/len(psnr_batch), epoch)
            #print("PSNR->"+str(-sum(psnr_batch)/len(psnr_batch)))
        
            #try:
            #    os.mkdir("outputs/testHyperparameter/"+str(comment))
            #except:
            #    pass
            mse_calc=losses.MaskedMSELoss()
            #Starting evaluation routine
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
                    #save_result_row(batch_data, output, "out_"+str(epoch)+".png", folder="outputs/testHyperparameter/"+str(comment)+"/")
            total_loss_val= sum(val_losses)/len(val_losses)
            total_psnr_val=-sum(psnr_batch)/len(psnr_batch)
            total_mse_val= sum(val_mses)/len(val_mses)
            writer.add_scalar("Val Loss- EPOCH", total_loss_val, epoch)
            writer.add_scalar("Val PSNR(dB)- EPOCH", -total_psnr_val, epoch)
            writer.add_scalar("Val MSE - EPOCH", total_mse_val, epoch)
            
        writer.add_hparams(
            {"lr": lr, "bsize": batch_size, "low_beta":low_beta,"loss_fun":loss_fun,"model":m,"attChannels":attChannels,"attLayers":attLayers,"deconvLayers":deconvLayers},
            {
                "PNSR_val": total_psnr_val,
                "loss_val": total_loss_val,
                "RMSE loss": total_mse_val
            },
        )
        writer.close()

if __name__ == '__main__':
    main()
