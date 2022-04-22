import torch
import time
from tqdm import tqdm
from helpers.helper import save_result_row, save_result_individual
from torch.profiler import profile, record_function, ProfilerActivity
from pynvml import *
import torch.nn.functional as F
import numpy as np
import imageio
import helpers.losses as losses
from helpers import graph_utils

def train_model(model, epochs, params, optimizer, logger, loader, loader_val, criterion, device, lr_scheduler,writer,azure_run):
    """
    Train the model 
    """

    # Empty cache on the GPU...just in case some more Mb appear
    
    latest_psnr=0
    
    val_psnr=[]
    val_loss=[]
    val_mse=[]
    
    train_psnr=[]
    train_loss=[]
    train_mse=[]

    folder_output="outputs/val/inceptionandatt/exp/"
    
        
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        print("------------EPOCH ("+str(epoch+1) +") of ("+str(epochs)+")------------")
        losses_batch, psnr_batch, gpu_time, mse_batch = train_one_epoch(
            model, optimizer, loader, criterion, logger, epoch,device,writer,azure_run)

        
        #Log the losses so that training graphs can be generated 
        
        train_loss.append(losses_batch)
        train_psnr.append(psnr_batch)
        train_mse.append(mse_batch)
        
        #log values to azure ml cloud
        azure_run.log("Train Loss",losses_batch)
        azure_run.log("Train PSNR(dB)",psnr_batch)
        azure_run.log("Train MSE",mse_batch)
        
        #"""log all the training!!!"""
        #writer.add_scalar("Train Loss- EPOCH", sum(losses_batch)/len(losses_batch))
        #writer.add_scalar("Train PSNR(dB)- EPOCH", -sum(psnr_batch)/len(psnr_batch))
        print("end epoch")
        
        #Starting evaluation routine
        with torch.no_grad():
            model.eval()
            val_losses=[]
            val_psnrs=[]
            val_mses=[]
            mse_loss=losses.MaskedMSELoss()
            for i,batch_data in enumerate(loader_val):
                batch_data = {
                    key: val.to(device) for key, val in batch_data.items() if val is not None
                }
                output=model(batch_data)

                print(f"\nLimits Output Depth low ({torch.min(output)} and high ({torch.max(output)}))")
                print(f"Limits Output GT low ({torch.min(batch_data['gt'])} and high ({torch.max(batch_data['gt'])}))")
                val_current_loss = criterion(output, batch_data['gt']).item()
                val_current_psnr = losses.psnr_loss(output, batch_data['gt']).item()
                val_current_mse = mse_loss(output, batch_data['gt']).item()
                val_losses.append(val_current_loss)
                val_psnrs.append(val_current_psnr)
                val_mses.append(val_current_mse)
                save_result_row(batch_data, output, "out_"+str(epoch)+"_"+str(i)+".png", folder=folder_output)

            val_mean_loss= sum(val_losses)/len(val_losses)
            val_mean_psnr= -sum(val_psnrs)/len(val_psnrs)
            val_mean_mse= sum(val_mses)/len(val_mses)
            
            val_loss.append(val_mean_loss)
            val_mse.append(val_mean_mse)
            val_psnr.append(val_mean_psnr)
            
            #log values to azureml cloud
            azure_run.log("Validation Loss",val_mean_loss)
            azure_run.log("Validation PSNR(dB)",val_mean_psnr)
            azure_run.log("Validation MSE",val_mean_mse)
            
            #writer.add_scalar("Val Loss- EPOCH", sum(losses_batch)/len(losses_batch))
            #writer.add_scalar("Val PSNR(dB)- EPOCH", -sum(psnr_batch)/len(psnr_batch))
            logger.logToFile(epoch, losses_batch, -psnr_batch, gpu_time)
            logger.logToFile(epoch, val_mean_loss, val_mean_psnr, gpu_time, False)
            if val_mean_psnr>latest_psnr:
                latest_psnr=val_mean_psnr
                torch.save(model.state_dict(), "model_best.pt")
                azure_run.upload_file(name="model_best.pt", path_or_stream="model_best.pt")
                print(latest_psnr)
            
        lr_scheduler(losses_batch)
    
    #y_ticker=(max(train_loss)-min(train_loss))/10
    #
    #y_ticker_mse=(max(train_mse)-min(train_mse))/10
    #
    #graph_utils.make_graph([train_loss, val_loss], ['train', 'val'], range(0,len(train_loss)), title="Loss values train&test", x_label="epochs",
    #                    y_label="Loss", x_lim_low=0, x_lim_high=len(train_loss), show=False, subtitle="", output_dir=folder_output,x_ticker=5,y_ticker=y_ticker)
    #graph_utils.make_graph([train_psnr, val_psnr], ['train', 'val'], range(0,len(val_psnr)), title="PSNR values train&test", x_label="epochs",
    #            y_label="PSNR(dB)", x_lim_low=0, x_lim_high=len(train_psnr), show=False, subtitle="", output_dir=folder_output,x_ticker=5)
    #graph_utils.make_graph([train_mse, val_mse], ['train', 'val'], range(0,len(val_psnr)), title="MSE loss values train&test", x_label="epochs",
    #            y_label="MSE", x_lim_low=0, x_lim_high=len(train_psnr), show=False, subtitle="", output_dir=folder_output,x_ticker=5,y_ticker=y_ticker_mse)
    return model


def train_one_epoch(model, optimizer, loader, criterion, logger, epoch,device,writer,azure_run):
    """
    It will train the model for only one epoch
    """
    model.train()
    t = tqdm(loader, total=int(len(loader)))
    running_loss = 0.0
    counter = 0
    loss_batch = []
    psnr_batch = []
    mse_batch = []
    mse_loss=losses.MaskedMSELoss()
    for i, batch_data in enumerate(t):
        data_start = time.time()
        batch_data = {
            key: val.to(device) for key, val in batch_data.items() if val is not None
        }
        data_time = time.time() - data_start
        azure_run.log("Data loading time for model inference (s)",data_time)
        gpu_time_start = time.time()

        # zero the parameter gradients
        optimizer.zero_grad()        
        # forward + backward + optimize
        output = model(batch_data)
        loss = criterion(output, batch_data['gt'])
        loss.backward()
        optimizer.step()

        gpu_time = time.time()-gpu_time_start
        
        azure_run.log("GPU time model forward pass (s)",gpu_time)
        
        #exp_lr_scheduler.step()
        current_loss = loss.item()
        loss_batch.append(current_loss)

        current_psnr = -losses.psnr_loss(output, batch_data['gt']).item()
        psnr_batch.append(current_psnr)
        
        current_mse=mse_loss(output, batch_data["gt"]).item()
        mse_batch.append(current_mse)

        running_loss += current_loss
        counter += 1
        t.set_postfix(loss=current_loss)
        #writer.add_scalar("Train Loss- LOOP", current_loss)
        #writer.add_scalar("Train PSNR(dB)- LOOP", -current_psnr)
        
    return sum(loss_batch)/len(loss_batch), sum(psnr_batch)/len(psnr_batch), gpu_time,sum(mse_batch)/len(mse_batch)


def test_model(model, loader, criterion, logger, device, folder="outputs/"):
    model.eval()
    t = tqdm(loader, total=int(len(loader)))
    running_loss=0
    i=0
    for i, batch_data in enumerate(t):
        batch_data = {
            key: val.to(device) for key, val in batch_data.items() if val is not None
        }
        gpu_time_start = time.time()
        with torch.no_grad():
            output = model(batch_data)
            """
            In the case of using TWISE, apply the below function
            
            output=smooth2chandep(1-output, {'depth_maxrange':80, 'threshold':100}, device)


            out_pred = output.detach().cpu().numpy()
            out_pred = np.squeeze(out_pred)
            pred_h, pred_w = np.shape(out_pred)   
            zero_mtrx = np.zeros((pred_h, pred_w))
            pred_dep = np.concatenate((zero_mtrx, out_pred))
            pred_dep_round = np.uint16(pred_dep*256)
            filename = 'TESTING_'+str(i)+'.png'
            imageio.imwrite(filename, pred_dep_round)
            """




            loss = criterion(output, batch_data['gt'])
            gpu_time = time.time()-gpu_time_start
            current_loss = loss.item()
            running_loss +=current_loss
            t.set_postfix(loss=current_loss)
            save_result_row(batch_data, output, "out_"+str(i)+".png", folder)
            save_result_individual(batch_data["rgb"], "rgb",name="rgb_"+str(i)+".png", folder="outputs/test/")
            save_result_individual(batch_data["d"], "d",name="depth_"+str(i)+".png", folder="outputs/test/")
            save_result_individual(batch_data["gt"], "gt",name="gt_"+str(i)+".png", folder="outputs/test/")
            save_result_individual(output, "pred",name="pred_"+str(i)+".png", folder="outputs/test/")
            
        #print(batch_data['d'].max())
        #print(batch_data['d'].min())
        #depth_mask=batch_data["d"]>2
        #depth_mask_2=batch_data["d"]<-1.3
        #depth=batch_data["d"]*depth_mask
        #depth_2=batch_data["d"]*depth_mask_2
        #save_result_individual(depth, "d",name="depth_"+str(i)+".png", folder="outputs/test_2/")
        #save_result_individual(depth_2, "d",name="depth_2_"+str(i)+".png", folder="outputs/test_2/")
    
    logger.logToFile(0, running_loss/i, 0, gpu_time, False)
    return running_loss/(i+1)


def smooth2chandep(chan_deps, params = None, device = None):
    if device is None:
        device = torch.device("cpu")
    split_deps = torch.split(chan_deps, 1, 1)        
    split_deps = list(split_deps)

    alpha = torch.sigmoid(split_deps[2])
    final_dep = alpha*F.relu(split_deps[0])*params['depth_maxrange'] + (1 - alpha)*F.relu(split_deps[1])*params['depth_maxrange']

    return final_dep
