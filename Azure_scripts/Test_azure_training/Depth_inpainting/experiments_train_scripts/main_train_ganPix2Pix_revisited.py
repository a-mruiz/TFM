"""
This file contains the main logic to train the model
Author:Alejandro
"""

from dataloaders.MiddleburyDataloaderFile import MiddleburyDataLoader
import torch
from helpers.helper import Logger, save_result_row
from helpers.helper import LRScheduler
import helpers.helper as helper 
import helpers.losses as losses
import model.models as models
import model.models_gan as models_gan
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
from torch.autograd import Variable
from helpers import graph_utils

sys.path.append("/media/beegfs/home/t588/t588188/.local/lib/python3.9/site-packages") 
cuda = torch.cuda.is_available()
if cuda:
    import torch.backends.cudnn as cudnn
    #cudnn.benchmark = True
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("===> Using '{}' for computation.".format(device))



def main():
    
    train_or_test = "train"

    print("===> Starting framework...")
    """Some parameters for the model"""
    #weight_decay = 1e-06
    weight_decay = 0
    epochs = 30
    params = {"mode": train_or_test, "lr": "0",
            "weight_decay": weight_decay, "epochs": epochs}
    params['bs']=1
    writer = SummaryWriter('runs/experiment_GAN')

   
    
    """#1. Load the model"""
    print("===> Loading model...")
    
    
    # Initialize generator and discriminator
    generator = models_gan.GeneratorRRDB(channels=4,filters=8, num_res_blocks=2)#R+G+B+D
    discriminator = models_gan.Discriminator(input_shape=(1,512,1024))
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        
    generator.apply(models.weights_init_normal_pix2pix)
    discriminator.apply(models.weights_init_normal_pix2pix)
    lr=0.0005
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.7,0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.7, 0.999))    
    


    """#2. Dataloaders"""
    print("===> Configuring Dataloaders...")

    dataset = MiddleburyDataLoader('train', augment=True, preprocess_depth=False,h=512,w=1024)
    dataset_test = MiddleburyDataLoader('test', augment=False, preprocess_depth=False,h=512,w=1024)
    dataset_option = "Bosch"
    params['dataset_option'] = dataset_option
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params['bs'],
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        sampler=None)
    test_dataloader = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=10,
        pin_memory=True,
        sampler=None)

    """#3. Create optimizer and criterion"""
    print("===> Creating optimizer and criterion...")
    # Losses
    criterion_GAN = torch.nn.BCEWithLogitsLoss().cuda()
    criterion_content = models_gan.NormalLoss().cuda()
    criterion_pixel = torch.nn.L1Loss().cuda()
    #criterion_GAN = losses.CombinedNew()
    criterion_pixel = losses.CombinedNew()
    #losses.CombinedLoss_2()
    
    """#4. If no errors-> Create Logger object to backup code and log training"""
    logger = Logger(params,deactivate=True)
    Tensor = torch.cuda.FloatTensor
    patch=(1,1024//2**4,1024//2**4)

    
    val_psnr=[]
    val_loss=[]
    
    train_loss_gen=[]

    
    
    for epoch in range(epochs):
        print("------------EPOCH ("+str(epoch+1) +") of ("+str(epochs)+")------------")
        
        #Adjust the LR (a little big brute, but it should work)
        if epoch==15 or epoch==25:
            optimizer_D.param_groups[0]['lr']*=0.5
            optimizer_G.param_groups[0]['lr']*=0.5
        train_loss_gen_intermediate=[]
        for i, batch_data in enumerate(train_dataloader): 
            batch_data = {
                key: val.to(device) for key, val in batch_data.items() if val is not None
            }
            batches_done = epoch * len(train_dataloader) + i+1
            rgbd=torch.cat((batch_data['rgb'],batch_data['d']),1)
            gt=batch_data['gt']
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((rgbd.size(0), *discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((rgbd.size(0), *discriminator.output_shape))), requires_grad=False)
            """
            Train generator
            """
            optimizer_G.zero_grad()
            # GAN loss
            #gen input=(RGB+DEPTH)
            gen_out = generator(rgbd)
            # Pixel-wise loss
            loss_pixel = criterion_pixel(gen_out, gt)
            train_loss_gen_intermediate.append(loss_pixel.item())
            #writer.add_scalar("Pixel_Loss/Train", loss_pixel.item(), batches_done)
            # log learning rate
            gen_lr = optimizer_G.param_groups[0]['lr']
            #writer.add_scalar("Generator_LR", gen_lr, batches_done)
            ## Pixel-wise loss 
            if batches_done < 2:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                optimizer_G.step()
                continue
            # Extract validity predictions from discriminator
            pred_real = discriminator(gt).detach()
            pred_fake = discriminator(gen_out)
            # Adversarial loss (relativistic average GAN)
            loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)
            #writer.add_scalar("GAN_Loss/Train", loss_GAN, batches_done)
            #Total loss for the generator
            loss_G=loss_pixel+loss_GAN
            #writer.add_scalar("Generator_Loss/Train", loss_G, batches_done)
            loss_G.backward()
            optimizer_G.step()
            
            """
            Train discriminator
            """
            optimizer_D.zero_grad()

            pred_real = discriminator(gt)
            pred_fake = discriminator(gen_out.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            #writer.add_scalar("Discriminator_RealLoss/Train", loss_real, batches_done)
            loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
            #writer.add_scalar("Discriminator_FakeLoss/Train", loss_fake, batches_done)
            
            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            #writer.add_scalar("Discriminator_Loss/Train", loss_D, batches_done)
            
            #Discriminator LR
            disc_lr = optimizer_D.param_groups[0]['lr']
            #writer.add_scalar("Discriminator_LR", disc_lr, batches_done)
            
            loss_D.backward()
            optimizer_D.step()
        train_loss_gen.append(sum(train_loss_gen_intermediate)/len(train_loss_gen_intermediate))
        intermediate_val_psnr=[]
        intermediate_val_loss=[]
        with torch.no_grad():
            #generator.eval()
            for i,batch_data in enumerate(test_dataloader):
                batch_data = {
                    key: val.to(device) for key, val in batch_data.items() if val is not None
                }
                rgbd=torch.cat((batch_data['rgb'],batch_data['d']),1)
                gt=batch_data['gt']
                     
                output=generator(rgbd)
                val_current_psnr = losses.psnr_loss(output, batch_data['gt']).item()
                intermediate_val_psnr.append(val_current_psnr)
                intermediate_val_loss.append(criterion_pixel(output,batch_data['gt']).item())
                save_result_row(batch_data, output, "out_"+str(epoch)+"_"+str(i)+".png", folder="outputs/val/gan/")
            #print("Val PSNR(db)->"+str(-sum(val_psnr)/len(val_psnr)))
            val_psnr.append(-sum(intermediate_val_psnr)/len(intermediate_val_psnr))
            val_loss.append(sum(intermediate_val_loss)/len(intermediate_val_loss))

    y_ticker=(max(train_loss_gen)-min(train_loss_gen))/10
    
    graph_utils.make_graph([val_psnr], ['val'], range(0,len(val_psnr)), title="PSNR values train&test", x_label="epochs",
                y_label="PSNR(dB)", x_lim_low=0, x_lim_high=len(val_psnr), show=False, subtitle="", output_dir="outputs/val/gan/",x_ticker=5)
    graph_utils.make_graph([train_loss_gen,val_loss], ['train','val'], range(0,len(val_psnr)), title="Loss values train&test", x_label="epochs",
                y_label="Loss(dB)", x_lim_low=0, x_lim_high=len(val_psnr), show=False, subtitle="", output_dir="outputs/val/gan/",x_ticker=5,y_ticker=y_ticker)

if __name__ == '__main__':
    main()
