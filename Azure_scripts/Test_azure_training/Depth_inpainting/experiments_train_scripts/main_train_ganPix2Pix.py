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
    #generator = models.Pix2PixGanGenerator()
    generator = models.InceptionLikeModel()
    discriminator = models.Pix2PixGanDiscriminator()
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        
    #generator.apply(models.weights_init_normal_pix2pix)
    discriminator.apply(models.weights_init_normal_pix2pix)
    lr=0.0002
    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.9,0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))    
    


    """#2. Dataloaders"""
    print("===> Configuring Dataloaders...")

    dataset = MiddleburyDataLoader('train', augment=True, preprocess_depth=False,h=1024,w=1024)
    dataset_test = MiddleburyDataLoader('test', augment=False, preprocess_depth=False,h=1024,w=1024)
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
    # Loss functions
    criterion_GAN = torch.nn.L1Loss()
    criterion_pixelwise = losses.CombinedNew()
    
    #losses.CombinedLoss_2()
    
    """#4. If no errors-> Create Logger object to backup code and log training"""
    logger = Logger(params)
    Tensor = torch.cuda.FloatTensor
    patch=(1,1024//2**4,1024//2**4)

    for epoch in range(epochs):
        print("------------EPOCH ("+str(epoch+1) +") of ("+str(epochs)+")------------")
        #generator.train()
        #discriminator.train()
        
        #Adjust the LR (a little big brute, but it should work)
        if epoch==15 or epoch==25:
            optimizer_D.param_groups[0]['lr']*=0.5
            optimizer_G.param_groups[0]['lr']*=0.5
        
        for i, batch_data in enumerate(train_dataloader): 
            batch_data = {
                key: val.to(device) for key, val in batch_data.items() if val is not None
            }
            
            rgbd=torch.cat((batch_data['rgb'],batch_data['d']),1)
            gt=batch_data['gt']
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((rgbd.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((rgbd.size(0), *patch))), requires_grad=False)
            """
            Train generator
            """
            optimizer_G.zero_grad()
            # GAN loss
            #gen input=(RGB+DEPTH)
            #gen_out = generator(rgbd)
            gen_out = generator(batch_data)
            
            #disc input=gen_out+(RGB+DEPTH)
            pred_fake = discriminator(gen_out, rgbd)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(gen_out, gt)

            # Total loss
            loss_G = loss_GAN + loss_pixel

            loss_G.backward()

            optimizer_G.step()
            """
            Train discriminator
            """
            optimizer_D.zero_grad()

            # Real loss
            pred_real = discriminator(gt, rgbd)
            loss_real = criterion_GAN(pred_real, valid)

            # Fake loss
            pred_fake = discriminator(gen_out.detach(), rgbd)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            loss_D.backward()
            optimizer_D.step()
        intermediate_val_psnr=[]
        intermediate_val_loss=[]
        with torch.no_grad():
            #generator.eval()
            val_psnrs=[]
            for i,batch_data in enumerate(test_dataloader):
                batch_data = {
                    key: val.to(device) for key, val in batch_data.items() if val is not None
                }
                rgbd=torch.cat((batch_data['rgb'],batch_data['d']),1)
                gt=batch_data['gt']
                     
                output=generator(batch_data)
                val_current_psnr = losses.psnr_loss(output, batch_data['gt']).item()
                val_psnrs.append(val_current_psnr)
                intermediate_val_psnr.append(val_current_psnr)
                #intermediate_val_loss.append(criterion_pixel(output,batch_data['gt']).item())
                save_result_row(batch_data, output, "out_"+str(epoch)+".png", folder="outputs/val/pix2pix/")
            print("Val PSNR(db)->"+str(-sum(val_psnrs)/len(val_psnrs)))
        
    
if __name__ == '__main__':
    main()
