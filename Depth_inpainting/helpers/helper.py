import time
import os
import math
import numpy as np
import shutil
import csv
import cv2
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torch
import helpers.graph_utils as graph_utils

#cmap = plt.cm.Greys
cmap = plt.cm.binary

def depth_colorize(depth):
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))
    depth = 255 * cmap(depth)[:, :, :3]  # H, W, C
    
    return depth.astype('uint8')   

def create_output_folder_name(params):
    lr=weight_decay=dataset_option=model_option=mode=""
    if "lr" in params:
        lr=params['lr']
    if "weight_decay" in params:
        weight_decay=params['weight_decay']
    if "dataset_option" in params:
        dataset_option=params['dataset_option']
    if "model_option" in params:
        model_option=params['model_option']
    if "bs" in params:
        bs=params['bs']
    current_time = time.strftime('%Y-%m-%d@%H-%M-%S;mode='+str(params['mode'])+';lr='+str(lr)+";bs="+str(bs)+";wd="+str(weight_decay)+";model="+str(model_option)+";data="+str(dataset_option))
    return os.path.join('results', 'time={}'.format(current_time))

def save_image_color(img_merge, filename):
    #print("--------"+str(filename))
    #print(img_merge.max())
    #print(img_merge.min())
    #print(img_merge)
    cv2.imwrite(filename, img_merge)

def save_test_result(batch_data, output, name,folder="outputs/"):
    resize=transforms.Resize((1080,1920))
    unorm_rgb=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
    unorm_d=transforms.Normalize(mean=[0],std=[1])
    unorm_gt=transforms.Normalize(mean=[0],std=[1])

    rgb=resize(unorm_rgb(batch_data['rgb'][0, ...]))
    depth=resize(unorm_d(batch_data['d']))
    gt=unorm_gt(batch_data['gt'])
    output=resize(unorm_d(output))
    
    
    img_list=[]
    
    rgb = np.squeeze(rgb.data.cpu().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))*255
    img_list.append(rgb)

    depth = depth_colorize(np.squeeze(depth.data.cpu().numpy()))
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)

    #print("OUTPUT SIZE BEFORE--->"+str(output.shape))
    output = depth_colorize(np.squeeze(output.data.cpu().numpy()))
    #output = np.moveaxis(np.squeeze(output.data.cpu().numpy()),0,2)
    #print("OUTPUT SIZE--->"+str(output.shape))
    img_list.append(output)
    
    img_merge = np.hstack([img_list[0], img_list[1], img_list[2]])
    img_merge= img_merge.astype('uint8')
    save_image_color(img_merge,folder+name)
    #print("saving img to "+str(folder+name))
    
    
def save_result_row(batch_data, output, name, folder="outputs/",azure_run=None):
    """Will save a row with the different images rgb+depth+gt+output

    Args:
        batch_data ([type]): [description]
        output ([type]): [description]
    """

    #unorm_rgb = transforms.Normalize(mean=[-0.4409/0.2676, -0.4570/0.2132, -0.3751/0.2345],
    #                         std=[1/0.2676, 1/0.2132, 1/0.2345])
    #unorm_d = transforms.Normalize(mean=[-0.2674/0.1949],
    #                         std=[1/0.1949])
    #unorm_gt = transforms.Normalize(mean=[-0.3073/0.1761],
    #                         std=[1/0.1761])

    unorm_rgb=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
    unorm_d=transforms.Normalize(mean=[0],std=[1])
    unorm_gt=transforms.Normalize(mean=[0],std=[1])

    rgb=unorm_rgb(batch_data['rgb'][0, ...])
    depth=unorm_d(batch_data['d'])
    gt=unorm_gt(batch_data['gt'])
    output=unorm_d(output)
    #depth=unorm_d(batch_data['d'])

    #rgb=batch_data['rgb'][0,...]
    #depth=batch_data['d']
    #gt=batch_data['gt']



    #print("OUTPUT Size------------------->"+str(output.shape))
    img_list=[]
    
    rgb = np.squeeze(rgb.data.cpu().numpy())
    rgb = np.transpose(rgb, (1, 2, 0))*255
    img_list.append(rgb)

    depth = depth_colorize(np.squeeze(depth.data.cpu().numpy()))
    #print("DEPTH SIZE--->"+str(depth.shape))
    img_list.append(depth)

    gt = depth_colorize(np.squeeze(gt.data.cpu().numpy()))
    img_list.append(gt)

    #print("OUTPUT SIZE BEFORE--->"+str(output.shape))
    output = depth_colorize(np.squeeze(output.data.cpu().numpy()))
    #output = np.moveaxis(np.squeeze(output.data.cpu().numpy()),0,2)
    #print("OUTPUT SIZE--->"+str(output.shape))
    img_list.append(output)
    
    img_merge_up = np.hstack([img_list[0], img_list[2]])
    img_merge_down = np.hstack([img_list[1], img_list[3]])
    img_merge = np.vstack([img_merge_up, img_merge_down])
    img_merge= img_merge.astype('uint8')
    if azure_run:
        """
        imgplot = plt.figure()
        plt.imshow(img_merge)
        
        azure_run.log_image(name=name,plot=imgplot)
        """
        pass
    else:
        save_image_color(img_merge,folder+name)
    #print("saving img to "+str(folder+name))

def save_result_individual(img, mode, name, folder="outputs/test/"):
    #unorm_rgb = transforms.Normalize(mean=[-0.4409/0.2676, -0.4570/0.2132, -0.3751/0.2345],
    #                         std=[1/0.2676, 1/0.2132, 1/0.2345])
    #unorm_d = transforms.Normalize(mean=[-0.2674/0.1949],
    #                         std=[1/0.1949])
    #unorm_gt = transforms.Normalize(mean=[-0.3073/0.1761],
    #                         std=[1/0.1761])

    unorm_rgb=transforms.Normalize(mean=[0,0,0],std=[1,1,1])
    unorm_d=transforms.Normalize(mean=[-0.5/1],std=[1])
    unorm_gt=transforms.Normalize(mean=[-0.5/1],std=[1])



    if mode=="rgb":
        img=unorm_rgb(img[0, ...])
        img = np.squeeze(img.data.cpu().numpy())
        img = np.transpose(img, (1, 2, 0))*255
    elif mode=="d":
        img=unorm_d(img)
        img= depth_colorize(np.squeeze(img.data.cpu().numpy()))
    elif mode=="gt":
        img=unorm_gt(img)
        img= depth_colorize(np.squeeze(img.data.cpu().numpy()))
    elif mode=="pred":
        #img=unorm_gt(img)
        img= depth_colorize(np.squeeze(img.data.cpu().numpy()))
    img= img.astype('uint8')
    save_image_color(img, folder+name)


def backup_source_code(backup_directory):
    """Will back-up all the source code without the data, git and other folders

    Args:
        backup_directory (str): path to create the backup
    """
    ignore_hidden = shutil.ignore_patterns(".", "..", ".git*", "*pycache*",
                                           "*build", "*.fuse*", "*_drive_*", "results", "data_*", "external_models", "data_bosch", "outputs", "data_gdem", "data_bosch_reduced","data_jaime_hovitron"
                                           ,"data_middlebury","runs","slurm-*")
    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)
    shutil.copytree('.', backup_directory, ignore=ignore_hidden)


class Result(object):
    """
    Class to hold the results
    """

    def __init__(self):
        self.rmse = 0
        self.mae = 0
        self.gpu_time = 0

    def set_to_worst(self):
        """Set the results to the worst"""
        self.rmse = np.inf
        self.mae = np.inf
        self.gpu_time = 0

    def update(self, rmse, mae, gpu_time):
        """Updates the saved results with the new ones"""
        self.rmse = rmse
        self.mae = mae
        self.gpu_time = gpu_time

    def evaluate(self, repro_v0, target_v0, repro_v2, target_v2, repro_v6, target_v6, repro_v8, target_v8):
        """
        NOT IMPLEMENTED YET
        """


class Logger:
    def __init__(self,params, deactivate=False):
        """Will create a logger process to generate a copy of the actual code structure into a 
            folder with the date of the execution and different parameters
            params-> dict containing important training params such as lr, weight_decay
        """
        self.deactivate=deactivate
        if not self.deactivate:            
            self.fieldnames = ['epoch','loss', 'psnr', 'gpu_time']

            self.output_directory = create_output_folder_name(params)
            self.best_result = Result()
            self.best_result.set_to_worst()
            os.makedirs(self.output_directory)
            self.train_csv = os.path.join(self.output_directory, 'train.csv')
            self.val_csv = os.path.join(self.output_directory, 'val.csv')
            self.best_txt = os.path.join(self.output_directory, 'best.txt')

            print("===> Creating source code backup ...")
            backup_directory = os.path.join(self.output_directory, "code_backup")
            self.backup_directory = backup_directory
            backup_source_code(backup_directory)
            # create new csv files with only header
            with open(self.train_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
            with open(self.val_csv, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()
            print("===> Finished creating source code backup.")
        
    def logToFile(self,epoch, rmse, mae, gpu_time, train=True):
        if not self.deactivate: 
            if train:
                csvfile_name=self.train_csv
            else:
                csvfile_name=self.val_csv

            with open(csvfile_name, 'a') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writerow({
                    'epoch': epoch,
                    'loss': rmse,
                    'psnr': mae,
                    'gpu_time': gpu_time,
                    })
    def generateTrainingGraphs(self, subtitle=""):
        if not self.deactivate: 
            #open training log
            x_axis=[]
            train_loss=[]
            train_psnr=[]
            val_loss=[]
            val_psnr=[]
            
            with open(self.train_csv, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader)  # skip the header
                for row in csvreader:
                    x_axis.append(row[0])
                    train_loss.append(float(row[1]))
                    train_psnr.append(float(row[2]))
                    
            with open(self.val_csv, newline='') as csvfile:
                csvreader = csv.reader(csvfile, delimiter=',')
                next(csvreader)  # skip the header
                for row in csvreader:
                    #x_axis.append(row[0])
                    val_loss.append(float(row[1]))
                    val_psnr.append(float(row[2]))

            graph_utils.make_graph([train_loss, val_loss], ['train', 'val'], x_axis, title="Loss values train&test", x_label="epochs",
                        y_label="Loss", x_lim_low=0, x_lim_high=len(x_axis), show=False, subtitle=subtitle, output_dir=self.output_directory,x_ticker=5,y_ticker=0.1)
            graph_utils.make_graph([train_psnr, val_psnr], ['train', 'val'], x_axis, title="PSNR values train&test", x_label="epochs",
                        y_label="PSNR(dB)", x_lim_low=0, x_lim_high=len(x_axis), show=False, subtitle=subtitle, output_dir=self.output_directory,x_ticker=5)

        
    
    """def conditional_save_img_comparison(self, mode, i, ele, pred, epoch, predrgb=None, predg=None, extra=None, extra2=None, extrargb=None):
        # save 8 images for visualization
        skip = 100
        if i == 0:
            self.img_merge = merge_into_row(
                ele, pred, predrgb, predg, extra, extra2, extrargb)
        elif i % skip == 0 and i < 8 * skip:
            row = merge_into_row(ele, pred, predrgb, predg,
                                 extra, extra2, extrargb)
            self.img_merge = add_row(self.img_merge, row)
        elif i == 8 * skip:
            filename = self._get_img_comparison_name(mode, epoch)
            save_image(self.img_merge, filename)"""
            
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=4, min_lr=1e-6, factor=0.5
    ):
        """
        new_lr = old_lr * factor
        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)