"""
This file defines the dataloader to load the images from Middlebury DATASET to feed the model
Author:Alejandro
"""

import glob
import numpy as np
from PIL import Image
import os
from pyrsistent import v
import torch
import torch.utils.data as data
import cv2
from torchvision import transforms
#from dataloaders.data_augmentation import DataAugmentation,DataAugmentation_Albu
from .occlussion import slow_remove_occlusions
import random
import torch.nn.functional as F

def read_rgb(path):
    rgb = cv2.resize((cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)),(1024,1024))#/255.0
    gray = np.array(Image.fromarray(cv2.imread(path)).convert('L'))#/255.0
    gray = np.expand_dims(gray, -1)
    return rgb, gray

def read_depth(path, preprocess_depth=False):
    """
    Reads the depth image
    """
    depth_original=np.array(cv2.resize(cv2.imread(path,cv2.IMREAD_GRAYSCALE),(1024,1024)))
    depth=np.expand_dims(depth_original,2)
    return depth,depth_original.copy()


def ToTensor(img):
    """Convert a ``numpy.ndarray`` to tensor.

    Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W).

    # swap color axis because
    # numpy image: H x W x C
    # torch image: C x H x W

    Convert a ``numpy.ndarray`` to tensor.

    Args:
        img (numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    def _is_numpy_image(img):
        return isinstance(img, np.ndarray) and (img.ndim in {2, 3})

    if not (_is_numpy_image(img)):
        raise RuntimeError('AlEJANDRO ERROR-->img should be ndarray or dimesions are wrong. Got type (' +
                           str(type(img))+') and dimensions ('+str(img.shape)+')')

    if isinstance(img, np.ndarray):
        # handle numpy array
        if img.ndim == 3:
            img = torch.from_numpy(img.transpose((2, 0, 1)).copy())
        elif img.ndim == 2:
            img = torch.from_numpy(img.copy())
        else:
            raise RuntimeError('AlEJANDRO ERROR-->img should be ndarray or dimesions are wrong. Got type (' +
                               str(type(img))+') and dimensions ('+str(img.shape)+')')
        return img


def get_paths():
    """Returns the paths for the data files

    Args:
        train_or_test (["train"/"test"]): To load files from train or test directory

    Returns:
        [dict]: {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    """

    path_gt = "test_data/middlebury/gt/*.png"
    path_rgb = "test_data/middlebury/rgb/*.png"

    paths_rgb = sorted(glob.glob(path_rgb))
    paths_gt = sorted(glob.glob(path_gt))
    if len(paths_rgb) == 0 or len(paths_gt) == 0:
        raise (RuntimeError(
            "ALEJANDRO ERROR--->No images in some or all of the paths of the dataloader.\n Len(RGB)="+str(len(paths_rgb))+"\n Len(gt)="+str(len(paths_gt))+"\n Path rgb->"+str(path_rgb)))
    paths = {"rgb": paths_rgb, 'gt': paths_gt}
    return paths


class MiddleburyDataLoader(data.Dataset):
    def __init__(self):
        """Inits the MiddleburyDataLoader dataloader

        Args:
            train_or_test (string): "train" or "test" to use different data (train and inference)
        """
        self.paths = get_paths()
        #self.data_aug=DataAugmentation(size=(1024,1024))

    def __getraw__(self, index):
        rgb, gray = read_rgb(self.paths['rgb'][index]) if (
            self.paths['rgb'][index] is not None) else None
        gt,gt_processed = read_depth(self.paths['gt'][index],True) if (
            self.paths['gt'][index] is not None) else None
        return rgb, gray, gt,gt_processed

    def __getitem__(self, index):
        rgb, gray, gt,gt_processed = self.__getraw__(index)
             
        DIAMOND_KERNEL_11 = np.asarray(
        [
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        ], dtype=np.uint8)         
 
        mask=(gt==0).astype(np.uint8)
        
        degraded_mask = cv2.dilate(mask, DIAMOND_KERNEL_11)
        
        degraded_mask=np.expand_dims(degraded_mask,2)
        degraded_depth=(1-degraded_mask)*gt
                
                
        items = {"rgb": rgb, "d": degraded_depth, 'gt':gt, 'g':gray}
    
        h=1024
        w=1024
        
        to_pil_image = transforms.ToPILImage()
        
        resize=transforms.Resize((h,w))    
        preprocess_rgb=transforms.Compose([
            #transforms.Resize((h,w)),
            #transforms.Normalize(mean=[0.4409, 0.4570, 0.3751], std=[0.2676, 0.2132, 0.2345])
            transforms.Normalize(mean=[0, 0, 0], std=[1,1,1])          
        ])
        preprocess_depth=transforms.Compose([
            #transforms.Resize((h,w)),
            #transforms.Normalize(mean=[0.2674], std=[0.1949])  
            transforms.Normalize(mean=[0], std=[1])          
        
        ])
        preprocess_gt=transforms.Compose([
            #transforms.Resize((h,w)),
            #transforms.Normalize(mean=[0.3073], std=[0.1761])
            transforms.Normalize(mean=[0], std=[1])             
        ])
        

        gt=slow_remove_occlusions(gt_processed)
        gt=np.expand_dims(gt,2)
        
        #print("init rgb shape->"+str(rgb.shape))
        rgb=ToTensor(rgb).float()/255
        sparse=ToTensor(degraded_depth).float()/255
        gt=ToTensor(gt).float()/255
        gray=ToTensor(gray).float()/255

        #print("med rgb shape->"+str(rgb.shape))
        #sparse=ToTensor(degraded_depth).float()/255
        #gt=ToTensor(gt).float()/255
        #gray=ToTensor(gray).float()/255
        
        
        #items={'rgb':preprocess_rgb(rgb), 'd':sparse, 'gt':preprocess_gt(gt), 'g':resize(gray), 'occlusion_mask':occlusion_mask}
        items={'rgb':preprocess_rgb(rgb), 'd':preprocess_depth(sparse), 'gt':preprocess_gt(gt), 'g':preprocess_depth(gray)}
    
        return items

    def __len__(self):
        return len(self.paths['rgb'])
