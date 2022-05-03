"""
This file defines the dataloader to load the images from GDEM DATASET to feed the model
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
from dataloaders.occlusions_removal import slow_remove_occlusions

def read_rgb(path):
    rgb=cv2.imread(path)
    #rgb=np.array(Image.open(path))
    print("RGB max value->"+str(rgb.max()))
    print("RGB min value->"+str(rgb.min()))
    #rgb = (rgb - np.min(rgb)) / (np.max(rgb) - np.min(rgb))*255
    #print("SHAPE JUST READED RGGB----->"+str(rgb.shape))
    #gray = np.array(Image.fromarray(rgb).convert('L'))
    #gray = np.expand_dims(gray, -1)
    #print(rgb.shape)
    #print("RGB->"+str(rgb.shape))
    return rgb,rgb


def read_depth(path, preprocess_depth=False):
    depth=np.array(cv2.imread(path,cv2.IMREAD_UNCHANGED))#.astype(np.int16)
    
    depth = (depth - np.min(depth)) / (np.max(depth) - np.min(depth))*255
    #depth = cv2.convertScaleAbs(depth)
    depth=cv2.imread(path)
    depth = np.array(Image.fromarray(depth).convert('L'))
    
    #print("DEPTH READ->"+str(depth.max()))
    #depth=np.array(depth).astype(np.uint8)
    #depth = np.array(Image.fromarray(depth).convert('L'))
    #depth = np.array(Image.open(path).convert("L"))*62
    if preprocess_depth:
        depth=slow_remove_occlusions(depth)
    depth=np.expand_dims(depth, 2)
    #print("Depth min value->"+str(depth.min()))
    #print("Depth max value->"+str(depth.max()))

    #depth=np.expand_dims(np.load(path)['arr_0'],2)
    #print(depth.shape)
    #print("DEPTH->"+str(depth.shape))
    return depth


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


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


def get_paths(train_or_test):
    """Returns the paths for the data files

    Args:
        train_or_test (["train"/"test"]): To load files from train or test directory

    Returns:
        [dict]: {"rgb": paths_rgb, "d": paths_d, "gt": paths_gt}
    """
    if train_or_test == "train":
        path_d = "data_jaime/train/sparse/*.png"
        path_gt = "data_jaime/train/gt/*.png"
        path_rgb = "data_jaime/train/rgb/*.png"
    elif train_or_test == "test":
        path_d = "data_jaime/test/depth/*.png"
        path_rgb = "data_jaime/test/rgb/*.png"

    paths_d = sorted(glob.glob(path_d))
    paths_rgb = sorted(glob.glob(path_rgb))

    if len(paths_d) == 0 or len(paths_rgb) == 0 :
        raise (RuntimeError(
            "ALEJANDRO ERROR--->No images in some or all of the paths of the dataloader.\n Len(RGB)="+str(len(paths_rgb))+"\n Len(Depth)="+str(len(paths_d))+"\n" ))
    paths = {"rgb": paths_rgb, "d": paths_d}
    return paths


class HoviDataLoader(data.Dataset):
    def __init__(self, train_or_test):
        """Inits the Hovitron dataloader

        Args:
            train_or_test (string): "train" or "test" to use different data (train and inference)
        """
        self.train_or_test = train_or_test
        self.paths = get_paths(train_or_test)

    def __getraw__(self, index):
        rgb, gray = read_rgb(self.paths['rgb'][index]) if (
            self.paths['rgb'][index] is not None) else None
        depth = read_depth(self.paths['d'][index], False) if (
            self.paths['d'][index] is not None) else None
        return rgb, gray, depth

    def __getitem__(self, index):
        rgb, gray, sparse = self.__getraw__(index)
        resize=transforms.Resize((1024,1024))
        
        preprocess_rgb=transforms.Compose([
            transforms.Resize((1024,1024)),
            #transforms.Normalize(mean=[0.4409, 0.4570, 0.3751], std=[0.2676, 0.2132, 0.2345])          
            transforms.Normalize(mean=[0,0,0],std=[1,1,1])
        ])
        preprocess_depth=transforms.Compose([
            transforms.Resize((1024,1024)),
            #transforms.Normalize(mean=[0.2674], std=[0.1949]),
            transforms.Normalize(mean=[0],std=[1])
         
        ])
        
        rgb=ToTensor(rgb).float()/255
        sparse=ToTensor(sparse).float()/255
        gray=ToTensor(gray).float()/255
        items={'rgb':preprocess_rgb(rgb), 'd':preprocess_depth(sparse), 'gt':preprocess_depth(sparse), 'gray':resize(gray)}
        
        #items = {"rgb": preprocess_rgb(ToTensor(rgb/255).float()), "d": preprocess_depth(ToTensor(sparse/255).float()), 'gt':preprocess_depth(ToTensor(gt/255).float()), 'g':resize(ToTensor(gray/255).float())}
        
        #def to_float_tensor(x): return resize(ToTensor(x).float())
        #candidates = {"rgb": rgb/255, "d": sparse/255, 'gt':gt/255, 'g':gray/255}
        #items = {
        #    key: to_float_tensor(val)
        #    for key, val in candidates.items() if val is not None
        #}
        return items

    def __len__(self):
        return len(self.paths['rgb'])
