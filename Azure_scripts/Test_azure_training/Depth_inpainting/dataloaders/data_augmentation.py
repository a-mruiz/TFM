import torch 
import torch.nn as nn
import kornia.augmentation as aug
from random import randint
#import albumentations as A




class DataAugmentation(nn.Module):
    """Module to perform data augmentation using Kornia on torch image tensors."""

    def __init__(self,size=(704,1280)) -> None:
        super().__init__()
        #self.transforms = nn.Sequential(
        #    aug.RandomHorizontalFlip(),
        #    aug.RandomCrop(size=(300,530)),
        #    aug.RandomAffine((-20, 20),(-20, 20),(-20, 20), p=1),
        #    aug.RandomEqualize(p=1),
        #    aug.RandomPerspective(p=1),
        #    #aug.RandomResizedCrop(p=1),
        #    #aug.RandomRotation(p=1),
        #    aug.RandomSharpness(p=1)
        #)
        
        #self.transforms_2 = aug.AugmentationSequential(
        #    aug.RandomHorizontalFlip(),
        #    #aug.RandomCrop(size=(300,530)),
        #    #aug.RandomAffine((-20, 20),(-20, 20),(-20, 20), p=1),
        #    #aug.RandomEqualize(p=1),
        #    #aug.RandomPerspective(p=1),
        #    ##aug.RandomResizedCrop(p=1),
        #    ##aug.RandomRotation(p=1),
        #    #aug.RandomSharpness(p=1),
        #    data_keys=["rgb", "gt", "d"],  # Just to define the future input here.
        #    return_transform=False,
        #    same_on_batch=True
        #    )
        #
        
        hor_flip=aug.RandomHorizontalFlip(p=1,keepdim=True)
        ver_flip=aug.RandomVerticalFlip(p=1,keepdim=True)
        self.crop=aug.RandomCrop(size=size,keepdim=True)
        self.affine=aug.RandomAffine((0.8, 1),(0.8, 1),(0.8, 1), p=1,keepdim=True)
        self.eq=aug.RandomEqualize(p=1)
        self.perspective=aug.RandomPerspective(0.2,p=1, keepdim=True)
        self.crop_2=aug.RandomResizedCrop(size=size,p=1, keepdim=True)
        self.rotation=aug.RandomRotation(p=1,degrees=[-20, 20], keepdim=True)
        self.sharp=aug.RandomSharpness(p=1, keepdim=True, sharpness=0.5)
        self.jitter = aug.ColorJitter(0.1, 0.1, 0.1, 0.1)
        
        self.transforms=[hor_flip, self.perspective, self.crop, self.rotation]
        self.transforms=[hor_flip, ver_flip, self.rotation, self.crop_2,self.sharp,self.affine]
        #self.transforms=[self.crop_2]
    @torch.no_grad()  # disable gradients for effiency
    def forward(self, input):
        rgb = input['rgb']
        d = input['d']
        gt = input['gt']
        g=input['g']
        
        #Apply the transformations with _params to use the same transform
        #on all the images!!!
        for t in self.transforms:
            if randint(0,10)>5:
                rgb=t(rgb)
                d=t(d, params=t._params)
                gt=t(gt, params=t._params)
                g=t(g, params=t._params)
        
        if torch.max(rgb)==torch.min(rgb) or torch.max(gt)==torch.min(gt) or torch.max(g)==torch.min(g) or torch.max(d)==torch.min(d):
            rgb = input['rgb']
            d = input['d']
            gt = input['gt']
            g=input['g']
    
        return rgb, d,gt, g
"""
class DataAugmentation_Albu(nn.Module):

    def __init__(self, h, w) -> None:
        super().__init__()
        self.h=h
        self.w=w
        hor_flip = A.Compose([A.HorizontalFlip(p=1)],
            additional_targets={
                'image':'image',
                'depth':'image',
                'gt':'image'
            })
        rotate = A.Compose([A.ShiftScaleRotate(p=1)],
            additional_targets={
                'image':'image',
                'depth':'image',
                'gt':'image'
            })
        vert_flip = A.Compose([A.VerticalFlip(p=1)],
            additional_targets={
                'image':'image',
                'depth':'image',
                'gt':'image'
            })

        crop = A.Compose([A.RandomResizedCrop(h,w,p=1)],
            additional_targets={
                    'image':'image',
                    'depth':'image',
                    'gt':'image'
            })

        elastic_transform= A.Compose([A.ElasticTransform(p=1)],
            additional_targets={
                    'image':'image',
                    'depth':'image',
                    'gt':'image'
            })
        perspective_transform = A.Compose([A.Perspective(p=1)],
            additional_targets={
                    'image':'image',
                    'depth':'image',
                    'gt':'image'
            })
        
        self.transforms=[hor_flip, rotate,vert_flip, crop, elastic_transform,perspective_transform]


    @torch.no_grad()  # disable gradients for effiency
    def forward(self, input):
        rgb = input['rgb']
        d = input['d']
        gt = input['gt']
        g=input['g']
        
        #Apply the transformations with _params to use the same transform
        #on all the images!!!
        for t in self.transforms:
            if randint(0,10)>7:
                composed=t(image=rgb,depth=d,gt=gt)
                rgb=composed['image']
                d=composed['depth']
                gt=composed['gt']
        return rgb, d,gt, g
"""