"""
This file defines different loss functions to try
Author:Alejandro
"""


import torch
import torch.nn as nn
import kornia.losses as k_losses
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

psnr_loss=k_losses.PSNRLoss(1)

MSE=nn.MSELoss()



class CombinedNewForLossLandscape(nn.Module):
    def __init__(self,alpha1=0.85,alpha2=0.85):
        super(CombinedNewForLossLandscape,self).__init__()
        self.SSIM=SSIMLoss()
        self.L2=MaskedMSELoss() 
        self.L1=MaskedL1Loss()
        self.alpha1=alpha1
        self.alpha2=alpha2
        self.MS_SSIM = MS_SSIM(data_range=1., size_average=True, channel=1)
        #self.SSIM=SSIM(data_range=1, size_average=True, channel=1) # channel=1 for grayscale images


    def forward(self,pred,target):
        x=pred
        y=target['gt']
        return self.alpha1*(self.alpha2*self.L2(x,y)+(1-self.alpha2)*self.L1(x,y))+(1-self.alpha1)*(1-self.MS_SSIM(x, y))


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    
class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
       
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff**2).mean()
        return self.loss


class CombinedNew(nn.Module):
    def __init__(self,alpha1=0.85,alpha2=0.85):
        super(CombinedNew,self).__init__()
        self.SSIM=SSIMLoss()
        self.L2=MaskedMSELoss() 
        self.L1=MaskedL1Loss()
        self.alpha1=alpha1
        self.alpha2=alpha2
        self.MS_SSIM = MS_SSIM(data_range=1., size_average=True, channel=1)
        #self.SSIM=SSIM(data_range=1, size_average=True, channel=1) # channel=1 for grayscale images


    def forward(self,pred,target):
        x=pred
        y=target
        return self.alpha1*(self.alpha2*self.L2(x,y)+(1-self.alpha2)*self.L1(x,y))+(1-self.alpha1)*(1-self.MS_SSIM(x, y))
    

class BerHuLoss(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold
    
    def forward(self, fake, real):
        mask = real>0
        if not fake.shape == real.shape:
            _,_,H,W = real.shape
            fake = F.upsample(fake, size=(H,W), mode='bilinear')
        fake = fake * mask
        diff = torch.abs(real-fake)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()#[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss

class BerHuLoss_2(nn.Module):
    def __init__(self, threshold=0.2):
        super(BerHuLoss, self).__init__()
        self.threshold = threshold
    
    def forward(self, gt, pred):
        mask = gt>0
        if not pred.shape == gt.shape:
            _,_,H,W = gt.shape
            pred = F.upsample(pred, size=(H,W), mode='bilinear')
        pred = pred * mask
        diff = torch.abs(gt-pred)
        delta = self.threshold * torch.max(diff).data.cpu().numpy()#[0]

        part1 = -F.threshold(-diff, -delta, 0.)
        part2 = F.threshold(diff**2 - delta**2, 0., -delta**2.) + delta**2
        part2 = part2 / (2.*delta)

        loss = part1 + part2
        loss = torch.sum(loss)
        return loss

def berhu(input, target, mask, apply_log=False):
    threshold = 0.2
    if apply_log:
        input = torch.log(1 + input)
        target = torch.log(1 + target)
    absdiff = torch.abs(target - input) * mask
    C = threshold * torch.max(absdiff).item()
    loss = torch.mean(torch.where(absdiff <= C,
                                  absdiff,
                                  (absdiff * absdiff + C * C) / (2 * C)))
    return loss


    
class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()

    def forward(self,x,y):
        criterion = nn.MSELoss()
        eps = 1e-6
        loss = torch.sqrt(criterion(x, y) + eps)
        return loss


class ScaleInvariantLoss(torch.nn.Module):
    def __init__(self):
        super(ScaleInvariantLoss,self).__init__()

    def forward(self,x,y):
        log_diff=torch.log(x)-torch.log(y)
        #print("LOG DIFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF-->"+str(log_diff.size))
        num_pixels = log_diff.size(3)*log_diff.size(2)
        return torch.sqrt(torch.sum(torch.square(log_diff))/num_pixels-torch.square(torch.sum(log_diff))/torch.square(num_pixels))
        #return np.sqrt(np.sum(np.square(log_diff)) / num_pixels - np.square(np.sum(log_diff)) / np.square(num_pixels))

class CombinedLoss_2(nn.Module):
    """
    Combines SSIM with BerHu
    """
    def __init__(self, alpha=0.7):
        super(CombinedLoss_2,self).__init__()
        self.SSIM=SSIMLoss()
        self.BerHu=BerHuLoss()
        self.alpha=alpha

    def forward(self,x,y):
        x=x
        y=y
        
        return self.alpha*(self.BerHu(x,y))+(1-self.alpha)*(self.SSIM(x, y))

class BerHuLoss_Occlusions(nn.Module):
    """
    Berhu loss for the general image, and also checks spetially the "oclussion areas" (white)
    """     
    def __init__(self):
        super(BerHuLoss_Occlusions,self).__init__()
        self.BerHu=BerHuLoss()

    def forward(self,x,y,occ_mask, alpha=0.5):
        x=x*255
        y=y*255
       
        x_new=x*occ_mask
        y_new=y*occ_mask
        loss_1=(self.BerHu(x,y))
        loss_2=(self.BerHu(x_new, y_new)) 
        loss_2=self.BerHu(x,y) if torch.isnan(loss_2) else loss_2
        loss=alpha*loss_1+(1-alpha)*loss_2
        return loss


class SSIMLoss(torch.nn.Module):
    def __init__(self):
        super(SSIMLoss,self).__init__()

    def forward(self,x,y):

        return k_losses.ssim_loss(x,y,11)


class CombinedLoss(torch.nn.Module):
    """
    Combines SSIM with RMSE
    """
    def __init__(self, alpha=0.8):
        super(CombinedLoss,self).__init__()
        self.SSIM=SSIMLoss()
        self.RMSE=RMSELoss()
        self.alpha=alpha

    def forward(self,x,y):
        x=x
        y=y
        return self.alpha*(self.RMSE(x,y))+(1-self.alpha)*(self.SSIM(x, y))
    
class CombinedLossL1(torch.nn.Module):
    """
    Combines SSIM with RMSE
    """
    def __init__(self, alpha=0.8):
        super(CombinedLoss,self).__init__()
        self.SSIM=SSIMLoss()
        self.RMSE=RMSELoss()
        self.alpha=alpha

    def forward(self,x,y):
        x=x
        y=y
        return self.alpha*(self.RMSE(x,y))+(1-self.alpha)*(self.SSIM(x, y))

class ExperimentalLoss_5(nn.Module):
    """
    Trying to create a better loss function based on second derivative 
    """
    def __init__(self):
        super(ExperimentalLoss_5, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        def second_derivative(x):
            assert x.dim(
            ) == 4, "expected 4-dimensional data, but instead got {}".format(
                x.dim())
            horizontal = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, 1:-1, :
                                                        -2] - x[:, :, 1:-1, 2:]
            vertical = 2 * x[:, :, 1:-1, 1:-1] - x[:, :, :-2, 1:
                                                    -1] - x[:, :, 2:, 1:-1]
            der_2nd = horizontal.abs() + vertical.abs()
            return der_2nd.mean()
        G1= second_derivative(pred)
        G2= second_derivative(target)
        G1=torch.tensor(G1).to(torch.device("cuda"))
        G2=torch.tensor(G2).to(torch.device("cuda"))
        self.loss = torch.tensor(G2-G1)
        return self.loss

