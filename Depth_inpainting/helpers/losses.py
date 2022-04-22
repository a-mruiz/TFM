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
from loss_landscapes.metrics.metric import Metric
from loss_landscapes.model_interface.model_parameters import rand_u_like
from loss_landscapes.model_interface.model_wrapper import ModelWrapper
psnr_loss=k_losses.PSNRLoss(1)

MSE=nn.MSELoss()



class LossForLandscape(Metric):
    """ Computes the loss for the loss-landscapes library to plot later the landscape """
    def __init__(self, loss_fn, inputs: torch.Tensor, target: torch.Tensor):
        super().__init__()
        self.loss_fn = loss_fn
        self.inputs = inputs
        self.target = target

    def __call__(self, model_wrapper: ModelWrapper) -> float:
        return self.loss_fn(model_wrapper.forward(self.inputs), self.target).item()


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
    
class KorniaPSNRLoss(nn.Module):
    def __init__(self, max_value=1):
        super(KorniaPSNRLoss, self).__init__()
        self.max_value = max_value
    
    def forward(self, real, fake):

        unorm_rgb = transforms.Normalize(mean=[-0.4409/0.2676, -0.4570/0.2132, -0.3751/0.2345],
                             std=[1/0.2676, 1/0.2132, 1/0.2345])
        unorm_d = transforms.Normalize(mean=[-0.2674/0.1949],
                                std=[1/0.1949])
        unorm_gt = transforms.Normalize(mean=[-0.3073/0.1761],
                                std=[1/0.1761])

        rgb=unorm_rgb(batch_data['rgb'][0, ...])
        depth=unorm_d(batch_data['d'])
        gt=unorm_gt(batch_data['gt'])


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

class ExperimentalLoss(nn.Module):
    """
    Trying to create a better loss function based on Sobel operator and borders
    """
    def __init__(self):
        super(ExperimentalLoss, self).__init__()

    def forward(self, pred, target, alpha=0.2):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        MAE = (diff**2).mean()
        pred_aux=torch.squeeze(pred).cpu().detach().numpy()
        target_aux=torch.squeeze(target).cpu().detach().numpy()
        #print(pred_aux.shape)
        pred_blur = cv2.GaussianBlur(pred_aux,(3,3),0)
        #pred_gray = cv2.cvtColor(pred_blur, cv2.COLOR_RGB2GRAY)
        target_blur = cv2.GaussianBlur(target_aux,(3,3),0)
        #target_gray = cv2.cvtColor(target_blur, cv2.COLOR_BGR2GRAY)
        canny_pred=cv2.Sobel(pred_blur, cv2.CV_32F, dx=1, dy=1)
        canny_target=cv2.Sobel(target_blur, cv2.CV_32F, dx=1, dy=1)

        
        #sobel_pred=ndimage.sobel(pred.cpu().detach().numpy())
        #sobel_tarjet=ndimage.sobel(target.cpu().detach().numpy())
        G= canny_target-canny_pred
        G=torch.tensor(G).to(torch.device("cuda"))
        self.loss = alpha*G + (1-alpha)*MAE
        return self.loss.mean()

class ExperimentalLoss_1(nn.Module):
    """
    Trying to create a better loss function based on Sobel operator and borders (improved, by taking x and y sobel independently)
    """
    def __init__(self):
        super(ExperimentalLoss_1, self).__init__()

    def forward(self, pred, target, alpha=0.2):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        MAE = (diff**2).mean()
        pred_aux=torch.squeeze(pred).cpu().detach().numpy()
        target_aux=torch.squeeze(target).cpu().detach().numpy()
        #print(pred_aux.shape)
        pred_blur = cv2.GaussianBlur(pred_aux,(3,3),0)
        #pred_gray = cv2.cvtColor(pred_blur, cv2.COLOR_RGB2GRAY)
        target_blur = cv2.GaussianBlur(target_aux,(3,3),0)
        #target_gray = cv2.cvtColor(target_blur, cv2.COLOR_BGR2GRAY)
        canny_pred_x=cv2.Sobel(pred_blur, cv2.CV_32F, dx=1, dy=0)
        canny_pred_y=cv2.Sobel(pred_blur, cv2.CV_32F, dx=0, dy=1)
        
        canny_target_x=cv2.Sobel(target_blur, cv2.CV_32F, dx=1, dy=0)
        canny_target_y=cv2.Sobel(target_blur, cv2.CV_32F, dx=0, dy=1)

        G = ((canny_pred_x-canny_target_x)**2 - (canny_pred_y-canny_target_y)**2).mean()
        
        #sobel_pred=ndimage.sobel(pred.cpu().detach().numpy())
        #sobel_tarjet=ndimage.sobel(target.cpu().detach().numpy())
        #G= canny_target-canny_pred
        G=torch.tensor(G).to(torch.device("cuda"))
        self.loss = alpha*G + (1-alpha)*MAE
        return self.loss.mean()

class ExperimentalLoss_2(nn.Module):
    """
    Trying to create a better loss function based on Canny operator and borders 
    """
    def __init__(self):
        super(ExperimentalLoss_2, self).__init__()

    def forward(self, pred, target):
        alpha=0.2
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        MAE = (diff**2).mean()
        print(pred.shape)
        pred_aux=torch.squeeze(pred).cpu().detach()
        target_aux=torch.squeeze(target).cpu().detach()
        canny_pred=feature.canny(pred_aux.numpy().astype('float32'), sigma=3)
        canny_target=feature.canny(pred_aux.numpy().astype('float32'), sigma=3)
        
        #canny_pred=feature.canny(torch.squeeze(pred).cpu().detach().numpy().astype('float32'), sigma=3)
        #canny_target=feature.canny(torch.squeeze(target).cpu().detach().numpy().astype('float32'), sigma=3)
        print(canny_pred)
        G= canny_target^canny_pred
        G=torch.tensor(G).to(torch.device("cuda"))
        self.loss = alpha*G + (1-alpha)*MAE
        return self.loss.mean()

class ExperimentalLoss_3(nn.Module):
    """
    Trying to create a better loss function based on Laplace operator and borders 
    """
    def __init__(self):
        super(ExperimentalLoss_3, self).__init__()

    def forward(self, pred, target):
        alpha=0.2
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        MAE = (diff**2).mean()
        #print(pred.shape)
        pred_aux=torch.squeeze(pred).cpu().detach().numpy()
        target_aux=torch.squeeze(target).cpu().detach().numpy()
        canny_pred=cv2.Laplacian(pred_aux, cv2.CV_32F)
        canny_target=cv2.Laplacian(target_aux, cv2.CV_32F)
        
        #canny_pred=feature.canny(torch.squeeze(pred).cpu().detach().numpy().astype('float32'), sigma=3)
        #canny_target=feature.canny(torch.squeeze(target).cpu().detach().numpy().astype('float32'), sigma=3)
        #print(canny_pred)
        G= canny_target-canny_pred
        G=torch.tensor(G).to(torch.device("cuda"))
        self.loss = alpha*G + (1-alpha)*MAE
        return self.loss.mean()

class ExperimentalLoss_4(nn.Module):
    """
    Trying to create a better loss function based on Canny operator and borders 
    """
    def __init__(self):
        super(ExperimentalLoss_4, self).__init__()

    def forward(self, pred, target):
        alpha=0.2
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        MAE = (diff**2).mean()
        #print(pred.shape)
        pred_aux=torch.squeeze(pred).cpu().detach().numpy()
        target_aux=torch.squeeze(target).cpu().detach().numpy()
        pred_blur = cv2.GaussianBlur(pred_aux,(3,3),0)
        pred_gray = cv2.cvtColor(pred_blur, cv2.COLOR_BGR2GRAY)
        target_blur = cv2.GaussianBlur(target_aux,(3,3),0)
        target_gray = cv2.cvtColor(target_blur, cv2.COLOR_BGR2GRAY)
        canny_pred=cv2.Laplacian(pred_gray, cv2.CV_32F)
        canny_target=cv2.Laplacian(target_gray, cv2.CV_32F)
        
        #canny_pred=feature.canny(torch.squeeze(pred).cpu().detach().numpy().astype('float32'), sigma=3)
        #canny_target=feature.canny(torch.squeeze(target).cpu().detach().numpy().astype('float32'), sigma=3)
        #print(canny_pred)
        G= canny_target-canny_pred
        G=torch.tensor(G).to(torch.device("cuda"))
        self.loss = alpha*G + (1-alpha)*MAE
        return self.loss.mean()

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

class ALEError(nn.Module):
    """
    Trying to create a better loss function based on Asymetric Linear Error(ALE) and Reflected Asymetric Linear Error(RALE)
    """
    def __init__(self):
        super(ALEError, self).__init__()

    def forward(self, pred, target, alpha=0.3):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        MAE = (diff**2).abs().mean()
        sobel_pred=ndimage.sobel(pred.cpu().detach().numpy())
        sobel_tarjet=ndimage.sobel(target.cpu().detach().numpy())
        G=torch.tensor(sobel_tarjet-sobel_pred).to(torch.device("cuda"))
        self.loss = alpha*G + (1-alpha)*MAE
        return self.loss.mean()