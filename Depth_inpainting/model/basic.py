"""
This file defines some basic blocks to help build the model
Author:Alejandro
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def init_weights(m):
    # Initialize filters with Gaussian random weights
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun




def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    layer= nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=True)
    init_weights(layer)
    return layer

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    layer= nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=True, dilation=dilation)
    init_weights(layer)
    return layer
    

def conv3x3_bn_relu(in_channels, out_channels, kernel_size=3,stride=1, padding=1,bn=True):
    """Convolution + BatchNorm + ReLU"""
    if bn:
        layers= nn.Sequential(
	    	nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
	    	#nn.BatchNorm2d(out_channels,eps=0.001),
            #nn.InstanceNorm2d(out_channels),
	    	nn.ReLU(inplace=True)
	    )
    else:
        layers= nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
	)
    for m in layers.modules():
        init_weights(m)
    return layers
        
def conv3x3_relu(in_channels, out_channels, kernel_size=3,stride=1, padding=1,relu=True):
    """Convolution + BatchNorm + ReLU"""
    if relu:
        layers= nn.Sequential(
	    	nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
	    	nn.ReLU(inplace=True)
	    )
    else:
        layers= nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
	)
    for m in layers.modules():
        init_weights(m)
    return layers


def conv3x3_bn_Leakyrelu(in_channels, out_channels, kernel_size=3,stride=1, padding=1,bn=True):
    """Convolution + BatchNorm + LeakyReLU"""
    if bn:
        layers=  nn.Sequential(
	    	nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
	    	nn.BatchNorm2d(out_channels),
	    	nn.LeakyReLU(0.2,inplace=True)
	    )
    else:
        layers=  nn.Sequential(
		nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
	)
    for m in layers.modules():
        init_weights(m)
    return layers

def deconv3x3_bn_relu(in_channels, out_channels, kernel_size=3, stride=2, padding=2, output_padding=1, bn=True):
    """Transpose convolution + BatchNorm + ReLU"""
    if bn:
        layers= nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    else:
        layers= nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True),
        )
    for m in layers.modules():
        init_weights(m)
    return layers

def deconv3x3_relu(in_channels, out_channels, kernel_size=3, stride=2, padding=2, output_padding=1, relu=True):
    """Transpose convolution + BatchNorm + ReLU"""
    if relu:
        layers= nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False),
            nn.ReLU(inplace=True)
        )
    else:
        layers= nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True),
        )
    for m in layers.modules():
        init_weights(m)
    return layers

def deconv3x3_bn_relu_no_artifacts(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, bn=True,scale_factor=2, dilation=1):
    """Transpose convolution + BatchNorm + ReLU"""
    if bn:
        layers= nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest" ),
            nn.Conv2d( in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            nn.InstanceNorm2d(out_channels)
            #nn.BatchNorm2d(out_channels,momentum=0.9)
        )
    else:
        layers= nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True),
        )
    for m in layers.modules():
        init_weights(m)
    return layers

class deconv3x3_relu_no_artifacts_class(nn.Module):
    
    def __init__(self,in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, relu=True,scale_factor=2, dilation=1) -> None:
        super(deconv3x3_relu_no_artifacts_class, self).__init__()
        if relu:
            self.layers= nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv2d( in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            #nn.InstanceNorm2d(out_channels),
            nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            #nn.InstanceNorm2d(out_channels)
            #nn.BatchNorm2d(out_channels,momentum=0.9)
        )
        else:
            self.layers= nn.Sequential(
                #nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, bias=True),
                nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1,bias=True),
                #nn.AdaptiveAvgPool2d((512,512))
            )
        for m in self.layers.modules():
            init_weights(m)
            
    def forward(self, input):
        return self.layers(input)









def deconv3x3_relu_no_artifacts(in_channels, out_channels, kernel_size=3, stride=1, padding=1, output_padding=1, relu=True,scale_factor=2, dilation=1):
    """Transpose convolution + BatchNorm + ReLU"""
    if relu:
        layers= nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode="nearest"),
            nn.Conv2d( in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
            #nn.InstanceNorm2d(out_channels),
            nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1),
            nn.ReLU(inplace=True),
            #nn.InstanceNorm2d(out_channels)
            #nn.BatchNorm2d(out_channels,momentum=0.9)
        )
    else:
        layers= nn.Sequential(
            #nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=0, bias=True),
            nn.Conv2d( in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding=1, dilation=1),
            #nn.AdaptiveAvgPool2d((512,512))
        )
    for m in layers.modules():
        init_weights(m)
    return layers





class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

        self.apply(weights_init(init_type='kaiming'))

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)

        # if self.update_mask.type() != input.type() or self.mask_ratio.type() != input.type():
        #     self.update_mask.to(input)
        #     self.mask_ratio.to(input)

        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)

        if self.return_mask:
            return output, self.update_mask.prod(dim=1, keepdim=True)
        else:
            return output

class ResNetBlock(nn.Module):
    """
    This class represents a basic extraction block based on conv_layers + batchnorm + ReLU.
    It is implemented following ResNet architecture
    """
    def __init__(self, in_channels, out_channels, stride=1,downsample=None,dilation=1,batch=True):
        super(ResNetBlock, self).__init__()
        self.batch=batch
        #First extraction layer
        #self.ext1=conv3x3_bn_relu(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv1= conv3x3(in_channels, out_channels, stride,dilation=dilation)
        self.bn1=nn.BatchNorm2d(out_channels)
        #Second extraction layer
        self.conv2 = conv3x3(out_channels, out_channels,dilation=dilation) #in_channels=out_channels
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            if batch:
                downsample = nn.Sequential(
                    conv1x1(in_channels, out_channels, stride),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
            )    
            init_weights(downsample)
        self.downsample = downsample
        self.relu=nn.ReLU(inplace=True)
        
        init_weights(self.bn1)
        init_weights(self.bn2)
        
        #self.relu=nn.SiLU(inplace=True)
        
    def forward(self, x):
        #out=self.ext1(x)
        out=self.conv1(x)
        if self.batch:
            out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        if self.batch:
            out=self.bn2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out+=x
        out=self.relu(out)
        return out

class ResNetBlockNoBatch(nn.Module):
    """
    This class represents a basic extraction block based on conv_layers + batchnorm + ReLU.
    It is implemented following ResNet architecture. Also, the Batch Normalization layers are avoided and the final ReLU.
    """
    def __init__(self, in_channels, out_channels, stride=1,downsample=None):
        super(ResNetBlockNoBatch, self).__init__()
        #First extraction layer
        self.ext1=conv_bn_relu(in_channels=in_channels, out_channels=out_channels, stride=stride)
        self.conv1= conv3x3(in_channels, out_channels, stride)
        self.bn1=nn.BatchNorm2d(out_channels)
        #Second extraction layer
        self.conv2 = conv3x3(out_channels, out_channels) #in_channels=out_channels
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride),
                #nn.BatchNorm2d(out_channels),
            )
        self.downsample = downsample
        self.relu=nn.ReLU(inplace=True)

    def forward(self, x):
        #out=self.ext1(x)
        out=self.conv1(x)
        #out=self.bn1(out)
        out=self.relu(out)
        out=self.conv2(out)
        out=self.bn2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out+=x
        #out=self.relu(out)
        return out



    
class ResNetBottleneckBlock(nn.Module):
    """
    This class represents a basic extraction block based on conv_layers + batchnorm + ReLU.
    It is implemented following ResNet architecture. It is the deeper convolution version, "Bottleneck"
    """
    def __init__(self, in_channels, out_channels, stride=1, groups=1, base_width=64, dilation=1):
        super(ResNetBottleneckBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        width = int(out_channels * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(in_channels, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, out_channels)
        self.bn3 = norm_layer(out_channels )
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample=nn.Sequential(
            nn.Conv2d(in_channels,width, kernel_size=1, bias=False),
            nn.BatchNorm2d(width)
        )

    def forward(self, x):
        
        identity = x
        #print("In->"+str(identity.shape))

        out = self.conv1(x)
        #print("Out1->"+str(out.shape))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #print("Out2->"+str(out.shape))
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #print("Out3->"+str(out.shape))
        out = self.bn3(out)

        out += self.downsample(identity)
        out = self.relu(out)

        return out
        
    
class ResNextBlock(nn.Module):
    """
    This class represents a basic extraction block based on conv_layers + batchnorm + ReLU.
    Follows the ResNext structure!
    """
    def __init__(self, in_channels, out_channels, stride=1, cardinality=6, base_width=6, dilation=1):
        super(ResNextBlock, self).__init__()

        width_ratio = out_channels / (dilation * 64.)
        D = cardinality * int(base_width * width_ratio)
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv',
                                     nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                                               bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)
    
class DownsamplerBlock (nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplerBlock,self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels-in_channels, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)   


class DS(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DS, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels-in_channels, (3, 3), stride=2, padding=1, bias=True)
    def forward(self, input):
        out=self.conv(input)
        return out
        
        
class non_bottleneck_1d (nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super(non_bottleneck_1d,self).__init__()

        self.conv3x1_1 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1, 0), bias=True)

        self.conv1x3_1 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1), bias=True)

        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)

        self.conv3x1_2 = nn.Conv2d(chann, chann, (3, 1), stride=1, padding=(1*dilated,0), bias=True, dilation=(dilated, 1))

        self.conv1x3_2 = nn.Conv2d(chann, chann, (1, 3), stride=1, padding=(0, 1*dilated), bias=True, dilation=(1, dilated))

        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)

        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):

        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output+input)
    
class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super(UpsamplerBlock,self).__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class ResidualBlock(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = nn.Conv2d(inp_dim, 
                          int(out_dim/2), 
                          1,padding=(1-1)//2)
        self.bn2 = nn.BatchNorm2d(int(out_dim/2))
        self.conv2 = nn.Conv2d(int(out_dim/2), 
                          int(out_dim/2), 
                          3,padding=(3-1)//2)
        self.bn3 = nn.BatchNorm2d(int(out_dim/2))
        self.conv3 = nn.Conv2d(int(out_dim/2),
                          out_dim, 
                          1,padding=(1-1)//2)
        self.skip_layer = nn.Conv2d(inp_dim,
                               out_dim, 
                               1,padding=(1-1)//2)

        modules=[self.relu, self.bn1, self.conv1, self.bn2, self.bn2, self.conv3, self.skip_layer]
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        for m in modules:
            init_weights(m)
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        #print("Out shape->"+str(out.shape))
        #print("Input shape->"+str(residual.shape))
        #print("Input with skip->"+str(self.skip_layer(residual).shape))
        out += residual
        #out += self.skip_layer(residual)
        return out

    
class HourglassModule(nn.Module):
    def __init__(self, n, f, bn=None, increase=0):
        super(HourglassModule, self).__init__()
        nf = f + increase
        self.up1 = ResidualBlock(f, f)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        self.low1 = ResidualBlock(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = HourglassModule(n-1, nf, bn=bn)
        else:
            self.low2 = ResidualBlock(nf, nf)
        self.low3 = ResidualBlock(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        
    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2




###################################################
#    This is from the Fast.ai implementation      #
#    consult to fix convlayer and NormType        #
# https://github.com/fastai/fastai/blob/f91e058f500fdcebb9af74654bf14a2edc430cc0/fastai/layers.py#L237 #
###################################################
class SelfAttention(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super(SelfAttention, self).__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def _conv(self,n_in,n_out):
        #return ConvLayer(n_in, n_out, ks=1, ndim=1, norm_type=NormType.Spectral, act_cls=None, bias=False)
        return conv3x3_bn_relu(n_in, n_out,kernel_size=1)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()
    
class DIYSelfAttention(nn.Module):
    def __init__(self,channels):
        super(DIYSelfAttention, self).__init__()
        
        self.conv1_0=conv1x1(channels, channels)
        self.conv1_1=conv1x1(channels, channels)
        self.conv1_2=conv1x1(channels, channels)
        
        self.softmax=nn.Softmax(1)
        
        self.conv2=conv1x1(channels,channels)
        
        modules=[self.conv1_0,self.conv1_1,self.conv1_2,self.conv2]
        for m in modules:
            init_weights(m)
        
    def forward(self, x):
       x1=self.conv1_0(x)
       x2=self.conv1_1(x)
       x3=self.conv1_2(x)
       
       joint1=self.softmax(x1*x2)
       joint2=joint1*x3
       
       return self.conv2(joint2)