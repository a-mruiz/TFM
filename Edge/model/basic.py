
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


def conv1x1(in_channels, out_channels, stride=1, groups=1, bias=False):
    """1x1 convolution"""
    layer= nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, groups=groups, bias=True)
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

class inceptionBlock_light(nn.Module):
    def __init__(self,in_channels, ch1x1,ch3x3,ch5x5,ch3x3_in,ch5x5_in,pool_proj=32) -> None:
        super(inceptionBlock_light,self).__init__()
        self.branch1=conv3x3_bn_relu(in_channels,ch1x1,kernel_size=1,padding=0)
        self.branch2=nn.Sequential(conv3x3_bn_relu(in_channels,ch3x3_in,kernel_size=1,padding=0),conv3x3_bn_relu(ch3x3_in,ch3x3,kernel_size=3))
        self.branch3=nn.Sequential(conv3x3_bn_relu(in_channels,ch5x5_in,kernel_size=1,padding=0),conv3x3_bn_relu(ch5x5_in,ch5x5,kernel_size=5,padding=2))
        #self.branch4=nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),conv3x3_bn_relu(in_channels,pool_proj,kernel_size=1,padding=0))
    def forward(self, input):
        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        outputs = [branch1, branch2, branch3]
        return torch.cat(outputs,1)
    
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

def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1, bias=False, padding=1):
    """3x3 convolution with padding"""
    if padding >= 1:
        padding = dilation
    layer= nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=True, dilation=dilation)
    init_weights(layer)
    return layer
###############################CBAM#################################

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=True):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out