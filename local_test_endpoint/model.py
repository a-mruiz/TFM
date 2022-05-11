
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


class InceptionAndAttentionModel(nn.Module):
    def __init__(self):
        super(InceptionAndAttentionModel, self).__init__()
        
        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
             
        self.conv_intermediate=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=64,stride=2, padding=1)
        
        self.inception_1=inceptionBlock_light(in_channels=64,ch1x1=32,ch3x3=64,ch5x5=32,ch3x3_in=32,ch5x5_in=16,pool_proj=32) #128
        #self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32) #256
        self.conv_intermediate_2=conv3x3_relu(in_channels=128,kernel_size=3,out_channels=256,stride=2, padding=1)
        
        self.att_1=DIYSelfAttention(256)
        
        self.conv_intermediate_3=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=1)
        
        self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32,pool_proj=64) #256
        
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)

        self.dec_5_res=conv1x1(in_channels=1,out_channels=1,bias=True)
           
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        #gray = input['g']
        d = input['d']

        #init branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_init_depth=self.first_layer_depth(d)
        #print("RGB_I->"+str(encoder_feature_init_rgb.shape))
        #print("Depth_I->"+str(encoder_feature_init_depth.shape))
        #Join both representations
        out=torch.cat((encoder_feature_init_rgb,encoder_feature_init_depth),1)
        out=self.conv_intermediate(out)
        
        out=self.inception_1(out)
        out=self.conv_intermediate_2(out)
        
        out=self.att_1(out)
        out=self.conv_intermediate_3(out)
        
        out=self.inception_2(out)
        
        #Decoder
        out=self.dec_1(out)
        out=self.dec_2(out)
        out=self.dec_3(out)
        out=self.dec_4(out)
        out=self.dec_5_res(out)
        return self.final_sigmoid(out) 
    
    
class SelfAttentionCBAM(nn.Module):
    def __init__(self):
        super(SelfAttentionCBAM,self).__init__()
        
        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)#B,16,516,1028

        self.att_1=CBAM(64)

        self.conv_intermediate_1=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=128,stride=2, padding=1)
        self.att_2=CBAM(128)
        
        self.conv_intermediate_2=conv3x3_relu(in_channels=128,kernel_size=3,out_channels=256,stride=2, padding=1)
        self.att_3=CBAM(256)
        
        self.conv_intermediate_3=conv3x3_relu(in_channels=256,kernel_size=3,out_channels=256,stride=2, padding=1)
        self.att_4=CBAM(256)
        
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)

        self.dec_5_res=conv3x3(in_channels=32,out_channels=2,bias=True)
           
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)

        encoder_feature_init_depth=self.first_layer_depth(d)

        #Join both representations
        out=torch.cat((encoder_feature_init_rgb,encoder_feature_init_depth),1)
        out=self.att_1(out)
        
        out=self.att_2(self.conv_intermediate_1(out))
        out=self.att_3(self.conv_intermediate_2(out))
        out=self.att_4(self.conv_intermediate_3(out))
        
        #Decoder
        out=self.dec_1(out)
        out=self.dec_2(out)
        out=self.dec_3(out)
        out=self.dec_4(out)
        out=self.dec_5_res(out)
        
        depth=out[:, 0:1, :, :]
        confidence=out[:, 1:2, :, :]

        out=depth*confidence
        
        return self.final_sigmoid(out) 
   
     
class BasicModelDeep(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers, but this time is deeper
    """
    def __init__(self):
        super(BasicModelDeep, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_bn_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.enc_2=ResNetBlock(in_channels=64, out_channels=64, stride=1)
        self.enc_3=ResNetBlock(in_channels=64, out_channels=128, stride=2)
        self.enc_4=ResNetBlock(in_channels=128, out_channels=128, stride=1)
        self.enc_5=ResNetBlock(in_channels=128, out_channels=256, stride=2)
        self.enc_6=ResNetBlock(in_channels=256, out_channels=256, stride=1)
        self.enc_7=ResNetBlock(in_channels=256, out_channels=512, stride=2)
        self.enc_8=ResNetBlock(in_channels=512, out_channels=512, stride=1)
        self.enc_9=ResNetBlock(in_channels=512, out_channels=1024, stride=2)
        self.enc_10=ResNetBlock(in_channels=1024, out_channels=1024, stride=1)

        #Decoder
        self.dec_1=deconv3x3_relu(in_channels=1024, out_channels=512,padding=1)
        self.dec_2=deconv3x3_relu(in_channels=512, out_channels=256,padding=1)
        self.dec_3=deconv3x3_relu(in_channels=256, out_channels=128,padding=1)
        self.dec_4=deconv3x3_relu(in_channels=128, out_channels=64,padding=1)
        self.dec_5=deconv3x3_relu(in_channels=64, out_channels=32,padding=1)
        self.dec_6=deconv3x3_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,relu=False)

        init_weights(self)
        self.final_sigmoid=nn.Sigmoid()
    def forward(self,input):

        rgb = input['rgb']
        d = input['d']

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        encoder_feature_5=self.enc_5(encoder_feature_4)
        encoder_feature_6=self.enc_6(encoder_feature_5)
        encoder_feature_7=self.enc_7(encoder_feature_6)
        encoder_feature_8=self.enc_8(encoder_feature_7)
        encoder_feature_9=self.enc_9(encoder_feature_8)
        encoder_feature_10=self.enc_10(encoder_feature_9)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_10)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4+encoder_feature_2 #skip connection

        decoder_feature_5=self.dec_5(decoder_feature_4_plus)
        decoder_feature_5_plus=decoder_feature_5#+encoder_feature_init #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_5_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#, depth, confidence