import torch
from torchvision.models import resnet
from model.basic import *
import torch.nn as nn


def init_decoder(model):
    #init conv1d
    n = model.conv1d.kernel_size[0] * model.conv1d.kernel_size[1] * model.conv1d.out_channels
    model.conv1d.weight.data.normal_(0, math.sqrt(2. / n))
    if model.conv1d.bias is not None:
            model.conv1d.bias.data.zero_()

    #init conv1rgbd
    n = model.conv1rgb.kernel_size[0] * model.conv1rgb.kernel_size[1] * model.conv1rgb.out_channels
    model.conv1rgb.weight.data.normal_(0, math.sqrt(2. / n))
    if model.conv1rgb.bias is not None:
            model.conv1rgb.bias.data.zero_()

    #init conv6
    n = model.conv6.kernel_size[0] * model.conv6.kernel_size[1] * model.conv6.out_channels
    model.conv6.weight.data.normal_(0, math.sqrt(2. / n))
    if model.conv6.bias is not None:
            model.conv6.bias.data.zero_()

    #init dec_6
    n = model.dec_6.kernel_size[0] * model.dec_6.kernel_size[1] * model.dec_6.out_channels
    model.dec_6.weight.data.normal_(0, math.sqrt(2. / n))
    if model.dec_6.bias is not None:
            model.dec_6.bias.data.zero_()


    #init dec_1
    n = model.dec_1.kernel_size[0] * model.dec_1.kernel_size[1] * model.dec_1.in_channels
    model.dec_1.weight.data.normal_(0, math.sqrt(2. / n))
    if model.dec_1.bias is not None:
            model.dec_1.bias.data.zero_()


    #init dec_2
    n = model.dec_2.kernel_size[0] * model.dec_2.kernel_size[1] * model.dec_2.in_channels
    model.dec_2.weight.data.normal_(0, math.sqrt(2. / n))
    if model.dec_2.bias is not None:
            model.dec_2.bias.data.zero_()

    #init dec_3
    n = model.dec_3.kernel_size[0] * model.dec_3.kernel_size[1] * model.dec_3.in_channels
    model.dec_3.weight.data.normal_(0, math.sqrt(2. / n))
    if model.dec_3.bias is not None:
            model.dec_3.bias.data.zero_()
    
    #init dec_4
    n = model.dec_4.kernel_size[0] * model.dec_4.kernel_size[1] * model.dec_4.in_channels
    model.dec_4.weight.data.normal_(0, math.sqrt(2. / n))
    if model.dec_4.bias is not None:
            model.dec_4.bias.data.zero_()
        
    #init dec_5
    n = model.dec_5.kernel_size[0] * model.dec_5.kernel_size[1] * model.dec_5.in_channels
    model.dec_5.weight.data.normal_(0, math.sqrt(2. / n))
    if model.dec_5.bias is not None:
            model.dec_5.bias.data.zero_()

class AutoEncoderPretrained(nn.Module):
    def __init__(self):
        super(AutoEncoderPretrained, self).__init__()

        #Encoder part
        self.conv1d=conv_bn_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)
        self.conv1rgb=conv_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)

        # load pretrained here
        pretrained_model = resnet.__dict__['resnet34'](pretrained=True)
        self.conv2 = pretrained_model._modules['layer1']
        self.conv3 = pretrained_model._modules['layer2']
        self.conv4 = pretrained_model._modules['layer3']
        self.conv5 = pretrained_model._modules['layer4']
        del pretrained_model  # clear memory
        self.conv6 = conv_bn_relu(512,512,kernel_size=3,stride=2,padding=1)

        #Decoder part
        self.dec_1=deconv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_2=deconv_bn_relu(in_channels=(512+256), out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_3=deconv_bn_relu(in_channels=(256+128), out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_4=deconv_bn_relu(in_channels=(128+64), out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_5=deconv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0)
        self.dec_6=nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0)

        #init_decoder(self)

        
    def forward(self, input):

        rgb=input['rgb']
        depth=input['d']

        depth_1=self.conv1d(depth)
        rgb_1=self.conv1rgb(rgb)

        conv1=torch.cat((depth_1, rgb_1),1)
        
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)  
        conv4 = self.conv4(conv3)  
        conv5 = self.conv5(conv4)  
        conv6 = self.conv6(conv5)

        # decoder
        convt5 = self.dec_1(conv6)
        y = torch.cat((convt5, conv5), 1)

        convt4 = self.dec_2(y)
        y = torch.cat((convt4, conv4), 1)

        convt3 = self.dec_3(y)
        y = torch.cat((convt3, conv3), 1)

        convt2 = self.dec_4(y)
        y = torch.cat((convt2, conv2), 1)

        convt1 = self.dec_5(y)
        y = torch.cat((convt1, conv1), 1)

        y = self.dec_6(y)

        return y

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        
        print("-----")
        print(x1.shape)
        print(x2.shape)
        
        #x = torch.cat([x2, x1], dim=1)
        x = x2+ x1
        return self.conv(x)


class AutoEncoderPretrained_2(nn.Module):
    """
    Using resnet34 pretrained as encoder and transposed conv as decoder 
    """
    def __init__(self):
        super(AutoEncoderPretrained_2, self).__init__()

        #Encoder part
        self.conv1d=conv_bn_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)
        self.conv1rgb=conv_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)

        # load pretrained here
        pretrained_model = resnet.__dict__['resnet34'](pretrained=True)
        self.conv2 = pretrained_model._modules['layer1'] #[b, 64, 704, 1280]
        self.conv3 = pretrained_model._modules['layer2'] #[b, 128, 352, 640]
        self.conv4 = pretrained_model._modules['layer3'] #[b, 256, 176, 320]
        self.conv5 = pretrained_model._modules['layer4'] #[b, 512, 88, 160]
        del pretrained_model  # clear memory
        self.conv6 = conv_bn_relu(512,1024,kernel_size=3,stride=2,padding=1) #[b, 1024, 44, 80]

        
        self.dec_1=deconv_bn_relu(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1) #[b, 512, 88, 160]
        self.dec_2=deconv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1) #[b, 256, 176, 320]
        self.dec_3=deconv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) #[b, 128, 352, 640]
        self.dec_4=deconv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) #[b, 64, 704, 1280]
        self.dec_5=deconv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0) #[b, 32, 704, 1280]
        self.dec_6=deconv_bn_relu(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, output_padding=0) #[b, 1, 704, 1280]
        
        
        
        
        
        
        ##Decoder part
        #self.dec_1=deconv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_2=deconv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_3=deconv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_4=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_5=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0)
        #self.dec_5_2=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=0)
        #self.dec_6=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        #init_decoder(self)

        
    def forward(self, input):
        
        rgb=input['rgb']
        depth=input['d']

        #print("init params->")
        #print(rgb.shape)
        #print(depth.shape)
        
        depth_1=self.conv1d(depth)
        rgb_1=self.conv1rgb(rgb)
        
        #conv1=depth_1 +rgb_1
        
        conv1=torch.cat((depth_1, rgb_1), 1)
        
        #print("Init resnet params->")
        #print(conv1.shape)
        
        conv2 = self.conv2(conv1)
        #print("2 resnet params->")
        #print(conv2.shape)
        conv3 = self.conv3(conv2)
        #print("3 resnet params->")
        #print(conv3.shape)  
        conv4 = self.conv4(conv3)
        #print("4 resnet params->")
        #print(conv4.shape)  
        conv5 = self.conv5(conv4)
        #print("5 resnet params->")
        #print(conv5.shape)    
        conv6 = self.conv6(conv5)
        #print("6 resnet params->")
        #print(conv6.shape)
          
        # decoder
        convt5 = self.dec_1(conv6)
        #print("Deconv 1 params->")
        #print(convt5.shape)
        y = convt5 + conv5

        convt4 = self.dec_2(y)
        #print("Deconv 2 params->")
        #print(convt4.shape)
        y = convt4 + conv4

        convt3 = self.dec_3(y)
        #print("Deconv 3 params->")
        #print(convt3.shape)
        y = convt3 + conv3

        convt2 = self.dec_4(y)
        #print("Deconv 4 params->")
        #print(convt2.shape)
        y = convt2 + conv2

        convt1 = self.dec_5(y)
        #print("Deconv 5 params->")
        #print(convt1.shape)
        y = convt1
        

        y = self.dec_6(y)

        #print("Output shape->")
        #print(y.shape)
        return y


class AutoEncoderPretrained_3_upsample(nn.Module):
    """
    Using resnet34 pretrained as encoder and upsample layers + conv as decoder 
    """
    def __init__(self):
        super(AutoEncoderPretrained_3_upsample, self).__init__()

        #Encoder part
        self.conv1d=conv_bn_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)
        self.conv1rgb=conv_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)

        # load pretrained here
        pretrained_model = resnet.__dict__['resnet34'](pretrained=True)
        self.conv2 = pretrained_model._modules['layer1'] #[b, 64, 704, 1280]
        self.conv3 = pretrained_model._modules['layer2'] #[b, 128, 352, 640]
        self.conv4 = pretrained_model._modules['layer3'] #[b, 256, 176, 320]
        self.conv5 = pretrained_model._modules['layer4'] #[b, 512, 88, 160]
        del pretrained_model  # clear memory
        self.conv6 = conv_bn_relu(512,1024,kernel_size=3,stride=2,padding=1) #[b, 1024, 44, 80]

        
        #self.dec_1=Up(in_channels=1024, out_channels=512) 
        self.dec_1=deconv_bn_relu(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_2=Up(in_channels=512, out_channels=256)
        self.dec_3=Up(in_channels=256, out_channels=128)
        self.dec_4=Up(in_channels=128, out_channels=64)
        self.dec_5=Up(in_channels=64, out_channels=32)
        self.dec_6=deconv_bn_relu(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, output_padding=0) 
        
        
               
        
        
        ##Decoder part
        #self.dec_1=deconv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_2=deconv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_3=deconv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_4=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_5=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0)
        #self.dec_5_2=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=0)
        #self.dec_6=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        #init_decoder(self)

        
    def forward(self, input):
        
        rgb=input['rgb']
        depth=input['d']

        #print("init params->")
        #print(rgb.shape)
        #print(depth.shape)
        
        depth_1=self.conv1d(depth)
        rgb_1=self.conv1rgb(rgb)
        
        #conv1=depth_1 +rgb_1
        
        conv1=torch.cat((depth_1, rgb_1), 1)
        
        #print("Init resnet params->")
        #print(conv1.shape)
        
        conv2 = self.conv2(conv1)
        #print("2 resnet params->")
        #print(conv2.shape)
        conv3 = self.conv3(conv2)
        #print("3 resnet params->")
        #print(conv3.shape)  
        conv4 = self.conv4(conv3)
        #print("4 resnet params->")
        #print(conv4.shape)  
        conv5 = self.conv5(conv4)
        #print("5 resnet params->")
        #print(conv5.shape)    
        conv6 = self.conv6(conv5)
        #print("6 resnet params->")
        #print(conv6.shape)
          
        # decoder
        convt5 = self.dec_1(conv6)
        #print("Dec 1 params->")
        #print(convt5.shape)
        y = self.dec_2(convt5, conv5)
        #print("Dec 2 params->")
        #print(y.shape)
        y = self.dec_3(y, conv4)
        #print("Dec 3 params->")
        #print(y.shape)
        y = self.dec_4(y, conv3)
        y = self.dec_5(y, conv2)
        y = self.dec_6(y)
        return y


class PretrainedEncoderTwoBranch_1(nn.Module):
    """
    Using resnet34 pretrained as encoder and another branch for Depth images
    """
    def __init__(self):
        super(PretrainedEncoderTwoBranch_1, self).__init__()

        #Encoder part
        self.conv1d=conv_bn_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)
        self.conv1rgb=conv_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)

        # load pretrained here
        pretrained_model = resnet.__dict__['resnet34'](pretrained=True)
        self.conv2 = pretrained_model._modules['layer1'] #[b, 64, 704, 1280]
        self.conv3 = pretrained_model._modules['layer2'] #[b, 128, 352, 640]
        self.conv4 = pretrained_model._modules['layer3'] #[b, 256, 176, 320]
        self.conv5 = pretrained_model._modules['layer4'] #[b, 512, 88, 160]
        del pretrained_model  # clear memory
        self.conv6 = conv_bn_relu(512,1024,kernel_size=3,stride=2,padding=1) #[b, 1024, 44, 80]

        
        self.dec_1=deconv_bn_relu(in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1) #[b, 512, 88, 160]
        self.dec_2=deconv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1) #[b, 256, 176, 320]
        self.dec_3=deconv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1) #[b, 128, 352, 640]
        self.dec_4=deconv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1) #[b, 64, 704, 1280]
        self.dec_5=deconv_bn_relu(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, output_padding=0) #[b, 32, 704, 1280]
        self.dec_6=deconv_bn_relu(in_channels=32, out_channels=1, kernel_size=1, stride=1, padding=0, output_padding=0) #[b, 1, 704, 1280]
        
        
        
        
        
        
        ##Decoder part
        #self.dec_1=deconv_bn_relu(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_2=deconv_bn_relu(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_3=deconv_bn_relu(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_4=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        #self.dec_5=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, output_padding=0)
        #self.dec_5_2=deconv_bn_relu(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=0)
        #self.dec_6=nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1, padding=0)

        #init_decoder(self)

        
    def forward(self, input):
        
        rgb=input['rgb']
        depth=input['d']

        #print("init params->")
        #print(rgb.shape)
        #print(depth.shape)
        
        depth_1=self.conv1d(depth)
        rgb_1=self.conv1rgb(rgb)
        
        #conv1=depth_1 +rgb_1
        
        conv1=torch.cat((depth_1, rgb_1), 1)
        
        #print("Init resnet params->")
        #print(conv1.shape)
        
        conv2 = self.conv2(conv1)
        #print("2 resnet params->")
        #print(conv2.shape)
        conv3 = self.conv3(conv2)
        #print("3 resnet params->")
        #print(conv3.shape)  
        conv4 = self.conv4(conv3)
        #print("4 resnet params->")
        #print(conv4.shape)  
        conv5 = self.conv5(conv4)
        #print("5 resnet params->")
        #print(conv5.shape)    
        conv6 = self.conv6(conv5)
        #print("6 resnet params->")
        #print(conv6.shape)
          
        # decoder
        convt5 = self.dec_1(conv6)
        #print("Deconv 1 params->")
        #print(convt5.shape)
        y = convt5 + conv5

        convt4 = self.dec_2(y)
        #print("Deconv 2 params->")
        #print(convt4.shape)
        y = convt4 + conv4

        convt3 = self.dec_3(y)
        #print("Deconv 3 params->")
        #print(convt3.shape)
        y = convt3 + conv3

        convt2 = self.dec_4(y)
        #print("Deconv 4 params->")
        #print(convt2.shape)
        y = convt2 + conv2

        convt1 = self.dec_5(y)
        #print("Deconv 5 params->")
        #print(convt1.shape)
        y = convt1
        

        y = self.dec_6(y)

        #print("Output shape->")
        #print(y.shape)
        return y