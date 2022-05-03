"""
This file defines different model architectures to try
Author:Alejandro
"""

from json import encoder
from model.attention import CBAM
from model.basic import *
from torchvision.models import resnet
import torch.nn as nn


class BasicModel(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers
    """
    def __init__(self):
        super(BasicModel, self).__init__()
        
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

        #Decoder
        self.dec_1=deconv3x3_bn_relu(in_channels=512, out_channels=256,padding=1)
        self.dec_2=deconv3x3_bn_relu(in_channels=256, out_channels=128,padding=1)
        self.dec_3=deconv3x3_bn_relu(in_channels=128, out_channels=64,padding=1)
        self.dec_4=deconv3x3_bn_relu(in_channels=64, out_channels=32,padding=1)
        self.dec_5=deconv3x3_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0, bn=False)

        init_weights(self)
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):

        rgb = input['rgb']
        d = input['d']
        #print(rgb.shape)
        #print(d.shape)

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

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_8)
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_6 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_4 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_2 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4+encoder_feature_init #skip connection

        decoder_feature_5=self.dec_5(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_5[:, 0:1, :, :]
        confidence=decoder_feature_5[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#depth, confidence, output 

class BasicModel_2(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers
    """
    def __init__(self):
        super(BasicModel_2, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv_bn_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

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

        self.dec_1=deconv_bn_relu(in_channels=1024, out_channels=512)
        self.dec_2=deconv_bn_relu(in_channels=512, out_channels=256)
        self.dec_3=deconv_bn_relu(in_channels=256, out_channels=128)
        self.dec_4=deconv_bn_relu(in_channels=128, out_channels=64)
        self.dec_5=deconv_bn_relu(in_channels=64, out_channels=32)
        self.dec_6=deconv_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0)

        init_weights(self)
        
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
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_7 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_5 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_3 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4+encoder_feature_1 #skip connection

        decoder_feature_5=self.dec_5(decoder_feature_4_plus)

        decoder_feature_6=self.dec_6(decoder_feature_5)

        ##Output
        #depth=decoder_feature_6[:, 0:1, :, :]
        #confidence=decoder_feature_6[:, 1:2, :, :]
        #output=depth*confidence

        return decoder_feature_6#depth, confidence, output 

class BasicModelDeep(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers, but this time is deeper
    """
    def __init__(self):
        super(BasicModelDeep, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv_bn_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

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
        self.dec_1=deconv_bn_relu(in_channels=1024, out_channels=512)
        self.dec_2=deconv_bn_relu(in_channels=512, out_channels=256)
        self.dec_3=deconv_bn_relu(in_channels=256, out_channels=128)
        self.dec_4=deconv_bn_relu(in_channels=128, out_channels=64)
        self.dec_5=deconv_bn_relu(in_channels=64, out_channels=32)
        self.dec_6=deconv_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0)

        init_weights(self)
        
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
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4+encoder_feature_2 #skip connection

        decoder_feature_5=self.dec_5(decoder_feature_4_plus)
        decoder_feature_5_plus=decoder_feature_5+encoder_feature_init #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_5_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return output#, depth, confidence 

class TwoBranchModel(nn.Module):
    """
    Two branch encoder-decoder model where one branch focuses on color while the other focuses on depth, then they merge
    """
    def __init__(self):
        super(TwoBranchModel, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv_bn_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

        #RGB Encoder
        self.rgb_enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.rgb_enc_2=ResNetBlock(in_channels=64, out_channels=64, stride=1)
        self.rgb_enc_3=ResNetBlock(in_channels=64, out_channels=128, stride=2)
        self.rgb_enc_4=ResNetBlock(in_channels=128, out_channels=128, stride=1)
        self.rgb_enc_5=ResNetBlock(in_channels=128, out_channels=256, stride=2)
        self.rgb_enc_6=ResNetBlock(in_channels=256, out_channels=256, stride=1)
        self.rgb_enc_7=ResNetBlock(in_channels=256, out_channels=512, stride=2)
        self.rgb_enc_8=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #RGB Decoder
        self.rgb_dec_1=deconv_bn_relu(in_channels=512, out_channels=256)
        self.rgb_dec_2=deconv_bn_relu(in_channels=256, out_channels=128)
        self.rgb_dec_3=deconv_bn_relu(in_channels=128, out_channels=64)
        self.rgb_dec_4=deconv_bn_relu(in_channels=64, out_channels=32)
        self.rgb_dec_5=deconv_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0)

        #First layer of the Depth Encoder
        self.depth_first_layer=conv_bn_relu(in_channels=2,kernel_size=5,out_channels=32,stride=1, padding=2)

        #Depth Encoder
        self.depth_enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.depth_enc_2=ResNetBlock(in_channels=64, out_channels=64, stride=1)
        self.depth_enc_3=ResNetBlock(in_channels=128, out_channels=128, stride=2)
        self.depth_enc_4=ResNetBlock(in_channels=128, out_channels=128, stride=1)
        self.depth_enc_5=ResNetBlock(in_channels=256, out_channels=256, stride=2)
        self.depth_enc_6=ResNetBlock(in_channels=256, out_channels=256, stride=1)
        self.depth_enc_7=ResNetBlock(in_channels=512, out_channels=512, stride=2)
        self.depth_enc_8=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #Depth Decoder
        self.depth_dec_1=deconv_bn_relu(in_channels=512, out_channels=256)
        self.depth_dec_2=deconv_bn_relu(in_channels=256, out_channels=128)
        self.depth_dec_3=deconv_bn_relu(in_channels=128, out_channels=64)
        self.depth_dec_4=deconv_bn_relu(in_channels=64, out_channels=32)
        
        self.depth_dec_5=conv_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1)
        
        self.softmax = nn.Softmax(dim=1)

        init_weights(self)
        
    def forward(self, input):

        rgb = input['rgb']
        d = input['d']

        #join the rgb and the sparse information
        rgb_encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        
        #RGB Encoder
        rgb_encoder_feature_1=self.rgb_enc_1(rgb_encoder_feature_init)
        rgb_encoder_feature_2=self.rgb_enc_2(rgb_encoder_feature_1)
        rgb_encoder_feature_3=self.rgb_enc_3(rgb_encoder_feature_2)
        rgb_encoder_feature_4=self.rgb_enc_4(rgb_encoder_feature_3)
        rgb_encoder_feature_5=self.rgb_enc_5(rgb_encoder_feature_4)
        rgb_encoder_feature_6=self.rgb_enc_6(rgb_encoder_feature_5)
        rgb_encoder_feature_7=self.rgb_enc_7(rgb_encoder_feature_6)
        rgb_encoder_feature_8=self.rgb_enc_8(rgb_encoder_feature_7)

        #RGB Decoder
        rgb_decoder_feature_1=self.rgb_dec_1(rgb_encoder_feature_8)
        rgb_decoder_feature_1_plus=rgb_decoder_feature_1+rgb_encoder_feature_6 #skip connection

        rgb_decoder_feature_2=self.rgb_dec_2(rgb_decoder_feature_1_plus)
        rgb_decoder_feature_2_plus=rgb_decoder_feature_2+rgb_encoder_feature_4 #skip connection

        rgb_decoder_feature_3=self.rgb_dec_3(rgb_decoder_feature_2_plus)
        rgb_decoder_feature_3_plus=rgb_decoder_feature_3+rgb_encoder_feature_2 #skip connection

        rgb_decoder_feature_4=self.rgb_dec_4(rgb_decoder_feature_3_plus)
        rgb_decoder_feature_4_plus=rgb_decoder_feature_4+rgb_encoder_feature_init #skip connection

        rgb_decoder_feature_5=self.rgb_dec_5(rgb_decoder_feature_4_plus)

        #RGB Output
        rgb_depth=rgb_decoder_feature_5[:, 0:1, :, :]
        rgb_confidence=rgb_decoder_feature_5[:, 1:2, :, :]

        #join the output of the rgb and the sparse information
        depth_encoder_feature_init=self.depth_first_layer(torch.cat((rgb_depth, d),dim=1))

        #Depth Encoder
        
        depth_encoder_feature_1=self.depth_enc_1(depth_encoder_feature_init)
        depth_encoder_feature_2=self.depth_enc_2(depth_encoder_feature_1)
        
        depth_encoder_feature_2_plus=torch.cat([rgb_decoder_feature_3_plus,depth_encoder_feature_2],dim=1)#skip connection
        depth_encoder_feature_3=self.depth_enc_3(depth_encoder_feature_2_plus)
        depth_encoder_feature_4=self.depth_enc_4(depth_encoder_feature_3)

        depth_encoder_feature_4_plus=torch.cat([rgb_decoder_feature_2_plus,depth_encoder_feature_4],dim=1)#skip connection
        depth_encoder_feature_5=self.depth_enc_5(depth_encoder_feature_4_plus)
        depth_encoder_feature_6=self.depth_enc_6(depth_encoder_feature_5)

        depth_encoder_feature_6_plus=torch.cat([rgb_decoder_feature_1_plus,depth_encoder_feature_6],dim=1)#skip connection
        depth_encoder_feature_7=self.depth_enc_7(depth_encoder_feature_6_plus)
        depth_encoder_feature_8=self.depth_enc_8(depth_encoder_feature_7)

        #Depth Decoder

        fusion1 = rgb_encoder_feature_8 + depth_encoder_feature_8#skip connection
        depth_decoder_feature_1 = self.depth_dec_1(fusion1)

        fusion2 = rgb_encoder_feature_6 + depth_decoder_feature_1#skip connection
        depth_decoder_feature_2 = self.depth_dec_2(fusion2)

        fusion3 = rgb_encoder_feature_4 + depth_decoder_feature_2#skip connection
        depth_decoder_feature_3 = self.depth_dec_3(fusion3)

        fusion4 = rgb_encoder_feature_2 + depth_decoder_feature_3#skip connection
        depth_decoder_feature_4 = self.depth_dec_4(fusion4)

        depth_decoder_feature_5 = self.depth_dec_5(depth_decoder_feature_4)
        
        #Depth Output
        depth_depth, depth_confidence = torch.chunk(depth_decoder_feature_5, 2, dim=1)

        rgb_confidence, depth_confidence = torch.chunk(self.softmax(torch.cat((rgb_confidence, depth_confidence), dim=1)), 2, dim=1)
        
        output = rgb_confidence*rgb_depth + depth_confidence*depth_depth

        return output#, rgb_depth, depth_depth, rgb_confidence, depth_confidence 
   

class BasicModelMultipleInputs(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers, but this time it supports multiple in and outputs
    """
    def __init__(self):
        super(BasicModelMultipleInputs, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv_bn_relu(in_channels=16,kernel_size=5,out_channels=32,stride=1, padding=2)

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

        #Decoder central
        self.dec_central_1=deconv_bn_relu(in_channels=1024, out_channels=512)
        self.dec_central_2=deconv_bn_relu(in_channels=512, out_channels=256)
        self.dec_central_3=deconv_bn_relu(in_channels=256, out_channels=128)
        self.dec_central_4=deconv_bn_relu(in_channels=128, out_channels=64)
        self.dec_central_5=deconv_bn_relu(in_channels=64, out_channels=32)
        self.dec_central_6=deconv_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0)

        init_weights(self)
        
    def forward(self,input):

        rgb_central = input['rgb_v4']
        d_central = input['depth_v4']

        rgb_v0 = input['rgb_v0']
        rgb_v1 = input['rgb_v2']
        rgb_v2 = input['rgb_v6']
        rgb_v3 = input['rgb_v8']
        """
        print(rgb_central.shape)
        print(rgb_v0.shape)
        print(rgb_v1.shape)
        print(rgb_v2.shape)
        print(rgb_v3.shape)
        print(d_central.shape)
        """
        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb_central, d_central, rgb_v0, rgb_v1, rgb_v2, rgb_v3),dim=1))
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
        
        #Decoder v3
        decoder_feature_central_1=self.dec_central_1(encoder_feature_10)
        decoder_feature_central_1_plus=decoder_feature_central_1+encoder_feature_8 #skip connection1
        decoder_feature_central_2=self.dec_central_2(decoder_feature_central_1_plus)
        decoder_feature_central_2_plus=decoder_feature_central_2+encoder_feature_6 #skip connection1
        decoder_feature_central_3=self.dec_central_3(decoder_feature_central_2_plus)
        decoder_feature_central_3_plus=decoder_feature_central_3+encoder_feature_4 #skip connection1
        decoder_feature_central_4=self.dec_central_4(decoder_feature_central_3_plus)
        decoder_feature_central_4_plus=decoder_feature_central_4+encoder_feature_2 #skip connection1
        decoder_feature_central_5=self.dec_central_5(decoder_feature_central_4_plus)
        decoder_feature_central_5_plus=decoder_feature_central_5+encoder_feature_init #skip connection1
        decoder_feature_central_6=self.dec_central_6(decoder_feature_central_5_plus)

        #Output central
        depth_central=decoder_feature_central_6[:, 0:1, :, :]
        confidence_central=decoder_feature_central_6[:, 1:2, :, :]
        output_central=depth_central*confidence_central
        return output_central 

class DummyModel(nn.Module):
    """
    Dummy model to try the correctness of the framework built for train and test the models
    """
    def __init__(self):
        super(DummyModel, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv_bn_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)
        
        #Decoder central
        self.dec_central_6=deconv_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0)

        init_weights(self)
        
    def forward(self,input):

        rgb=input['rgb']
        depth=input['d']
        
        
        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, depth),dim=1))   
        
        #Decoder v3
        decoder_feature_central_6=self.dec_central_6(encoder_feature_init)

        #Output central
        depth_central=decoder_feature_central_6[:, 0:1, :, :]
        confidence_central=decoder_feature_central_6[:, 1:2, :, :]
        output=depth_central*confidence_central
        return output 



class BasicModelDeepResNext(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers, but this time is deeper and using ResNext architecture instead of ResNet
    """
    def __init__(self, cardinality=6, base_width=6, dilation=1):
        super(BasicModelDeepResNext, self).__init__()
        
        self.cardinality=cardinality
        self.base_width=base_width
        self.dilation=dilation

        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_bn_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)

        #Encoder
        self.enc_1=ResNextBlock(in_channels=32, out_channels=64, stride=2, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_2=ResNextBlock(in_channels=64, out_channels=64, stride=1, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_3=ResNextBlock(in_channels=64, out_channels=128, stride=2, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_4=ResNextBlock(in_channels=128, out_channels=128, stride=1, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_5=ResNextBlock(in_channels=128, out_channels=256, stride=2, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_6=ResNextBlock(in_channels=256, out_channels=256, stride=1, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_7=ResNextBlock(in_channels=256, out_channels=512, stride=2, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_8=ResNextBlock(in_channels=512, out_channels=512, stride=1, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_9=ResNextBlock(in_channels=512, out_channels=1024, stride=2, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)
        self.enc_10=ResNextBlock(in_channels=1024, out_channels=1024, stride=1, cardinality=self.cardinality, base_width=self.base_width, dilation=self.dilation)

        #Decoder
        self.dec_1=deconv3x3_bn_relu(in_channels=1024, out_channels=512)
        self.dec_2=deconv3x3_bn_relu(in_channels=512, out_channels=256)
        self.dec_3=deconv3x3_bn_relu(in_channels=256, out_channels=128)
        self.dec_4=deconv3x3_bn_relu(in_channels=128, out_channels=64)
        self.dec_5=deconv3x3_bn_relu(in_channels=64, out_channels=32)
        self.dec_6=deconv3x3_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0)

        init_weights(self)
        
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
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_8 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_6 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_4 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4+encoder_feature_2 #skip connection

        decoder_feature_5=self.dec_5(decoder_feature_4_plus)
        decoder_feature_5_plus=decoder_feature_5+encoder_feature_init #skip connection

        decoder_feature_6=self.dec_6(decoder_feature_5_plus)

        #Output
        depth=decoder_feature_6[:, 0:1, :, :]
        confidence=decoder_feature_6[:, 1:2, :, :]

        output=depth*confidence

        return output#, depth, confidence 

class BasicModelLight(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers and less parameters
    """
    def __init__(self):
        super(BasicModelLight, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_bn_relu(in_channels=4,kernel_size=5,out_channels=32,stride=1, padding=2)
        #self.first_layer_rgb=conv_bn_relu(in_channels=3,kernel_size=3,out_channels=16,stride=1, padding=1)
        #self.first_layer_depth=conv_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)

        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.enc_2=ResNetBlock(in_channels=64, out_channels=128, stride=2)
        self.enc_3=ResNetBlock(in_channels=128, out_channels=256, stride=2)
        self.enc_4=ResNetBlock(in_channels=256, out_channels=512, stride=2)
        #self.enc_5=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #Decoder
        self.dec_1=deconv3x3_bn_relu(in_channels=512, out_channels=256,padding=1)
        self.dec_2=deconv3x3_bn_relu(in_channels=256, out_channels=128,padding=1)
        self.dec_3=deconv3x3_bn_relu(in_channels=128, out_channels=64,padding=1)
        self.dec_4=deconv3x3_bn_relu(in_channels=64, out_channels=32,padding=1)
        self.dec_5=deconv3x3_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1, output_padding=0,bn=False)
        #self.dec_5=conv_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=1, padding=1,bn=False)

        #init_weights(self)
        self.apply(weights_init(init_type='kaiming'))

        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']
        #print(rgb.shape)
        #print(d.shape)

        #join the rgb and the sparse information
        #encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        #encoder_feature_init_depth=self.first_layer_depth(d)
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        #encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_4)
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_3 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_2 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_1 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4+encoder_feature_init #skip connection

        decoder_feature_5=self.dec_5(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_5[:, 0:1, :, :]
        confidence=decoder_feature_5[:, 1:2, :, :]

        output=depth*confidence

        return output#depth, confidence, output 

class BasicModelUltraLight(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers and leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeess parameters
    """
    def __init__(self, in_channels=4):
        super(BasicModelUltraLight, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.in_channels=in_channels
        if self.in_channels>1:
            self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=8,stride=1, padding=1)
            self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=8,stride=1, padding=1)
        else:#4 channels
            self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)
            
        #Encoder
        self.enc_1=ResNetBlock(in_channels=16, out_channels=32, stride=2)
        self.enc_2=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.enc_3=ResNetBlock(in_channels=64, out_channels=128, stride=2)
        self.enc_4=ResNetBlock(in_channels=128, out_channels=256, stride=2)
        #self.enc_5=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #Decoder
        self.dec_1=deconv3x3_bn_relu(in_channels=256, out_channels=128,padding=1)
        self.dec_2=deconv3x3_bn_relu(in_channels=128, out_channels=64,padding=1)
        self.dec_3=deconv3x3_bn_relu(in_channels=64, out_channels=32,padding=1)
        self.dec_4=deconv3x3_bn_relu(in_channels=32, out_channels=1,bn=False,padding=1)
        #self.dec_5=deconv_bn_relu(in_channels=16, out_channels=1,kernel_size=3,bn=False,padding=1)
        #self.dec_5=conv3x3_bn_relu(in_channels=16, out_channels=2,kernel_size=3, stride=1, padding=1,bn=False)

        #init_weights(self)
        #self.apply(weights_init(init_type='kaiming'))
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']
        #print(rgb.shape)
        #print(d.shape)

        #join the rgb and the sparse information
        if self.in_channels>1:
            encoder_feature_init_rgb=self.first_layer_rgb(rgb)
            encoder_feature_init_depth=self.first_layer_depth(d)
            encoder_feature_init=torch.cat((encoder_feature_init_rgb, encoder_feature_init_depth),1)        #print(encoder_feature_init.shape)
        else:
            encoder_feature_init_depth=self.first_layer_depth(d)
        #Encoder
        #print(encoder_feature_init.shape)
        encoder_feature_1=self.enc_1(encoder_feature_init)
        #print(encoder_feature_1.shape)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        #print(encoder_feature_2.shape)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        #print(encoder_feature_3.shape)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        #encoder_feature_5=self.enc_5(encoder_feature_4)
        #print(encoder_feature_4.shape)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_4)
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_3 #skip connection
        #print(decoder_feature_1_plus.shape)
        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_2 #skip connection
        #print(decoder_feature_2_plus.shape)
        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_1 #skip connection
        #print(decoder_feature_3_plus.shape)
        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_init #skip connection
        #print(decoder_feature_4_plus.shape)
        #decoder_feature_5=self.dec_5(decoder_feature_4_plus)
        #print(decoder_feature_5.shape)

        #Output
        #depth=decoder_feature_5[:, 0:1, :, :]
        #confidence=decoder_feature_5[:, 1:2, :, :]
        #output=depth*confidence

        return decoder_feature_4_plus#depth, confidence, output 

class BasicModelUltraLight_revisited(nn.Module):
    def __init__(self, in_channels=4):
        super(BasicModelUltraLight_revisited, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.in_channels=in_channels
        if self.in_channels>1:
            self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=16,stride=2, padding=1)
            self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=2, padding=1)
        else:#4 channels
            self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)
        self.relu_1=nn.ReLU(inplace=True)
        
        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.enc_2=ResNetBlock(in_channels=64, out_channels=128, stride=2)
        self.enc_3=ResNetBlock(in_channels=128, out_channels=256, stride=2)
        self.enc_4=ResNetBlock(in_channels=256, out_channels=256, stride=1)
        self.enc_5=ResNetBlock(in_channels=256, out_channels=256, stride=1)
        #self.enc_5=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #Decoder
        self.dec_1=deconv3x3_bn_relu_no_artifacts(in_channels=256, out_channels=256,padding=1,stride=1,output_padding=0, scale_factor=1)
        self.dec_2=deconv3x3_bn_relu_no_artifacts(in_channels=256, out_channels=128,padding=1,stride=1,output_padding=0, scale_factor=2)
        self.dec_3=deconv3x3_bn_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,stride=1,output_padding=0, scale_factor=2)
        self.dec_4=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_5=deconv3x3_bn_relu_no_artifacts(in_channels=32, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=2)
        #self.dec_6=conv3x3(in_channels=1, out_channels=1,padding=1, stride=1)
        
      
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']


        #join the rgb and the sparse information
        if self.in_channels>1:
            encoder_feature_init_rgb=self.first_layer_rgb(rgb)
            encoder_feature_init_depth=self.first_layer_depth(d)
            encoder_feature_init=torch.cat((encoder_feature_init_rgb, encoder_feature_init_depth),1)        #print(encoder_feature_init.shape)
        else:
            encoder_feature_init_depth=self.first_layer_depth(d)
            
        encoder_feature_init=self.relu_1(encoder_feature_init)
        #Encoder
        #print(encoder_feature_init.shape)
        encoder_feature_1=self.enc_1(encoder_feature_init)
        #print(encoder_feature_1.shape)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        #print(encoder_feature_2.shape)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        #print(encoder_feature_3.shape)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        
        encoder_feature_5=self.enc_5(encoder_feature_4)
        #print(encoder_feature_4.shape)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=decoder_feature_1#+encoder_feature_4 #skip connection
        #print(decoder_feature_1_plus.shape)
        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2#+encoder_feature_2 #skip connection
        #print(decoder_feature_2_plus.shape)
        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3#+encoder_feature_1 #skip connection
        #print(decoder_feature_3_plus.shape)
        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_init #skip connection
        #print(decoder_feature_4_plus.shape)
        decoder_feature_5=self.dec_5(decoder_feature_4_plus)
        #print(decoder_feature_5.shape)
        #decoder_feature_6=self.dec_6(decoder_feature_5)              
        #Output
        #depth=decoder_feature_5[:, 0:1, :, :]
        #confidence=decoder_feature_5[:, 1:2, :, :]
        #output=depth*confidence

        return decoder_feature_5#depth, confidence, output 

class BasicModelUltraLight_revisited_concat(nn.Module):
    def __init__(self, in_channels=4):
        super(BasicModelUltraLight_revisited_concat, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.in_channels=in_channels
        if self.in_channels>1:
            self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=16,stride=2, padding=1)
            self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=2, padding=1)
        else:#4 channels
            self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)
        
        self.relu_1=nn.ReLU(inplace=True)#B,32,256,512
        
        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)#B,64,128,256
        self.enc_2=ResNetBlock(in_channels=64, out_channels=128, stride=1)#B,128,128,256
        self.enc_3=ResNetBlock(in_channels=128, out_channels=256, stride=2)#B,256,64,128
        self.enc_4=ResNetBlock(in_channels=256, out_channels=512, stride=1)#B,512,64,128
        self.enc_5=ResNetBlock(in_channels=512, out_channels=512, stride=1)#B,512,64,128
        #self.enc_5=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #Decoder
        self.dec_1=deconv3x3_bn_relu_no_artifacts(in_channels=512, out_channels=256,padding=1,stride=1,output_padding=0, scale_factor=1)#B,256,64,128
        self.dec_2=deconv3x3_bn_relu_no_artifacts(in_channels=768, out_channels=256,padding=1,stride=1,output_padding=0, scale_factor=2)#B,256,128,256
        self.dec_3=deconv3x3_bn_relu_no_artifacts(in_channels=384, out_channels=128,padding=1,stride=1,output_padding=0, scale_factor=2)#B,128,256,512
        self.dec_4=deconv3x3_bn_relu_no_artifacts(in_channels=160, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)#B,64,512,1024
        self.dec_5=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=2,padding=1, stride=1,output_padding=1, scale_factor=1)#B,1,512,1024
        #self.dec_6=conv3x3(in_channels=1, out_channels=1,padding=1, stride=1)
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']


        #join the rgb and the sparse information
        if self.in_channels>1:
            encoder_feature_init_rgb=self.first_layer_rgb(rgb)
            encoder_feature_init_depth=self.first_layer_depth(d)
            encoder_feature_init=torch.cat((encoder_feature_init_rgb, encoder_feature_init_depth),1)        #print(encoder_feature_init.shape)
        else:
            encoder_feature_init_depth=self.first_layer_depth(d)
            
        encoder_feature_init=self.relu_1(encoder_feature_init)
        #Encoder
        #print(encoder_feature_init.shape)
        encoder_feature_1=self.enc_1(encoder_feature_init)
        #print(encoder_feature_1.shape)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        #print(encoder_feature_2.shape)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        #print(encoder_feature_3.shape)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        
        encoder_feature_5=self.enc_5(encoder_feature_4)
        #print(encoder_feature_4.shape)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_5)
        decoder_feature_1_plus=torch.cat((decoder_feature_1,encoder_feature_4),1)#B,768,64,128
        #print(decoder_feature_1_plus.shape)
        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=torch.cat((decoder_feature_2,encoder_feature_2),1)#B,384,64,128
        #print(decoder_feature_2_plus.shape)
        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=torch.cat((decoder_feature_3,encoder_feature_init),1)#B,384,64,128
        #print(decoder_feature_3_plus.shape)
        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_init #skip connection
        #print(decoder_feature_4_plus.shape)
        decoder_feature_5=self.dec_5(decoder_feature_4_plus)
        #print(decoder_feature_5.shape)
        #decoder_feature_6=self.dec_6(decoder_feature_5)              
        #Output
        depth=decoder_feature_5[:, 0:1, :, :]
        confidence=decoder_feature_5[:, 1:2, :, :]
        output=depth*confidence

        #return self.final_sigmoid(decoder_feature_5)#depth, confidence, output 
        return output

class BasicModelUltraLight_revisited_2branch(nn.Module):
    def __init__(self, in_channels=4):
        super(BasicModelUltraLight_revisited_2branch, self).__init__()

        #Encoder RGB
        self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512

        self.enc_1_rgb=conv3x3_bn_relu(in_channels=32,kernel_size=7,out_channels=32,stride=1, padding=5)#B,32,256,512
        self.enc_2_rgb=ResNetBlock(in_channels=32, out_channels=32, stride=1)#B,32,256,512
        self.enc_3_rgb=ResNetBlock(in_channels=32, out_channels=64, stride=2)#B,64,128,256
        
        self.enc_4_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_5_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_6_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        
        self.enc_7_rgb=ResNetBlock(in_channels=64, out_channels=128, stride=1)#B,128,128,256
        self.enc_8_rgb=ResNetBlock(in_channels=128, out_channels=128, stride=1,dilation=2)#B,128,128,256
        
        #Encoder Depth
        self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=7,out_channels=16,stride=1, padding=5)#B,16,516,1028
        self.enc_1_depth=conv3x3_bn_relu(in_channels=16,kernel_size=5,out_channels=32,stride=2, padding=2)#B,32,256,512
        self.enc_2_depth=conv3x3_bn_relu(in_channels=32,kernel_size=3,out_channels=64,stride=2, padding=2)#B,64,128,256
        self.enc_3_depth=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_4_depth=ResNetBlock(in_channels=64, out_channels=64, stride=1, dilation=2)#B,64,128,256
        
        #Decoder
        self.dec_1=deconv3x3_bn_relu_no_artifacts(in_channels=192, out_channels=128,padding=1,stride=2,output_padding=0, scale_factor=2,dilation=2)#B,128,256,512
        self.dec_2=deconv3x3_bn_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,stride=1,output_padding=0, scale_factor=1)#B,256,256,512
        self.dec_3=deconv3x3_bn_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,stride=1,output_padding=0, scale_factor=2)#B,128,512,1024
        self.dec_4=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_5=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=2,dilation=3)#B,1,512,1024
        #self.dec_6=conv3x3(in_channels=1, out_channels=1,padding=1, stride=1)
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_1_rgb=self.enc_1_rgb(encoder_feature_init_rgb)
        encoder_feature_2_rgb=self.enc_2_rgb(encoder_feature_1_rgb)
        encoder_feature_3_rgb=self.enc_3_rgb(encoder_feature_2_rgb)
        encoder_feature_4_rgb=self.enc_4_rgb(encoder_feature_3_rgb)
        encoder_feature_5_rgb=self.enc_5_rgb(encoder_feature_4_rgb)
        encoder_feature_6_rgb=self.enc_6_rgb(encoder_feature_5_rgb)
        encoder_feature_7_rgb=self.enc_7_rgb(encoder_feature_6_rgb)
        encoder_feature_8_rgb=self.enc_8_rgb(encoder_feature_7_rgb)
        #print("Init rgb->"+str(encoder_feature_init_rgb.shape))
        #print("1 rgb->"+str(encoder_feature_1_rgb.shape))
        #print("2 rgb->"+str(encoder_feature_2_rgb.shape))
        #print("3 rgb->"+str(encoder_feature_3_rgb.shape))
        #print("4 rgb->"+str(encoder_feature_4_rgb.shape))
        #print("5 rgb->"+str(encoder_feature_5_rgb.shape))
        #print("6 rgb->"+str(encoder_feature_6_rgb.shape))
        #print("7 rgb->"+str(encoder_feature_7_rgb.shape))
        #print("8 rgb->"+str(encoder_feature_8_rgb.shape))
        #Depth branch
        encoder_feature_init_depth=self.first_layer_depth(d)
        encoder_feature_1_depth=self.enc_1_depth(encoder_feature_init_depth)
        encoder_feature_2_depth=self.enc_2_depth(encoder_feature_1_depth)
        encoder_feature_3_depth=self.enc_3_depth(encoder_feature_2_depth)
        encoder_feature_4_depth=self.enc_4_depth(encoder_feature_3_depth)
        #print("Init depth->"+str(encoder_feature_init_depth.shape))
        #print("1 depth->"+str(encoder_feature_1_depth.shape))
        #print("2 depth->"+str(encoder_feature_2_depth.shape))
        #print("3 depth->"+str(encoder_feature_3_depth.shape))
        #print("4 depth->"+str(encoder_feature_4_depth.shape))

        #Join both representations
        encoder_final_joint=torch.cat((encoder_feature_8_rgb,encoder_feature_4_depth),1) #B,192,130,258   
        #print("Joint imgs->"+str(encoder_final_joint.shape))
        #Decoder
        dec_1=self.dec_1(encoder_final_joint)
        dec_2=self.dec_2(dec_1)
        dec_3=self.dec_3(dec_2)
        dec_4=self.dec_4(dec_3)
        dec_5=self.dec_5(dec_4)
        #print("Dec 1->"+str(dec_1.shape))
        #print("Dec 2->"+str(dec_2.shape))
        #print("Dec 3->"+str(dec_3.shape))
        #print("Dec 4->"+str(dec_4.shape))
        #print("Dec 5->"+str(dec_5.shape))
        return dec_5#self.final_sigmoid(decoder_feature_5)#depth, confidence, output 

class BasicModelUltraLight_revisited_2branch_2_cat(nn.Module):
    def __init__(self, in_channels=4):
        super(BasicModelUltraLight_revisited_2branch_2_cat, self).__init__()

        #Encoder RGB
        self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512

        self.enc_1_rgb=conv3x3_bn_relu(in_channels=32,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        self.enc_2_rgb=ResNetBlock(in_channels=32, out_channels=32, stride=1)#B,32,256,512
        self.enc_3_rgb=ResNetBlock(in_channels=32, out_channels=64, stride=2)#B,64,128,256
        
        self.enc_4_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_5_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_6_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        
        self.enc_7_rgb=ResNetBlock(in_channels=64, out_channels=128, stride=1)#B,128,128,256
        self.enc_8_rgb=ResNetBlock(in_channels=128, out_channels=128, stride=1)#B,128,128,256
        
        #Encoder Depth
        self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)#B,16,516,1028
        self.enc_1_depth=conv3x3_bn_relu(in_channels=16,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512
        self.enc_2_depth=conv3x3_bn_relu(in_channels=32,kernel_size=3,out_channels=64,stride=2, padding=1)#B,64,128,256
        self.enc_3_depth=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_4_depth=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        
        #Decoder
        self.dec_1=deconv3x3_bn_relu_no_artifacts(in_channels=192, out_channels=128,padding=1,stride=2,output_padding=0, scale_factor=2)#B,128,256,512
        self.dec_2=deconv3x3_bn_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,stride=1,output_padding=0, scale_factor=1)#B,256,256,512
        self.dec_3=deconv3x3_bn_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,stride=1,output_padding=0, scale_factor=2)#B,128,512,1024
        self.dec_4=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_5=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=2)#B,1,512,1024
        #self.dec_6=conv3x3(in_channels=1, out_channels=1,padding=1, stride=1)
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_1_rgb=self.enc_1_rgb(encoder_feature_init_rgb)
        encoder_feature_2_rgb=self.enc_2_rgb(encoder_feature_1_rgb)
        encoder_feature_3_rgb=self.enc_3_rgb(encoder_feature_2_rgb)
        encoder_feature_4_rgb=self.enc_4_rgb(encoder_feature_3_rgb)
        encoder_feature_5_rgb=self.enc_5_rgb(encoder_feature_4_rgb)
        encoder_feature_6_rgb=self.enc_6_rgb(encoder_feature_5_rgb)
        encoder_feature_7_rgb=self.enc_7_rgb(encoder_feature_6_rgb)
        encoder_feature_8_rgb=self.enc_8_rgb(encoder_feature_7_rgb)
        #print("Init rgb->"+str(encoder_feature_init_rgb.shape))
        #print("1 rgb->"+str(encoder_feature_1_rgb.shape))
        #print("2 rgb->"+str(encoder_feature_2_rgb.shape))
        #print("3 rgb->"+str(encoder_feature_3_rgb.shape))
        #print("4 rgb->"+str(encoder_feature_4_rgb.shape))
        #print("5 rgb->"+str(encoder_feature_5_rgb.shape))
        #print("6 rgb->"+str(encoder_feature_6_rgb.shape))
        #print("7 rgb->"+str(encoder_feature_7_rgb.shape))
        #print("8 rgb->"+str(encoder_feature_8_rgb.shape))
        #Depth branch
        encoder_feature_init_depth=self.first_layer_depth(d)
        encoder_feature_1_depth=self.enc_1_depth(encoder_feature_init_depth)
        encoder_feature_2_depth=self.enc_2_depth(encoder_feature_1_depth)
        encoder_feature_3_depth=self.enc_3_depth(encoder_feature_2_depth)
        encoder_feature_4_depth=self.enc_4_depth(encoder_feature_3_depth)
        #print("Init depth->"+str(encoder_feature_init_depth.shape))
        #print("1 depth->"+str(encoder_feature_1_depth.shape))
        #print("2 depth->"+str(encoder_feature_2_depth.shape))
        #print("3 depth->"+str(encoder_feature_3_depth.shape))
        #print("4 depth->"+str(encoder_feature_4_depth.shape))

        #Join both representations
        encoder_final_joint=torch.cat((encoder_feature_8_rgb,encoder_feature_4_depth),1) #B,192,130,258   
        #print("Joint imgs->"+str(encoder_final_joint.shape))
        #Decoder
        dec_1=self.dec_1(encoder_final_joint)
        dec_2=self.dec_2(dec_1)
        dec_3=self.dec_3(dec_2)
        dec_4=self.dec_4(dec_3)
        dec_5=self.dec_5(dec_4)
        #print("Dec 1->"+str(dec_1.shape))
        #print("Dec 2->"+str(dec_2.shape))
        #print("Dec 3->"+str(dec_3.shape))
        #print("Dec 4->"+str(dec_4.shape))
        #print("Dec 5->"+str(dec_5.shape))
        return dec_5#self.final_sigmoid(decoder_feature_5)#depth, confidence, output 

class BasicModelUltraLight_revisited_2branch_2_add(nn.Module):
    def __init__(self, in_channels=4):
        super(BasicModelUltraLight_revisited_2branch_2_add, self).__init__()

        #Encoder RGB
        self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512

        self.enc_1_rgb=conv3x3_bn_relu(in_channels=32,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        self.enc_2_rgb=ResNetBlock(in_channels=32, out_channels=32, stride=1)#B,32,256,512
        self.enc_3_rgb=ResNetBlock(in_channels=32, out_channels=64, stride=2)#B,64,128,256
        
        self.enc_4_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_5_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_6_rgb=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        
        self.enc_7_rgb=ResNetBlock(in_channels=64, out_channels=128, stride=1)#B,128,128,256
        self.enc_8_rgb=ResNetBlock(in_channels=128, out_channels=128, stride=1)#B,128,128,256
        
        #Encoder Depth
        self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)#B,16,516,1028
        self.enc_1_depth=conv3x3_bn_relu(in_channels=16,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512
        self.enc_2_depth=conv3x3_bn_relu(in_channels=32,kernel_size=3,out_channels=64,stride=2, padding=1)#B,64,128,256
        self.enc_3_depth=ResNetBlock(in_channels=64, out_channels=64, stride=1)#B,64,128,256
        self.enc_4_depth=ResNetBlock(in_channels=64, out_channels=128, stride=1)#B,64,128,256
        
        #Decoder
        self.dec_1=deconv3x3_bn_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,stride=2,output_padding=0, scale_factor=2)#B,128,256,512
        self.dec_2=deconv3x3_bn_relu_no_artifacts(in_channels=128, out_channels=128,padding=1,stride=1,output_padding=0, scale_factor=1)#B,256,256,512
        self.dec_3=deconv3x3_bn_relu_no_artifacts(in_channels=128, out_channels=64,padding=1,stride=1,output_padding=0, scale_factor=2)#B,128,512,1024
        self.dec_4=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_5=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=2)#B,1,512,1024
        #self.dec_6=conv3x3(in_channels=1, out_channels=1,padding=1, stride=1)
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_1_rgb=self.enc_1_rgb(encoder_feature_init_rgb)
        encoder_feature_2_rgb=self.enc_2_rgb(encoder_feature_1_rgb)
        encoder_feature_3_rgb=self.enc_3_rgb(encoder_feature_2_rgb)
        encoder_feature_4_rgb=self.enc_4_rgb(encoder_feature_3_rgb)
        encoder_feature_5_rgb=self.enc_5_rgb(encoder_feature_4_rgb)
        encoder_feature_6_rgb=self.enc_6_rgb(encoder_feature_5_rgb)
        encoder_feature_7_rgb=self.enc_7_rgb(encoder_feature_6_rgb)
        encoder_feature_8_rgb=self.enc_8_rgb(encoder_feature_7_rgb)
        #print("Init rgb->"+str(encoder_feature_init_rgb.shape))
        #print("1 rgb->"+str(encoder_feature_1_rgb.shape))
        #print("2 rgb->"+str(encoder_feature_2_rgb.shape))
        #print("3 rgb->"+str(encoder_feature_3_rgb.shape))
        #print("4 rgb->"+str(encoder_feature_4_rgb.shape))
        #print("5 rgb->"+str(encoder_feature_5_rgb.shape))
        #print("6 rgb->"+str(encoder_feature_6_rgb.shape))
        #print("7 rgb->"+str(encoder_feature_7_rgb.shape))
        #print("8 rgb->"+str(encoder_feature_8_rgb.shape))
        #Depth branch
        encoder_feature_init_depth=self.first_layer_depth(d)
        encoder_feature_1_depth=self.enc_1_depth(encoder_feature_init_depth)
        encoder_feature_2_depth=self.enc_2_depth(encoder_feature_1_depth)
        encoder_feature_3_depth=self.enc_3_depth(encoder_feature_2_depth)
        encoder_feature_4_depth=self.enc_4_depth(encoder_feature_3_depth)
        #print("Init depth->"+str(encoder_feature_init_depth.shape))
        #print("1 depth->"+str(encoder_feature_1_depth.shape))
        #print("2 depth->"+str(encoder_feature_2_depth.shape))
        #print("3 depth->"+str(encoder_feature_3_depth.shape))
        #print("4 depth->"+str(encoder_feature_4_depth.shape))

        #Join both representations
        encoder_final_joint=encoder_feature_8_rgb+encoder_feature_4_depth #B,192,130,258   
        #print("Joint imgs->"+str(encoder_final_joint.shape))
        #Decoder
        dec_1=self.dec_1(encoder_final_joint)
        dec_2=self.dec_2(dec_1)
        dec_3=self.dec_3(dec_2)
        dec_4=self.dec_4(dec_3)
        dec_5=self.dec_5(dec_4)
        #print("Dec 1->"+str(dec_1.shape))
        #print("Dec 2->"+str(dec_2.shape))
        #print("Dec 3->"+str(dec_3.shape))
        #print("Dec 4->"+str(dec_4.shape))
        #print("Dec 5->"+str(dec_5.shape))
        return dec_5#self.final_sigmoid(decoder_feature_5)#depth, confidence, output 


class TwoBranch_newModel(nn.Module):
    def __init__(self, in_channels=4,debug=False):
        super(TwoBranch_newModel, self).__init__()
        self.debug=debug
        ################################
        #         Encoder RGB          #
        ################################
        self.enc_1_rgb=conv3x3(3,32,padding=1)
        self.relu_1_rgb=nn.LeakyReLU(inplace=True)
        self.enc_2_rgb=conv3x3(32,32,stride=2,padding=1)
        self.relu_2_rgb=nn.LeakyReLU(inplace=True)
        self.enc_3_rgb=conv3x3(32,32,padding=1)
        self.relu_3_rgb=nn.LeakyReLU(inplace=True)
        
        ################################
        #         Encoder Depth        #
        ################################
        self.enc_1_depth=conv3x3(1,32,padding=1)
        self.relu_1_depth=nn.LeakyReLU(inplace=True)
        self.enc_2_depth=conv3x3(32,32,stride=2,padding=1)
        self.relu_2_depth=nn.LeakyReLU(inplace=True)
        self.enc_3_depth=conv3x3(32,32,padding=1)
        self.relu_3_depth=nn.LeakyReLU(inplace=True)
        
        ################################
        #          Decoder             #
        ################################
        self.dec_1=nn.ConvTranspose2d(64,64,3,padding=1)
        self.relu_1_dec=nn.LeakyReLU(inplace=True)
        self.dec_2=nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1)
        self.relu_2_dec=nn.LeakyReLU(inplace=True)
        self.dec_3=nn.ConvTranspose2d(32,32,3,padding=1)
        self.relu_3_dec=nn.LeakyReLU(inplace=True)
        self.dec_4=nn.ConvTranspose2d(32,1,3,padding=1)
        self.relu_4_dec=nn.LeakyReLU(inplace=True)
        self.dec_final=nn.Conv2d(1,1,1,padding=0)
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        depth = input['d']

        ################################
        #         Encoder RGB          #
        ################################
        rgb=self.relu_1_rgb(self.enc_1_rgb(rgb))
        #print("Enc_1_rgb->"+str(rgb.shape))
        rgb=self.relu_2_rgb(self.enc_2_rgb(rgb))
        #print("Enc_2_rgb->"+str(rgb.shape))
        rgb=self.relu_3_rgb(self.enc_3_rgb(rgb))
        #print("Enc_3_rgb->"+str(rgb.shape))
        
        ################################
        #         Encoder Depth        #
        ################################
        depth=self.relu_1_depth(self.enc_1_depth(depth))
        #print("Enc_1_depth->"+str(depth.shape))
        depth=self.relu_2_depth(self.enc_2_depth(depth))
        #print("Enc_2_depth->"+str(depth.shape))
        depth=self.relu_3_depth(self.enc_3_depth(depth))
        #print("Enc_3_depth->"+str(depth.shape))
            
        ################################
        #          Concat              #
        ################################
        encoder_final=torch.cat((rgb, depth),1)
        #print("Joint->"+str(encoder_final.shape))
        
        ################################
        #          Decoder             #
        ################################
        dec_out=self.relu_1_dec(self.dec_1(encoder_final))
        #print("Dec_1->"+str(dec_out.shape))
        dec_out=self.relu_2_dec(self.dec_2(dec_out))
        #print("Dec_2->"+str(dec_out.shape))
        dec_out=self.relu_3_dec(self.dec_3(dec_out))
        #print("Dec_3->"+str(dec_out.shape))
        dec_out=self.relu_4_dec(self.dec_4(dec_out))
        #print("Dec_4->"+str(dec_out.shape))
        dec_out=self.dec_final(dec_out)
        #print("Dec_5->"+str(dec_out.shape))
        return self.final_sigmoid(dec_out)


class TwoBranch_newModel_in(nn.Module):
    def __init__(self, in_channels=4,debug=False):
        super(TwoBranch_newModel_in, self).__init__()
        self.debug=debug
        ################################
        #         Encoder RGB          #
        ################################
        self.enc_1_rgb=conv3x3(3,32,padding=1)
        self.relu_1_rgb=nn.LeakyReLU(inplace=True)
        self.enc_2_rgb=conv3x3(32,32,stride=2,padding=1)
        self.relu_2_rgb=nn.LeakyReLU(inplace=True)
        self.norm_1_rgb=nn.InstanceNorm2d(32)
        self.enc_3_rgb=conv3x3(32,32,padding=1)
        self.relu_3_rgb=nn.LeakyReLU(inplace=True)
        self.norm_2_rgb=nn.InstanceNorm2d(32)
        ################################
        #         Encoder Depth        #
        ################################
        self.enc_1_depth=conv3x3(1,32,padding=1)
        self.relu_1_depth=nn.LeakyReLU(inplace=True)
        self.enc_2_depth=conv3x3(32,32,stride=2,padding=1)
        self.relu_2_depth=nn.LeakyReLU(inplace=True)
        self.norm_1_depth=nn.InstanceNorm2d(32)
        self.enc_3_depth=conv3x3(32,32,padding=1)
        self.relu_3_depth=nn.LeakyReLU(inplace=True)
        self.norm_2_depth=nn.InstanceNorm2d(32)
        ################################
        #          Decoder             #
        ################################
        self.dec_1=nn.ConvTranspose2d(64,64,3,padding=1)
        self.relu_1_dec=nn.LeakyReLU(inplace=True)
        self.norm_1_dec=nn.InstanceNorm2d(64)
        self.dec_2=nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1)
        self.relu_2_dec=nn.LeakyReLU(inplace=True)
        self.norm_2_dec=nn.InstanceNorm2d(32)
        self.dec_3=nn.ConvTranspose2d(32,32,3,padding=1)
        self.relu_3_dec=nn.LeakyReLU(inplace=True)
        self.norm_3_dec=nn.InstanceNorm2d(32)
        self.dec_4=nn.ConvTranspose2d(32,1,3,padding=1)
        self.relu_4_dec=nn.LeakyReLU(inplace=True)
        self.dec_final=nn.Conv2d(1,1,1,padding=0)
        
        self.final_sigmoid=nn.Sigmoid()
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        depth = input['d']

        ################################
        #         Encoder RGB          #
        ################################
        rgb=self.relu_1_rgb(self.enc_1_rgb(rgb))
        #print("Enc_1_rgb->"+str(rgb.shape))
        rgb=self.relu_2_rgb(self.enc_2_rgb(rgb))
        #print("Enc_2_rgb->"+str(rgb.shape))
        rgb=self.relu_3_rgb(self.enc_3_rgb(rgb))
        #print("Enc_3_rgb->"+str(rgb.shape))
        
        ################################
        #         Encoder Depth        #
        ################################
        depth=self.relu_1_depth(self.enc_1_depth(depth))
        #print("Enc_1_depth->"+str(depth.shape))
        depth=self.relu_2_depth(self.enc_2_depth(depth))
        #print("Enc_2_depth->"+str(depth.shape))
        depth=self.relu_3_depth(self.enc_3_depth(depth))
        #print("Enc_3_depth->"+str(depth.shape))
            
        ################################
        #          Concat              #
        ################################
        encoder_final=torch.cat((rgb, depth),1)
        #print("Joint->"+str(encoder_final.shape))
        
        ################################
        #          Decoder             #
        ################################
        dec_out=self.norm_1_dec(self.relu_1_dec(self.dec_1(encoder_final)))
        #print("Dec_1->"+str(dec_out.shape))
        dec_out=self.norm_2_dec(self.relu_2_dec(self.dec_2(dec_out)))
        #print("Dec_2->"+str(dec_out.shape))
        dec_out=self.norm_3_dec(self.relu_3_dec(self.dec_3(dec_out)))
        #print("Dec_3->"+str(dec_out.shape))
        dec_out=self.relu_4_dec(self.dec_4(dec_out))
        #print("Dec_4->"+str(dec_out.shape))
        dec_out=self.dec_final(dec_out)
        #print("Dec_5->"+str(dec_out.shape))
        return self.final_sigmoid(dec_out)


class TwoBranch_newModel_bn(nn.Module):
    def __init__(self, in_channels=4,debug=False):
        super(TwoBranch_newModel_bn, self).__init__()
        self.debug=debug
        ################################
        #         Encoder RGB          #
        ################################
        self.enc_1_rgb=conv3x3(3,32,padding=1)
        self.relu_1_rgb=nn.LeakyReLU(inplace=True)
        self.enc_2_rgb=conv3x3(32,32,stride=2,padding=1)
        self.relu_2_rgb=nn.LeakyReLU(inplace=True)
        self.norm_1_rgb=nn.BatchNorm2d(32)
        self.enc_3_rgb=conv3x3(32,32,padding=1)
        self.relu_3_rgb=nn.LeakyReLU(inplace=True)
        self.norm_2_rgb=nn.BatchNorm2d(32)
        ################################
        #         Encoder Depth        #
        ################################
        self.enc_1_depth=conv3x3(1,32,padding=1)
        self.relu_1_depth=nn.LeakyReLU(inplace=True)
        self.enc_2_depth=conv3x3(32,32,stride=2,padding=1)
        self.relu_2_depth=nn.LeakyReLU(inplace=True)
        self.norm_1_depth=nn.BatchNorm2d(32)
        self.enc_3_depth=conv3x3(32,32,padding=1)
        self.relu_3_depth=nn.LeakyReLU(inplace=True)
        self.norm_2_depth=nn.BatchNorm2d(32)
        ################################
        #          Decoder             #
        ################################
        self.dec_1=nn.ConvTranspose2d(64,64,3,padding=1)
        self.relu_1_dec=nn.LeakyReLU(inplace=True)
        self.norm_1_dec=nn.BatchNorm2d(64)
        self.dec_2=nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1)
        self.relu_2_dec=nn.LeakyReLU(inplace=True)
        self.norm_2_dec=nn.BatchNorm2d(32)
        self.dec_3=nn.ConvTranspose2d(32,32,3,padding=1)
        self.relu_3_dec=nn.LeakyReLU(inplace=True)
        self.norm_3_dec=nn.BatchNorm2d(32)
        self.dec_4=nn.ConvTranspose2d(32,1,3,padding=1)
        self.relu_4_dec=nn.LeakyReLU(inplace=True)
        self.dec_final=nn.Conv2d(1,1,1,padding=0)
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        depth = input['d']

        ################################
        #         Encoder RGB          #
        ################################
        rgb=self.relu_1_rgb(self.enc_1_rgb(rgb))
        #print("Enc_1_rgb->"+str(rgb.shape))
        rgb=self.relu_2_rgb(self.enc_2_rgb(rgb))
        #print("Enc_2_rgb->"+str(rgb.shape))
        rgb=self.relu_3_rgb(self.enc_3_rgb(rgb))
        #print("Enc_3_rgb->"+str(rgb.shape))
        
        ################################
        #         Encoder Depth        #
        ################################
        depth=self.relu_1_depth(self.enc_1_depth(depth))
        #print("Enc_1_depth->"+str(depth.shape))
        depth=self.relu_2_depth(self.enc_2_depth(depth))
        #print("Enc_2_depth->"+str(depth.shape))
        depth=self.relu_3_depth(self.enc_3_depth(depth))
        #print("Enc_3_depth->"+str(depth.shape))
            
        ################################
        #          Concat              #
        ################################
        encoder_final=torch.cat((rgb, depth),1)
        #print("Joint->"+str(encoder_final.shape))
        
        ################################
        #          Decoder             #
        ################################
        dec_out=self.norm_1_dec(self.relu_1_dec(self.dec_1(encoder_final)))
        #print("Dec_1->"+str(dec_out.shape))
        dec_out=self.norm_2_dec(self.relu_2_dec(self.dec_2(dec_out)))
        #print("Dec_2->"+str(dec_out.shape))
        dec_out=self.norm_3_dec(self.relu_3_dec(self.dec_3(dec_out)))
        #print("Dec_3->"+str(dec_out.shape))
        dec_out=self.relu_4_dec(self.dec_4(dec_out))
        #print("Dec_4->"+str(dec_out.shape))
        dec_out=self.dec_final(dec_out)
        #print("Dec_5->"+str(dec_out.shape))
        return self.final_sigmoid(dec_out)


class OnlyCNN(nn.Module):
    def __init__(self):
        super(OnlyCNN, self).__init__()

        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        self.enc_1_rgb=conv3x3_relu(in_channels=32,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)#B,16,516,1028
        self.enc_1_depth=conv3x3_relu(in_channels=32,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)#B,1,512,1024
        self.dec_6=conv1x1(in_channels=1, out_channels=1, stride=1)
           
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        #print("1 RGB->"+str(encoder_feature_init_rgb.shape))
        encoder_feature_1_rgb=self.enc_1_rgb(encoder_feature_init_rgb)
        #print("2 RGB->"+str(encoder_feature_1_rgb.shape))
        #Depth branch
        encoder_feature_init_depth=self.first_layer_depth(d)
        #print("1 Depth->"+str(encoder_feature_init_rgb.shape))
        encoder_feature_1_depth=self.enc_1_depth(encoder_feature_init_depth)
        #print("2 Depth->"+str(encoder_feature_1_depth.shape))

        #Join both representations
        encoder_final_joint=torch.cat((encoder_feature_1_rgb,encoder_feature_1_depth),1)
        #print("Joint imgs->"+str(encoder_final_joint.shape))
        
        #Decoder
        dec_1=self.dec_1(encoder_final_joint)
        dec_2=self.dec_2(dec_1)
        dec_3=self.dec_3(dec_2)
        return self.final_sigmoid(self.dec_6(dec_3))#self.final_sigmoid(decoder_feature_5)#depth, confidence, output 

class OnlyCNN_big_filters(nn.Module):
    def __init__(self):
        super(OnlyCNN_big_filters, self).__init__()

        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        self.enc_1_rgb=conv3x3_relu(in_channels=32,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)#B,16,516,1028
        self.enc_1_depth=conv3x3_relu(in_channels=32,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(kernel_size=3,in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_2=deconv3x3_relu_no_artifacts(kernel_size=3,in_channels=32, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)#B,1,512,1024
        self.dec_6=conv1x1(in_channels=1, out_channels=1, stride=1)
           
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        #print("1 RGB->"+str(encoder_feature_init_rgb.shape))
        encoder_feature_1_rgb=self.enc_1_rgb(encoder_feature_init_rgb)
        #print("2 RGB->"+str(encoder_feature_1_rgb.shape))
        #Depth branch
        encoder_feature_init_depth=self.first_layer_depth(d)
        #print("1 Depth->"+str(encoder_feature_init_rgb.shape))
        encoder_feature_1_depth=self.enc_1_depth(encoder_feature_init_depth)
        #print("2 Depth->"+str(encoder_feature_1_depth.shape))

        #Join both representations
        encoder_final_joint=torch.cat((encoder_feature_1_rgb,encoder_feature_1_depth),1)
        #print("Joint imgs->"+str(encoder_final_joint.shape))
        
        #Decoder
        dec_1=self.dec_1(encoder_final_joint)
        dec_2=self.dec_2(dec_1)
        dec_3=self.dec_3(dec_2)
        return self.final_sigmoid(self.dec_6(dec_3))#self.final_sigmoid(decoder_feature_5)#depth, confidence, output 


class AttModel(nn.Module):
    def __init__(self,attLayers=4,deconvLayers=3,attentionChannels=64):
        super(AttModel, self).__init__()

        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,16,516,1028
        
        self.conv_intermediate=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=64,stride=2, padding=1)
        

        self.att_1=DIYSelfAttention(64)
        self.conv_intermediate_2=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=128,stride=2, padding=1)
        self.att_2=DIYSelfAttention(128)
        #self.att_3=DIYSelfAttention(128)
        self.conv_intermediate_3=conv3x3_relu(in_channels=128,kernel_size=3,out_channels=256,stride=1, padding=1)
        #self.att_3=DIYSelfAttention(256)
        #self.conv_intermediate_4=conv3x3_relu(in_channels=256,kernel_size=3,out_channels=512,stride=1, padding=1)
        #self.att_4=DIYSelfAttention(512)
        
        #self.dec_1=deconv3x3_relu_no_artifacts(in_channels=512, out_channels=256,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_5=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=2,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)

        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_init_depth=self.first_layer_depth(d)

        #Join both representations
        out=torch.cat((encoder_feature_init_rgb,encoder_feature_init_depth),1)
        out=self.conv_intermediate(out)
                
        out=self.conv_intermediate_2(self.att_1(out))
        out=self.conv_intermediate_3(self.att_2(out))
        #out=self.att_3(out)
        #out=self.conv_intermediate_4(self.att_3(out))
        #out=self.att_4(out)        
                
        #out=self.dec_1(out)
        out=self.dec_2(out)
        out=self.dec_3(out)
        out=self.dec_4(out)
        out=self.dec_5(out)
        
        depth=out[:, 0:1, :, :]
        confidence=out[:, 1:2, :, :]

        out=depth*confidence
                
        return self.final_sigmoid(out) 

class SelfAttentionModel(nn.Module):
    def __init__(self,attLayers=4,deconvLayers=3,attentionChannels=64):
        super(SelfAttentionModel, self).__init__()
        self.deconvLayers=deconvLayers
        self.attLayers=attLayers
        out_channels=int(attentionChannels/2)

        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=out_channels,stride=2, padding=1)#B,32,256,512
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=out_channels,stride=2, padding=1)#B,16,516,1028
        
        
        #Self-attention
        #self.modules_attention=[]
        #for att in range(attLayers):
        #    self.modules_attention.append(DIYSelfAttention(attentionChannels))
        self.att_1=DIYSelfAttention(attentionChannels)
        if attLayers==2:
            self.att_2=DIYSelfAttention(attentionChannels)
        if attLayers==3:
            self.att_2=DIYSelfAttention(attentionChannels)
            self.att_3=DIYSelfAttention(attentionChannels)
        if attLayers==4:
            self.att_2=DIYSelfAttention(attentionChannels)
            self.att_3=DIYSelfAttention(attentionChannels)
            self.att_4=DIYSelfAttention(attentionChannels)    
        if deconvLayers==1:
            self.dec_1=deconv3x3_relu_no_artifacts(in_channels=attentionChannels, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=2,relu=False)#B,1,512,1024
        if deconvLayers==2:
            self.dec_1=deconv3x3_relu_no_artifacts(in_channels=attentionChannels, out_channels=out_channels,padding=1, stride=1,output_padding=1, scale_factor=2)
            self.dec_2=deconv3x3_relu_no_artifacts(in_channels=out_channels, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)
        if deconvLayers==3:
            self.dec_1=deconv3x3_relu_no_artifacts(in_channels=attentionChannels, out_channels=out_channels,padding=1, stride=1,output_padding=1, scale_factor=2)
            self.dec_2=deconv3x3_relu_no_artifacts(in_channels=out_channels, out_channels=int(out_channels/2),padding=1, stride=1,output_padding=1, scale_factor=1)
            self.dec_3=deconv3x3_relu_no_artifacts(in_channels=int(out_channels/2), out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)
        if deconvLayers==4:
            self.dec_1=deconv3x3_relu_no_artifacts(in_channels=attentionChannels, out_channels=out_channels,padding=1, stride=1,output_padding=1, scale_factor=2)
            self.dec_2=deconv3x3_relu_no_artifacts(in_channels=out_channels, out_channels=int(out_channels/2),padding=1, stride=1,output_padding=1, scale_factor=1)
            self.dec_3=deconv3x3_relu_no_artifacts(in_channels=int(out_channels/2), out_channels=int(out_channels/2),padding=1, stride=1,output_padding=1, scale_factor=1)
            self.dec_4=deconv3x3_relu_no_artifacts(in_channels=int(out_channels/2), out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)
            
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
        
        #Attention layers
        out=self.att_1(out)
        if self.attLayers==2:
            out=self.att_2(out)
        if self.attLayers==3:
            out=self.att_2(out)
            out=self.att_3(out)
        if self.attLayers==4:
            out=self.att_2(out)
            out=self.att_3(out)
            out=self.att_4(out)
            
        #Decoder
        out=self.dec_1(out)
        if self.deconvLayers==2:
            out=self.dec_2(out)
        if self.deconvLayers==3:
            out=self.dec_2(out)
            out=self.dec_3(out)
        if self.deconvLayers==4:
            out=self.dec_2(out)
            out=self.dec_3(out)
            out=self.dec_4(out)
                
        return self.final_sigmoid(out) 

class testceptionBlock(nn.Module):
    def __init__(self,in_channels, out_channels) -> None:
        super(testceptionBlock,self).__init__()
        self.branch1=conv3x3_bn_relu(in_channels,out_channels,kernel_size=1,padding=0)
        self.branch2_1=conv3x3_bn_relu(in_channels,out_channels,kernel_size=1,padding=0)
        self.branch2_2=conv3x3_bn_relu(in_channels,out_channels,kernel_size=3)
        self.branch3_1=conv3x3_bn_relu(in_channels,out_channels,kernel_size=1,padding=0)
        self.branch3_2=conv3x3_bn_relu(in_channels,out_channels,kernel_size=5,padding=2)
        self.branch4_1=nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True)
        self.branch4_2=conv3x3_bn_relu(in_channels,out_channels,kernel_size=1,padding=0)
    def forward(self, input):
        branch1 = self.branch1(input)#[1, 64, 800, 800]
        #print("Branch 1->"+str(branch1.shape))
        branch2_1 = self.branch2_1(input)#[1, 64, 800, 800]
        branch2_2 = self.branch2_2(input)#[1, 64, 800, 800]
        #print("Branch 2_1->"+str(branch2_1.shape))
        #print("Branch 2_2->"+str(branch2_2.shape))
        branch3_1 = self.branch3_1(input)#[1, 64, 800, 800]
        branch3_2 = self.branch3_2(input)#[1, 64, 800, 800]
        #print("Branch 3_1->"+str(branch3_1.shape))
        #print("Branch 3_2->"+str(branch3_2.shape))
        branch4_1 = self.branch4_1(input)#[1, 64, 800, 800]
        branch4_2 = self.branch4_2(input)#[1, 64, 800, 800]
        #print("Branch 4_1->"+str(branch4_1.shape))
        #print("Branch 4_2->"+str(branch4_2.shape))
        outputs = [branch1,branch2_2,branch3_2,branch4_2]
        return torch.cat(outputs,1)
    
class inceptionBlock(nn.Module):
    def __init__(self,in_channels, ch1x1,ch3x3,ch5x5,ch3x3_in,ch5x5_in,pool_proj) -> None:
        super(inceptionBlock,self).__init__()
        self.branch1=conv3x3_bn_relu(in_channels,ch1x1,kernel_size=1)
        self.branch2=nn.Sequential(conv3x3_bn_relu(in_channels,ch3x3_in,kernel_size=1,padding=0),conv3x3_bn_relu(ch3x3_in,ch3x3,kernel_size=3))
        self.branch3=nn.Sequential(conv3x3_bn_relu(in_channels,ch5x5_in,kernel_size=1,padding=0),conv3x3_bn_relu(ch5x5_in,ch5x5,kernel_size=5))
        self.branch4=nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),conv3x3_bn_relu(in_channels,pool_proj,kernel_size=1,padding=0))
    def forward(self, input):
        branch1 = self.branch1(input)
        branch2 = self.branch2(input)
        branch3 = self.branch3(input)
        branch4 = self.branch4(input)
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs,1)
    
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
    
class InceptionLikeModel(nn.Module):
    def __init__(self):
        super(InceptionLikeModel, self).__init__()
        
        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        
        self.conv_intermediate=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=64,stride=2, padding=1)
        
        
        self.inception_1=inceptionBlock_light(in_channels=64,ch1x1=32,ch3x3=64,ch5x5=32,ch3x3_in=32,ch5x5_in=16)
        self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32) #256
        #self.inception_3=inceptionBlock_light(in_channels=256,ch1x1=128,ch3x3=256,ch5x5=128,ch3x3_in=128,ch5x5_in=64) #512

        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)
           
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
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
        #print("INCEPTION OUTPUT 1->"+str(out.shape))
        out=self.inception_2(out)
        #print("INCEPTION OUTPUT 2->"+str(out.shape))    
        #Decoder
        out=self.dec_1(out)
        out=self.dec_2(out)
        out=self.dec_3(out)
        return self.final_sigmoid(out) 

class InceptionLikeModelDeeper(nn.Module):
    def __init__(self):
        super(InceptionLikeModelDeeper, self).__init__()
        
        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        
        self.conv_intermediate=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=64,stride=2, padding=1)
        
        
        self.inception_1=inceptionBlock_light(in_channels=64,ch1x1=32,ch3x3=64,ch5x5=32,ch3x3_in=32,ch5x5_in=16)
        self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32) #256
        self.inception_3=inceptionBlock_light(in_channels=256,ch1x1=128,ch3x3=256,ch5x5=128,ch3x3_in=128,ch5x5_in=64) #512

        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=512, out_channels=256,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=1)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)
           
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
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
        out=self.inception_2(out)
        out=self.inception_3(out)
        #Decoder
        out=self.dec_1(out)
        out=self.dec_2(out)
        out=self.dec_3(out)
        out=self.dec_4(out)
        return self.final_sigmoid(out) 


class InceptionLikeModelDeeper2(nn.Module):
    def __init__(self):
        super(InceptionLikeModelDeeper2, self).__init__()
        
        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        
        self.conv_intermediate=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=64,stride=2, padding=1)
        
        
        self.inception_1=inceptionBlock_light(in_channels=64,ch1x1=32,ch3x3=64,ch5x5=32,ch3x3_in=32,ch5x5_in=16)
        self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32) #256
        self.inception_3=inceptionBlock_light(in_channels=256,ch1x1=128,ch3x3=256,ch5x5=128,ch3x3_in=128,ch5x5_in=64) #512
        self.inception_4=inceptionBlock_light(in_channels=512,ch1x1=256,ch3x3=512,ch5x5=256,ch3x3_in=256,ch5x5_in=128) #1024

        
        
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=1024, out_channels=512,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=512, out_channels=256,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=1)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)
           
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
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
        out=self.inception_2(out)
        out=self.inception_3(out)
        out=self.inception_4(out)
        #Decoder
        out=self.dec_1(out)
        out=self.dec_2(out)
        out=self.dec_3(out)
        out=self.dec_4(out)
        return self.final_sigmoid(out) 

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

class InceptionAndAttentionModel_2(nn.Module):
    def __init__(self):
        super(InceptionAndAttentionModel_2, self).__init__()
        
        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
             
        self.conv_intermediate=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=64,stride=2, padding=1)
        
        self.inception_1=inceptionBlock_light(in_channels=64,ch1x1=32,ch3x3=64,ch5x5=32,ch3x3_in=32,ch5x5_in=16,pool_proj=32) #128
        #self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32) #256
        self.conv_intermediate_2=conv3x3_relu(in_channels=128,kernel_size=3,out_channels=256,stride=2, padding=1)
        
        self.att_1=DIYSelfAttention(256)
        self.att_2=DIYSelfAttention(256)
        
        self.conv_intermediate_3=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=1)
        
        self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32,pool_proj=64) #256
        
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)

        self.dec_5_res=conv3x3(in_channels=32,out_channels=2,bias=True)
           
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']

        #init branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_init_depth=self.first_layer_depth(d)

        #Join both representations
        out=torch.cat((encoder_feature_init_rgb,encoder_feature_init_depth),1)
        out=self.conv_intermediate(out)
        
        out=self.inception_1(out)
        out=self.conv_intermediate_2(out)
        
        out=self.att_1(out)
        out=self.att_2(out)
        
        out=self.conv_intermediate_3(out)
        
        out=self.inception_2(out)
        
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

class InceptionAndAttentionModel_3(nn.Module):
    def __init__(self):
        super(InceptionAndAttentionModel_3, self).__init__()
        
        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
             
        self.conv_intermediate=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=64,stride=2, padding=1)
        
        self.inception_1=inceptionBlock_light(in_channels=64,ch1x1=32,ch3x3=64,ch5x5=32,ch3x3_in=32,ch5x5_in=16,pool_proj=32) #128
        #self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32) #256
        self.conv_intermediate_2=conv3x3_relu(in_channels=128,kernel_size=3,out_channels=256,stride=2, padding=1)
        
        self.att_1=DIYSelfAttention(256)
        self.att_2=DIYSelfAttention(256)
        
        self.conv_intermediate_3=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        
        self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32,pool_proj=64) #256
        
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=1)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)

        self.dec_5_res=conv3x3(in_channels=32,out_channels=2,bias=True)
           
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']

        #init branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_init_depth=self.first_layer_depth(d)

        #Join both representations
        out=torch.cat((encoder_feature_init_rgb,encoder_feature_init_depth),1)
        out=self.conv_intermediate(out)
        
        out_i=self.inception_1(out)
        out=self.conv_intermediate_2(out_i)
        
        out=self.att_1(out)
        out=self.att_2(out)
        
        out=self.conv_intermediate_3(out)
        #print(f' {out.shape}+{out_i.shape} ')
        out_improv=out+out_i
        #print(f'= {out_improv}')
        out=self.inception_2(out_improv)
        
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

class InceptionAndAttentionModel_4(nn.Module):
    def __init__(self):
        super(InceptionAndAttentionModel_4, self).__init__()
        
        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,800,800
             
        self.conv_intermediate=conv3x3_relu(in_channels=64,kernel_size=3,out_channels=64,stride=2, padding=1)
        
        self.inception_1=inceptionBlock_light(in_channels=64,ch1x1=32,ch3x3=64,ch5x5=32,ch3x3_in=32,ch5x5_in=16,pool_proj=32) #128
        #self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32) #256
        self.conv_intermediate_2=conv3x3_relu(in_channels=128,kernel_size=3,out_channels=256,stride=2, padding=1)
        
        self.att_1=DIYSelfAttention(256)
        self.att_2=DIYSelfAttention(256)
        
        self.conv_intermediate_3=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        
        self.inception_2=inceptionBlock_light(in_channels=128,ch1x1=64,ch3x3=128,ch5x5=64,ch3x3_in=64,ch5x5_in=32,pool_proj=64) #256
        self.inception_3=inceptionBlock_light(in_channels=256,ch1x1=128,ch3x3=256,ch5x5=128,ch3x3_in=128,ch5x5_in=64,pool_proj=128) #512

        
        self.dec_0=deconv3x3_relu_no_artifacts(in_channels=512, out_channels=256,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=256, out_channels=128,padding=1, stride=1,output_padding=1, scale_factor=2)
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=128, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=1)
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=64,padding=1, stride=1,output_padding=1, scale_factor=1)
        self.dec_4=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)

        self.dec_5_res=conv3x3(in_channels=32,out_channels=2,bias=True)
           
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']

        #init branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_init_depth=self.first_layer_depth(d)

        #Join both representations
        out=torch.cat((encoder_feature_init_rgb,encoder_feature_init_depth),1)
        out=self.conv_intermediate(out)
        
        out_i=self.inception_1(out)
        out=self.conv_intermediate_2(out_i)
        
        out=self.att_1(out)
        out=self.att_2(out)
        
        out=self.conv_intermediate_3(out)
        #print(f' {out.shape}+{out_i.shape} ')
        out_improv=out+out_i
        #print(f'= {out_improv}')
        out=self.inception_2(out_improv)
        out=self.inception_3(out)
        #Decoder
        out=self.dec_0(out)
        out=self.dec_1(out)
        out=self.dec_2(out)
        out=self.dec_3(out)
        out=self.dec_4(out)
        out=self.dec_5_res(out)
        
        depth=out[:, 0:1, :, :]
        confidence=out[:, 1:2, :, :]

        out=depth*confidence
        
        
        return self.final_sigmoid(out) 


class FCN(nn.Module):
    """
    Idea from the paper "Depth Map Inpainting Using a Fully Convolutional Network" in https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8961820
    """
    def __init__(self):
        super(OnlyCNN, self).__init__()

        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=7,out_channels=31,stride=2, padding=1)#B,32,256,512
        
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=32,kernel_size=7,out_channels=64,stride=1, padding=1)#B,16,516,1028
        self.enc_1_depth=conv3x3_relu(in_channels=32,kernel_size=3,out_channels=32,stride=1, padding=1)#B,32,256,512
        
        #Decoder
        self.dec_1=deconv3x3_relu_no_artifacts(in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_2=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=1)#B,64,512,1024
        self.dec_3=deconv3x3_relu_no_artifacts(in_channels=32, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)#B,1,512,1024
        self.dec_6=conv3x3(in_channels=1, out_channels=1,padding=1, stride=1)
           
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_final_joint=torch.cat((encoder_feature_init_rgb,d),1) #32channels

        
        
        
        
        #Join both representations
        encoder_final_joint=torch.cat((encoder_feature_1_rgb,encoder_feature_1_depth),1)
        #print("Joint imgs->"+str(encoder_final_joint.shape))
        
        #Decoder
        dec_1=self.dec_1(encoder_final_joint)
        dec_2=self.dec_2(dec_1)
        dec_3=self.dec_3(dec_2)
        return self.final_sigmoid(self.dec_6(dec_3))#self.final_sigmoid(decoder_feature_5)#depth, confidence, output 


class SimplestModel(nn.Module):
    def __init__(self):
        super(SimplestModel, self).__init__()

        #Encoder RGB
        self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512
        self.enc_1_rgb=conv3x3_bn_relu(in_channels=32,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512
        
        #Encoder Depth
        self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=32,stride=2, padding=1)#B,16,516,1028
        self.enc_1_depth=conv3x3_bn_relu(in_channels=32,kernel_size=3,out_channels=32,stride=2, padding=1)#B,32,256,512
        
        #Decoder
        self.dec_1=deconv3x3_bn_relu_no_artifacts(in_channels=64, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=2)#B,64,512,1024
        self.dec_2=deconv3x3_bn_relu_no_artifacts(in_channels=32, out_channels=32,padding=1, stride=1,output_padding=1, scale_factor=2)#B,64,512,1024
        self.dec_3=deconv3x3_bn_relu_no_artifacts(in_channels=32, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1)#B,1,512,1024
        #self.dec_6=conv3x3(in_channels=1, out_channels=1,padding=1, stride=1)
        
        self.final_sigmoid=nn.Sigmoid()
        
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']

        #Rgb branch
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_1_rgb=self.enc_1_rgb(encoder_feature_init_rgb)

        #Depth branch
        encoder_feature_init_depth=self.first_layer_depth(d)
        encoder_feature_1_depth=self.enc_1_depth(encoder_feature_init_depth)

        #Join both representations
        encoder_final_joint=torch.cat((encoder_feature_1_rgb,encoder_feature_1_depth),1)
        #print("Joint imgs->"+str(encoder_final_joint.shape))
        #Decoder
        dec_1=self.dec_1(encoder_final_joint)
        dec_2=self.dec_2(dec_1)
        dec_3=self.dec_3(dec_2)
        return self.final_sigmoid(dec_3)#self.final_sigmoid(decoder_feature_5)#depth, confidence, output 


class BasicModelUltraLight_MoreHidden(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers and leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeess parameters
    """
    def __init__(self, in_channels=4):
        super(BasicModelUltraLight_MoreHidden, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.in_channels=in_channels
        if self.in_channels>1:
            self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=16,stride=1, padding=1)
            self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)
        else:#4 channels
            self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)
            self.first_layer_rgb=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)

        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2)
        self.enc_2=ResNetBlock(in_channels=64, out_channels=128, stride=2)
        self.enc_3=ResNetBlock(in_channels=128, out_channels=256, stride=2)
        self.enc_4=ResNetBlock(in_channels=256, out_channels=512, stride=2)
        #self.enc_5=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #Decoder
        self.dec_1=deconv3x3_bn_relu(in_channels=512, out_channels=256,padding=1)
        self.dec_2=deconv3x3_bn_relu(in_channels=256, out_channels=128,padding=1)
        self.dec_3=deconv3x3_bn_relu(in_channels=128, out_channels=64,padding=1)
        self.dec_4=deconv3x3_bn_relu(in_channels=64, out_channels=1,bn=False,padding=1)
        #self.dec_5=deconv_bn_relu(in_channels=16, out_channels=1,kernel_size=3,bn=False,padding=1)
        #self.dec_5=conv3x3_bn_relu(in_channels=16, out_channels=2,kernel_size=3, stride=1, padding=1,bn=False)

        #init_weights(self)
        #self.apply(weights_init(init_type='kaiming'))
        self.final_sigmoid=nn.Sigmoid()
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']
        #print(rgb.shape)
        #print(d.shape)

        #join the rgb and the sparse information
        if self.in_channels>1:
            encoder_feature_init_rgb=self.first_layer_rgb(rgb)
            encoder_feature_init_depth=self.first_layer_depth(d)
            encoder_feature_init=torch.cat((encoder_feature_init_rgb, encoder_feature_init_depth),1)        #print(encoder_feature_init.shape)
        else:
            encoder_feature_init_depth=self.first_layer_depth(d)
        #Encoder
        #print(encoder_feature_init.shape)
        encoder_feature_1=self.enc_1(encoder_feature_init)
        #print(encoder_feature_1.shape)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        #print(encoder_feature_2.shape)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        #print(encoder_feature_3.shape)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        #encoder_feature_5=self.enc_5(encoder_feature_4)
        #print(encoder_feature_4.shape)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_4)
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_3 #skip connection
        #print(decoder_feature_1_plus.shape)
        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_2 #skip connection
        #print(decoder_feature_2_plus.shape)
        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_1 #skip connection
        #print(decoder_feature_3_plus.shape)
        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_init #skip connection
        #print(decoder_feature_4_plus.shape)
        #decoder_feature_5=self.dec_5(decoder_feature_4_plus)
        #print(decoder_feature_5.shape)

        #Output
        #depth=decoder_feature_5[:, 0:1, :, :]
        #confidence=decoder_feature_5[:, 1:2, :, :]
        #output=depth*confidence

        return self.final_sigmoid(decoder_feature_4_plus)#depth, confidence, output 


class BasicModelUltraLightTwoBranch(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers and leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeess parameters
    """
    def __init__(self, in_channels=4):
        super(BasicModelUltraLightTwoBranch, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.in_channels=in_channels
        if self.in_channels>1:
            self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=16,stride=1, padding=1)
            self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)
        else:#4 channels
            self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)
            self.first_layer_rgb=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=16,stride=1, padding=1)
        #Encoder
        self.enc_1=ResNetBlock(in_channels=32, out_channels=64, stride=2,batch=False)
        self.enc_2=ResNetBlock(in_channels=64, out_channels=128, stride=2,batch=False)
        self.enc_3=ResNetBlock(in_channels=128, out_channels=256, stride=2,batch=False)
        self.enc_4=ResNetBlock(in_channels=256, out_channels=512, stride=2,batch=False)
        #self.enc_5=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #Decoder
        self.dec_1=deconv3x3_relu(in_channels=512, out_channels=256,padding=1)
        self.dec_2=deconv3x3_relu(in_channels=256, out_channels=128,padding=1)
        self.dec_3=deconv3x3_relu(in_channels=128, out_channels=64,padding=1)
        self.dec_4=deconv3x3_relu(in_channels=64, out_channels=1,bn=False,padding=1)
        #self.dec_5=deconv_bn_relu(in_channels=16, out_channels=1,kernel_size=3,bn=False,padding=1)
        #self.dec_5=conv3x3_bn_relu(in_channels=16, out_channels=2,kernel_size=3, stride=1, padding=1,bn=False)

        #init_weights(self)
        self.apply(weights_init(init_type='kaiming'))
        self.relu= nn.ReLU(inplace=True)
        
        self.final_sigmoid=nn.Sigmoid()
    def forward(self,input):
        rgb = input['rgb']
        gray = input['g']
        d = input['d']
        #join the rgb and the sparse information

        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_init_depth=self.first_layer_depth(d)
        encoder_feature_init=torch.cat((encoder_feature_init_rgb, encoder_feature_init_depth),1)        #print(encoder_feature_init.shape)
        encoder_feature_init=self.relu(encoder_feature_init)
        #Encoder
        #print(encoder_feature_init.shape)
        encoder_feature_1=self.enc_1(encoder_feature_init)
        #print(encoder_feature_1.shape)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        #print(encoder_feature_2.shape)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        #print(encoder_feature_3.shape)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        #encoder_feature_5=self.enc_5(encoder_feature_4)
        #print(encoder_feature_4.shape)

        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_4)
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_3 #skip connection
        #print(decoder_feature_1_plus.shape)
        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_2 #skip connection
        #print(decoder_feature_2_plus.shape)
        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_1 #skip connection
        #print(decoder_feature_3_plus.shape)
        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_init #skip connection
        #print(decoder_feature_4_plus.shape)
        #decoder_feature_5=self.dec_5(decoder_feature_4_plus)
        #print(decoder_feature_5.shape)

        #Output
        #depth=decoder_feature_5[:, 0:1, :, :]
        #confidence=decoder_feature_5[:, 1:2, :, :]
        #output=depth*confidence

        return self.final_sigmoid(decoder_feature_4_plus)#depth, confidence, output 


class BasicModelUltraLightBottleneck(nn.Module):
    """
    Basic Encoder-Decoder model with skip connections between layers and leeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeess parameters and a bottleneck resnet architecture instead of normal resnet
    """
    def __init__(self):
        super(BasicModelUltraLightBottleneck, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer=conv3x3_bn_relu(in_channels=4,kernel_size=5,out_channels=16,stride=1, padding=2)

        #Encoder
        self.enc_1=ResNetBottleneckBlock(in_channels=16, out_channels=32 )
        self.enc_2=ResNetBottleneckBlock(in_channels=32, out_channels=64 )
        self.enc_3=ResNetBottleneckBlock(in_channels=64, out_channels=128)
        self.enc_4=ResNetBottleneckBlock(in_channels=128, out_channels=256)
        #self.enc_5=ResNetBlock(in_channels=512, out_channels=512, stride=1)

        #Decoder
        self.dec_1=deconv3x3_bn_relu(in_channels=256, out_channels=128,padding=1,stride=1,output_padding=0)
        self.dec_2=deconv3x3_bn_relu(in_channels=128, out_channels=64,padding=1,stride=1,output_padding=0)
        self.dec_3=deconv3x3_bn_relu(in_channels=64, out_channels=32,padding=1,stride=1,output_padding=0)
        self.dec_4=deconv3x3_bn_relu(in_channels=32, out_channels=16,padding=1,stride=1,output_padding=0)
        self.dec_5=conv3x3_bn_relu(in_channels=16, out_channels=2,kernel_size=3, stride=1, padding=1,bn=False)

        init_weights(self)
        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']
        #print(rgb.shape)
        #print(d.shape)

        #join the rgb and the sparse information
        encoder_feature_init=self.first_layer(torch.cat((rgb, d),dim=1))
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        #encoder_feature_5=self.enc_5(encoder_feature_4)

        #Decoder
        #print("Enc1->"+str(encoder_feature_1.shape))
        #print("Enc2->"+str(encoder_feature_2.shape))
        #print("Enc3->"+str(encoder_feature_3.shape))
        #print("Enc4->"+str(encoder_feature_4.shape))
        decoder_feature_1=self.dec_1(encoder_feature_4)
        #print("Dec1->"+str(decoder_feature_1.shape))
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_3 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_2 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_1 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4+encoder_feature_init #skip connection

        decoder_feature_5=self.dec_5(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_5[:, 0:1, :, :]
        confidence=decoder_feature_5[:, 1:2, :, :]

        output=depth*confidence

        return output#depth, confidence, output 
    
class TFGmodel(nn.Module):
    """
    Model from the TFG, based on self-supervised sparse to dense model. Changed cat for sum and reduce hidden layers and neurons
    """
    def __init__(self):
        super(TFGmodel, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)
        self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)

        pretrained_model = resnet.__dict__['resnet18'](pretrained=True)
        
        #Encoder
        self.enc_1=pretrained_model._modules['layer1']
        self.enc_2=pretrained_model._modules['layer2']
        self.enc_3=pretrained_model._modules['layer3']
        self.enc_4=pretrained_model._modules['layer4'] #512
        del pretrained_model
        
        #Decoder
        self.dec_1=deconv3x3_bn_relu(in_channels=512, out_channels=256, kernel_size=3, padding=1)
        self.dec_2=deconv3x3_bn_relu(in_channels=256, out_channels=128, kernel_size=3, padding=1)
        self.dec_3=deconv3x3_bn_relu(in_channels=128, out_channels=64, kernel_size=3, padding=1)
        self.dec_4=deconv3x3_bn_relu(in_channels=64, out_channels=32, kernel_size=3, padding=1)
        self.dec_5=conv3x3_bn_relu(in_channels=32, out_channels=2,kernel_size=3, stride=2, padding=1,bn=False)

        self.final_sigmoid=nn.Sigmoid()
        
        init_weights(self)
        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']
        #print(rgb.shape)
        #print(d.shape)

        #join the rgb and the sparse information
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_init_depth=self.first_layer_depth(d)
        encoder_feature_init=torch.cat((encoder_feature_init_rgb, encoder_feature_init_depth),1)
        #print(encoder_feature_init.shape)
        #Encoder
        encoder_feature_1=self.enc_1(encoder_feature_init)
        encoder_feature_2=self.enc_2(encoder_feature_1)
        encoder_feature_3=self.enc_3(encoder_feature_2)
        encoder_feature_4=self.enc_4(encoder_feature_3)
        #encoder_feature_5=self.enc_5(encoder_feature_4)
        #print(encoder_feature_4.shape)
        #Decoder
        decoder_feature_1=self.dec_1(encoder_feature_4)
        decoder_feature_1_plus=decoder_feature_1+encoder_feature_3 #skip connection

        decoder_feature_2=self.dec_2(decoder_feature_1_plus)
        decoder_feature_2_plus=decoder_feature_2+encoder_feature_2 #skip connection

        decoder_feature_3=self.dec_3(decoder_feature_2_plus)
        decoder_feature_3_plus=decoder_feature_3+encoder_feature_1 #skip connection

        decoder_feature_4=self.dec_4(decoder_feature_3_plus)
        decoder_feature_4_plus=decoder_feature_4#+encoder_feature_init #skip connection

        decoder_feature_5=self.dec_5(decoder_feature_4_plus)

        #Output
        depth=decoder_feature_5[:, 0:1, :, :]
        confidence=decoder_feature_5[:, 1:2, :, :]

        output=depth*confidence

        return self.final_sigmoid(output)#depth, confidence, output 
    

class EncDecDropout(nn.Module):
    """
    Adapted from->
    https://github.com/wvangansbeke/Sparse-Depth-Completion/blob/090d56ac7977d9079e21fb57b21b4abdc3ddce15/Models/ERFNet.py#L98
    """
    def __init__(self):
        super(EncDecDropout, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer_1=DS(in_channels=4,out_channels=4+16)
        self.first_layer_2=DS(in_channels=16,out_channels=64+16)
                                               
        #Encoder
        self.enc_1=non_bottleneck_1d(64, 0.03, 1)
        self.enc_2=non_bottleneck_1d(64, 0.03, 1)
        self.enc_3=non_bottleneck_1d(64, 0.03, 1)
        self.enc_4=non_bottleneck_1d(64, 0.03, 1) 
        
        self.enc_5=DS(64,128+64)
        
        self.enc_6=non_bottleneck_1d(128, 0.03,2)
        self.enc_7=non_bottleneck_1d(128, 0.03,4)
        self.enc_8=non_bottleneck_1d(128, 0.03,8)
        self.enc_9=non_bottleneck_1d(128, 0.03,16)
        
        #Decoder
        self.dec_1=UpsamplerBlock(128,64)
        
        self.dec_2=non_bottleneck_1d(64, 0,1)
        self.dec_3=non_bottleneck_1d(64, 0,1) #64x64x304

        self.dec_4 = UpsamplerBlock(64, 32)
        
        self.dec_5 = non_bottleneck_1d(32, 0, 1)
        self.dec_6 = non_bottleneck_1d(32, 0, 1) # 32x128x608

        self.dec_7 = nn.ConvTranspose2d(32, 1, 2, stride=2, padding=0, output_padding=0, bias=True)
        
        self.apply(weights_init(init_type='xavier'))
        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']
        #Encoder
        out=self.first_layer_1(torch.cat((rgb,d),1))
        out=self.first_layer_2(out)
        
        out=self.enc_1(out)
        out=self.enc_2(out)
        out=self.enc_3(out)
        out=self.enc_4(out)
        
        out=self.enc_5(out)
        
        out=self.enc_6(out)
        out=self.enc_7(out)
        out=self.enc_8(out)
        out=self.enc_9(out)
        #Decoder
        out=self.dec_1(out)
        
        out=self.dec_2(out)
        out=self.dec_3(out)
        
        out=self.dec_4(out)

        out=self.dec_5(out)
        out=self.dec_6(out)

        out=self.dec_7(out)
        
        return out
    
class EncDecDropoutFull(nn.Module):
    """
    https://github.com/wvangansbeke/Sparse-Depth-Completion/blob/090d56ac7977d9079e21fb57b21b4abdc3ddce15/Models/ERFNet.py#L98
    """
    def __init__(self):
        super(EncDecDropoutFull, self).__init__()
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer_1=DS(in_channels=4,out_channels=4+16)
        self.first_layer_2=DS(in_channels=16,out_channels=64+16)
        
        
        self.layers_enc = nn.ModuleList()
        
        for x in range(0, 5):
            self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1)) 

        self.layers_enc.append(DownsamplerBlock(64, 128))

        for x in range(0, 2):
            self.layers_enc.append(non_bottleneck_1d(128, 0.3, 2))
            self.layers_enc.append(non_bottleneck_1d(128, 0.3, 4))
            self.layers_enc.append(non_bottleneck_1d(128, 0.3, 8))
            self.layers_enc.append(non_bottleneck_1d(128, 0.3, 16))
                                          
        #Decoder
        self.dec_1=UpsamplerBlock(128,64)
        self.dec_2=non_bottleneck_1d(64, 0,1)
        self.dec_3=non_bottleneck_1d(64, 0,1) #64x64x304

        self.dec_4 = UpsamplerBlock(64, 32)
        self.dec_5 = non_bottleneck_1d(32, 0, 1)
        self.dec_6 = non_bottleneck_1d(32, 0, 1) # 32x128x608

        self.dec_7 = nn.ConvTranspose2d(32, 1, 2, stride=2, padding=0, output_padding=0, bias=True)
        
        self.final_sigmoid=nn.Sigmoid()
        
        self.apply(weights_init(init_type='xavier'))
        
    def forward(self,input):
        rgb = input['rgb']
        d = input['d']
        #Encoder
        out=self.first_layer_1(torch.cat((rgb,d),1))
        out=self.first_layer_2(out)
        for layer in self.layers_enc:
            out = layer(out)

        #Decoder
        out=self.dec_1(out)
        
        out=self.dec_2(out)
        out=self.dec_3(out)
        
        out=self.dec_4(out)

        out=self.dec_5(out)
        out=self.dec_6(out)

        out=self.dec_7(out)
        
        return self.final_sigmoid(out)
    
class HourGlassNetwork(nn.Module):
    def __init__(self,mode="Light"):
        super(HourGlassNetwork,self).__init__()
        
        self.mode=mode
        
        #First layer of the network, where the rgb and depth values are introduced
        self.first_layer_rgb=conv3x3_bn_relu(in_channels=3,kernel_size=3,out_channels=32,stride=1, padding=1)
        self.first_layer_depth=conv3x3_bn_relu(in_channels=1,kernel_size=3,out_channels=32,stride=1, padding=1)
        
        self.hourglass_1=HourglassModule(1,64)
        self.hourglass_2=HourglassModule(1,64)
        self.hourglass_3=HourglassModule(1,64)
        self.hourglass_4=HourglassModule(1,64)
        
        
        self.dec_1=deconv3x3_bn_relu(in_channels=64, out_channels=1, kernel_size=3, padding=1,stride=1,output_padding=0)
        self.final_sigmoid=nn.Sigmoid()
        init_weights(self)
        
    def forward(self, input):
        rgb = input['rgb']
        d = input['d']
        
        #join the rgb and the sparse information
        encoder_feature_init_rgb=self.first_layer_rgb(rgb)
        encoder_feature_init_depth=self.first_layer_depth(d)
        encoder_feature_init=torch.cat((encoder_feature_init_rgb, encoder_feature_init_depth),1)
        
        
        if self.mode=="Light":
            hg=self.hourglass_1(encoder_feature_init)
            hg=self.hourglass_2(hg)
            hg=self.hourglass_3(hg)
            hg=self.hourglass_4(hg)
        if self.mode=="DeepLinear":
            pass
        #print(hg.shape)
        return self.final_sigmoid(self.dec_1(hg))        
    
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        #print("X->"+str(x.shape))
        #print("Skip->"+str(skip_input.shape))
        x = torch.cat((x, skip_input), 1)

        return x
    
class Pix2PixGanGenerator(nn.Module):
    def __init__(self):
        super(Pix2PixGanGenerator,self).__init__()
        #First layer of the network, where the rgb and depth values are introduced
        #self.first_layer=conv3x3_bn_relu(in_channels=4,kernel_size=3,out_channels=32,stride=1, padding=1)
        
        self.down1 = UNetDown(4, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 1, 4, padding=1),
            nn.Tanh(),
        )
        
        
    def forward(self, input):

        #join the rgb and the sparse information
        #encoder_feature_init=self.first_layer(input)
        
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        #print("inputs init->"+str(input.shape))
        #print("inputs d1->"+str(d1.shape))
        #print("inputs d2->"+str(d2.shape))
        #print("inputs d3->"+str(d3.shape))
        #print("inputs d4->"+str(d4.shape))
        #print("inputs d5->"+str(d5.shape))
        #print("inputs d6->"+str(d6.shape))
        #print("inputs d7->"+str(d7.shape))
        #print("inputs d8->"+str(d8.shape))
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)
        
        return self.final(u7)        

class Pix2PixGanDiscriminator(nn.Module):
    def __init__(self, in_channels=5):
        super(Pix2PixGanDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    
def weights_init_normal_pix2pix(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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
        
        

class SelfAttentionModelCBAM(nn.Module):
    def __init__(self,attLayers=4,deconvLayers=3,attentionChannels=64):
        super(SelfAttentionModelCBAM, self).__init__()
        self.deconvLayers=deconvLayers
        self.attLayers=attLayers
        out_channels=int(attentionChannels/2)

        #Encoder RGB
        self.first_layer_rgb=conv3x3_relu(in_channels=3,kernel_size=3,out_channels=out_channels,stride=1, padding=1)#B,32,256,512
        #Encoder Depth
        self.first_layer_depth=conv3x3_relu(in_channels=1,kernel_size=3,out_channels=out_channels,stride=1, padding=1)#B,16,516,1028
        
        
        #Self-attention
        #self.modules_attention=[]
        #for att in range(attLayers):
        #    self.modules_attention.append(DIYSelfAttention(attentionChannels))
        self.att_1=CBAM(attentionChannels)
        if attLayers==2:
            self.att_2=CBAM(attentionChannels)
        if attLayers==3:
            self.att_2=CBAM(attentionChannels)
            self.att_3=CBAM(attentionChannels)
        if attLayers==4:
            self.att_2=CBAM(attentionChannels)
            self.att_3=CBAM(attentionChannels)
            self.att_4=CBAM(attentionChannels)    
        if deconvLayers==1:
            self.dec_1=deconv3x3_relu_no_artifacts(in_channels=attentionChannels, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)#B,1,512,1024
        if deconvLayers==2:
            self.dec_1=deconv3x3_relu_no_artifacts(in_channels=attentionChannels, out_channels=out_channels,padding=1, stride=1,output_padding=1, scale_factor=1)
            self.dec_2=deconv3x3_relu_no_artifacts(in_channels=out_channels, out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)
        if deconvLayers==3:
            self.dec_1=deconv3x3_relu_no_artifacts(in_channels=attentionChannels, out_channels=out_channels,padding=1, stride=1,output_padding=1, scale_factor=1)
            self.dec_2=deconv3x3_relu_no_artifacts(in_channels=out_channels, out_channels=int(out_channels/2),padding=1, stride=1,output_padding=1, scale_factor=1)
            self.dec_3=deconv3x3_relu_no_artifacts(in_channels=int(out_channels/2), out_channels=1,padding=1, stride=1,output_padding=1, scale_factor=1,relu=False)
            
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
        
        #Attention layers
        out=self.att_1(out)
        if self.attLayers==2:
            out=self.att_2(out)
        if self.attLayers==3:
            out=self.att_2(out)
            out=self.att_3(out)
        if self.attLayers==4:
            out=self.att_2(out)
            out=self.att_3(out)
            out=self.att_4(out)
            
        #Decoder
        out=self.dec_1(out)
        if self.deconvLayers==2:
            out=self.dec_2(out)
        if self.deconvLayers==3:
            out=self.dec_2(out)
            out=self.dec_3(out)
                
        return self.final_sigmoid(out) 