import torch
from .basic import *
import torch.nn as nn

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
   
    
    
    
    