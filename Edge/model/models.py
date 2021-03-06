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
        rgb=input['rgb']
        depth=input['d']
        encoder_feature_init=torch.cat((rgb, depth),1)

        d1 = self.down1(encoder_feature_init)
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
