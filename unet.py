# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:03:30 2023

@author: micha
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:25:54 2023

@author: michael lobjoit
"""

import torch
import torchvision

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=2, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        if activation:
            self.lrelu = torch.nn.LeakyReLU(0.2)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(output_size)

    def forward(self, x):
        if self.activation:
            out = self.conv(self.lrelu(x))
        else:
            out = self.conv(x)

        if self.batch_norm:
            return self.bn(out)
        else:
            return out
        
class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation = True, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.dropout = dropout
        self.activation = activation
        self.batch_norm = batch_norm
        if dropout:
            self.drop = torch.nn.Dropout(0.5)
        if activation:
            self.relu = torch.nn.ReLU()
        if batch_norm:
            self.bn = torch.nn.BatchNorm2d(output_size)
        

    def forward(self, x):
        if self.activation:
            if self.batch_norm:
                out = self.bn(self.deconv(self.relu(x)))
            else:
                out = self.deconv(self.relu(x))
        else:
            if self.batch_norm:
                out = self.bn(self.deconv(x))
            else:
                out = self.deconv(x)
            
        if self.dropout:
                return self.drop(out)
        else:
                return out

class Unet(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim, kernel_size = 5):
        super(Unet, self).__init__()

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter * 2, kernel_size, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter * 2, num_filter * 4, kernel_size)
        self.conv3 = ConvBlock(num_filter * 4, num_filter * 8, kernel_size)
        self.conv4 = ConvBlock(num_filter * 8, num_filter * 8, kernel_size)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 16, kernel_size)
        self.conv6 = ConvBlock(num_filter * 16, num_filter * 32, kernel_size, batch_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 32, num_filter * 16, dropout=True)
        self.deconv2 = DeconvBlock(num_filter * 16 * 2, num_filter * 8, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv5 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv6 = DeconvBlock(num_filter * 4, output_dim, activation = False, batch_norm=False)

    def forward(self, x):
        # Encoder
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        # Decoder with skip-connections
        dec1 = self.deconv1(enc6)
        dec1 = torch.cat([dec1, enc5], 1)
        dec2 = self.deconv2(dec1)
        dec2 = torch.cat([dec2, enc4], 1)
        dec3 = self.deconv3(dec2)
        dec3 = torch.cat([dec3, enc3], 1)
        dec4 = self.deconv4(dec3)
        dec4 = torch.cat([dec4, enc2], 1)
        dec5 = self.deconv5(dec4)
        dec5 = torch.cat([dec5, enc1], 1)
        dec6 = self.deconv6(dec5)
        out = torch.nn.Tanh()(dec6)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal_(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal_(m.deconv.weight, mean, std)
                
def load_checkpoint(model, optim, path):
  checkpoint = torch.load(path, map_location=torch.device('cuda'))
  model.load_state_dict(checkpoint['model_state_dict'])
  optim.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']
  
  
  return model, optim, loss

def get_num_of_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    return total_params