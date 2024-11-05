# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:25:54 2023

@author: michael lobjoit
"""

import torch
import torchvision

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.batch_norm = batch_norm
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
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, batch_norm=True, dropout=False):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding)
        self.bn = torch.nn.BatchNorm2d(output_size)
        self.drop = torch.nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        self.batch_norm = batch_norm
        self.dropout = dropout

    def forward(self, x):
        if self.batch_norm:
            out = self.bn(self.deconv(self.relu(x)))
        else:
            out = self.deconv(self.relu(x))

        if self.dropout:
            return self.drop(out)
        else:
            return out
        
class InceptionBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_sizes, stride = 2, padding = 1, activation=True, batch_norm=True):
        super(InceptionBlock, self).__init__()
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2)
        self.batch_norm = batch_norm
        self.bn = torch.nn.BatchNorm2d(output_size)
        
        self.conv1 = ConvBlock(input_size, output_size, kernel_sizes[0], stride, padding)
        self.conv2 = ConvBlock(input_size, output_size, kernel_sizes[1], stride, padding+1)
        self.conv3 = ConvBlock(input_size, output_size, kernel_sizes[2], stride, padding+2)
        self.conv4 = ConvBlock(output_size * 3, output_size, 1, 1, 0)
        
        
    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out3 = self.conv3(x)
        out  = torch.cat((out1,out2,out3), dim = 1)
            
        return self.conv4(out)

class Steminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim, kernel_sizes = (4,6,8)):
        super(Steminator, self).__init__()

        # Encoder
        self.conv1 = InceptionBlock(input_dim, num_filter * 2, kernel_sizes, activation=False, batch_norm=False)
        self.conv2 = InceptionBlock(num_filter * 2, num_filter * 4, kernel_sizes)
        self.conv3 = InceptionBlock(num_filter * 4, num_filter * 8, kernel_sizes)
        self.conv4 = InceptionBlock(num_filter * 8, num_filter * 8, kernel_sizes)
        self.conv5 = InceptionBlock(num_filter * 8, num_filter * 8, kernel_sizes)
        self.conv6 = InceptionBlock(num_filter * 8, num_filter * 8, kernel_sizes, batch_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8, num_filter * 8, dropout=True)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv5 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv6 = DeconvBlock(num_filter * 4, output_dim, batch_norm=False)

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