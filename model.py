# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 17:35:47 2022

@author: Michael Lobjoit

Code adapted from: https://github.com/togheppi/pix2pix

"""

import torch
import torchvision

import hparams

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, activation=True, batch_norm=True):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding)
        self.activation = activation
        self.lrelu = torch.nn.LeakyReLU(0.2, True)
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
        self.relu = torch.nn.ReLU(True)
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

class Generator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Generator, self).__init__()

        #FC
        self.lin1 = torch.nn.Linear(256,256)
        self.lin2 = torch.nn.Linear(256,112)

        # Encoder
        self.conv1 = ConvBlock(input_dim, num_filter * 2, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv3 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv4 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, num_filter * 8)
        self.conv6 = ConvBlock(num_filter * 8, num_filter * 8, batch_norm=False)
        # Decoder
        self.deconv1 = DeconvBlock(num_filter * 8 + 1, num_filter * 8, dropout=True)
        self.deconv2 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv3 = DeconvBlock(num_filter * 8 * 2, num_filter * 8, dropout=True)
        self.deconv4 = DeconvBlock(num_filter * 8 * 2, num_filter * 4)
        self.deconv5 = DeconvBlock(num_filter * 4 * 2, num_filter * 2)
        self.deconv6 = DeconvBlock(num_filter * 4, output_dim, batch_norm=False)

    def forward(self, x, z):
        #FC z decoder
        z = self.lin1(z)
        z = self.lin2(z)
        z = z.view(z.size(0),1,7,16)
        # Encoder
        x = torch.nn.functional.pad(x, (0,0,17,18), mode='constant', value=0.0)
        enc1 = self.conv1(x)
        enc2 = self.conv2(enc1)
        enc3 = self.conv3(enc2)
        enc4 = self.conv4(enc3)
        enc5 = self.conv5(enc4)
        enc6 = self.conv6(enc5)
        dec0 = torch.cat([z, enc6], 1)
        # Decoder with skip-connections
        dec1 = self.deconv1(dec0)
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
        dec6 = dec6[:,:,18:18+413,:]
        out = torch.nn.Tanh()(dec6)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal_(m.conv.weight, mean, std)
            if isinstance(m, DeconvBlock):
                torch.nn.init.normal_(m.deconv.weight, mean, std)

class Discriminator(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)
        return out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal_(m.conv.weight, mean, std)

class Discriminator_AC(torch.nn.Module):
    def __init__(self, input_dim, num_filter, output_dim):
        super(Discriminator_AC, self).__init__()

        self.conv1 = ConvBlock(input_dim, num_filter, activation=False, batch_norm=False)
        self.conv2 = ConvBlock(num_filter, num_filter * 2)
        self.conv3 = ConvBlock(num_filter * 2, num_filter * 4)
        self.conv4 = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv5 = ConvBlock(num_filter * 8, output_dim, stride=1, batch_norm=False)

        self.conv1_ac = ConvBlock(num_filter * 4, num_filter * 8)
        self.conv2_ac = ConvBlock( num_filter * 8, 1)
        self.flatten = torch.nn.Flatten()
        self.lin_ac = torch.nn.Linear(192,len(hparams.instrument_list))
        self.softmax = torch.nn.Softmax(dim = -1)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        y = self.conv1_ac(torch.clone(x))
        y = self.conv2_ac(y)
        y = self.flatten(y)
        y = self.lin_ac(y)
        aux_out = self.softmax(y)
        x = self.conv4(x)
        x = self.conv5(x)
        out = torch.nn.Sigmoid()(x)

        
        return out,aux_out

    def normal_weight_init(self, mean=0.0, std=0.02):
        for m in self.children():
            if isinstance(m, ConvBlock):
                torch.nn.init.normal_(m.conv.weight, mean, std)