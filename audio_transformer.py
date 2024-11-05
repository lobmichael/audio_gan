# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:58:35 2023

@author: micha
"""

import torch
import unet

model = torch.nn.TransformerEncoderLayer(352, 16, dim_feedforward=512,
                                         dropout=0.1,  batch_first=True, 
                                         norm_first=False, device='cpu')

model = torch.nn.Transformer(d_model=352, nhead=8, num_encoder_layers=6, num_decoder_layers=6, 
                     dim_feedforward=512, dropout=0.1, custom_encoder=None, 
                     custom_decoder=None, layer_norm_eps=1e-05, 
                     batch_first=True, norm_first=False, device='cuda')

src = torch.rand((32, 128, 352)).cuda()
tgt = torch.rand((32, 128, 352)).cuda()


out = model(src,tgt)
print(out.shape)
print(unet.get_num_of_params(model))