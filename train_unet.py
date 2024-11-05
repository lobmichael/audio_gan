# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:35:07 2023

@author: micha
"""

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
import matplotlib.pyplot as plt
import musdb

import unet
import dataloader
import hparams
import utils



if __name__ == '__main__':
    
  # Load Dataset
  mus = musdb.DB(root=hparams.musdb_path, is_wav = True)
  tracks = mus.load_mus_tracks(subsets=["train"])  # You can choose other subsets like "test" or "validation"
  musdb_dataset = dataloader.MusDBDataset(tracks)
  
  batch_size = hparams.batch_size # Adjust as needed
  dataloader = DataLoader(musdb_dataset, batch_size=batch_size, shuffle=True,
                          num_workers = 4, prefetch_factor = 2, pin_memory=True)
  print("Dataset loaded...")
  
  # Models 
  model = unet.Unet(hparams.input_channels, hparams.num_of_filters, hparams.output_channels)
  model.cuda()

  model.normal_weight_init(mean=0.0, std=0.02)
  print('Model initialized...')
    
  # Loss function
  L2_loss = torch.nn.MSELoss().cuda()

  # Optimizers
  model_optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, betas=(hparams.beta1, hparams.beta2))
  
  # Training Unet
  model_avg_losses = []

  print('Starting to train the network...')
  for epoch in range(0, hparams.number_of_epochs):
      model_losses = []

      # training
      for i, batch in enumerate(dataloader):

          # input & target image data
          x_ = utils.normalize(batch[0]).cuda()
          y_ = utils.normalize(batch[1]).cuda()
          
          
          # Append zeros to the 
          zeros_vector = torch.zeros(x_.size(0), x_.size(1), x_.size(2), 4).cuda()
          x_ = torch.cat((x_, zeros_vector), dim=-1)
          
          # Forward Pass
          y_hat = model(x_)[:,:,:,:y_.shape[-1]]

          # L2 loss
          l2_loss = L2_loss(y_hat, y_)

          # Back propagation
          model.zero_grad()
          l2_loss.backward()
          model_optimizer.step()

          # loss value
          model_losses.append(l2_loss.item())

          #print('Epoch [%d/%d], Step [%d/%d], Steminator L1 Loss: %.4f'
          #      % (epoch+1, hparams.number_of_epochs, i+1, len(dataloader), l2_loss.item()))

          # ============ TensorBoard logging ============#
          
          #D_logger.scalar_summary('losses', D_loss.data[0], step + 1)
          #G_logger.scalar_summary('losses', G_loss.data[0], step + 1)
          #step += 1

      model_avg_loss = torch.mean(torch.FloatTensor(model_losses))
      print('Epoch [%d/%d], Steminator Loss: %.4f'
            % (epoch+1, hparams.number_of_epochs, model_avg_loss.item()))    
      # avg loss values for plot
      if ((epoch+1) % hparams.checkpoint_frequency)  == 0:
          torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': model_optimizer.state_dict(),
                'loss': model_avg_loss,
                }, hparams.model_save_path + hparams.source + '_steminator_epoch_' + str(epoch+1) + '.pt')
      model_avg_losses.append(model_avg_loss)
      #log.write_losses(G_avg_loss, D_avg_loss, epoch)