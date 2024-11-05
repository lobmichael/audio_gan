# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:35:02 2022

@author: Michael Lobjoit
"""
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import  musdb

import hparams
import utils
import unet
import dataloader
import utils

if __name__ == '__main__':

  mus = musdb.DB(root="musdb/", is_wav = True)
  tracks = mus.load_mus_tracks(subsets=["test"])  # You can choose other subsets like "test" or "validation"
  musdb_dataset = dataloader.MusDBDataset(tracks)
  
  batch_size = hparams.batch_size # Adjust as needed
  dataloader = DataLoader(musdb_dataset, batch_size=batch_size, shuffle=True)
  print("Dataset loaded...")

  print(musdb_dataset.__len__())

  model = unet.Unet(hparams.input_channels, hparams.num_of_filters, hparams.output_channels)
  optim = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, betas=(hparams.beta1, hparams.beta2))

  model, optim, loss = unet.load_checkpoint(model,optim, 
                                                  "models/bass_steminator_epoch_100.pt")
  model.cuda()
  print('Loaded model...')
  model.eval()

  for i, batch in enumerate(dataloader):
      x_ = utils.normalize(batch[0]).cuda()
      y_ = utils.normalize(batch[1]).cuda()
      
      
      
      zeros_vector = torch.zeros(x_.size(0), x_.size(1), x_.size(2), 4).cuda()
      x_ = torch.cat((x_, zeros_vector), dim=-1)
      
      # Forward Pass
      y_hat = model(x_)[:,:,:,:y_.shape[-1]]
      
      print(x_[0,0,:,:].shape)
      #Plot Specs
      utils.plot_mag_spec(x_[0,0,:,:].detach().cpu().numpy(), 'input.png')
      utils.plot_mag_spec(y_hat[0,0,:,:].detach().cpu().numpy(), 'estimated.png' )
      utils.plot_mag_spec(y_[0,0,:,:].detach().cpu().numpy(), 'ground_truth.png')
      break
      
      
      
      
      
    



