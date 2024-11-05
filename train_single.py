# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 16:52:20 2022

@author: sbp247
"""

import torch

from torch.autograd import Variable
from model import Generator, Discriminator
import nsynth_loader
import hparams
import spectrogram_utils
import os
import numpy as np
import matplotlib.pyplot as plt
import evaluate
import logger

def load_checkpoint(model, optim, path):
  checkpoint = torch.load(path, )
  print('Loaded checkpoint: '+path)
  model.load_state_dict(checkpoint['model_state_dict'])
  optim.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']

  return model, optim, loss, epoch

if __name__ == '__main__':
  # Load Dataset
  CQT = spectrogram_utils.NSGTCQT_Transform()
  dataset = nsynth_loader.Nsynth(transform = CQT, split = 'train', instrument_source = 'acoustic',
      instrument_family = 'keyboard')
  train_data_loader = torch.utils.data.DataLoader(dataset,batch_size=hparams.batch_size,
      shuffle=True, num_workers=2, drop_last = True)
                        
  print(dataset.__len__())

  inference_audio_gen = evaluate.Evaluate()
  log = logger.Tensorboard_Logger()

  # Models 
  G = Generator(hparams.input_channels, hparams.generator_filters, hparams.output_channels)
  D = Discriminator(2*hparams.output_channels, hparams.discriminator_filters, 1)
  G.cuda()
  D.cuda()
  G.normal_weight_init(mean=0.0, std=0.02)
  D.normal_weight_init(mean=0.0, std=0.02)


  # Loss function
  BCE_loss = torch.nn.BCELoss().cuda()
  L1_loss = torch.nn.L1Loss().cuda()

  # Optimizers
  G_optimizer = torch.optim.Adam(G.parameters(), lr=hparams.learning_rate, betas=(hparams.beta1, hparams.beta2))
  D_optimizer = torch.optim.Adam(D.parameters(), lr=hparams.learning_rate, betas=(hparams.beta1, hparams.beta2))

  # Training GAN
  D_avg_losses = []
  G_avg_losses = []

  last = -1
  if hparams.load_checkpoint == True:

    G, G_optimizer, G_loss, last = load_checkpoint(G, G_optimizer, 
      '/content/drive/MyDrive/phd/nsynth-nsgt/model_checkpoints/generator_acoustic_keyboard.pt')

    D, D_optimizer, D_loss, last = load_checkpoint(D, D_optimizer, 
      '/content/drive/MyDrive/phd/nsynth-nsgt/model_checkpoints/discriminator_acoustic_keyboard.pt')


  for epoch in range(last+1, hparams.number_of_epochs):
      D_losses = []
      G_losses = []

      # training
      for i, sample in enumerate(train_data_loader):

          y_mid, y = sample
          
          # input & target image data
          x_ = Variable(y_mid.cuda())
          y_ = Variable(y.cuda())

          z = torch.randn(hparams.batch_size, hparams.latent_dim).cuda()
          

          # Train discriminator with real data
          D_real_decision = D(x_, y_).squeeze()
          real_ = Variable(torch.ones(D_real_decision.size()).cuda())
          D_real_loss = BCE_loss(D_real_decision, real_)

          # Train discriminator with fake data
          gen_image = G(x_,z)
          D_fake_decision = D(x_, gen_image).squeeze()
          fake_ = Variable(torch.zeros(D_fake_decision.size()).cuda())
          D_fake_loss = BCE_loss(D_fake_decision, fake_)

          # Back propagation
          D_loss = (D_real_loss + D_fake_loss) * 0.5
          D.zero_grad()
          D_loss.backward()
          D_optimizer.step()

          # Train generator
          gen_image = G(x_, z)
          D_fake_decision = D(x_, gen_image).squeeze()
          G_fake_loss = BCE_loss(D_fake_decision, real_)

          # L1 loss
          l1_loss = hparams.lamda * L1_loss(gen_image, y_)

          # Back propagation
          G_loss = G_fake_loss + l1_loss
          G.zero_grad()
          G_loss.backward()
          G_optimizer.step()

          # loss values
          D_losses.append(D_loss.item())
          G_losses.append(G_loss.item())

          print('Epoch [%d/%d], Step [%d/%d], D_loss: %.4f, G_loss: %.4f'
                % (epoch+1, hparams.number_of_epochs, i+1, len(train_data_loader), D_loss.item(), G_loss.item()))

          # ============ TensorBoard logging ============#
          
          #D_logger.scalar_summary('losses', D_loss.data[0], step + 1)
          #G_logger.scalar_summary('losses', G_loss.data[0], step + 1)
          #step += 1


      D_avg_loss = torch.mean(torch.FloatTensor(D_losses))
      G_avg_loss = torch.mean(torch.FloatTensor(G_losses))

      # avg loss values for plot
      D_avg_losses.append(D_avg_loss)
      G_avg_losses.append(G_avg_loss)
      log.write_losses(G_avg_loss, D_avg_loss, epoch)

      if (epoch+1) % hparams.checkpoint_frequency == 0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': G_optimizer.state_dict(),
                'loss': G_loss,
                }, (hparams.model_output_path + 'generator_acoustic_bass.pt'))

        torch.save({
                'epoch': epoch,
                'model_state_dict': D.state_dict(),
                'optimizer_state_dict': D_optimizer.state_dict(),
                'loss': D_loss,
                }, (hparams.model_output_path + 'discriminator_acoustic_bass.pt'))