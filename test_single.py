# -*- coding: utf-8 -*-
"""
Created on Fri May 14 15:35:02 2022

@author: Michael Lobjoit
"""
import torch
import hparams
import nsynth_loader
import spectrogram_utils
from model import Generator, Discriminator
from nsgt import NSGT, LogScale, LinScale, MelScale, OctScale
import numpy as np
import utils
import librosa
import evaluate
import logger

def load_checkpoint(model, optim, path):
  checkpoint = torch.load(path, map_location=torch.device('cuda'))
  model.load_state_dict(checkpoint['model_state_dict'])
  optim.load_state_dict(checkpoint['optimizer_state_dict'])
  epoch = checkpoint['epoch']
  loss = checkpoint['loss']

  print('Epoch: ' + str(epoch))

  return model, optim, loss

if __name__ == '__main__':

  CQT = spectrogram_utils.NSGTCQT_Transform()
  testset = nsynth_loader.Nsynth(transform = CQT, split = 'train', instrument_source = 'acoustic',
      instrument_family = 'keyboard')
  test_data_loader = torch.utils.data.DataLoader(testset,batch_size=hparams.test_batch_size,
      shuffle=True, num_workers=2, drop_last = True)

  print(testset.__len__())

  model = Generator(hparams.input_channels, hparams.generator_filters, hparams.output_channels)
  optim = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, betas=(hparams.beta1, hparams.beta2))

  model, optim, loss = load_checkpoint(model,optim, hparams.checkpoint_path)
  model.cuda()

  model.eval()

  inv_spec = evaluate.Inference()
  log = logger.Tensorboard_Logger()

  for i, track in enumerate(test_data_loader):
    y_mid, y = track
    z = torch.randn(hparams.test_batch_size, hparams.latent_dim).cuda()

    in_specs = y_mid.cuda()
    print(in_specs.shape)

    log.draw_model(model, (in_specs,z))

    #in_spec = torch.cat((in_specs[0], in_specs[1]),dim = 0)
    output = model(in_specs, z)
    y_hat = inv_spec(librosa.util.fix_length(output[0].detach().cpu().numpy(),testset.orig_size,-1), i)[:hparams.sr*4]
    y_real = inv_spec(librosa.util.fix_length(y[0],testset.orig_size,-1), i)[:hparams.sr*4]
    log.audio_out(y_hat[:hparams.sr*4], y_real[:hparams.sr*4])
    log.specs_out(y_mid.cpu().detach().numpy(),output.cpu().detach().numpy(),y.cpu().detach().numpy())
    if i==10:
      break
    #plt.imshow(output[0])



