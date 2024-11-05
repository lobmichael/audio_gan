# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 14:58:34 2022

@author: Michael Lobjoit
"""

import hparams
import spectrogram_utils
import utils
import data_preparation

import torch
import torchaudio

import librosa
import numpy as np

import matplotlib.pyplot as plt

class Slakh_dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Slakh_dataset).__init__()
        self.paths = load_dataset()

    def __getitem__(self, idx):
        metadata = torchaudio.info(self.paths['path_to_wav'].iloc[idx])
        offset = random.randint(0, metadata.num_frames -  metadata.sample_rate*hparams.sample_length)
        if offset <= 0:
          y, sr = torchaudio.load(self.paths['path_to_wav'].iloc[idx])
          y = torch.nn.functional.pad(y, (0,metadata.num_frames - metadata.sample_rate*hparams.sample_length), mode='constant', value=0.0)
          return y[0], metadata.num_frames/hparams.sr
        y, sr = torchaudio.load(self.paths['path_to_wav'].iloc[idx],
          frame_offset = offset, num_frames = hparams.sample_length*hparams.sr)
        return y, metadata.num_frames/hparams.sr

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':

  dataset = data_preparation.Slakh_dataset(transform = spectrogram_utils.NSGTCQT_Transform())

  dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,
                        shuffle=False, num_workers=1)


  for index, track in enumerate(dataloader):
    print(index)
    #print(track[0].shape)
    print(track[0])
    print(track[1][0].shape)
    spectrogram_utils.plot_mag_spec(track[1][0][0].numpy())
    plt.savefig('/content/sample_data/gen_sample.png')
    spectrogram_utils.plot_mag_spec(track[2][0][0].numpy())
    plt.savefig('/content/sample_data/real_sample.png')    
    
    break

