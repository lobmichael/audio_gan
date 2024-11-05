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

class Slakh_dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(Slakh_dataset).__init__()
        self.paths = load_dataset()

    def __getitem__(self, idx):
        metadata = torchaudio.info(self.paths['path_to_wav'].iloc[idx])
        offset = random.randint(0, metadata.num_frames -  metadata.sample_rate*hparams.sample_length)
        y, sr = torchaudio.load(self.paths['path_to_wav'].iloc[idx],
          frame_offset = offset, num_frames = hparams.sample_length*hparams.sr)
        return y

    def __len__(self):
        return len(self.paths)


if __name__ == '__main__':

  dataset = data_preparation.Slakh_dataset()

  dataloader = torch.utils.data.DataLoader(dataset,batch_size=1,
                        shuffle=False, num_workers=2)
    
  for index, track in enumerate(dataloader):
    print(index)
    print(track.shape)