# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:43:47 2022

@author: Michael Lobjoit

Nsynth Dataset Handler

"""

import hparams
import spectrogram_utils
import utils

import glob

import torch
import librosa
import torchaudio
import numpy as np

class Nsynth(torch.utils.data.Dataset):
    def __init__(self, transform, split, instrument_source = None, instrument_family = None, render = False):
        super().__init__()
        self.split = split
        self.dataset_root_dir = hparams.dataset_root_dir
        self.instrument_family = instrument_family
        self.instrument_source = instrument_source
        self.paths = self.load_dataset()
        self.paths = self.filter_dataset()
        self.transform = transform
        self.device = 'cpu'
        self.orig_size = 3187
        self.sine_table = self.load_notes()
    
    def __getitem__(self, idx):
      y = torch.load(self.paths[idx], map_location = self.device)
      note = self.paths[idx].split('/')[-1].split('.')[0].split('-')[1].split('_')[-1]
      velocity = self.paths[idx].split('/')[-1].split('.')[0].split('-')[-1]
      note = (127/int(velocity))*self.sine_table[str(int(note))]
      return np.swapaxes(note,0,1),np.swapaxes(y,0,1)


    def load_dataset(self):
      path = self.dataset_root_dir + 'nsynth-' + self.split + '/cqt/*.pt' 
      paths = sorted(glob.glob(path))
      return paths

    def __len__(self):
        return len(self.paths)

    def load_notes(self):
      sine_table = torch.load(hparams.path_to_notes_table + 'notes.pt', 
        map_location = self.device)     
      return sine_table

    def filter_dataset(self):
      self.paths = list(filter(lambda sub: self.instrument_family in sub, self.paths))
      self.paths = list(filter(lambda sub: self.instrument_source in sub, self.paths))

      return self.paths

class Nsynth_render_cqt(torch.utils.data.Dataset):
    def __init__(self, transform, split, instrument_source = None, instrument_family = None):
        super().__init__()
        self.split = split
        self.dataset_root_dir = hparams.dataset_root_dir
        self.instrument_family = instrument_family
        self.instrument_source = instrument_source
        self.paths = self.load_dataset()
        self.transform = transform
        self.paths = self.filter_dataset()
        self.render_path = self.dataset_root_dir + 'nsynth-' + self.split + '/cqt/'
        utils.create_dir(self.render_path)
        self.orig_size = 3187
        #self.create_inputs()
    
    def __getitem__(self, idx):
      y, sr = torchaudio.load(self.paths[idx])
      y = torch.nn.functional.pad(y, 
        (0,-len(y[0]) + hparams.sr*hparams.padded_sample_length),
         mode='constant', value=0.0)
      print(y.shape)
      assert(sr == hparams.sr)
      save_path = self.render_path + self.paths[idx].split('/')[-1].split('.')[0] + '.pt'
      y = self.transform(y.numpy())[:,:,:hparams.frame_length]
      self.save_tensor(save_path, y)
      return idx

    def load_dataset(self):
      path = self.dataset_root_dir + 'nsynth-' + self.split + '/audio/*.wav' 
      paths = sorted(glob.glob(path))
      return paths

    def save_tensor(self, path, tensor):
      torch.save(tensor, path)

    def __len__(self):
        return len(self.paths)

    def filter_dataset(self):
      self.paths = list(filter(lambda sub: self.instrument_family in sub, self.paths))
      self.paths = list(filter(lambda sub: self.instrument_source in sub, self.paths))

      return self.paths

    def create_inputs(self):
      sine_tables = {}
      sine_table = {}
      for i in range(1,150):
        hz = librosa.midi_to_hz(i)
        sine = torch.from_numpy(np.expand_dims(librosa.tone(hz, sr = hparams.sr, duration = 3),0).astype(np.float32))
        sine = torch.nn.functional.pad(sine, 
                (0, hparams.sr*hparams.padded_sample_length - sine.shape[-1]),
                mode='constant', value=0.0)
        c = self.transform(sine.numpy())
        sine_table[str(i)] = c[:,:,:hparams.frame_length]
      print('Computed sine table...')
      self.save_tensor(hparams.path_to_notes_table + 'notes.pt', sine_table)
      return sine_table

 
if __name__ == '__main__':

  if hparams.render == True:
    print('Rendering cqt..')
    CQT = spectrogram_utils.NSGTCQT_Transform()
    dataset = Nsynth_render_cqt(transform = CQT, split = 'train', instrument_source = 'acoustic',
      instrument_family = 'keyboard')
    print(dataset.__len__())

    dataloader = torch.utils.data.DataLoader(dataset,batch_size=2,
                          shuffle=False, num_workers=2, drop_last = False)

    for i, sample in enumerate(dataloader):
        print(sample)
  exit(1)
  CQT = spectrogram_utils.NSGTCQT_Transform()
  dataset = Nsynth(transform = CQT, split = 'train', instrument_source = 'acoustic',
      instrument_family = 'keyboard')
  dataloader = torch.utils.data.DataLoader(dataset,batch_size=10,
      shuffle=True, num_workers=2, drop_last = False)

  for i, sample in enumerate(dataloader):
    mid, y = sample
    print(mid.shape)
    print(y.shape)

  spectrogram_utils.plot_mag_spec(mid.numpy()[0,0,:], 'mid.png')
  spectrogram_utils.plot_mag_spec(y.numpy()[0,0,:], 'mag.png')









