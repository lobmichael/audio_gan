# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import glob
import yaml
import csv
import random
import os

import pandas as pd
import numpy as np
from scipy.io.wavfile import write as write_wav


import hparams
import spectrogram_utils
import utils

import torch
import torchaudio

import librosa


def parse_dataset(path, trimmed = False):
    
    
    dataset = []
    keys = ['instrument_type', 'midi_program_name', 'path_to_midi', 'path_to_wav']
    dataset_entry = dict(zip(keys, [None]*len(keys)))
    
    
    path = path + 'Track*/metadata.yaml'
    
    files = sorted(glob.glob(path))
    
    filenames = []
    data_descriptions = []
    list_of_unique_instruments = []
    list_of_unique_midi_programs = []
    suffix = '.flac'
    if trimmed:
      suffix = '_trimmed.flac'


    
    for i in range(0,len(files)):
      with open(files[i]) as f:
          data = yaml.load(f, Loader=yaml.loader.SafeLoader)
          
          data_descriptions = data['stems']
          
          for key in data_descriptions.keys():
              list_of_unique_instruments.append(data_descriptions[key]["inst_class"])
              list_of_unique_midi_programs.append(data_descriptions[key]["midi_program_name"]) 
              dataset_entry = dict(zip(keys, [data_descriptions[key]["inst_class"], data_descriptions[key]["midi_program_name"],
                                                  files[i].split("metadata.yaml")[-2] + 'midi_stem/' + str(key) + suffix,
                                                  files[i].split("metadata.yaml")[-2] + 'stems/' + str(key) + suffix]))
              if os.path.exists(dataset_entry['path_to_wav']) & os.path.exists(dataset_entry['path_to_midi']):
                dataset.append(dataset_entry)
                            
    with open('dataset/dataset_train_trimmed.csv', 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
                              
    with open('dataset/list_of_instruments_train_trimmed.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(list(set(list_of_unique_instruments)))          
    
    with open('dataset/list_of_vst_train_trimmed.csv', 'w', newline='') as output_file:
        writer = csv.writer(output_file)
        writer.writerow(list(set(list_of_unique_midi_programs)))                        

    return dataset,set(list_of_unique_instruments)

def load_dataset(split = 'test', instrument_class = None):
    if split == 'test':
      dataset_file_lists = pd.read_csv("dataset/dataset_test.csv")
      if instrument_class is not None: 
        return dataset_file_lists[dataset_file_lists['instrument_type'] == instrument_class]
      return dataset_file_lists
    if split == 'valid':
      dataset_file_lists = pd.read_csv("dataset/dataset_valid.csv")
      if instrument_class is not None: 
        return dataset_file_lists[dataset_file_lists['instrument_type'] == instrument_class]
      return dataset_file_lists
    if split == 'train':
      dataset_file_lists = pd.read_csv("dataset/dataset_train_trimmed.csv") ##WARNING, change line to train
      if instrument_class is not None: 
        return dataset_file_lists[dataset_file_lists['instrument_type'] == instrument_class]
      return dataset_file_lists

    return 0

def load_unique_instruments():
    file = open('dataset/list_of_instruments.csv')
    csvreader = csv.reader(file)
    unique_instruments = next(csvreader)
    #unique_instruments = pd.read_csv("dataset/list_of_instruments.csv")

    return unique_instruments

class Slakh_dataset(torch.utils.data.Dataset):
    def __init__(self, transform, split = 'train', instrument_class = None):
        super().__init__()
        self.split = split
        self.instrument_class = instrument_class
        self.paths = load_dataset(self.split, self.instrument_class)
        self.instrument_list = hparams.instrument_list#load_unique_instruments()
        self.instrument_index = np.arange(0,len(self.instrument_list))
        self.instrument_lookup = dict(zip(self.instrument_list,self.instrument_index))
        self.tf_trans = transform
        self.paths = self.paths[self.paths['midi_program_name'].isin(hparams.instruments)]

    def __getitem__(self, idx):
        y, offset = self.load_audio_segment(self.paths['path_to_wav'].iloc[idx])
        y_mid = self.load_audio_segment_mid(self.paths['path_to_midi'].iloc[idx], offset)
        instrument_label = self.decode_label(self.paths['instrument_type'].iloc[idx])
        return instrument_label, self.tf_trans(y_mid), self.tf_trans(y)

    def __len__(self):
        return len(self.paths)

    def load_audio_segment(self,path):
        metadata = torchaudio.info(path)
        if metadata.num_frames <= metadata.sample_rate*hparams.sample_length:
          y, sr = torchaudio.load(path)
          y = torch.nn.functional.pad(y, (0,int(-metadata.num_frames + metadata.sample_rate*hparams.sample_length)), mode='constant', value=0.0)
          offset = -1
          return y, offset
        offset = random.randint(0, metadata.num_frames -  metadata.sample_rate*hparams.sample_length)
        y, sr = torchaudio.load(path,
          frame_offset = offset, num_frames =  int(hparams.sample_length*hparams.sr))
        return y, offset

    def load_audio_segment_mid(self, path, offset):
        metadata = torchaudio.info(path)
        if offset == -1:
          y, sr = torchaudio.load(path)
          y = torch.nn.functional.pad(y, (0,int(-metadata.num_frames + metadata.sample_rate*hparams.sample_length)), mode='constant', value=0.0)
          return y
        y, sr = torchaudio.load(path,
          frame_offset = offset, num_frames =  int(hparams.sample_length*hparams.sr))
        return y

    def decode_label(self, label):
      return self.instrument_lookup[label]


class Slakh_loader(torch.utils.data.Dataset):
    def __init__(self, transform, split = 'test'):
        super().__init__()
        self.split = split
        self.paths = load_dataset(self.split)
        self.instrument_list = load_unique_instruments()
        self.instrument_index = np.arange(0,len(self.instrument_list))
        self.instrument_lookup = dict(zip(self.instrument_list,self.instrument_index))
        self.tf_trans = transform

    def __getitem__(self, idx):
        y, offset = self.load_audio_segment(self.paths['path_to_wav'].iloc[idx])
        y_mid = self.load_audio_segment_mid(self.paths['path_to_midi'].iloc[idx], offset)
        print(y_mid.shape)
        instrument_label = self.decode_label(self.paths['instrument_type'].iloc[idx])
        return instrument_label, self.tf_trans(y_mid), y

    def __len__(self):
        return len(self.paths)

    def load_audio(self, path):
      y, sr = torchaudio.load(path)
      if sr != hparams.sr:
        y = torchaudio.functional.resample(y, sr, hparams.sr)
      return y

    def load_audio_segment(self,path):
        metadata = torchaudio.info(path)
        if metadata.num_frames <= metadata.sample_rate*hparams.sample_length:
          y, sr = torchaudio.load(path)
          if sr != hparams.sr:
            y = torchaudio.functional.resample(y, sr, hparams.sr)
          y = torch.nn.functional.pad(y, (0,int(-metadata.num_frames + metadata.sample_rate*hparams.sample_length)), mode='constant', value=0.0)
          offset = -1
          return y, offset
        y, sr = torchaudio.load(path,
          frame_offset = offset, num_frames =  int(hparams.sample_length*hparams.sr))
        if sr != hparams.sr:
          y = torchaudio.functional.resample(y, sr, hparams.sr)
        offset = random.randint(0, metadata.num_frames -  metadata.sample_rate*hparams.sample_length)
        return y, offset

    def load_audio_segment_mid(self, path, offset):
        metadata = torchaudio.info(path)
        if offset == -1:
          y, sr = torchaudio.load(path)
          y = torch.nn.functional.pad(y, (0,int(-metadata.num_frames + metadata.sample_rate*hparams.sample_length)), mode='constant', value=0.0)
          return y
        y, sr = torchaudio.load(path,
          frame_offset = offset, num_frames =  int(hparams.sample_length*hparams.sr))
        return y

      

    def decode_label(self, label):
      return self.instrument_lookup[label]
                                
class Trim_Dataset(torch.utils.data.Dataset):
  def __init__(self, split = 'test', instrument_class = None):
        super().__init__()
        self.split = split
        self.instrument_class = instrument_class
        self.paths = load_dataset(self.split, instrument_class = self.instrument_class)

  def __getitem__(self,idx):
    stem_path = (self.paths['path_to_wav'].iloc[idx].split('.')[0])+ '_trimmed.flac'
    midi_path = (self.paths['path_to_midi'].iloc[idx].split('.')[0])+ '_trimmed.flac'
    y_trimmed, indices = self.compute_trimmed_stem(idx)
    y_midi_trimmed = self.compute_trimmed_midi(idx,indices)
    #utils.create_dir(''.join(stem_path.split('stems')[0:-1]) + 'stems')
    #utils.create_dir(''.join(midi_path.split('midi_stem')[0:-1]) + 'midi_stem')
    torchaudio.save(stem_path, torch.from_numpy(np.expand_dims(y_trimmed,0)), hparams.sr, format = 'flac', bits_per_sample = 16)
    torchaudio.save(midi_path, torch.from_numpy(np.expand_dims(y_midi_trimmed,0)), hparams.sr, format = 'flac', bits_per_sample = 16)
    out = 'Processed ' + str(idx+1)+' out of ' + str(self.__len__()) #Written file '.join(str(idx)).join('/').join(str(self.__len__()))

    return out

  def __len__(self):
    return len(self.paths)

  def compute_trimmed_stem(self,idx):
    y, sr = torchaudio.load(self.paths['path_to_wav'].iloc[idx])
    if sr != hparams.sr:
      y = torchaudio.functional.resample(y, sr, hparams.sr)
    indices = librosa.effects.split(y.numpy()[0], 
      frame_length = hparams.sr, hop_length = hparams.sr//8)
    y_trimmed = take_slice(y,indices[0,0],indices[0,1])
    for i in range(1,len(indices)):
      y_trimmed = np.concatenate((y_trimmed, take_slice(y,indices[i,0],indices[i,1])),0)
    return y_trimmed, indices

  def compute_trimmed_midi(self,idx,indices):
    y, sr = torchaudio.load(self.paths['path_to_midi'].iloc[idx])
    y_trimmed = take_slice(y,indices[0,0],indices[0,1])
    for i in range(1,len(indices)):
      y_trimmed = np.concatenate((y_trimmed, take_slice(y,indices[i,0],indices[i,1])),0)
    return y_trimmed


    

def take_slice(y,start,end):
  y_slice = y[0][start:end]

  return y_slice

def write_trimmed_wav(y, path):
    write_wav(path, hparams.sr, y)


if __name__ == '__main__':
    
    
  if hparams.overwrite_data_csv == True:
        dataset, inst_count = parse_dataset(hparams.path_to_dataset, trimmed = True)   

  #ds = Trim_Dataset(split = 'train', instrument_class = 'Guitar')

  #print(ds.__len__())

  #trim_loader = torch.utils.data.DataLoader(ds,batch_size=16,
                       #shuffle=False, num_workers=2, drop_last = False)


  #for i, batch in enumerate(trim_loader):
    #print(batch[-1])


 
    


          