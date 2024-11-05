# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:29:37 2023

@author: Michael Lobjoit
"""

import musdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from cqt_nsgt_pytorch import CQT_nsgt

import hparams
import utils

class MusDBDatasetTransformed(Dataset):
    def __init__(self,path = hparams.path_to_transformed_musdb18, 
                 slice_length=hparams.slice_length, sr=hparams.sr, sources = ['mixture', 'vocals']):
        self.dataset = torchaudio.datasets.MUSDB_HQ(path, 'train')
        self.paths = self.dataset.names#str(self.dataset._get_track(self.dataset.names, sources[0])).replace('.wav','.pt')
        self.sources = sources      
        self.slice_length = slice_length
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx): 
        path_x = str(self.dataset._get_track(self.paths[idx], self.sources[0])).replace('.wav','.pt')                                                                                                     
        path_y = str(self.dataset._get_track(self.paths[idx], self.sources[1])).replace('.wav','.pt')
        x = torch.load(path_x)
        y = torch.load(path_y)
        
        # Randomly select a starting point for the excerpt
        track_duration = x.shape[-1]  # Duration of the track in seconds
        start_time = np.random.uniform(1, track_duration - self.slice_length)
        
        # Extract the excerpts
        start_sample = int(start_time)
        end_sample = int((start_time + self.slice_length))
        x = x[:,:,start_sample:end_sample]
        y = y[:,:,start_sample:end_sample]
        
        
        return x,y

    
if __name__ == '__main__':
    
   musdb_dataset = MusDBDatasetTransformed()
    
   batch_size = 1 # Adjust as needed
   dataloader = DataLoader(musdb_dataset, batch_size=batch_size,
                            shuffle=True, num_workers = 0)
    

   for x_batch, y_batch in dataloader:
        print(x_batch.shape)
        print(y_batch.shape)
