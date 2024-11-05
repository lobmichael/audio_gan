# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 17:02:02 2023

@author: michael lobjoit
"""

import os
import musdb
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from cqt_nsgt_pytorch import CQT_nsgt

import hparams
import utils

class MusDBDataset(Dataset):
    def __init__(self, tracks, duration=hparams.sample_length, sr=hparams.sr,
                 source = hparams.source):
        self.tracks = tracks
        self.duration = duration  # Desired duration in seconds
        self.sr = sr  # Sample rate
        self.numocts = hparams.num_of_octaves
        self.binsoct = hparams.bins_per_octave
        self.fs = sr
        self.Ls = int(self.duration * self.sr)
        self.source = source
        self.cqt=CQT_nsgt(self.numocts, self.binsoct, mode="matrix",fs=self.fs, 
                          audio_len=self.Ls, device="cpu", dtype=torch.float32)
        

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        
        #Load the tracks
        track = self.tracks[idx]
        source_path = track.path.replace('mixture',self.source)
        waveform, sample_rate = torchaudio.load(track.path)
        src_waveform, _ = torchaudio.load(source_path)
         
        # Randomly select a starting point for the excerpt
        track_duration = waveform.shape[1] / self.sr  # Duration of the track in seconds
        start_time = np.random.uniform(1, track_duration - self.duration)

        # Extract the excerpts
        start_sample = int(start_time * self.sr)
        end_sample = int((start_time + self.duration) * self.sr)
        excerpt = waveform[:,start_sample:end_sample]
        src_excerpt = src_waveform[:,start_sample:end_sample]
        
        # Perform the NSGCQT
        transf_exerpt = self.forward(torch.unsqueeze(excerpt,0))
        transf_src_exerpt = self.forward(torch.unsqueeze(src_excerpt,0))
        
        # Convert to magnitude spectrogram and power db spectrogram
        #x = torchaudio.functional.amplitude_to_DB(torch.abs(transf_exerpt[0,:,:,:]),20, 10e-12, 2)
        #y = torchaudio.functional.amplitude_to_DB(torch.abs(transf_src_exerpt[0,:,:,:]),20, 10e-12, 2)
        x = utils.db_scale(torch.abs(transf_exerpt[0,:,:,:]))
        y = utils.db_scale(torch.abs(transf_src_exerpt[0,:,:,:]))
        #x = torch.abs(transf_exerpt[0,:,:,:])
        #y = torch.abs(transf_src_exerpt[0,:,:,:])
        #print(torch.max(x))
        #print(torch.max(y))
        
    
        return utils.normalize(x),utils.normalize(y)
    
    # Transform input
    def forward(self, audio):
        X=self.cqt.fwd(audio)# forward transform
        return X
    
if __name__ == '__main__':
    
    mus = musdb.DB(root=hparams.musdb_path, is_wav = True)
    tracks = mus.load_mus_tracks(subsets=["train"])  # You can choose other subsets like "test" or "validation"
    musdb_dataset = MusDBDataset(tracks)
    
    batch_size = 1 # Adjust as needed
    dataloader = DataLoader(musdb_dataset, batch_size=batch_size,
                            shuffle=True, num_workers = 4, prefetch_factor = 2)
    

    for x_batch, y_batch in dataloader:
        print("Batch shape:", x_batch.shape)
        print("Batch dtype:", torch.max(y_batch[0,:,:,:]))
        
        print(torch.max(x_batch))
                
        utils.plot_mag_spec(x_batch[0,0,:,:].detach().cpu().numpy(), 'input.png')
        utils.plot_mag_spec(y_batch[0,0,:,:].detach().cpu().numpy(), 'output.png')      
