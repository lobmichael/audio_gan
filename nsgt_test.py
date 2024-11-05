# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:02:36 2023

@author: micha
"""

import torch
import numpy as np
import torchaudio

import os

from nsgt import NSGT, NSGT_sliced, LogScale, LinScale, MelScale, OctScale
import pyfftw

import hparams



dataset = torchaudio.datasets.MUSDB_HQ(hparams.musdb_path, 'train')
sources = ['mixture', 'vocals']
for source in sources:
    for i,track_name in enumerate(dataset.names):
        path = dataset._get_track(track_name, source)
        y, fs = torchaudio.load(path)
    
        scale = OctScale(hparams.fmin, hparams.fmin*2**10,hparams.bins_per_octave)
        nsgt = NSGT(scale = scale, Ls = y.shape[-1], fs = fs, 
                    real=True, matrixform=True, reducedform=2, multichannel=True)
        Y = np.asarray(nsgt.forward(y))
        save_path = str(path).replace('.wav', '.pt')
        save_path = str(save_path).replace('musdb18hq', 'musdb18hq_cqt')
        save_path = save_path.replace('C:','E:')
        save_dir = save_path.split(source)[0]
        save_dir = save_dir.replace('C:','E:')
        os.makedirs(save_dir, exist_ok = True)
        torch.save(np.abs(Y), save_path)
        print("Saved track " + str(i+1) + "/100")


