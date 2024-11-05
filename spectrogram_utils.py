# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 17:01:48 2022

@author: sbp247
"""

from librosa.core.constantq import cqt
import matplotlib.pyplot as plt
import numpy as np
import hparams
import utils

import librosa
import madmom
import torch
import torchaudio
from nsgt import NSGT, NSGT_sliced, LogScale, LinScale, MelScale, OctScale
import pyfftw

class CQT_Transform():
    
    def __init__(self):
        self.n_bins = hparams.bins
        self.cqt_freqs = librosa.cqt_frequencies(hparams.bins, hparams.fmin,
        hparams.bins_per_octave, tuning=0.0)
        print('CQT MIN and MAX frequencies: ' + str(self.cqt_freqs[0]) + 'Hz , '
        + str(self.cqt_freqs[-1]) + ' Hz')
        
    def __call__(self, y):
      cqt = librosa.cqt(np.squeeze(y.numpy()), sr = hparams.sr, hop_length = hparams.hop,
                   n_bins = hparams.bins, 
                   fmin = hparams.fmin,
                   bins_per_octave = hparams.bins_per_octave)

      y_hat = librosa.icqt(cqt, sr = hparams.sr, hop_length = hparams.hop,
                   fmin = hparams.fmin,
                   bins_per_octave = hparams.bins_per_octave)

      #cqt = librosa.util.fix_length(cqt, 1280, -1)
      #cqt = librosa.util.frame(cqt, )
      #print(cqt.shape)

      return torch.abs(torch.from_numpy(cqt))

class NSGTCQT_Transform():
    
    def __init__(self):
        self.n_bins = hparams.bins_per_octave
        self.fs = hparams.sr
        self.Ls = int(hparams.padded_sample_length*self.fs)
        self.scale = OctScale(hparams.fmin, hparams.fmax, self.n_bins)
        self.nsgt = NSGT(scale = self.scale, Ls = self.Ls, fs = self.fs, real=True, matrixform=True, reducedform=True)

        
    def __call__(self, y):
      cqt = np.asarray(self.nsgt.forward(y[0]))
      mag = (utils.normalize(self.db_scale(np.abs(cqt))))
      phase = np.diff(np.unwrap((np.angle(cqt))),prepend = 0).astype(np.float32)
      phase = (utils.normalize((phase)))
      return torch.squeeze(torch.from_numpy(np.concatenate((np.expand_dims(mag,1),np.expand_dims(phase,1)),1)))#torch.squeeze((torch.from_numpy(mag)))

    def db_scale(self, c):
      C = 20 * np.log10(c**2 + 0.000001)
      return C
    
    def inv_db_scale(self, C):
      c = 10**(C/20) - 0.000001
      return np.sqrt(c)

    def inverse(self, c):
      mag = c[0]
      
      print(mag.shape)
      phase = c[1]
      #phase[:,1024:] = -1
      mag = utils.normalize(mag, -120, 20)
      mag[:,1024:] = -120# mag[:,1024:] - 1

      mag = self.inv_db_scale(mag)
      phase = utils.normalize(phase,-np.math.pi,np.math.pi).cumsum(-1)
      phase = np.math.e**(1j*phase)
      y = self.nsgt.backward(mag*phase)
      return utils.normalize(y, -1,1)


    def griffin_lim(self, c, momentum = 0.99):
      cqt = self.inv_db_scale(utils.normalize(np.hstack(c[:8]), -100, 40))
      rng = np.random
      phase = np.empty(cqt.shape, dtype=np.complex64)

      phase[:] = np.exp(2j * np.pi * rng.rand(*cqt.shape))
      rebuilt = 0
      for i in range(hparams.griffin_lim_iter):
        tprev = rebuilt


        inverse = self.nsgt.backward(np.complex64(cqt)*phase)

        rebuilt = np.asarray(self.nsgt.forward(inverse))

        phase[:] = rebuilt - (momentum / (1 + momentum)) * tprev
        phase[:] /= np.abs(phase) + np.finfo(float).eps

      return self.nsgt.backward(cqt*phase)

    def make_frames(self, c):
      c_framed = np.asarray(np.hsplit(c,8))
      return c_framed
        
def plot_mag_spec(spec, save_path):
    
    plt.imshow(np.flip(spec,0), cmap = 'magma')
    plt.savefig('/content/' + save_path)

def TF_transform(y, mode):
    
    cqt = librosa.cqt(y, sr = hparams.sr, hop_length = hparams.hop,
                   n_bins = hparams.bins, 
                   fmin = hparams.fmin,
                   bins_per_octave = hparams.bins_per_octave)
    
    return cqt

def read_write_midi(midi_path, stem_no_of_samples, sr):
    mid = madmom.io.midi.MIDIFile(filename=midi_path)

    track_duration = np.rint(mid.notes[-1,0] + mid.notes[-1,2])
    
    track = np.zeros(stem_no_of_samples)
    
    for i in range(0,len(mid.notes)):
        midi_note = mid.notes[i,1]
        hz = librosa.midi_to_hz(midi_note)
        start = np.rint(mid.notes[i,0]*sr).astype(int)
        end = mid.notes[i,2]
        velocity = mid.notes[i,3]
        sine = (velocity/127)*librosa.tone(hz, sr = sr, duration = end.astype(float))
        track[start:len(sine)+start] += sine
        
    return track


if __name__ == '__main__':

  cqt_transf = NSGTCQT_Transform()
  











  

