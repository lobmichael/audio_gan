# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:29:36 2022

@author: Michael Lobjoit
"""

import madmom
import glob
import math
import os

import hparams

import librosa
#from scipy.io.wavfile import write as write_wav
import torchaudio
import torch
import numpy as np

def vec_sine(midi, sine_table):
  midi_note = midi[1]
  start = np.rint(midi[0]*hparams.sr).astype(int)
  end = midi[2]
  velocity = midi[3]
  if midi_note < 1 or midi_note > 133:
    return np.zeros(int(np.ceil(end.astype(float)*hparams.sr))),0
  sine = (velocity/127)*sine_table[str(int(midi_note))][:int(np.ceil(end.astype(float)*hparams.sr))]

  return sine,start

def decode_midi(midi_path, stem_no_of_samples, sr, sine_table):
    mid = madmom.io.midi.MIDIFile(filename=midi_path)

    track_duration = np.rint(mid.notes[-1,0] + mid.notes[-1,2])
   
    track = np.zeros(stem_no_of_samples)
    sines = np.apply_along_axis(vec_sine, axis = 1, arr = mid.notes, sine_table = sine_table)
    for sine, start in sines:
      track[start:len(sine)+start] += sine
    track = normalize(track)
    track = float_to_pcm16(track)
    return track

def precompute_sine_table(max_duration = 60):
    sine_table = {}

    for i in range(1,133):
        hz = librosa.midi_to_hz(i)
        sine_table[str(i)] = librosa.tone(hz, sr = hparams.sr, duration = max_duration)
    print('Computed sine table...')    
    return sine_table

def read_midi_files():
    files = sorted(glob.glob(hparams.path_to_dataset + 'Track*/MIDI/*.mid'))
    return files

def normalize(x,a = -1, b = 1):
  x = (b-a)*(x-np.min(x))/(np.max(x)-np.min(x)) - a
  return x

def float_to_pcm16(x):
  x = x*np.iinfo(np.int16).max
  return np.int16(x)

def read_stem_files():
    files = sorted(glob.glob(hparams.path_to_dataset + 'Track*/stems/*.flac'))
    return files

def read__midi_durations(files):
    durations = []
    for file in files:
        path = sorted(glob.glob(file.split('MIDI')[-2] + 'stems/*.flac'))[0]
        durations.append(int(math.ceil(librosa.get_duration(filename = path, sr = hparams.sr)*hparams.sr)))
    return durations

def write_midi_wav(y, output_path, path):
    path1 = path.split('/')[-3]
    path2 = '/midi_stem/' + path.split('/')[-1].split('.')[-2] + '.flac'
    path = output_path + path1 + path2 
    create_dir(os.path.dirname(path))
    #write_wav(path, hparams.sr, y)
    torchaudio.backend.sox_io_backend.save(path, torch.from_numpy(np.expand_dims(y,0)), hparams.sr, 
      format = 'flac', bits_per_sample = 16)
    
def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        
def convert_midi(output_path):
    files = read_midi_files()
    durs = read__midi_durations(files)
    sine_table = precompute_sine_table()
    for idx,path in enumerate(files):
        y = decode_midi(path, durs[idx], hparams.sr, sine_table)
        write_midi_wav(y, output_path, path)

        
        

if __name__ == '__main__':
    
    convert_midi(hparams.midi_output_path)
    


    