# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 14:29:36 2022

@author: Michael Lobjoit
"""

from librosa.core.convert import midi_to_note
import madmom
import glob
import math
import os
import time

import multiprocessing

import hparams

import librosa
from scipy.io.wavfile import write as write_wav
import numpy as np

def vec_sine(midi, sine_table):
  midi_note = midi[1]
  start = np.rint(midi[0]*hparams.sr).astype(int)
  end = midi[2]
  velocity = midi[3]
  sine = (velocity/127)*sine_table[str(int(midi_note))][:int(np.ceil(end.astype(float)*hparams.sr))]

  return sine,start

def decode_midi(midi_path, stem_no_of_samples, sr, sine_table):
    mid = madmom.io.midi.MIDIFile(filename=midi_path)

    sines = []
    starts = []
    ends = []

    track_duration = np.rint(mid.notes[-1,0] + mid.notes[-1,2])
    #vfunc = np.vectorize(vec_sine, otypes = [list], excluded = ['sine_table'], signature='(n)->(),()')
    
    track = np.zeros(stem_no_of_samples)
    start = time.time()
    #print(len(mid.notes))
    #print(mid.notes.shape)
    prev_time = start   
    #for i in range(0,len(mid.notes)):
    #sines.append(vfunc(list(mid.notes), sine_table))
    sines = np.apply_along_axis(vec_sine, axis = 1, arr = mid.notes, sine_table = sine_table)
    for sine, start in sines:
      track[start:len(sine)+start] += sine
    track = normalize(track)
    track = float_to_pcm16(track)
    curr = time.time()
    print(curr - start)
    return track

def decode_midi_v1(midi_path, stem_no_of_samples, sr, sine_table):
     
    mid = madmom.io.midi.MIDIFile(filename=midi_path)

    track_duration = np.rint(mid.notes[-1,0] + mid.notes[-1,2])
    
    track = np.zeros(stem_no_of_samples)
    start = time.time()
    print(len(mid.notes))
    for i in range(0,len(mid.notes)):
        midi_note = mid.notes[i,1]
        hz = librosa.midi_to_hz(midi_note)
        start = np.rint(mid.notes[i,0]*sr).astype(int)
        end = mid.notes[i,2]
        velocity = mid.notes[i,3]
        sine = (velocity/127)*sine_table[str(int(midi_note))][:int(np.ceil(end.astype(float)*sr))]
        track[start:len(sine)+start] += sine
    track = normalize(track)
    track = float_to_pcm16(track)
    end = time.time()
    print('Time elapse for 1 track: ' + str(end-start))
    return track

def precompute_sine_table(max_duration = 60):
    sine_table = {}

    for i in range(11,133):
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
    files = sorted(glob.glob(hparams.path_to_dataset + 'Track*/stems/*.wav'))
    return files

def read__midi_durations(files):
    durations = []
    for file in files:
        durations.append(int(math.ceil(librosa.get_duration(filename = file.split('MIDI')[-2] + 'stems/S00.wav', sr = hparams.sr)*hparams.sr)))
    return durations

def write_midi_wav(y, output_path, path):
    path1 = path.split('/')[-3]
    path2 = '/midi_stem/' + path.split('/')[-1].split('.')[-2] + '.wav'
    path = output_path + path1 + path2 
    create_dir(os.path.dirname(path))
    write_wav(path, hparams.sr, y)
    
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
    


    