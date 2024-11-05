# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 20:55:20 2022

@author: sbp247
"""

import librosa

###
###
###
### Paths, dataset, environment setup variables
###
###

musdb_path = "C:\\Users\\micha\Documents\\datasets\\"
path_to_transformed_musdb18 = "E:\\Users\micha\\Documents\\datasets\\"
model_save_path = 'E:\\root\\mss_software\\models\\'

###
###
###
### Audio Parameters
###
###
###

source = 'bass'
sample_length = 4*1.08
slice_length = 2048
sr = 44100
bins_per_octave  = 32#64
num_of_octaves = 11
fmin = librosa.note_to_hz('e0')
fmax = librosa.note_to_hz('e10')

###
###
###
### Network Hyperparameters
###
###
###

number_of_epochs = 100
checkpoint_frequency = 10

batch_size = 5
input_channels = 2
output_channels = 2
num_of_filters = 16

learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999