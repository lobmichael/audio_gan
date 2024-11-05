import numpy as np
import hparams
import spectrogram_utils
import torchaudio
import torch


class Evaluate():
  def __init__(self):
    self.NSGT = spectrogram_utils.NSGTCQT_Transform()

  def __call__(self, input_spec, gen_output, real_output):
    self.make_audio(input_spec, '/content/input_audio.flac')
    self.make_audio(gen_output, '/content/gen_audio.flac')
    self.make_audio(real_output, '/content/real_audio.flac')
    

  def make_audio(self, x, path):
    y = self.NSGT.griffin_lim(x)
    torchaudio.save(path, torch.from_numpy(np.expand_dims(y,0)), 
      hparams.sr, 
      format ="flac", bits_per_sample=16)

class Inference():
  def __init__(self):
    self.NSGT = spectrogram_utils.NSGTCQT_Transform()

  def __call__(self, spec, i):
    y = self.make_audio(spec, '/content/generated_sample_' + str(i+1) + '_drums' +'.flac')
    return y

  def make_audio(self, x, path):
    y = self.NSGT.inverse(x)
    torchaudio.save(path, torch.from_numpy(np.expand_dims(y,0)), 
      hparams.sr, 
      format ="flac", bits_per_sample=16)
    return y