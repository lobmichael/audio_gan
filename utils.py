import os
import numpy as np
import matplotlib.pyplot as plt

import torch

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

def normalize(x,a = -1, b = 1):
  eps = torch.finfo(float).eps
  x = (b-a)*(x-torch.min(x))/(torch.max(x)-torch.min(x)+eps) + a
  return x

def db_scale(c):
    C = 20 * torch.log10(c**2 + 0.000001)
    return C
    
def inv_db_scale(C):
    c = 10**(C/20) - 0.000001
    return torch.sqrt(c)

def plot_mag_spec(spec, save_path):
    
    plt.imshow(np.flip(spec,0), cmap = 'magma')
    plt.savefig('plots/' + save_path)