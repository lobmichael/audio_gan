import torch
import hparams
from torch.utils.tensorboard import SummaryWriter
import torchvision
import matplotlib.pyplot as plt
import numpy as np 

def matplotlib_imshow(img, one_channel=False):
    plt.imshow(np.transpose(img.numpy(),(1,2,0)), cmap="magma")

class Tensorboard_Logger():
  def __init__(self, experiment_description = 'tensor_test'):
    self.desc = experiment_description
    self.log_path = hparams.log_path
    self.writer = torch.utils.tensorboard.SummaryWriter(self.log_path + experiment_description)
    self.idx = 0

  def write_losses(self, g_loss, d_loss, epoch):
   self.writer.add_scalar('Generator training loss',
                            g_loss,
                            epoch)
   self.writer.add_scalar('Discriminator training loss',
                            d_loss,
                            epoch)
  
  def draw_model(self, model,dummy_input):
    self.writer.add_graph(model, dummy_input)
    self.writer.close()

  def audio_out(self, y, y_ground_truth):
    self.writer.add_audio(tag = 'Ground Truth ' + str(self.idx) , global_step = self.idx,
      snd_tensor = y_ground_truth, sample_rate = hparams.sr)
    self.writer.add_audio(tag = 'Generated ' + str(self.idx) , global_step = self.idx,
      snd_tensor = y, sample_rate = hparams.sr)
    self.idx += 1
    return 0

  def specs_out(self, y_in, y_gen, y_ground_truth):
    images = np.asarray((y_in[0,0], y_gen[0,0], y_ground_truth[0,0], y_in[0,1], y_gen[0,1], y_ground_truth[0,1]), dtype = np.float32)
    print(images.shape)
    img_grid = torchvision.utils.make_grid(torch.unsqueeze(torch.Tensor(images),1))
    matplotlib_imshow(img_grid, one_channel=True)
    self.writer.add_image(tag = 'Generated spectrograms' + str(self.idx) , global_step = self.idx, img_tensor = img_grid)
    return 0
    

