import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

import time
from pathlib import Path

from torchvision.datasets import MNIST

import nn as nn_utils
import layers
 
class ConvolutionalModel(nn.Module):
  def __init__(self, in_channels, conv1_width, conv2_width, batch_size, fc1_width, class_count):
    super(ConvolutionalModel,self).__init__()
    # in_channels vjv batch*1?*28*28
    self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool1 = nn.MaxPool2d(2)
    # relu
    self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
    self.maxpool2 = nn.MaxPool2d((2,2))
    # relu
    self.fc1 = nn.Linear(7*7*conv2_width, fc1_width, bias=True)
    # relu
    self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

    # parametri su već inicijalizirani pozivima Conv2d i Linear
    # ali možemo ih drugačije inicijalizirati
    self.reset_parameters()

  def reset_parameters(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
      elif isinstance(m, nn.Linear) and m is not self.fc_logits:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.constant_(m.bias, 0)
    self.fc_logits.reset_parameters()

  def forward(self, x):
    h = self.conv1(x)
    h = self.maxpool1(h)
    h = torch.relu(h)  # može i h.relu() ili nn.functional.relu(h)
    h = self.conv2(h)
    h = self.maxpool2(h)
    h = torch.relu(h)
    h = h.view(h.shape[0], -1) # flatten?
    h = self.fc1(h)
    h = torch.relu(h)
    logits = self.fc_logits(h)

    return logits

# nauci na mnist (regularized?)
# pohrani filter i gubitak u datoteku
# vizualiziraj loss (spremi)

DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
SAVE_DIR = Path(__file__).parent / 'out'

config = {}
config['max_epochs'] = 8
config['batch_size'] = 50
config['save_dir'] = SAVE_DIR
config['lr_policy'] = {1:{'lr':1e-1}, 3:{'lr':1e-2}, 5:{'lr':1e-3}, 7:{'lr':1e-4}}

def dense_to_one_hot(y, class_count):
    return np.eye(class_count)[y]

#np.random.seed(100) 
np.random.seed(int(time.time() * 1e6) % 2**31)

ds_train, ds_test = MNIST(DATA_DIR, train=True, download=True), MNIST(DATA_DIR, train=False)
train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]
test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(float) / 255
test_y = ds_test.targets.numpy()
train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))
train_y, valid_y, test_y = (dense_to_one_hot(y, 10) for y in (train_y, valid_y, test_y))



def train(model, x_train, y_train, x_val, y_val, epochs, batch_size, delta):

    optimizer = torch.optim.SGD(model.parameters(), lr=delta)

    for epoch in range(epochs):
      model.train()

      rand_indices = torch.randperm(x_train.shape[0])
      shuffled_x = x_train[rand_indices]
      shuffled_y = y_train[rand_indices]

      for iter in range(0,x_train.shape[0], batch_size):
          batch_x = shuffled_x[iter:iter+batch_size]
          batch_y = shuffled_y[iter:iter+batch_size]

          #forward
          #loss
          #loss back
          #grad step

if __name__ == '__main__':
    
    model = ConvolutionalModel(1, 16, 32, 16, 512, 10)

    loss = layers.SoftmaxCrossEntropyWithLogits() #?nn
    # pisem svoje funkcije za train i eval?
    nn_utils.train(train_x, train_y, valid_x, valid_y, model, loss, config)
    nn_utils.evaluate("Test", test_x, test_y, model, loss, config)