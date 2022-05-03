import numpy as np
import torch

from torch import nn

import ipdb

def start():
  true_w = torch.tensor([2, -3.4])
  true_b = 4.2
  features, labels = synthetic_data(true_w, true_b, 1000)

  batch_size = 10
  data_iter = load_array((features, labels), batch_size)
  net = nn.Sequential(nn.Linear(2, 1))

  net[0].weight.data.normal_(0, 0.01)
  net[0].bias.data.fill_(0)

  loss = nn.MSELoss()
  trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

  for epoch in range(3):
    for X, y in data_iter:
      l = loss(net(X), y)
      trainer.zero_grad()
      l.backward()
      trainer.step()

    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

  w = net[0].weight.data
  b = net[0].bias.data

  print('error in estimating w:', true_w - w.reshape(true_w.shape))
  print('error in estimating b:', true_b - b)

def load_array(data_arrays, batch_size, is_train = True):
  dataset = torch.utils.data.TensorDataset(*data_arrays)

  return torch.utils.data.DataLoader(dataset, batch_size, shuffle = is_train)

def synthetic_data(true_w, true_b, size):
  features = torch.normal(0, 1, (size, len(true_w)))
  labels = torch.matmul(features, true_w) + true_b
  noise = torch.normal(0, 0.01, labels.shape)
  labels += noise
  labels = labels.reshape((-1, 1))

  return features, labels

start()
