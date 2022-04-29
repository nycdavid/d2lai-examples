import random
import torch

def gen_features(rows, cols):
  return torch.normal(0, 1, size = (rows, cols))

def gen_labels(w, b, features):
  y = torch.matmul(features, w) + b
  noise = torch.normal(0, 0.01, size = y.shape)

  return (y + noise).reshape((-1, 1))
