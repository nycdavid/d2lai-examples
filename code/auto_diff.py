import torch

# 1. create a tensor
x = torch.arange(4.0)

x.requires_grad_(True) # sets the tensor up for calculating Gradients

y = 2 * torch.dot(x, x) # calculate y with the dot product
