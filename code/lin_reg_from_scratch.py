import random
import torch
import ipdb

def gen_features(rows, cols):
  return torch.normal(0, 1, size = (rows, cols))

def gen_labels(w, b, features):
  y = torch.matmul(features, w) + b
  noise = torch.normal(0, 0.01, size = y.shape)

  return (y + noise).reshape((-1, 1))

def linreg(X, w, b):
  return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
  return ((y_hat - y)**2) / 2

def data_iter(batch_size, features, labels):
  num_examples = len(features)
  indices = list(range(num_examples))

  random.shuffle(indices)

  for i in range(0, num_examples, batch_size):
    batch_indices = torch.tensor(
      indices[i:min(i + batch_size, num_examples)],
    )

    yield features[batch_indices], labels[batch_indices]

def sgd(params, lr, batch_size):
  with torch.no_grad():
    for param in params:
      param -= lr * param.grad / batch_size
      param.grad.zero_()

# Procedural work
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
w = torch.normal(0, 0.01, size = (2, 1), requires_grad = True)
true_w = torch.tensor([2, -3.4])
true_b = 4.2
b = torch.zeros(1, requires_grad = True)
batch_size = 10
features = torch.normal(0, 1, (1000, len(true_w)))
labels = torch.matmul(features, true_w) + true_b
noise = torch.normal(0, 0.01, labels.shape)
labels += noise
labels = labels.reshape((-1, 1))

for epoch in range(num_epochs):
  for X, y in data_iter(batch_size, features, labels):
    l = loss(net(X, w, b), y)
    l.sum().backward()
    sgd([w, b], lr, batch_size)
  with torch.no_grad():
    train_l = loss(net(features, w, b), labels)
    print(f"epoch {epoch + 1}, loss {float(train_l.mean()):f}")
