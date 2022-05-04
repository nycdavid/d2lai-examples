import pandas as pd

import torch
from torch import nn

from lin_reg_concise import load_array

import ipdb

def predict(x1, x2):
  return 14.6162 * x1 - 33.4602 * x2 + 31.1143

columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv(
  "boston-housing.csv",
  header = None,
  delimiter = r"\s+",
  names = columns,
)

features = torch.tensor(df.loc[:400, ["RM", "LSTAT"]].values, dtype=torch.float32)
features = (features - features.min()) / (features.max() - features.min())
labels = torch.tensor(df.loc[:400, "MEDV"].values, dtype=torch.float32)
labels += torch.normal(0, 0.01, labels.shape)
labels = labels.reshape((-1, 1))

# ipdb.set_trace()

batch_size = 10

data_iterator = load_array((features, labels), batch_size)
net = nn.Sequential(nn.Linear(features.shape[-1], 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()
trainer = torch.optim.SGD(net.parameters(), lr = 0.03)

for epoch in range(60):
  for X, y in data_iterator:
    l = loss(net(X), y)
    trainer.zero_grad()
    l.backward()
    trainer.step()

  l = loss(net(features), labels)
  print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
b = net[0].bias.data

print("w: ", w)
print("b: ", b)

ipdb.set_trace()
