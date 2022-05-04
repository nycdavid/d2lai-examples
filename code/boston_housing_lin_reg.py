import pandas as pd

import ipdb

columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.read_csv(
  "boston-housing.csv",
  header = None,
  delimiter = r"\s+",
  names = columns,
)

ipdb.set_trace()
