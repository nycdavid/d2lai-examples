import numpy as np
import math

def normal(x, mu, sigma):
  p = 1 / math.sqrt(2 * math.pi * sigma**2)
  exp = np.exp(-0.5 / sigma**2 * (x -mu)**2)

  return p * exp
