## Imports
import matplotlib.pyplot as plt
import numpy as np
import random as rd

from datagen import seeds



## Misc. functions
def read_one_hot(hot):
  """
  Given a one hot encoded array, returns the indice of the only 1.
  """
  for i in range(len(hot)):
    if hot[i]: return i



## Main functions
def plot_fakes(n,
          gen, ldim, p,
          title = None, ):
  """
  See (n) fake images generated with the categorical generator (gen),
  with latent space of dimension (ldim) and (p) categories.
  """
  X, Y = seeds(n, ldim, p)
  X = gen.predict(X)
  fig, axs = plt.subplots(1, n, figsize = (15, 3))
  for i in range(n):
    axs[i].imshow(X[i, :, :, 0], cmap = 'gray_r')
    axs[i].axis('off')
    axs[i].set_title('Fake ' + str(read_one_hot(Y[i, :]) - 1))
  if title != None:
    fig.suptitle(title)
  plt.show()