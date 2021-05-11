## Imports
import numpy as np
import random as rd

from tensorflow.keras.datasets import mnist



## Main function
def load_dataset(softness = 0.0):
  """
  Loads the MNSIT dataset into two array X for images and Y for labels.
  Converts the X array to have the right shape, dtype and normalized format.
  Converts the Y array to be one hot encoded with fake first assumed.
  """
  X, pre_Y = mnist.load_data()[0]
  X = np.expand_dims(X, axis = -1).astype('float32') / 255.0
  Y = np.zeros((pre_Y.shape[0], 2 * 10))
  for i in range(Y.shape[0]):
    Y[i, pre_Y[i] + 10] = 1 + (rd.random() - 0.5) * softness
  return X, Y