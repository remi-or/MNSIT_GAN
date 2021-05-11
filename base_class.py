## Imports
import numpy as np
import random as rd

import tensorflow.keras as ks



## Misc. functions
def switch_labels(Y):
  """
  Given an array of labels (Y), replaces fake labels by real ones and vice-versa.
  This is assuming fakes first.
  """
  n, two_p = Y.shape
  p = two_p // 2
  for i in range(n):
    label = read_one_hot(Y[i, :])
    Y[i, label] = 0
    if label >= p:
      Y[i, label - p] = 1
    else:
      Y[i, label + p] = 1
  return Y

def read_one_hot(hot):
  """
  Given a one hot encoded array, returns the indice of the only 1.
  """
  for i in range(len(hot)):
    if hot[i]: return i



## Generator class
class Generator:
    """
    Generates fakes images from random seeds given a target category.
    """

    def __init__(self, ldim, p, shape):
        """
        ldim  : size of the latent space
        p     : number of categories
        shape : shape of the generated images, black and white assumed
        """
        self.ldim = ldim
        self.p = p
        self.shape = shape
        self.model = None

    def set_model(self, model):
        """
        Sets the keras model that is the crux of the generator.
        """
        self.model = model

    def seeds(self, n, categories = None, pretend_real = False, softness = 0.0):
        """
        Returns (n) seeds and their labels for the generator, fake first assumed.
        One can specify the categories of the seeds with the (categories) array.
        The seeds are marked fake but can but marked real with (pretend_real).
        The labels are hard by default, but can be made soft with the (softness) argument.
        The labels then are uniformly spread around 1 with (softness) / 2.
        """
        X, Y = np.random.randn(n, self.p + self.ldim), np.zeros((n, 2 * self.p))
        X[:, : self.p] = 0
        if categories == None:
            categories = np.random.randint(low = 0, high = self.p - 1, size = (n, ))
        for i in range(n):
            c = categories[i]
            X[i, c], Y[i, c] = 1, 1 + (rd.random() - 0.5) * softness
        if pretend_real:
            Y = switch_labels(Y)
        return X, Y


    def fakes(self, n, categories = None, pretend_real = False, softness = 0.0):
        """
        Return (n) fake images and their labels, fake first assumed.
        One can specify the categories of the seeds with the (categories) array.
        The images are marked fake but can but marked real with (pretend_real).
        The labels are hard by default, but can be made soft with the (softness) argument.
        The labels then are uniformly spread around 1 with (softness) / 2.
        """
        if self.model == None:
            ValueError("Can't generate fakes without having a model.")
        X, Y = self.seeds(n, categories, pretend_real, softness)
        X = self.model.predict(X)
        return X, Y



## Dicriminator class
class Discriminator:
    """
    Discriminates between real and fake images.
    """

    def __init__(self, p, shape):
        """
        p     : the number of image categories
        shape : the shape of the given images, 2D assumed
        """
        self.p = p
        self.shape = shape
        self.model = None
        self.memory_size = 0


    def set_model(self, model):
        """
        Sets the keras model that is the crux of the discriminator.
        Then compile the model with a pre-set loss and optimizer.
        """
        self.model = model
        self.compile()


    def compile(self):
        """
        Compiles the model with a pre-set loss and optimizer.
        """
        self.model.compile(loss='binary_crossentropy',
                           optimizer=ks.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
                           metrics=['accuracy'], )























