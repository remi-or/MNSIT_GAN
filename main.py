## Locate
import os
os.chdir("C:/Users/meri2/Documents/Projects/MNSIT_GAN/")



## Imports
#Local
from gan import Gan
from datagen import load_dataset

# Global
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from tensorflow import keras as ks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LeakyReLU, Reshape, Conv2DTranspose, Conv2D, Flatten, Dropout



## Parameters and dataset
Ldim = 100
P = 10
Shape = (28, 28, 1)

X, Y = load_dataset()



## Gan
Gan = Gan(ldim = Ldim, p = P, shape = Shape)
Gan.load('C:/Users/meri2/Documents/Projects/MNSIT_GAN/Attempt_0')
Gan.make_gan()


losses, accuracies, times = Gan.train(X, Y, epochs = 0, batch_size = 256, )

Gan.samples(7)
plt.plot([i for i in range(len(losses))], losses)
plt.plot([i for i in range(len(losses))], accuracies)
plt.show()












