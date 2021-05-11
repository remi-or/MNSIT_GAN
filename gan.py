## Locate
import os
os.chdir("C:/Users/meri2/Documents/Projects/MNSIT_GAN/")



## Imports
#Local
from base_class import Discriminator, Generator


#Global
import numpy as np
import random as rd
import matplotlib.pyplot as plt
from time import perf_counter

import tensorflow.keras as ks



## Misc.function
def temp_print(txt):
    print("\r{}".format(txt), end = "")



## Gan class
class Gan:

    def __init__(self, ldim, p, shape):
        self.generator = Generator(ldim, p, shape)
        self.discriminator = Discriminator(p, shape)
        self.gan = None


    def set_discriminator(self, model):
        self.discriminator.set_model(model)


    def set_generator(self, model):
        self.generator.set_model(model)


    def make_gan(self):
        if self.generator.model == None or self.discriminator.model == None:
            ValueError("Cannot make the gan if the generator model or discriminator model has not been specified.")
        self.discriminator.model.trainable = False
        gan = ks.models.Sequential()
        gan.add(self.generator.model)
        gan.add(self.discriminator.model)
        gan.layers[0].trainable = False
        opt = ks.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        gan.compile(loss='binary_crossentropy', optimizer=opt)
        self.gan = gan


    def train_discriminator(self, real_X, real_Y):
        real_accuracy = self.discriminator.model.train_on_batch(real_X, real_Y)[1]
        fake_X, fake_Y = self.generator.fakes(real_X.shape[0])
        fake_accuracy = self.discriminator.model.train_on_batch(fake_X, fake_Y)[1]
        return real_accuracy, fake_accuracy


    def train_generator(self, n):
        X, Y = self.generator.seeds(n, pretend_real = True)
        return self.gan.train_on_batch(X, Y)


    def train(self, X, Y, epochs = 10, batch_size = 256, ):
        losses, accuracies, times = [], [], []
        if self.gan == None : self.make_gan()
        batches, half_batch_size = int(X.shape[0] / batch_size), int(batch_size / 2)
        keys = [k for k in range(X.shape[0])]
        for i in range(epochs):
            t0, loss_acc, real_accuracy_acc, fake_accuracy_acc = perf_counter(), 0, 0, 0
            rd.shuffle(keys)
            for j in range(batches):
                temp_print('Epoch ' + str(i) + ' batch ' + str(j))
                batch_keys = keys[j *  half_batch_size : (j + 1) * half_batch_size]

                real_accuracy, fake_accuracy = self.train_discriminator(X[batch_keys], Y[batch_keys])
                real_accuracy_acc += real_accuracy
                fake_accuracy_acc += fake_accuracy

                loss = self.train_generator(half_batch_size)
                loss_acc += loss

            losses.append(loss_acc / batches)
            accuracies.append(real_accuracy_acc / batches + fake_accuracy_acc / batches)
            times.append((perf_counter() - t0) / batches)
            temp_print('Epoch ' + str(i) + ' finished in ' + str(perf_counter() - t0))
            print("\nLoss :", loss_acc / batches)
            print("Accuracy on real images:", real_accuracy_acc/ batches)
            print("Accuracy on fake images:", fake_accuracy_acc/ batches, '\n')
        return losses, accuracies, times


    def samples(self, n = 1):
        fig, axs = plt.subplots(n, self.generator.p, figsize = (10, 6))
        if n == 1:
            X, Y = self.generator.fakes(self.generator.p, [i for i in range(self.generator.p)])
            for i in range(self.generator.p):
                axs[i].axis('off')
                axs[i].imshow(X[i], cmap = 'Greys')
        else:
            for row in range(n):
                X, Y = self.generator.fakes(self.generator.p, [i for i in range(self.generator.p)])
                for i in range(self.generator.p):
                    axs[row, i].axis('off')
                    axs[row, i].imshow(X[i], cmap = 'Greys')
        plt.show()


    def save(self, name, path):
        self.generator.model.save(path + '/' + name + '/generator/')
        self.discriminator.model.save(path + '/' + name + '/discriminator/')


    def load(self, path):
        self.generator.model = ks.models.load_model(path + '/generator/')
        self.discriminator.model = ks.models.load_model(path + '/discriminator/')
        self.discriminator.compile()


    def add_memory(self, memory_size):
        X, Y = self.generator(memory_size)
        self.discriminator.add_memory(X, Y)












