from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


import pickle
import glob
import numpy as np
import os
import time
import sys
import matplotlib.pyplot as plt
import tarfile

import tensorflow as tf

from PIL import Image
from scipy.misc import imresize
from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split

# # Definición de función que importa los datos

# esta parte del codigo hace de decifrador de datos, esto se hace ya
# que puede ser mucho consumo computacional hacer las imagenes a arreglos
# nuevamente.
#

# In[ ]:

# SE CAMBIO LA DIRECCION POR LA DE CARPETA DE DESBALANCe
DIR_BINARIES = 'Datos_desb/asiaticos_20/'
print(DIR_BINARIES)

def unpickle(filename):
    f = open(filename, 'rb')
    dic = pickle.load(f, encoding='latin1')
    f.close()
    return dic


def batch_to_bc01(batch):
    ''' Converts CIFAR sample to bc01 tensor'''
    return batch.reshape([-1, 3, 64, 64])  # CAMBIADO SHAPE 32 POR 64


def batch_to_b01c(batch):
    ''' Converts CIFAR sample to b01c tensor'''
    return batch_to_bc01(batch).transpose(0, 2, 3, 1)


def labels_to_one_hot(labels):
    ''' Converts list of integers to numpy 2D array with one-hot encoding'''
    N = len(labels)
    one_hot_labels = np.zeros([N, 2], dtype=int)  # CAMBIADO 10 CLASES POR 2
    one_hot_labels[np.arange(N), labels] = 1
    return one_hot_labels


class CIFAR10:
    def __init__(self, batch_size=100, validation_proportion=0.1, test_proportion=0.1, augment_data=False):

        # Training set
        train_data_list = []
        train_labels_list = []
        r = range(1, 5)
        for bi in r:
            d = unpickle(DIR_BINARIES + 'datos' + str(bi) + '.pkl')
            train_data_list.append(d['data'])
            train_labels_list.append(d['labels'])
        self.train_data = np.concatenate(train_data_list, axis=0).astype(np.float32)
        self.train_labels = np.concatenate(train_labels_list, axis=0).astype(np.uint8)

        # Check set
        check_proportion = validation_proportion + test_proportion
        assert check_proportion > 0. and check_proportion < 1.
        self.train_data, self.check_data, self.train_labels, self.check_labels = train_test_split(
            self.train_data, self.train_labels, test_size=check_proportion, random_state=1)

        # Validation set and Test set
        self.test_data, self.validation_data, self.test_labels, self.validation_labels = train_test_split(
            self.check_data, self.check_labels, test_size=validation_proportion, random_state=1)

        # Normalize data
        mean = self.train_data.mean(axis=0)
        std = self.train_data.std(axis=0)
        self.train_data = (self.train_data - mean) / std
        self.validation_data = (self.validation_data - mean) / std
        self.test_data = (self.test_data - mean) / std

        # Converting to b01c and one-hot encoding
        self.train_data = batch_to_b01c(self.train_data)
        self.validation_data = batch_to_b01c(self.validation_data)
        self.test_data = batch_to_b01c(self.test_data)

        self.train_labels_edad = np.transpose(self.train_labels)[0]
        self.validation_labels_edad = np.transpose(self.validation_labels)[0]
        self.test_labels_edad = np.transpose(self.test_labels)[0]

        self.train_labels_raza = np.transpose(self.train_labels)[2]
        self.validation_labels_raza = np.transpose(self.validation_labels)[2]
        self.test_labels_raza = np.transpose(self.test_labels)[2]

        self.train_labels_genero = np.transpose(self.train_labels)[1]
        self.validation_labels_genero = np.transpose(self.validation_labels)[1]
        self.test_labels_genero = np.transpose(self.test_labels)[1]

        self.train_labels_genero = labels_to_one_hot(self.train_labels_genero)
        self.validation_labels_genero = labels_to_one_hot(self.validation_labels_genero)
        self.test_labels_genero = labels_to_one_hot(self.test_labels_genero)

        np.random.seed(seed=1)
        self.augment_data = augment_data

        # Batching & epochs
        self.batch_size = batch_size
        self.n_batches = len(self.train_labels_genero) // self.batch_size
        self.current_batch = 0
        self.current_epoch = 0

    def nextBatch(self):
        ''' Returns a tuple with batch and batch index '''
        start_idx = self.current_batch * self.batch_size
        end_idx = start_idx + self.batch_size
        batch_data = self.train_data[start_idx:end_idx]
        batch_labels_genero = self.train_labels_genero[start_idx:end_idx]
        batch_labels_raza = self.train_labels_raza[start_idx:end_idx]
        batch_labels_edad = self.train_labels_edad[start_idx:end_idx]
        batch_idx = self.current_batch

        if self.augment_data:
            if np.random.randint(0, 2) == 0:
                batch_data = batch_data[:, :, ::-1, :]
            batch_data += np.random.randn(self.batch_size, 1, 1, 3) * 0.05

        # Update self.current_batch and self.current_epoch
        self.current_batch = (self.current_batch + 1) % self.n_batches
        if self.current_batch != batch_idx + 1:
            self.current_epoch += 1

            # shuffle training data
            new_order = np.random.permutation(np.arange(len(self.train_labels_genero)))
            self.train_data = self.train_data[new_order]
            self.train_labels_genero = self.train_labels_genero[new_order]
            self.train_labels_raza = self.train_labels_raza[new_order]
            self.train_labels_edad = self.train_labels_edad[new_order]

        return ((batch_data, batch_labels_genero, batch_labels_raza, batch_labels_edad), batch_idx)

    def getEpoch(self):
        return self.current_epoch

    # TODO: refactor getTestSet and getValidationSet to avoid code replication
    def getTestSet(self, asBatches=False):
        if asBatches:
            batches = []
            for i in range(len(self.test_labels_genero) // self.batch_size):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_data = self.test_data[start_idx:end_idx]
                batch_labels_genero = self.test_labels_genero[start_idx:end_idx]
                batch_labels_raza = self.test_labels_raza[start_idx:end_idx]
                batch_labels_edad = self.test_labels_edad[start_idx:end_idx]

                batches.append((batch_data, batch_labels_genero, batch_labels_raza, batch_labels_edad))
            return batches
        else:
            return (self.test_data, self.test_labels_genero, self.test_labels_raza, self.test_labels_edad)

    def getValidationSet(self, asBatches=False):
        if asBatches:
            batches = []
            for i in range(len(self.validation_labels_genero) // self.batch_size):
                start_idx = i * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_data = self.validation_data[start_idx:end_idx]
                batch_labels_genero = self.validation_labels_genero[start_idx:end_idx]
                batch_labels_raza = self.validation_labels_raza[start_idx:end_idx]
                batch_labels_edad = self.validation_labels_edad[start_idx:end_idx]

                batches.append((batch_data, batch_labels_genero, batch_labels_raza, batch_labels_edad))
            return batches
        else:
            return (self.validation_data, self.validation_labels_genero, self.validation_labels_raza,
                    self.validation_labels_edad)

    def shuffleValidation(self):
        new_order = np.random.permutation(np.arange(len(self.validation_labels_genero)))
        self.validation_labels_genero = self.validation_labels_genero[new_order]
        self.validation_labels_raza = self.validation_labels_raza[new_order]
        self.validation_labels_edad = self.validation_labels_edad[new_order]
        self.validation_data = self.validation_data[new_order]

    def reset(self):
        self.current_batch = 0
        self.current_epoch = 0


if __name__ == '__main__':
    cifar10 = CIFAR10(batch_size=64)
    while cifar10.getEpoch() < 2:
        batch, batch_idx = cifar10.nextBatch()
        # print(batch_idx, cifar10.n_batches, cifar10.getEpoch())
    batch = cifar10.getTestSet(asBatches=True)
    # print(len(batch))
    for ba in batch:
        data, genero, raza, etnia = ba
        print(data)
    data, genero, raza, edad = cifar10.getValidationSet()
    # print(genero.sum(axis=0))
    data, genero, raza, edad = cifar10.getTestSet()
    # print(genero.sum(axis=0))

