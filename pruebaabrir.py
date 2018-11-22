from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
from cifar10mod import CIFAR10


def unpickle(filename):
    f = open(filename, 'rb')
    dic = pickle.load(f, encoding='latin1')
    f.close()
    return dic

train_data_list = []
train_labels = []
with open('datos3.pkl', 'rb') as f:
    d = pickle.load(f)  # SACADO EL FOR E IMPORTADO IMAGENES Y LABELS CON DICT UNPICKLE
    train_data_list.append(d['data'])
    train_labels += d['labels']
train_labels = np.asarray(train_labels)
train_data = np.concatenate(train_data_list, axis=0).astype(np.float32)

# cambiar por numero de foto a probar
nro_foto = 3718
imgplot = plt.imshow(train_data[nro_foto].astype(np.uint8))
plt.savefig('fotodepruebadescomprimida.png')

print(train_labels[0, nro_foto], train_labels[1, nro_foto])
print('1 = mujer, 2 = hombre')