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

# In[ ]:


# importar etiquetas e imagenes
filelist = glob.glob('UTKFace/*.jpg', recursive=True)

imagenes = np.array([np.array(Image.open(fname)) for fname in filelist])


def prepare_image(image, target_height=64, target_width=64):
    image = imresize(image, (target_width, target_height))
    return image


# IMAGENES REESCALADAS A 64
print('escalando imagenes...')
imagenes_chicas = []
for i in range(imagenes.__len__()):
    imagenes_chicas.append(prepare_image(imagenes[i]))
imagenes_chicas = np.array(imagenes_chicas)
# imgplot = plt.imshow(imagenes_chicas[0])


# In[ ]:


edad = []
genero = []
raza = []
c = 0
for fname in filelist:
    print(fname)
    edad.append(int(fname.split('_')[0].split('/')[1]))
    genero.append(int(fname.split('_')[1]))
    raza.append(int(fname.split('_')[2]))
    c += 1
    print(c)
edad = np.array(edad)
genero = np.array(genero)
raza = np.array(raza)
# print(filelist.__len__())
# print(edad[1])
# print(genero[1])
print(raza)

# # Exploración del dataset

# In[ ]:


h_blancos = np.zeros(117, dtype=int)
h_negros = np.zeros(117, dtype=int)
h_asiaticos = np.zeros(117, dtype=int)
h_indios = np.zeros(117, dtype=int)
h_otros = np.zeros(117, dtype=int)
m_blancos = np.zeros(117, dtype=int)
m_negros = np.zeros(117, dtype=int)
m_asiaticos = np.zeros(117, dtype=int)
m_indios = np.zeros(117, dtype=int)
m_otros = np.zeros(117, dtype=int)
for ident in range(genero.__len__()):
    if genero[ident] == 0:
        if raza[ident] == 0:
            e = edad[ident]
            h_blancos[e] += 1
        elif raza[ident] == 1:
            e = edad[ident]
            h_negros[e] += 1
        elif raza[ident] == 2:
            e = edad[ident]
            h_asiaticos[e] += 1
        elif raza[ident] == 3:
            e = edad[ident]
            h_indios[e] += 1
        elif raza[ident] == 4:
            e = edad[ident]
            h_otros[e] += 1
    elif genero[ident] == 1:
        if raza[ident] == 0:
            e = edad[ident]
            m_blancos[e] += 1
        elif raza[ident] == 1:
            e = edad[ident]
            m_negros[e] += 1
        elif raza[ident] == 2:
            e = edad[ident]
            m_asiaticos[e] += 1
        elif raza[ident] == 3:
            e = edad[ident]
            m_indios[e] += 1
        elif raza[ident] == 4:
            e = edad[ident]
            m_otros[e] += 1

        # In[ ]:

plt.plot(h_blancos[0:40], label="blancos")
plt.plot(h_negros[0:40], label="negros")
plt.plot(h_asiaticos[0:40], label="asiaticos")
plt.plot(h_indios[0:40], label="indios")
plt.plot(h_otros[0:40], label="otros")
plt.ylabel('Cantidad de datos')
plt.xlabel('Edad')
plt.title('Distribución de razas por edad en Hombres')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# In[ ]:


plt.plot(m_blancos[0:40], label="blanca")
plt.plot(m_negros[0:40], label="negra")
plt.plot(m_asiaticos[0:40], label="asiatica")
plt.plot(m_indios[0:40], label="india")
plt.plot(m_otros[0:40], label="otra")
plt.ylabel('Cantidad de datos')
plt.xlabel('Edad')
plt.title('Distribución de razas por edad en Mujeres')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# In[ ]:


plt.plot(h_blancos, label="blancos")
plt.plot(h_negroslabel="negros")
plt.plot(h_asiaticos, label="asiaticos")
plt.plot(h_indios, label="indios")
plt.plot(h_otros, label="otros")
plt.ylabel('Cantidad de datos')
plt.xlabel('Edad')
plt.title('Distribución de razas por edad en Hombres')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# In[ ]:


plt.plot(m_blancos, label="blancas")
plt.plot(m_negros, label="negras")
plt.plot(m_asiaticos, label="asiaticas")
plt.plot(m_indios, label="indias")
plt.plot(m_otros, label="otras")
plt.ylabel('Cantidad de datos')
plt.xlabel('Edad')
plt.title('Distribución de razas por edad en Mujeres')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()

# # Guardar datos
#

# In[ ]:


datos1 = {'data': imagenes_chicas[0:5541],
          'labels': np.transpose([edad[0:5541], genero[0:5541], raza[0:5541]])
          }

filename1 = "datos1.pkl"
file1 = open(filename1, 'wb')
pickle.dump(datos1, file1)
file1.close()

datos2 = {'data': imagenes_chicas[5541:11082],
          'labels': np.transpose([edad[5541:11082], genero[5541:11082], raza[5541:11082]])
          }

filename2 = "datos2.pkl"
file2 = open(filename2, 'wb')
pickle.dump(datos2, file2)
file2.close()

datos3 = {'data': imagenes_chicas[11082:16624],
          'labels': np.transpose([edad[11082:16624], genero[11082:16624], raza[11082:16624]])
          }

filename3 = "datos3.pkl"
file3 = open(filename3, 'wb')
pickle.dump(datos3, file3)
file3.close()

datos4 = {'data': imagenes_chicas[16624:22165],
          'labels': np.transpose([edad[16624:22165], genero[16624:22165], raza[16624:22165]])
          }
filename4 = "datos4.pkl"
file4 = open(filename4, 'wb')
pickle.dump(datos4, file4)
file4.close()

datos5 = {'data': imagenes_chicas[22165:27707],
          'labels': np.transpose([edad[22165:27707], genero[22165:27707], raza[22165:27707]])
          }
filename5 = "datos5.pkl"
file5 = open(filename5, 'wb')
pickle.dump(datos5, file5)
file5.close()

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

# In[ ]:


for i in range(1, 4):
    print(i)