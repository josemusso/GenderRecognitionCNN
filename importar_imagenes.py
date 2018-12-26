# coding: utf-8

# In[5]:


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
import random

import tensorflow as tf
from PIL import Image
from scipy.misc import imresize

from urllib.request import urlretrieve
from sklearn.model_selection import train_test_split
from cifar10mod import CIFAR10

# In[6]:


# importar etiquetas e imagenes
filelist = glob.glob('UTKFace/*.jpg', recursive=True)

imagenes = np.array([np.array(Image.open(fname)) for fname in filelist])


def prepare_image(image, target_height=64, target_width=64):
    image = imresize(image, (target_width, target_height))
    return image


# In[4]:


print(filelist[1])

# In[1]:


# IMAGENES REESCALADAS A 64
print('escalando imagenes...')
imagenes_chicas = []
edad = []
genero = []
raza = []

# raza a modificar:
'''     
blanco = 0
negro = 1
asiatico = 2
indio = 3
otro = 4
'''

razamod = 0
porcentaje = 20        # DE 0 A 100

for i in range(imagenes.__len__()):
    if 14 <= int(filelist[i].split('_')[0].split('/')[1]) <= 59:
        if int(filelist[i].split('_')[2]) == razamod:
            if random.randint(1,100) <= porcentaje:
                edad.append(int(filelist[i].split('_')[0].split('/')[1]))
                genero.append(int(filelist[i].split('_')[1]))
                raza.append(int(filelist[i].split('_')[2]))
                imagenes_chicas.append(prepare_image(imagenes[i]))

        else:
            edad.append(int(filelist[i].split('_')[0].split('/')[1]))
            genero.append(int(filelist[i].split('_')[1]))
            raza.append(int(filelist[i].split('_')[2]))
            imagenes_chicas.append(prepare_image(imagenes[i]))

imagenes_chicas = np.array(imagenes_chicas)
imgplot = plt.imshow(imagenes_chicas[0])
edad = np.array(edad)
genero = np.array(genero)
raza = np.array(raza)

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


print('Muestras Totales: ' + str(sum(h_blancos)+sum(h_negros)+sum(h_asiaticos)+sum(h_indios)+sum(h_otros)+
      sum(m_blancos)+sum(m_negros)+sum(m_asiaticos)+sum(m_indios)+sum(m_otros)))
# PLOT DESCRIPCION CONJUNTO DEL TEST (FALTA EL TOTAL)

# ARREGLAR

menMeans = [sum(h_blancos), sum(h_negros), sum(h_asiaticos), sum(h_indios), sum(h_otros)]
womenMeans = [sum(m_blancos), sum(m_negros), sum(m_asiaticos), sum(m_indios), sum(m_otros)]

ind = [0, 2000, 4000, 6000, 8000]  # the x locations for the groups
width = 1000  # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, menMeans, width, color='#d62728')
p2 = plt.bar(ind, womenMeans, width,
             bottom=menMeans)

plt.ylabel('Muestras')
plt.title('Composicion Dataset con raza ' +  str(razamod) + ' presente en un ' + str(porcentaje) + '%')
plt.xticks(ind, ('Blancos', 'Negros', 'Asiaticos', 'Indios', 'Otros'))
plt.legend((p1[0], p2[0]), ('Hombres', 'Mujeres'))
plt.gca().invert_yaxis()
plt.gca().set_ylim([0,7500])

plt.show()

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
# ARREGLAR LOS INDICES A VALORES DINAMICOS

datos1 = {'data': imagenes_chicas[0:int(len(imagenes_chicas)/4)],
          'labels': np.transpose([edad[0:int(len(imagenes_chicas)/4)], genero[0:int(len(imagenes_chicas)/4)],
                                  raza[0:int(len(imagenes_chicas)/4)]])
          }

filename1 = "datos1.pkl"
file1 = open(filename1, 'wb')
pickle.dump(datos1, file1)
file1.close()

datos2 = {'data': imagenes_chicas[int(len(imagenes_chicas)/4)+1:int(2*len(imagenes_chicas)/4)],
          'labels': np.transpose([edad[int(len(imagenes_chicas)/4)+1:int(2*len(imagenes_chicas)/4)],
                                  genero[int(len(imagenes_chicas)/4)+1:int(2*len(imagenes_chicas)/4)],
                                  raza[int(len(imagenes_chicas)/4)+1:int(2*len(imagenes_chicas)/4)]])
          }

filename2 = "datos2.pkl"
file2 = open(filename2, 'wb')
pickle.dump(datos2, file2)
file2.close()

datos3 = {'data': imagenes_chicas[int(2*len(imagenes_chicas)/4)+1:int(3*len(imagenes_chicas)/4)],
          'labels': np.transpose([edad[int(2*len(imagenes_chicas)/4)+1:int(3*len(imagenes_chicas)/4)],
                                  genero[int(2*len(imagenes_chicas)/4)+1:int(3*len(imagenes_chicas)/4)],
                                  raza[int(2*len(imagenes_chicas)/4)+1:int(3*len(imagenes_chicas)/4)]])
          }

filename3 = "datos3.pkl"
file3 = open(filename3, 'wb')
pickle.dump(datos3, file3)
file3.close()

datos4 = {'data': imagenes_chicas[int(3*len(imagenes_chicas)/4)+1:int(len(imagenes_chicas))-1],
          'labels': np.transpose([edad[int(3*len(imagenes_chicas)/4)+1:int(len(imagenes_chicas))-1],
                                  genero[int(3*len(imagenes_chicas)/4)+1:int(len(imagenes_chicas))-1],
                                  raza[int(3*len(imagenes_chicas)/4)+1:int(len(imagenes_chicas))-1]])
          }
filename4 = "datos4.pkl"
file4 = open(filename4, 'wb')
pickle.dump(datos4, file4)
file4.close()
