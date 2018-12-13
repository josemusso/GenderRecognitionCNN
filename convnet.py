from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

# %matplotlib inline
import tensorflow as tf

from cifar10mod import CIFAR10

tf.set_random_seed(1)
sess = tf.Session()

"""### Constructor del dataset. Modifique aquí el *flag* de *data augmentation*"""

# Load dataset
batch_size = 64
cifar10 = CIFAR10(batch_size=batch_size, validation_proportion=0.1, test_proportion=0.1, augment_data=False)


# Model blocks
def conv_layer(input_tensor, kernel_shape, layer_name):
    # input_tensor b01c
    # kernel_shape 01-in-out
    weights = tf.get_variable("weights", kernel_shape,
                              initializer=tf.contrib.layers.xavier_initializer_conv2d())
    biases = tf.get_variable("biases", [kernel_shape[3]],
                             initializer=tf.constant_initializer(0.05))

    tf.summary.histogram(layer_name + "/weights", weights)
    tf.summary.histogram(layer_name + "/biases", biases)

    # Other options are to use He et. al init. for weights and 0.01
    # to init. biases.
    conv = tf.nn.conv2d(input_tensor, weights,
                        strides=[1, 1, 1, 1], padding='SAME')
    return tf.nn.relu(conv + biases)


def fc_layer(input_tensor, weights_shape, layer_name):
    # weights_shape in-out
    weights = tf.get_variable("weights", weights_shape,
                              initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable("biases", [weights_shape[1]],
                             initializer=tf.constant_initializer(0.0))
    tf.summary.histogram(layer_name + "/weights", weights)
    tf.summary.histogram(layer_name + "/biases", biases)
    mult_out = tf.matmul(input_tensor, weights)
    return tf.nn.relu(mult_out + biases)


"""## Elija aquí el directorio donde guardará los registros de Tensorboard"""

PARENT_DIR = './summaries/'
SUMMARIES_DIR = PARENT_DIR + 'conv_2_layer_with_dropout'

"""## Construcción del grafo

### Construcción del modelo + función de costos
"""

# Model
use_convnet = True
n_conv_layers = 3

n_filters_convs = [32, 64, 128]

model_input = tf.placeholder(tf.float32, name='model_input',
                             shape=(batch_size, 64, 64, 3))

keep_prob = tf.placeholder(tf.float32, name='dropout_prob', shape=())

target = tf.placeholder(tf.float32, name='target', shape=(batch_size, 2))

# NUEVO PLACEHOLDER PARA EDAD Y RAZA:

target_edad = tf.placeholder(tf.float32, name='edad', shape=(batch_size))

target_raza = tf.placeholder(tf.float32, name='edad', shape=(batch_size))

if use_convnet:
    layer_input = model_input
    previous_n_feature_maps = 3
    for layer_index in range(n_conv_layers):
        layer_name = 'conv%d' % layer_index
        with tf.variable_scope(layer_name):
            conv_out = conv_layer(
                layer_input,
                [5, 5, previous_n_feature_maps, n_filters_convs[layer_index]],
                layer_name)
        if layer_index == 0:
            with tf.variable_scope(layer_name, reuse=True):
                conv1_filters = tf.get_variable("weights")
                tf.summary.image(
                    'conv1_filters',
                    tf.transpose(conv1_filters, perm=[3, 0, 1, 2]),
                    max_outputs=n_filters_convs[layer_index]
                )
        previous_n_feature_maps = n_filters_convs[layer_index]
        pool_out = tf.nn.max_pool(
            conv_out,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME',
            name='pool%d' % layer_index)
        layer_input = pool_out

    fc_input = tf.layers.flatten(pool_out, name='fc_input')

    feature_map_height = int(64 / (2 ** n_conv_layers))

    # First fully connected layer
    layer_name = 'fc1'
    with tf.variable_scope(layer_name):
        fc1_out = fc_layer(
            fc_input,
            [(feature_map_height ** 2) * previous_n_feature_maps, 50],
            layer_name)

    fc1_out_drop = tf.nn.dropout(fc1_out, keep_prob)

    # Second fully connected layer
    layer_name = 'fc2'
    with tf.variable_scope(layer_name):
        fc2_out = fc_layer(fc1_out_drop, [50, 2], layer_name)
    model_output = fc2_out

else:
    # Reshape tensor to MLP
    first_layer_input = tf.reshape(model_input, [-1, 3072], name='first_layer_input')

    # First layer
    layer_name = 'fc1'
    with tf.variable_scope(layer_name):
        fc1_out = fc_layer(first_layer_input, [3072, 100], layer_name)

    fc1_out_drop = tf.nn.dropout(fc1_out, keep_prob)

    # Second layer
    layer_name = 'fc2'
    with tf.variable_scope(layer_name):
        fc2_out = fc_layer(fc1_out_drop, [100, 10], layer_name)
    model_output = fc2_out

with tf.name_scope('loss_function'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model_output, labels=target,
                                                name='cross_entropy'))
    xentropy_summary = tf.summary.scalar('cross_entropy', cross_entropy)

"""### Construcción del optimizador + funciones auxiliares"""

# Optimization
with tf.name_scope('optimizer'):
    optimizer = tf.train.RMSPropOptimizer(0.00005)              # REDUCIDA 1 MAGNITUD
    grads_vars = optimizer.compute_gradients(cross_entropy)
    optimizer.apply_gradients(grads_vars)
    train_step = optimizer.minimize(cross_entropy)

# Metrics
correct_prediction = tf.equal(tf.argmax(model_output, 1),
                              tf.argmax(target, 1))

print(correct_prediction.shape)
print(target.shape)
print(model_output.shape)
print(tf.argmax(target, 1).shape)

con_mat = tf.confusion_matrix(labels=tf.argmax(target, 1), predictions=tf.argmax(model_output, 1), num_classes=None,
                              dtype=tf.int32, name=None)

#  con_vec = [Raza ,Edad, EtiquetaGenero, PrediccionGenero]
con_vec = [target_raza, target_edad,tf.argmax(target, 1),tf.argmax(model_output, 1)]

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
accuracy_summary = tf.summary.scalar('accuracy', accuracy)
learning_summaries = tf.summary.merge((xentropy_summary, accuracy_summary))
merged = tf.summary.merge_all()


# Useful training functions
def validate():
    cifar10.shuffleValidation()
    batches = cifar10.getValidationSet(asBatches=True)
    accs = []
    xent_vals = []
    for batch in batches:
        data, genero, raza, edad = batch
        acc, xentropy_val = sess.run((accuracy, cross_entropy),
                                     feed_dict={
                                         model_input: data,
                                         target: genero,
                                         keep_prob: 1.0
                                     })
        accs.append(acc)
        xent_vals.append(xentropy_val)
    mean_xent = np.array(xent_vals).mean()
    mean_acc = np.array(accs).mean()
    summary = sess.run(
        merged,
        feed_dict={
            model_input: data,
            target: genero,
            keep_prob: 1.0
        })
    return summary, mean_acc, mean_xent

def test():
    batches = cifar10.getTestSet(asBatches=True)
    accs = []
    matsum = np.zeros((2, 2))
    vecsum = np.asarray([['Raza' ,'Edad', 'EtiquetaGenero', 'PrediccionGenero']])
    for batch in batches:
        data, genero, raza, edad = batch
        acc, mat, vec = sess.run((accuracy, con_mat, con_vec),
                            feed_dict={
                                model_input: data,
                                target: genero,
                                target_edad: edad,
                                target_raza: raza,
                                keep_prob: 1.0
                            })
        accs.append(acc)
        matsum = matsum + mat
        # print(vecsum.shape)
        # print(np.transpose(np.asarray(vec)).shape)

        vecsum = np.concatenate((vecsum, np.transpose(np.asarray(vec))), axis=0)

    mean_acc = np.array(accs).mean()
    return mean_acc, matsum, vecsum


# PARA GENERAR MATRICES DE CONFUSION SEGUN CADA ETNIA

def testrazas():
    batches = cifar10.getTestSet(asBatches=True)
    accs0 = []
    accs1 = []
    accs2 = []
    accs3 = []
    accs4 = []

    matsum0 = np.zeros((2, 2))
    matsum1 = np.zeros((2, 2))
    matsum2 = np.zeros((2, 2))
    matsum3 = np.zeros((2, 2))
    matsum4 = np.zeros((2, 2))

    mean_acc0 =0
    mean_acc1 =0
    mean_acc2 =0
    mean_acc3 =0
    mean_acc4 =0

    data0 = []
    genero0 = []
    data1 = []
    genero1 = []
    data2 = []
    genero2 = []
    data3 = []
    genero3 = []
    data4 = []
    genero4 = []

    for batch in batches:
        data, genero, raza, edad = batch

        '''     
        blanco = 0
        negro = 1
        asiatico = 2
        indio = 3
        otro = 4
        '''

        for i in range(len(raza)):
            if raza[i] == 0:
                data0.append(data[i])
                genero0.append(genero[i])
            if raza[i] == 1:
                data1.append(data[i])
                genero1.append(genero[i])
            if raza[i] == 2:
                data2.append(data[i])
                genero2.append(genero[i])
            if raza[i] == 3:
                data3.append(data[i])
                genero3.append(genero[i])
            if raza[i] == 4:
                data4.append(data[i])
                genero4.append(genero[i])

        if len(data0)>64:
            acc0, mat0 = sess.run((accuracy, con_mat),
                                feed_dict={
                                    model_input: data0[0:64],
                                    target: genero0[0:64],
                                    keep_prob: 1.0
                                })
            accs0.append(acc0)
            matsum0 = matsum0 + mat0
            mean_acc0 = np.array(accs0).mean()
            data0 = data0[65:len(data0)-1]
            genero0 = genero0[65:len(genero0)-1]

        if len(data1)>64:
            acc1, mat1 = sess.run((accuracy, con_mat),
                                feed_dict={
                                    model_input: data1[0:64],
                                    target: genero1[0:64],
                                    keep_prob: 1.0
                                })
            accs1.append(acc1)
            matsum1 = matsum1 + mat1
            mean_acc1 = np.array(accs1).mean()
            data1 = data1[65:len(data1)-1]
            genero1 = genero1[65:len(genero1)-1]

        if len(data2)>64:
            acc2, mat2 = sess.run((accuracy, con_mat),
                                feed_dict={
                                    model_input: data2[0:64],
                                    target: genero2[0:64],
                                    keep_prob: 1.0
                                })
            accs2.append(acc2)
            matsum2 = matsum2 + mat2
            mean_acc2 = np.array(accs2).mean()
            data2 = data2[65:len(data2)-1]
            genero2 = genero2[65:len(genero2)-1]

        if len(data3)>64:
            acc3, mat3 = sess.run((accuracy, con_mat),
                                feed_dict={
                                    model_input: data3[0:64],
                                    target: genero3[0:64],
                                    keep_prob: 1.0
                                })
            accs3.append(acc3)
            matsum3 = matsum3 + mat3
            mean_acc3 = np.array(accs3).mean()
            data3 = data3[65:len(data3)-1]
            genero3 = genero3[65:len(genero3)-1]

        if len(data4)>64:
            acc4, mat4 = sess.run((accuracy, con_mat),
                                feed_dict={
                                    model_input: data4[0:64],
                                    target: genero4[0:64],
                                    keep_prob: 1.0
                                })
            accs4.append(acc4)
            matsum4 = matsum4 + mat4
            mean_acc4 = np.array(accs4).mean()
            data4 = data4[65:len(data4)-1]
            genero4 = genero4[65:len(genero4)-1]

    return mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4, matsum0, matsum1, matsum2, matsum3, matsum4


# Tensorboard writers
train_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/train',
                                     sess.graph)
validation_writer = tf.summary.FileWriter(SUMMARIES_DIR + '/validation')

"""## Entrenar
### Modifique el valor de *keep_prob* usado al computar *train_step* para activar/desactivar el *dropout*.
### NO MODIFIQUE EL KEEP_PROB USADO EN OTROS SITIOS, ESOS CORRESPONDEN A INFERENCIA.
"""

sess.run(tf.global_variables_initializer())
cifar10.reset()
print("Trainable variables")
for n in tf.trainable_variables():
    print(n.name)
if use_convnet:
    epochs = 5
else:
    epochs = 50

t_i = time.time()
n_batches = cifar10.n_batches
val_acc_vals = []
test_acc_vals = []
test_acc_vals0 = []
test_acc_vals1 = []
test_acc_vals2 = []
test_acc_vals3 = []
test_acc_vals4 = []

hist_loss = [1.0]
patience_cnt =0

while cifar10.getEpoch() < epochs:
    epoch = cifar10.getEpoch()
    batch, batch_idx = cifar10.nextBatch()
    batch_data = batch[0]
    batch_genero = batch[1]
    batch_raza = batch[2]
    batch_edad = batch[3]

    # just a training iteration
    _ = sess.run(train_step,
                 feed_dict={
                     model_input: batch_data,
                     target: batch_genero,
                     keep_prob: 0.8  ###### Modifique el dropout aqui y solo aqui. #####
                 })

    step = batch_idx + epoch * n_batches

    # Write training summary
    if step % 50 == 0:
        summary = sess.run(learning_summaries,
                           feed_dict={
                               model_input: batch_data,
                               target: batch_genero,
                               keep_prob: 1.0  # set to 1.0 at inference time
                           })
        train_writer.add_summary(summary, step)

    # gradient (by layer) statistics over last training batch & validation summary
    if batch_idx == 0:
        loss, acc, grads = sess.run((cross_entropy, accuracy, grads_vars),
                                    feed_dict={
                                        model_input: batch_data,
                                        target: batch_genero,
                                        keep_prob: 1.0
                                    })

        summary, validation_accuracy, validation_loss = validate()
        validation_writer.add_summary(summary, step)
        print('[Epoch %d, it %d] Training acc. %.3f, loss %.3f. \
Valid. acc. %.3f, loss %.3f' % (
            epoch,
            step,
            acc,
            loss,
            validation_accuracy,
            validation_loss
        ))
        val_acc_vals.append(validation_accuracy)
        test_accuracy, mat, vec = test()

        test_acc_vals.append(test_accuracy)


        # IMPLEMENTACION EARLY STOPPING

        hist_loss.append(validation_loss)
        patience = 5
        min_delta = 0.001

        if epoch > 0 and hist_loss[epoch - 1] - hist_loss[epoch] > min_delta:
            patience_cnt = 0
        else:
            patience_cnt += 1
        print('Early Stopping checks: %d/%d' %
              (patience_cnt, patience))
        if patience_cnt > patience:
            print("Early Stopping Activado")
            break

        print("Time elapsed %.2f minutes" % ((time.time() - t_i) / 60.0))


# TESTEA SOLO EN EL ULTIMO
mean_acc0, mean_acc1, mean_acc2, mean_acc3, mean_acc4, matsum0, matsum1, matsum2, matsum3, matsum4 = testrazas()


test_acc_vals0.append(mean_acc0)
test_acc_vals1.append(mean_acc1)
test_acc_vals2.append(mean_acc2)
test_acc_vals3.append(mean_acc3)
test_acc_vals4.append(mean_acc4)

train_writer.flush()
validation_writer.flush()

val_acc_vals = np.array(val_acc_vals)
test_acc_vals = np.array(test_acc_vals)
test_acc_vals0 = np.array(test_acc_vals0)
test_acc_vals1 = np.array(test_acc_vals1)
test_acc_vals2 = np.array(test_acc_vals2)
test_acc_vals3 = np.array(test_acc_vals3)
test_acc_vals4 = np.array(test_acc_vals4)

best_epoch = np.argmax(val_acc_vals)
test_acc_at_best = test_acc_vals[best_epoch]
print('*' * 30)
print("Testing set accuracy @ epoch %d (best validation acc): %.4f" % (best_epoch, test_acc_at_best))
print('*' * 30)
print("Testing set accuracy ultima Epoca: %.4f" % (test_accuracy))
print('*' * 30)

# CREAR MATRICES DE CONFUSION:

'''     
blanco = 0
negro = 1
asiatico = 2
indio = 3
otro = 4
'''

matriz_Blanco = np.zeros((2, 2))
matriz_Negro = np.zeros((2, 2))
matriz_Asiatico = np.zeros((2, 2))
matriz_Indio = np.zeros((2, 2))
matriz_Otro = np.zeros((2, 2))
print("lolxd:",vec[1])
for muestra in vec:
    if muestra[0] == '0.0':
        if (muestra[2] == '0.0') & (muestra[3] == '0.0'):
            matriz_Blanco[0][0] += 1
        if (muestra[2] == '0.0') & (muestra[3] == '1.0'):
            matriz_Blanco[1][0] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '0.0'):
            matriz_Blanco[0][1] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '1.0'):
            matriz_Blanco[1][1] += 1

    if muestra[0] == '1.0':
        if (muestra[2] == '0.0') & (muestra[3] == '0.0'):
            matriz_Negro[0][0] += 1
        if (muestra[2] == '0.0') & (muestra[3] == '1.0'):
            matriz_Negro[1][0] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '0.0'):
            matriz_Negro[0][1] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '1.0'):
            matriz_Negro[1][1] += 1

    if muestra[0] == '2.0':
        if (muestra[2] == '0.0') & (muestra[3] == '0.0'):
            matriz_Asiatico[0][0] += 1
        if (muestra[2] == '0.0') & (muestra[3] == '1.0'):
            matriz_Asiatico[1][0] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '0.0'):
            matriz_Asiatico[0][1] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '1.0'):
            matriz_Asiatico[1][1] += 1


    if muestra[0] == '3.0':
        if (muestra[2] == '0.0') & (muestra[3] == '0.0'):
            matriz_Indio[0][0] += 1
        if (muestra[2] == '0.0') & (muestra[3] == '1.0'):
            matriz_Indio[1][0] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '0.0'):
            matriz_Indio[0][1] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '1.0'):
            matriz_Indio[1][1] += 1

    if muestra[0] == '4.0':
        if (muestra[2] == '0.0') & (muestra[3] == '0.0'):
            matriz_Otro[0][0] += 1
        if (muestra[2] == '0.0') & (muestra[3] == '1.0'):
            matriz_Otro[1][0] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '0.0'):
            matriz_Otro[0][1] += 1
        if (muestra[2] == '1.0') & (muestra[3] == '1.0'):
            matriz_Otro[1][1] += 1

# NORMALIZAR
mat_norm = mat/np.linalg.norm(mat)

matriz_Blanco_norm = matriz_Blanco/np.linalg.norm(matriz_Blanco)
matriz_Negro_norm = matriz_Negro/np.linalg.norm(matriz_Negro)
matriz_Asiatico_norm = matriz_Asiatico/np.linalg.norm(matriz_Asiatico)
matriz_Indio_norm = matriz_Indio/np.linalg.norm(matriz_Indio)
matriz_Otro_norm = matriz_Otro/np.linalg.norm(matriz_Otro)

# CONSTRUCCION GRAFICOS DE BARRA

# Accuracy Rate por Raza

ACC_Blancos_hombre = matriz_Blanco[0][0]/(matriz_Blanco[0][0]+matriz_Blanco[0][1])
ACC_Blancos_mujeres = matriz_Blanco[1][1]/(matriz_Blanco[1][1]+matriz_Blanco[1][0])
ACC_Blancos = (matriz_Blanco[0][0]+matriz_Blanco[1][1])/(matriz_Blanco[0][0]+matriz_Blanco[0][1]+matriz_Blanco[1][0]+matriz_Blanco[1][1])

ACC_Negro_hombre = matriz_Negro[0][0]/(matriz_Negro[0][0]+matriz_Negro[0][1])
ACC_Negro_mujeres = matriz_Negro[1][1]/(matriz_Negro[1][1]+matriz_Negro[1][0])
ACC_Negro = (matriz_Negro[0][0]+matriz_Negro[1][1])/(matriz_Negro[0][0]+matriz_Negro[0][1]+matriz_Negro[1][0]+matriz_Negro[1][1])

ACC_Asiatico_hombre = matriz_Asiatico[0][0]/(matriz_Asiatico[0][0]+matriz_Asiatico[0][1])
ACC_Asiatico_mujeres = matriz_Asiatico[1][1]/(matriz_Asiatico[1][1]+matriz_Asiatico[1][0])
ACC_Asiatico = (matriz_Asiatico[0][0]+matriz_Asiatico[1][1])/(matriz_Asiatico[0][0]+matriz_Asiatico[0][1]+matriz_Asiatico[1][0]+matriz_Asiatico[1][1])

ACC_Indio_hombre = matriz_Indio[0][0]/(matriz_Indio[0][0]+matriz_Indio[0][1])
ACC_Indio_mujeres = matriz_Indio[1][1]/(matriz_Indio[1][1]+matriz_Indio[1][0])
ACC_Indio = (matriz_Indio[0][0]+matriz_Indio[1][1])/(matriz_Indio[0][0]+matriz_Indio[0][1]+matriz_Indio[1][0]+matriz_Indio[1][1])

ACC_Total = np.mean([ACC_Blancos, ACC_Negro, ACC_Asiatico, ACC_Indio])

ACC_plot = [ACC_Blancos_hombre, ACC_Blancos_mujeres, ACC_Negro_hombre, ACC_Negro_mujeres, ACC_Asiatico_hombre,
            ACC_Asiatico_mujeres, ACC_Indio_hombre, ACC_Indio_mujeres, ACC_Blancos, ACC_Negro, ACC_Asiatico, ACC_Indio,
            ACC_Total]

N = 4

fig, ax = plt.subplots()

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars

# PLOT GENEROS SEPARADOS
menMeans = [ACC_Blancos_hombre, ACC_Negro_hombre, ACC_Asiatico_hombre, ACC_Indio_hombre]
p1 = ax.bar(ind, menMeans, width, color='b')


womenMeans = [ACC_Blancos_mujeres, ACC_Negro_mujeres, ACC_Asiatico_mujeres, ACC_Indio_mujeres]
p2 = ax.bar(ind + width, womenMeans, width,
            color='c')

generalMeans = [ACC_Blancos, ACC_Negro, ACC_Asiatico, ACC_Indio]


# PLOT GENEROS JUNTOS
ax.bar(4 , generalMeans[0], width, color='g')
ax.bar(4 + width, generalMeans[1], width, color='y')
ax.bar(4 + 2*width, generalMeans[2], width, color='g')
ax.bar(4 + 3*width, generalMeans[3], width, color='y')

# PLOT RAZAS JUNTAS
ax.bar(6 + width/2, ACC_Total, width, color='r')

ax.set_title('Accuracy por Raza')
pos_ejes = [0 +width/2, 1 +width/2, 2 +width/2, 3 +width/2, 4+2*width/2, 6 +width/2]
ax.set_xticks(pos_ejes)

ax.set_xticklabels(('ACC_Blancos', 'ACC_Negro', 'ACC_Asiatico', 'ACC_Indio', 'Blancos\n Negros\n Asiaticos\n Indios', 'Total'))
plt.xticks(rotation=45)

ax.legend((p1[0], p2[0]), ('Hombres', 'Mujeres'))
ax.autoscale_view()

plt.show()

# False Discovery Rate
FDR_Blancos_hombre = matriz_Blanco[0][1]/(matriz_Blanco[0][0]+matriz_Blanco[0][1])
FDR_Blancos_mujeres = matriz_Blanco[1][0]/(matriz_Blanco[1][1]+matriz_Blanco[1][0])
FDR_Blancos = [FDR_Blancos_hombre, FDR_Blancos_mujeres]

FDR_Negro_hombre = matriz_Negro[0][1]/(matriz_Negro[0][0]+matriz_Negro[0][1])
FDR_Negro_mujeres = matriz_Negro[1][0]/(matriz_Negro[1][1]+matriz_Negro[1][0])
FDR_Negro = [FDR_Negro_hombre, FDR_Negro_mujeres]

FDR_Asiatico_hombre = matriz_Asiatico[0][1]/(matriz_Asiatico[0][0]+matriz_Asiatico[0][1])
FDR_Asiatico_mujeres = matriz_Asiatico[1][0]/(matriz_Asiatico[1][1]+matriz_Asiatico[1][0])
FDR_Asiatico = [FDR_Asiatico_hombre, FDR_Asiatico_mujeres]

FDR_Indio_hombre = matriz_Indio[0][1]/(matriz_Indio[0][0]+matriz_Indio[0][1])
FDR_Indio_mujeres = matriz_Indio[1][0]/(matriz_Indio[1][1]+matriz_Indio[1][0])
FDR_Indio = [FDR_Indio_hombre, FDR_Indio_mujeres]

FDR_Total = [FDR_Blancos_hombre, FDR_Blancos_mujeres, FDR_Negro_hombre, FDR_Negro_mujeres, FDR_Asiatico_hombre, FDR_Asiatico_mujeres, FDR_Indio_hombre, FDR_Indio_mujeres]

x = np.arange(8)
fig, ax = plt.subplots()
plt.bar(x, FDR_Total)
plt.xticks(x, ('FDR_Blancos_hombre', 'FDR_Blancos_mujeres', 'FDR_Negro_hombre', 'FDR_Negro_mujeres', 'FDR_Asiatico_hombre', 'FDR_Asiatico_mujeres', 'FDR_Indio_hombre', 'FDR_Indio_mujeres'))
plt.xticks(rotation=45)
plt.show()

# False Omission Rate

FOR_Blancos_hombre = matriz_Blanco[1][0]/(matriz_Blanco[1][0]+matriz_Blanco[1][1])
FOR_Blancos_mujeres = matriz_Blanco[0][1]/(matriz_Blanco[0][1]+matriz_Blanco[0][0])
FOR_Blancos = [FDR_Blancos_hombre, FDR_Blancos_mujeres]

FOR_Negro_hombre = matriz_Negro[1][0]/(matriz_Negro[1][0]+matriz_Negro[1][1])
FOR_Negro_mujeres = matriz_Negro[0][1]/(matriz_Negro[0][1]+matriz_Negro[0][0])
FOR_Negro = [FDR_Negro_hombre, FDR_Negro_mujeres]

FOR_Asiatico_hombre = matriz_Asiatico[1][0]/(matriz_Asiatico[1][0]+matriz_Asiatico[1][1])
FOR_Asiatico_mujeres = matriz_Asiatico[0][1]/(matriz_Asiatico[0][1]+matriz_Asiatico[0][0])
FOR_Asiatico = [FDR_Asiatico_hombre, FDR_Asiatico_mujeres]

FOR_Indio_hombre = matriz_Indio[1][0]/(matriz_Indio[1][0]+matriz_Indio[1][1])
FOR_Indio_mujeres = matriz_Indio[0][1]/(matriz_Indio[0][1]+matriz_Indio[0][0])
FOR_Indio = [FDR_Indio_hombre, FDR_Indio_mujeres]

FOR_Total = [FOR_Blancos_hombre, FOR_Blancos_mujeres, FOR_Negro_hombre, FOR_Negro_mujeres, FOR_Asiatico_hombre, FOR_Asiatico_mujeres, FOR_Indio_hombre, FOR_Indio_mujeres]

x = np.arange(8)
fig, ax = plt.subplots()
plt.bar(x, FOR_Total)
plt.xticks(x, ('FOR_Blancos_hombre', 'FOR_Blancos_mujeres', 'FOR_Negro_hombre', 'FOR_Negro_mujeres', 'FOR_Asiatico_hombre', 'FOR_Asiatico_mujeres', 'FOR_Indio_hombre', 'FOR_Indio_mujeres'))
plt.xticks(rotation=45)
plt.show()

# False Negative Rate

FNR_Blancos_hombre = matriz_Blanco[1][0]/(matriz_Blanco[1][0]+matriz_Blanco[0][0])
FNR_Blancos_mujeres = matriz_Blanco[1][0]/(matriz_Blanco[1][0]+matriz_Blanco[1][1])
FNR_Blancos = [FNR_Blancos_hombre, FNR_Blancos_mujeres]

FNR_Negro_hombre = matriz_Negro[1][0]/(matriz_Negro[1][0]+matriz_Negro[0][0])
FNR_Negro_mujeres = matriz_Negro[1][0]/(matriz_Negro[1][0]+matriz_Negro[1][1])
FNR_Negro = [FNR_Negro_hombre, FNR_Negro_mujeres]

FNR_Asiatico_hombre = matriz_Asiatico[1][0]/(matriz_Asiatico[1][0]+matriz_Asiatico[0][0])
FNR_Asiatico_mujeres = matriz_Asiatico[1][0]/(matriz_Asiatico[1][0]+matriz_Asiatico[1][1])
FNR_Asiatico = [FNR_Asiatico_hombre, FNR_Asiatico_mujeres]

FNR_Indio_hombre = matriz_Indio[1][0]/(matriz_Indio[1][0]+matriz_Indio[0][0])
FNR_Indio_mujeres = matriz_Indio[1][0]/(matriz_Indio[1][0]+matriz_Indio[1][1])
FNR_Indio = [FNR_Indio_hombre, FNR_Indio_mujeres]

FNR_Total = [FNR_Blancos_hombre, FNR_Blancos_mujeres, FNR_Negro_hombre, FNR_Negro_mujeres, FNR_Asiatico_hombre, FNR_Asiatico_mujeres, FNR_Indio_hombre, FNR_Indio_mujeres]

x = np.arange(8)
fig, ax = plt.subplots()
plt.bar(x, FNR_Total)
plt.xticks(x, ('FNR_Blancos_hombre', 'FNR_Blancos_mujeres', 'FNR_Negro_hombre', 'FNR_Negro_mujeres', 'FNR_Asiatico_hombre', 'FNR_Asiatico_mujeres', 'FNR_Indio_hombre', 'FNR_Indio_mujeres'))
plt.xticks(rotation=45)
plt.show()

# False Positive Rate

with sess.as_default():
    np.set_printoptions(threshold=np.nan)
    print('Vector Confusion: \n', vec)
    # print('Confusion Matrix de Blancos: \n', matsum0)
    # print('Confusion Matrix de Negros: \n', matsum1)
    # print('Confusion Matrix de Asiaticos: \n', matsum2)
    # print('Confusion Matrix de Indios: \n', matsum3)
    # print('Confusion Matrix Otros: \n', matsum4)
    print('Confusion Matrix: \n', mat)
    print('Confusion Matrix Normalizada: \n', mat_norm)
    print('MATRIZ NORMALIZADA Blancos: \n', matriz_Blanco_norm)
    print('MATRIZ NORMALIZADA Negros: \n', matriz_Negro_norm)
    print('MATRIZ NORMALIZADA Asiaticos: \n', matriz_Asiatico_norm)
    print('MATRIZ NORMALIZADA Indios: \n', matriz_Indio_norm)
    print('MATRIZ NORMALIZADA Otros: \n', matriz_Otro_norm)

