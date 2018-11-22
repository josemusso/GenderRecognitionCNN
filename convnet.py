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
cifar10 = CIFAR10(batch_size=batch_size, validation_proportion=0.1, test_proportion=0.1, augment_data=True)


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
n_conv_layers = 2

n_filters_convs = [32, 64, 128]

model_input = tf.placeholder(tf.float32, name='model_input',
                             shape=(batch_size, 64, 64, 3))

keep_prob = tf.placeholder(tf.float32, name='dropout_prob', shape=())

target = tf.placeholder(tf.float32, name='target', shape=(batch_size, 2))

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
    optimizer = tf.train.RMSPropOptimizer(0.0005)
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
    for batch in batches:
        data, genero, raza, edad = batch
        acc, mat = sess.run((accuracy, con_mat),
                            feed_dict={
                                model_input: data,
                                target: genero,
                                keep_prob: 1.0
                            })
        accs.append(acc)
        matsum = matsum + mat
    mean_acc = np.array(accs).mean()
    return mean_acc, matsum


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
    epochs = 4
else:
    epochs = 50

t_i = time.time()
n_batches = cifar10.n_batches
val_acc_vals = []
test_acc_vals = []
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
        test_accuracy, mat = test()
        test_acc_vals.append(test_accuracy)
        print("Time elapsed %.2f minutes" % ((time.time() - t_i) / 60.0))
train_writer.flush()
validation_writer.flush()

val_acc_vals = np.array(val_acc_vals)
test_acc_vals = np.array(test_acc_vals)
best_epoch = np.argmax(val_acc_vals)
test_acc_at_best = test_acc_vals[best_epoch]
print('*' * 30)
print("Testing set accuracy @ epoch %d (best validation acc): %.4f" % (best_epoch, test_acc_at_best))
print('*' * 30)

with sess.as_default():
    print('Confusion Matrix: \n', mat)

# test_confusion = sess.run(con_mat)
# print(test_confusion)

# print(tf.confusion_matrix(
#     target,
#     model_output,
#     num_classes=None,
#     dtype=tf.int32,
#     name=None,
#     weights=None
# ))

# def confusion_matrix(labels, predictions):
#     """Calcula la matriz de confusion.
#
#     Args:
#         labels: Array binario 1-D con las etiquetas reales.
#         predictions: Array binario 1-D con las predicciones.
#
#     Returns:
#         TP: Numero de verdaderos positivos.
#         FP: Numero de falsos positivos.
#         FN: Numero de falsos negativos.
#         TN: Numero de verdaderos negativos.
#     """
#     # Map labels and predictions to {0, 1, 2, 3}
#     encoded_data = 2 * labels + predictions
#     TN = np.sum(encoded_data == 0)  # True negatives
#     FP = np.sum(encoded_data == 1)  # False positives
#     FN = np.sum(encoded_data == 2)  # False negatives
#     TP = np.sum(encoded_data == 3)  # True positives
#     print('TP = ',TP, 'FP = ',FP, 'FN = ',FN, 'TN = ',TN)
#     return TP, FP, FN, TN
#
# confusion_matrix(target, tf.cast(correct_prediction, tf.float32))