
# coding: utf-8


import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,Dropout, Flatten, Dense
from keras.datasets import mnist
from keras.layers import Dense, Activation, BatchNormalization
from keras.constraints import min_max_norm
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.callbacks import LearningRateScheduler


from activations import binary_sigmoid as binary_sig
from activations import binary_tanh as binary_tanh_op
from activations import step_func as step
from binary_layers import BinaryDense, Clip

from keras.models import load_model

import matplotlib.pyplot as plt


class DropoutNoScale(Dropout):
    '''Keras Dropout does scale the input in training phase, which is undesirable here.
    '''
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed) * (1 - self.rate)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs


#Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# Binarizing the Inputs
for i in range(len(X_train)):
    for train_value in range(784):
        if(X_train[i][train_value]) > 0.0:
            X_train[i][train_value] = 1.0
        else:
            X_train[i][train_value] = 0.0
for j in range(len(X_test)):
    for test_value in range(784):
        if(X_test[j][test_value]) > 0.0:
            X_test[j][test_value] = 1.0
        else:
            X_test[j][test_value] = 0.0

print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)* 2 - 1
y_test = keras.utils.to_categorical(y_test, 10)* 2 - 1

#import to obtain _hard_tanh activation
def binary_tanh(x):
    return binary_tanh_op(x)
#import to obtain binary_sigmoid activation
def binary_sigmoid(x):
    return binary_sig(x)
#import to obtain step function
def step_func(x):
    return step(x)

batch_size = 100
epochs = 400  
nb_classes = 10

H = 'Glorot'
kernel_lr_multiplier = 'Glorot'

# network
num_unit = 512
num_hidden = 3
use_bias = False

# learning rate schedule
lr_start = 1e-3
lr_end = 1e-4
lr_decay = (lr_end / lr_start)**(1. / epochs)

# BN
epsilon = 1e-6
momentum = 0.9

# dropout
drop_in = 0.2
drop_hidden = 0.5

# Step 2: Build the Model

model = Sequential()
model.add(BinaryDense(512, input_shape=(784,), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
model.add(Activation(binary_tanh))
model.add(BinaryDense(512, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
model.add(Activation(binary_tanh))
model.add(BinaryDense(10, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))

# Printing the model
print(model.summary())
for layer in model.layers:
    h = layer.get_weights()
print(h)

opt = Adam(lr=lr_start)


# Step 3: Compile the Model
model.compile(loss='squared_hinge',optimizer=opt,metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)


# Step 4: Train the Model

history = model.fit(X_train,y_train,epochs=epochs,batch_size=100,validation_data=(X_test,y_test),verbose=1,callbacks=[lr_scheduler])


import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.subplot(2, 1, 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Accuracy History')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training','validation'])

#plt.figure()

plt.subplot(2, 1, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Loss History')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training','validation'])

plt.show()



# Step 5: Evaluate the Model
loss,accuracy = model.evaluate(X_test,y_test)
print("Loss = ",loss)
print("Accuracy = ",accuracy)


datafile = './models/mnist_nn_quantized_zeroone_FC.h5' # Change this if you want to rename your weights file
#Step 6: Save the Model and weights

#model.save('./models/mnist_nn_quantized_zeroone_FC.h5')
model.save_weights(datafile) # Edit this to save your own weights file




import h5py
import numpy as np
import pandas as pd
 

file=h5py.File(datafile,'r+')

print("Done")
################### Getting base directory ###################
base_items = list(file.items())
print("Items in base dir ", base_items)
print("\n")
dense_get_1 = file.get('binary_dense_2')
dense_items_1 = list(dense_get_1.items())
print("Items in first group ", dense_items_1)
print("\n")
################### First Layer #############################
dense_get_11 = file.get('/binary_dense_2/binary_dense_2')
dense_items_11 = list(dense_get_11.items())
print("Items in first group ", dense_items_11)
dataset1 = np.array(dense_get_11.get('kernel:0'))

w, h = 512, 784
Matrix1 = [[0 for x in range(w)] for y in range(h)]
print(np.shape(Matrix1))
#print(Matrix1)

def binarize():

    for i in range(784):
        for j in range(512):
            x = dataset1[i][j]
            if(x>0):
                f = 1
            else:
                f = -1
            Matrix1[i][j] = f
    
    return Matrix1

a = binarize()
print(a)

###################### Second Layer ###########################
dense_get_21 = file.get('/binary_dense_4/binary_dense_4')
dense_items_21 = list(dense_get_21.items())
print("Items in first group ", dense_items_21)
dataset2 = np.array(dense_get_21.get('kernel:0'))

h1 = 512
Matrix2 = [[0 for x in range(w)] for y in range(h1)]
print(np.shape(Matrix2))


def binarize2():

    for i in range(512):
        for j in range(512):
            x = dataset2[i][j]
            if(x>0):
                f1 = 1
            else:
                f1 = -1
            Matrix2[i][j] = f1
    
    return Matrix2

b = binarize2()
print(b)
############################# Final Layer ##################################################
dense_get_41 = file.get('/binary_dense_6/binary_dense_6')
dense_items_41 = list(dense_get_41.items())
print("Items in first group ", dense_items_41)
dataset4 = np.array(dense_get_41.get('kernel:0'))

h2 = 10
Matrix4 = [[0 for x in range(h2)] for y in range(w)]
print(np.shape(Matrix4))


def binarize4():

    for i in range(512):
        for j in range(10):
            x = dataset4[i][j]
            if(x>0):
                f3 = 1
            else:
                f3 = -1
            Matrix4[i][j] = f3
    
    return Matrix4

d = binarize4()
print(d)

########### Re-writing of h5 weights file with the -1, +1 matrix obtained from above ###########
with h5py.File(datafile, 'r+') as hdf:

    #Layer1
    dense_get_11 = hdf.get('/binary_dense_2/binary_dense_2')
    dataset1 = np.array(dense_get_11.get('kernel:0'))
    #print(dataset1)
    data = hdf.get('/binary_dense_2/binary_dense_2/kernel:0')
    data[...]=Matrix1
    dataset1 = np.array(dense_get_11.get('kernel:0'))
    print(dataset1)
    #Layer2
    data2 = hdf.get('/binary_dense_4/binary_dense_4/kernel:0')
    data2[...]=Matrix2
    dataset2 = np.array(hdf.get('/binary_dense_4/binary_dense_4/kernel:0'))
    print(dataset2)
    #Final Layer
    data4 = hdf.get('/binary_dense_6/binary_dense_6/kernel:0')
    data4[...]=Matrix4
    dataset4 = np.array(hdf.get('/binary_dense_6/binary_dense_6/kernel:0'))
    print(dataset4)

