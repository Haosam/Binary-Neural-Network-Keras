
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


from activations import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, Clip

from keras.models import load_model


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
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)* 2 - 1
y_test = keras.utils.to_categorical(y_test, 10)* 2 - 1

#import to obtain _hard_tanh activation
def binary_tanh(x):
    return binary_tanh_op(x)

batch_size = 100
epochs = 20  
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
#model.add(DropoutNoScale(drop_in, input_shape=(784,)))
#model.add(BinaryDense(128, activation=binary_tanh, kernel_initializer=keras.initializers.RandomUniform(minval=-1., maxval=1., seed=None), bias_initializer='zeros'))
#model.add(DropoutNoscale(0.5))
#BatchNormalization(momentum=0.9,epsilon=0.000001)
#model.add(BinaryDense(64,activation=binary_tanh))
#model.add(DropoutNoscale(0.5))
#BatchNormalization(momentum=0.9,epsilon=0.000001)
model.add(BinaryDense(512, input_shape=(784,), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
model.add(Activation(binary_tanh))
#model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
#model.add(DropoutNoScale(drop_hidden))
model.add(BinaryDense(512, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
model.add(Activation(binary_tanh))
#model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
#model.add(DropoutNoScale(drop_hidden))
model.add(BinaryDense(512, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
model.add(Activation(binary_tanh))
#model.add(BatchNormalization(epsilon=epsilon, momentum=momentum))
#model.add(DropoutNoScale(drop_hidden))
model.add(BinaryDense(10, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))


print(model.summary())
for layer in model.layers:
    h = layer.get_weights()
print(h)

opt = Adam(lr=lr_start)


# Step 3: Compile the Model
model.compile(loss='squared_hinge',optimizer=opt,metrics=['accuracy'])

lr_scheduler = LearningRateScheduler(lambda e: lr_start * lr_decay ** e)


# Step 4: Train the Model

history = model.fit(X_train,y_train,epochs=200,batch_size=100,validation_data=(X_test,y_test),verbose=1,callbacks=[lr_scheduler])


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



#Step 6: Save the Model and weights
model.save('./models/mnist_nn.h5')
model.save_weights('./models/mnist_nn_weights.h5') # Edit this to save your own weights file

