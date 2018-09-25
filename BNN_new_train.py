import keras
from keras.models import load_model
import keras.backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD, Adam, RMSprop
from keras.callbacks import LearningRateScheduler
from keras.utils import np_utils

from activations import binary_tanh as binary_tanh_op
from binary_layers import BinaryDense, Clip

from keras.models import load_model


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

######### Not really needed as we are doing fully-connected in this case
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

def build_model():
	model = Sequential()
	model.add(BinaryDense(512, input_shape=(784,), H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
	model.add(Activation(binary_tanh))
	model.add(BinaryDense(512, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
	model.add(Activation(binary_tanh))
	model.add(BinaryDense(10, H=H, kernel_lr_multiplier=kernel_lr_multiplier, use_bias=use_bias))
	opt = Adam(lr=lr_start)
	model.compile(loss='squared_hinge',optimizer=opt,metrics=['accuracy'])
	return model


model1 = build_model()
model1.load_weights('./models/mnist_nn_binaryweights2.h5') # Please put your weights file into this area

loss,accuracy = model1.evaluate(X_test,y_test)
print("Loss = ",loss*100,"%")
print("Accuracy = ",accuracy*100,"%")

