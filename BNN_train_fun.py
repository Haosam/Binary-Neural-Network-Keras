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




# MNIST Live Demo

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
os.environ['TF_ENABLE_WINOGRAD_NONE_USED']='1'

from keras.models import load_model
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFilter

# Model Path
path = "./models/mnist_nn_fun.h5"

class ImageGenerator:

    def __init__(self, parent, *kwargs):
        self.parent = parent
        self.posx = 10
        self.posy = 10
        self.sizex = 200
        self.sizey = 200
        self.b1 = "up"
        self.xold = None
        self.yold = None
        self.drawing_area = tk.Canvas(self.parent, width=self.sizex, height=self.sizey)
        self.drawing_area.place(x=self.posx, y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button = tk.Button(self.parent, text="Done!", width=10, bg='white', command=self.minst_nn_pred)
        self.button.place(x=self.sizex / 7, y=self.sizey + 20)
        self.button1 = tk.Button(self.parent, text="Clear!", width=10, bg='white', command=self.clear)
        self.button1.place(x=(self.sizex / 7) + 80, y=self.sizey + 20)

        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def save(self):
        img = self.image.resize((28,28)).convert('L')
        self.minst_nn_pred(img)

    def clear(self):
        self.drawing_area.delete("all")
        self.image = Image.new("RGB", (200, 200), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image)

    def b1down(self, event):
        self.b1 = "down"

    def b1up(self, event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self, event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold, self.yold, event.x, event.y, smooth='true', width=20, fill='black')
                self.draw.line(((self.xold, self.yold), (event.x, event.y)), (0, 128, 0), width=20)

        self.xold = event.x
        self.yold = event.y


    def minst_nn_pred(self):

        model1 = build_model()
        model1.load_weights('./models/mnist_nn_binaryweights2.h5') # Load your weights file into this area

        img = self.image.convert('L').resize((28, 28))
        X_test = np.asarray(img)
        X_test = X_test.astype('float32')
        X_test = (255.0 - X_test) / 255.0

        X_test = X_test.reshape(1,784)

        prediction = model1.predict(X_test)
        print("Predicted digit is : ", prediction.argmax(axis=1))


if __name__ == "__main__":
    root = tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (250, 250, 10, 10))
    root.config(bg='white')
    ImageGenerator(root)
    root.mainloop()

