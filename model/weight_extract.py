import h5py
import numpy as np
import pandas as pd

file=h5py.File('./mnist_nn_weights.h5','r+')

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

########################### Third Layer ##############################################
dense_get_31 = file.get('/binary_dense_6/binary_dense_6')
dense_items_31 = list(dense_get_31.items())
print("Items in first group ", dense_items_31)
dataset3 = np.array(dense_get_31.get('kernel:0'))


Matrix3 = [[0 for x in range(w)] for y in range(h1)]
print(np.shape(Matrix3))


def binarize3():

	for i in range(512):
		for j in range(512):
			x = dataset3[i][j]
			if(x>0):
				f2 = 1
			else:
				f2 = -1
			Matrix3[i][j] = f2
	
	return Matrix3

c = binarize3()
print(c)

############################# Final Layer ##################################################
dense_get_41 = file.get('/binary_dense_8/binary_dense_8')
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

########### Re-writing of h5 weights file with the -1, +1 matrix obtained from above ###########################
########### Please edit your weight file location accordingly ##################################################
########### This weight extract literally re-writes the weights inside the original quantized weights file######
########### Please make a copy of your weights file if you want the original trained weights ###################
########### Please edit your layers accordingly, by removing or adding the layers ##############################
#(Original file was written as a 4 fully connected layer. New code is using a 3 fully connected layer, please edit)#
with h5py.File('./mnist_nn_binaryweights2.h5', 'r+') as hdf:
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
	#Layer3
	data3 = hdf.get('/binary_dense_6/binary_dense_6/kernel:0')
	data3[...]=Matrix3
	dataset3 = np.array(hdf.get('/binary_dense_6/binary_dense_6/kernel:0'))
	print(dataset3)
	#Layer4
	data4 = hdf.get('/binary_dense_8/binary_dense_8/kernel:0')
	data4[...]=Matrix4
	dataset4 = np.array(hdf.get('/binary_dense_8/binary_dense_8/kernel:0'))
	print(dataset4)
