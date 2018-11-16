import h5py
import numpy as np
import pandas as pd


file_path = './model_mnist_cnn_binary.h5'
file=h5py.File(file_path,'r+')

print("Done")
################### Getting base directory ###################
base_items = list(file.items())
print("Items in base dir ", base_items)
print("\n")
dense_get_1 = file.get('conv1')
dense_items_1 = list(dense_get_1.items())
print("Items in first group ", dense_items_1)
print("\n")
################### First Layer #############################
dense_get_11 = file.get('/conv1/conv1')
dense_items_11 = list(dense_get_11.items())
print("Items in first group ", dense_items_11)
dataset1 = np.array(dense_get_11.get('kernel:0'))

w, h, d = 3, 3, 128
Matrix1 = [[[[0 for a in range(d)] for b in range(1)] for x in range(w)] for y in range(h)]
print(np.shape(Matrix1))
#print(Matrix1)

def binarize():

	for i in range(3):
		for j in range(3):
			for k in range(128):
				x = dataset1[i][j][0][k]
				if(x>0):
					f = 1
				else:
					f = -1
				Matrix1[i][j][0][k] = f
	
	return Matrix1

a = binarize()
#print(a)

###################### Second Layer ###########################
dense_get_21 = file.get('/conv2/conv2')
dense_items_21 = list(dense_get_21.items())
print("Items in first group ", dense_items_21)
dataset2 = np.array(dense_get_21.get('kernel:0'))

h1 = 3
Matrix2 = [[[[0 for a in range(d)] for b in range(128)] for x in range(w)] for y in range(h)]
print(np.shape(Matrix2))


def binarize2():

	for i in range(3):
		for j in range(3):
			for l in range(128):
				for k in range(128):
					x = dataset2[i][j][l][k]
					if(x>0):
						f = 1
					else:
						f = -1
					Matrix2[i][j][l][k] = f
	
	return Matrix2

b = binarize2()
print(b)

###################### Second Layer ###########################
dense_get_31 = file.get('/conv3/conv3')
dense_items_31 = list(dense_get_31.items())
print("Items in first group ", dense_items_31)
dataset3 = np.array(dense_get_31.get('kernel:0'))

h2 = 3
Matrix3 = [[[[0 for a in range(256)] for b in range(128)] for x in range(w)] for y in range(h)]
print(np.shape(Matrix3))


def binarize3():

	for i in range(3):
		for j in range(3):
			for l in range(128):
				for k in range(256):
					x = dataset3[i][j][l][k]
					if(x>0):
						f = 1
					else:
						f = -1
					Matrix3[i][j][l][k] = f
	
	return Matrix3

c = binarize3()
print(c)

###################### Second Layer ###########################
dense_get_41 = file.get('/conv4/conv4')
dense_items_41 = list(dense_get_41.items())
print("Items in first group ", dense_items_41)
dataset4 = np.array(dense_get_41.get('kernel:0'))

h3 = 3
Matrix4 = [[[[0 for a in range(256)] for b in range(256)] for x in range(w)] for y in range(h)]
print(np.shape(Matrix4))


def binarize4():

	for i in range(3):
		for j in range(3):
			for l in range(256):
				for k in range(256):
					x = dataset4[i][j][l][k]
					if(x>0):
						f = 1
					else:
						f = -1
					Matrix4[i][j][l][k] = f
	
	return Matrix4

d = binarize4()
print(d)
########################### Third Layer ##############################################
dense_get_51 = file.get('/dense5/dense5')
dense_items_51 = list(dense_get_51.items())
print("Items in first group ", dense_items_51)
dataset5 = np.array(dense_get_51.get('kernel:0'))

width,height = 1024, 12544
Matrix5 = [[0 for x in range(width)] for y in range(height)]
print(np.shape(Matrix5))


def binarize5():

	for i in range(12544):
		for j in range(1024):
			x = dataset5[i][j]
			if(x>0):
				f4 = 1
			else:
				f4 = -1
			Matrix5[i][j] = f4
	
	return Matrix5

e = binarize5()
#print(e)

############################# Final Layer ##################################################
dense_get_61 = file.get('/dense6/dense6')
dense_items_61 = list(dense_get_61.items())
print("Items in first group ", dense_items_61)
dataset6 = np.array(dense_get_61.get('kernel:0'))

width1,height1 = 10, 1024
Matrix6 = [[0 for x in range(width1)] for y in range(height1)]
print(np.shape(Matrix6))


def binarize6():

	for i in range(1024):
		for j in range(10):
			x = dataset6[i][j]
			if(x>0):
				f5 = 1
			else:
				f5 = -1
			Matrix6[i][j] = f5
	
	return Matrix6

f = binarize6()
#print(d)

########### Re-writing of h5 weights file with the -1, +1 matrix obtained from above ###########
with h5py.File(file_path, 'r+') as hdf:
	"""
	a1 = hdf.create_group('activation_1')
	a2 = hdf.create_group('activation_2')
	a3 = hdf.create_group('activation_3')

	b2 = hdf.create_group('binary_dense_2/binary_dense_2')
	b2.create_dataset('kernel:0', data=Matrix1)
	b4 = hdf.create_group('binary_dense_4/binary_dense_4')
	b4.create_dataset('kernel:0', data=Matrix2)
	b6 = hdf.create_group('binary_dense_6/binary_dense_6')
	b6.create_dataset('kernel:0', data=Matrix3)
	b8 = hdf.create_group('binary_dense_8/binary_dense_8')
	b8.create_dataset('kernel:0', data=Matrix4)
	"""
	#Layer1
	dense_get_11 = hdf.get('/conv1/conv1')
	dataset1 = np.array(dense_get_11.get('kernel:0'))
	#print(dataset1)
	data = hdf.get('/conv1/conv1/kernel:0')
	data[...]=Matrix1
	dataset1 = np.array(dense_get_11.get('kernel:0'))
	print(dataset1)
	#Layer2
	data2 = hdf.get('/conv2/conv2/kernel:0')
	data2[...]=Matrix2
	dataset2 = np.array(hdf.get('/conv2/conv2/kernel:0'))
	print(dataset2)
	#Layer3
	data3 = hdf.get('/conv3/conv3/kernel:0')
	data3[...]=Matrix3
	dataset3 = np.array(hdf.get('/conv3/conv3/kernel:0'))
	print(dataset3)
	#Layer4
	data4 = hdf.get('/conv4/conv4/kernel:0')
	data4[...]=Matrix4
	dataset4 = np.array(hdf.get('/conv4/conv4/kernel:0'))
	print(dataset4)
	#Layer4
	data5 = hdf.get('/dense5/dense5/kernel:0')
	data5[...]=Matrix5
	dataset5 = np.array(hdf.get('/dense5/dense5//kernel:0'))
	print(dataset5)
	#Layer4
	data6 = hdf.get('/dense6/dense6/kernel:0')
	data6[...]=Matrix6
	dataset6 = np.array(hdf.get('/dense6/dense6/kernel:0'))
	print(dataset4)
