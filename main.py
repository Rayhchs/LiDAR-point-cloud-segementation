import numpy as np
import tifffile as tiff
from numpy import genfromtxt
import keras
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import EarlyStopping

def Conv3(length, width, height, n_classes):

	inputs = Input((length, width, height, 1))
	conv = Conv3D(12, (3,3,3), padding = 'same', activation = 'relu')(inputs)
	conv = Conv3D(12, (3,3,3), padding = 'same', activation = 'relu')(conv)
	pool = MaxPooling3D(pool_size=(1, 1, 4))(conv)
	
	conv2 = Conv3D(12, (3,3,3), padding = 'same', activation = 'relu')(pool)
	conv2 = Conv3D(12, (3,3,3), padding = 'same', activation = 'relu')(conv2)
	pool2 = MaxPooling3D(pool_size=(1, 1, 4))(conv2)
	
	conv3 = Conv3D(12, (3,3,3), padding = 'same', activation = 'relu')(pool2)
	conv3 = Conv3D(12, (3,3,3), padding = 'same', activation = 'relu')(conv3)
	pool3 = MaxPooling3D(pool_size=(1, 1, 4))(conv3)
	
	conv4 = Conv3D(12, (3,3,3), padding = 'same', activation = 'relu')(pool3)
	conv4 = Conv3D(12, (3,3,3), padding = 'same', activation = 'relu')(conv4)
	pool4 = MaxPooling3D(pool_size=(1, 1, conv4.shape[3]))(conv4)
	res = Reshape((length,width,12))(pool4)
	
	conv5 = Conv2D(12,(3,3),padding = 'same', activation = 'relu')(res)
	conv5 = Conv2D(12,(3,3),padding = 'same', activation = 'relu')(conv5)

	output = Conv2D(n_classes,1, activation = 'softmax')(conv5)
	model = Model(inputs = inputs, outputs = output)
	model.compile(loss='categorical_crossentropy', optimizer = Adam(lr=0.001) ,metrics=['accuracy'])
	model.summary()
	
	return model

def clip_data(p_train, label, dataset, img_sz, zmax):
	p_train = p_train
	label = label
	count = -1
	train_all = []
	label_all = []
	hx = np.max(p_train[:,0])
	lx = np.min(p_train[:,0])
	hy = np.max(p_train[:,1])
	ly = np.min(p_train[:,1])
	p_train[:,0] = p_train[:,0] - lx
	p_train[:,1] = p_train[:,1] - ly
	
	while True:
		count += 1
		for j in range(int((hy-ly)/img_sz)):
			x_train = []
			label_y = []
			for i in range(p_train.shape[0]):
				if p_train[i,0] > + img_sz*count and p_train[i,0] <= img_sz+img_sz*count:
					if p_train[i,1] > img_sz+img_sz*j:
						pass
					elif p_train[i,1] > 0+img_sz*j and p_train[i,1] <= img_sz+img_sz*j:
						x = p_train[i,:]
						y = label[i,:]
						x_train.append(x)
						label_y.append(y)
				else:
					pass
			if len(x_train) < img_sz:
				pass
			elif len(x_train) > img_sz:
				label_all.append(label_y)
				train_all.append(x_train)
		if count == int((hx-lx)/img_sz):
			break
		
	train_all = np.array(train_all)
	label_all = np.array(label_all)
	space = np.zeros(shape=(train_all.shape[0],img_sz,img_sz,zmax,1),dtype=np.float32)
	y_label = np.zeros(shape=(train_all.shape[0],img_sz,img_sz,1),dtype=np.float32)
	
	for i in range(space.shape[0]):
		a = np.array(train_all[i])
		b = np.array(label_all[i])
		a[:,0] = a[:,0] % img_sz
		a[:,1] = a[:,1] % img_sz
		for j in range(a.shape[0]):
			space[i,int(a[j,0]),int(a[j,1]),int(a[j,2]),:] = a[j,2]
			y_label[i,int(a[j,0]),int(a[j,1])] = b[j,0]

	if dataset == "Vaihegen":
		mask_gt = np.zeros(shape=(y_label.shape[0],img_sz,img_sz,5))
		for i in range(y_label.shape[0]):
			for j in range(img_sz):
				for k in range(img_sz):
					if y_label[i,j,k,0] == 0:
						mask_gt[i,j,k,0] = 1
					elif y_label[i,j,k,0] == 1:
						mask_gt[i,j,k,1] = 1
					elif y_label[i,j,k,0] == 2:
						mask_gt[i,j,k,2] = 1
					elif y_label[i,j,k,0] == 3:
						mask_gt[i,j,k,3] = 1
					elif y_label[i,j,k,0] == 4:
						mask_gt[i,j,k,4] = 1
					else:
						print('trouble')
						break
	elif dataset == "Tornoto":
		from keras.utils.np_utils import to_categorical
		mask_gt = to_categorical(y_label[:,:],3)

	return space, mask_gt

if __name__ == '__main__':
	
	length = 50
	width = 50
	height = 400
	train_x = []
	train_y = []

	print("Please input which dataset you want? (Tornoto or Vaihegen)")
	dataset = input()

	if dataset == "Tornoto":
		n_classes = 3
		b1 = genfromtxt('Tornoto/train/sb1.csv', delimiter=',')
		b2 = genfromtxt('Tornoto/train/sb2.csv', delimiter=',')
		nb1 = genfromtxt('Tornoto/train/snb1.csv', delimiter=',')
		nb2 = genfromtxt('Tornoto/train/snb2.csv', delimiter=',')
		b1 = np.delete(b1, 0, axis=0)
		b2 = np.delete(b2, 0, axis=0)
		nb1 = np.delete(nb1, 0, axis=0)
		nb2 = np.delete(nb2, 0, axis=0)
		b = np.vstack((b1, b2))
		nb = np.vstack((nb1, nb2))
		train = np.vstack((b, nb))
		y1 = np.ones(shape=(b.shape[0], 1), dtype=np.float32)
		y2 = np.zeros(shape=(nb.shape[0], 1), dtype=np.float32)
		label = np.vstack((y1, y2))
		del b, b1, b2, nb, nb1, nb2, y1, y2
		train_x, train_y = clip_data(train, label, dataset, length, height)

	elif dataset == "Vaihegen": #Run Vaihegen dataset
		#Prepare for point cloud
		n_classes = 5
		for i in range(1,9):
			data = genfromtxt('Vaihegen/train/{}.csv'.format(i), delimiter=',')
			train = np.zeros(shape=(data.shape[0], 3), dtype=np.float32)
			label = np.zeros(shape=(data.shape[0], 1), dtype=np.float32)
			train[:, :3] = data[:, :3]
			label[:, 0] = data[:, 3]
			train, label = clip_data(train, label, dataset, length, height)
			train_x.append(train)
			train_y.append(label)
		train_x = np.concatenate(train_x)
		train_x = np.stack(train_x, axis=0)
		train_y = np.concatenate(train_y)
		train_y = np.stack(train_y, axis=0)

		for i in range(train_x.shape[0]):
			trainx = np.stack(train_x[i], axis=0)

	else:
		print("Please type Tornoto or Vaihegen")

	#Setting hyperparameter
	BATCH_SIZE = 1
	epoch = 10000

	#Training
	model = Conv3(length,width,height,n_classes)
	print('Start Train Net!')
	early_stopping = EarlyStopping(monitor='loss', min_delta=0.01, patience=100, mode='min', baseline=5, verbose=1)
	run_model = model.fit(train_x, train_y, batch_size=BATCH_SIZE, 
		shuffle=True, epochs=epoch, callbacks=[early_stopping])
	if dataset == "Tornoto":
		model.save_weights('Tornoto/model')
	elif dataset == "Vaihegen":
		model.save_weights('Vaihegen/model')



