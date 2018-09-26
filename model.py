import csv
import numpy as np
import cv2
import os
import sklearn

samples = []
with open('data2r/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader: 
		samples.append(line)

samples = samples[1:]
#images = []
#measurements = []

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size = 0.2)

from sklearn.utils import shuffle

def generator(samples, batch_size = 128):
	a = 0
	num_samples = len(samples)
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]

			images = []
			angles = []
			 


			for batch_sample in batch_samples:
				steering_center = float(batch_sample[3])

				correction = 0.18
				steering_left = steering_center + correction
				steering_right = steering_center - correction
	
				for i in range(3):
					source_path = batch_sample[i]
					filename = source_path.split('/')[-1]
					#filename = source_path
					current_path = 'data2r/IMG/' + filename
					image = cv2.imread(current_path)
					#angle = float(batch_sample[3])
					images.append(image)
				angles.append(steering_center)
				angles.append(steering_left)
				angles.append(steering_right)
				aug_images, aug_angles = [], [] 
				for r,t in zip(images,angles):
					aug_images.append(r)
					aug_angles.append(t)
					aug_images.append(cv2.flip(r,1))
					aug_angles.append(t * -1.0)
				#a += len(aug_angles)			
			X_train = np.array(images)
			y_train = np.array(angles)
			#print(a)
			yield sklearn.utils.shuffle(X_train, y_train)
	#source_path = line[1]
	#filename = source_path.split('/')[-1]
	#current = 'data2/IMG/'+ filename
	#image = cv2.imread(current)
	
	#if i == 0:
		#measurement = float(line[3])
	#else:
		#measurement = float(line[3]) + 0.18
				#measurements.append(steering_center)
				#measurements.append(steering_left)
				#measurements.append(steering_right)

	#augmented_images, augmented_measurements = [], [] 
	#for image, measurement in zip(images, measurements): 
	#	augmented_images.append(image)
	#	augmented_measurements.append(measurement)
	#	augmented_images.append(cv2.flip(image,1))
	#	augmented_measurements.append(measurement * -1.0)


#print(len(images))	
#print(len(measurements))
	#X_train = np.array(augmented_images)
	#y_train = np.array(augmented_measurements)
#print(X_train[1])
	#yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size = 128)
validation_generator = generator(validation_samples, batch_size = 128)

from keras.models import Sequential, Model
import matplotlib.pyplot as plt 
from keras.layers import Flatten,Dense,Lambda, Cropping2D
from keras.layers import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3))) 
model.add(Cropping2D(cropping = ((70,25), (0,0))))
model.add(Convolution2D(24,5,5,activation = 'relu', subsample = (2,2)))
#model.add(MaxPooling2D())
model.add(Convolution2D(36,5,5,activation = 'relu', subsample = (2,2)))
#model.add(MaxPooling2D())
model.add(Convolution2D(48,5,5, activation = 'relu', subsample = (2,2)))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation = 'relu'))
#model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3,activation = 'relu'))
model.add(Flatten())
#model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
#model.add(Flatten())
model.add(Dense(1))

model.compile(loss = 'mse', optimizer = 'adam')
#model.fit(X_train, y_train, shuffle = True, validation_split = 0.2, nb_epoch = 5, batch_size = 128)
#num_val = int(0.2*len(X_train))
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), validation_data = validation_generator, nb_val_samples = len(validation_samples), nb_epoch = 5)
#print(history_object.history.keys())
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val loss'])
#plt.show()
model.save('model2r.h5')
