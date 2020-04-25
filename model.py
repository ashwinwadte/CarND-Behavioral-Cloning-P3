import csv
import cv2
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras import backend as K


lines=[]

with open('data/driving_log.csv', 'r') as f:
    reader = csv.reader(f)
    for line in reader:
        lines.append(line)
        
images = []
measurements = []

for line in lines:
        # read center, left, and right images of camera
        center_image_path = line[0]
        left_image_path = line[1]
        right_image_path = line[2]
        
        # Read steering measurement
        measurement_center = float(line[3])
        measurement_left = measurement_center + 0.2
        measurement_right = measurement_center - 0.2
        
        # Store paths
        filename_center = center_image_path.split('/')[-1]
        filename_left = left_image_path.split('/')[-1]
        filename_right = right_image_path.split('/')[-1]
        path_center = 'data/IMG/' + filename_center 
        path_left = 'data/IMG/' + filename_left 
        path_right = 'data/IMG/' + filename_right 
        
        # Append images and measurements
        img_center = cv2.imread(path_center)
        img_left = cv2.imread(path_left)
        img_right = cv2.imread(path_right)
        images.append(img_center)
        images.append(img_left)
        images.append(img_right)
        
        measurements.append(measurement_center)
        measurements.append(measurement_left)
        measurements.append(measurement_right)
        
        
X_train = np.array(images)         # training images
y_train = np.array(measurements)   # labels

model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,3)))
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=36, kernel_size=(5, 5),strides=(2, 2), activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3) ,activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(filters=64, kernel_size=(3, 3),activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(units=100, activation='relu'))
model.add(Dense(units=50, activation='relu'))
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=1))
print(model.summary())

model.compile(loss='mse',optimizer='adam')
model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True)
model.save('model.h5')