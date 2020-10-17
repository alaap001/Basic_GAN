# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 13:01:12 2019

@author: wwech
"""

## image_resizer.py
# Importing required libraries
import os
import numpy as np
from PIL import Image

# Defining an image size and image channel
# We are going to resize all our images to 128X128 size and since our images are colored images
# We are setting our image channels to 3 (RGB)

IMAGE_SIZE = 128
IMAGE_CHANNELS = 3
IMAGE_DIR = 'dataset/'

# Defining image dir path. Change this if you have different directory
images_path = IMAGE_DIR 

training_data = []

# Iterating over the images inside the directory and resizing them using
# Pillow's resize method.
print('resizing...')

for filename in os.listdir(images_path):
    path = os.path.join(images_path, filename)
    image = Image.open(path).resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)

    training_data.append(np.asarray(image))

training_data = np.reshape(
    training_data, (-1, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNELS))
training_data = training_data / 127.5 - 1

print('saving file...')
np.save('cubism_data.npy', training_data)



# Keras libs
from keras.preprocessing.image import ImageDataGenerator  # to generate more training data by augmentation
from keras.layers import Dense, Dropout, Flatten, ZeroPadding2D, Conv2D, MaxPooling2D, Activation, \
                      BatchNormalization ,GlobalAveragePooling2D, concatenate, AveragePooling2D, Input,Reshape,UpSampling2D
                      
from keras.models import Sequential , Model ,load_model
from keras.optimizers import Adam, SGD #For Optimizing the Neural Network
from sklearn.metrics import confusion_matrix # confusion matrix to carry out error analysis
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, History # import callback functions for model
from keras.applications import DenseNet121
from keras.applications.resnet50 import ResNet50, preprocess_input
import numpy as np
from PIL import Image
import os

# Preview image Frame
PREVIEW_ROWS = 4
PREVIEW_COLS = 7
PREVIEW_MARGIN = 4
SAVE_FREQ = 100 # Size vector to generate images from
NOISE_SIZE = 100 # Configuration
EPOCHS = 10000 # number of iterations
BATCH_SIZE = 32
GENERATE_RES = 3
IMAGE_SIZE = 128 # rows/colsIMAGE_CHANNELS = 3

training_data = np.load("cubism_data.npy")


def build_discriminator(image_shape):    
    model = Sequential()    
    model.add(Conv2D(256, kernel_size=3, strides=2,input_shape=image_shape, padding="same", activation='relu'))
    model.add(Dropout(0.25))    
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same", activation='relu'))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.50))    
    model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"), activation='relu')
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.5))    
    model.add(Conv2D(512, kernel_size=3, strides=1, padding="same"), activation='relu')
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"), activation='relu')
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"), activation='relu')
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    input_image = Input(shape=image_shape)
    validity = model(input_image)
    return Model(input_image, validity)


def build_generator(noise_size, channels):
    model = Sequential()
    model.add(Dense(4 * 4 * 256, activation="relu", input_dim=noise_size))
    model.add(Reshape((4, 4, 256)))    
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))    
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))    
    for i in range(GENERATE_RES):
         model.add(UpSampling2D())
         model.add(Conv2D(256, kernel_size=3, padding=”same”))
         model.add(BatchNormalization(momentum=0.8))
         model.add(Activation("relu"))
         model.summary()
    model.add(Conv2D(channels, kernel_size=3, padding=”same”))
    model.add(Activation(“tanh”))    input = Input(shape=(noise_size,))
    generated_image = model(input)
    
    return Model(input, generated_image)

