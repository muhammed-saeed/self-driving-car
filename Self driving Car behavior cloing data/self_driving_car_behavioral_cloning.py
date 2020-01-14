# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 16:58:13 2019

@author: Muhammed Yahia
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Convolution2D, MaxPooling2D, Dropout, Flatten, Dense
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import ntpath
import random

datadir = 'track'
columns = ['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed']
data = pd.read_csv('file:///C:/Users/Muhammed Yahia/Desktop/Self driving Car behavior cloing data/track/driving_log.csv', names = columns)
pd.set_option('display.max_colwidth', -1)
data.head()

#preprocessing the datapath file name
def path_leaf(path):
  head, tail = ntpath.split(path)
  return tail
data['center'] = data['center'].apply(path_leaf)
data['left'] = data['left'].apply(path_leaf)
data['right'] = data['right'].apply(path_leaf)
data.head()

#bins and hist both used for the following
#showing our data again now show the data in the correct formate
#and show which steering 

#for every set of images (center, left, right) we are going to draw a histogram
#and see what steering angle is the most common through out the trainnig
#this allows us to determine the errors in the training datasets and provide us with a way to solve their problems


num_bins = 25
samples_per_bin = 200
hist, bins = np.histogram(data['steering'], num_bins)
#note bins go from -1 to 1 since all angles aer between -1 and -1

center = (bins[:-1]+ bins[1:]) * 0.5
#start adding from the value at index one to the very end

plt.bar(center, hist, width=0.05)
# the center of each histogram, the value of each historgram and hte width of each bar
# the vertical represent the frequancy of each angle through the training process

plt.ylabel('Frequancy of angle_through_training_process')
plt.xlabel('the value of the angle rotated')
plt.title('Histogram of angles and its frequancy during training process')
#note after the first process we have detected that angle zero has encountered far more than the other angles
#so we make threshold value of 200 and reject all the values above this 200
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))
# the above line is responsible of rejecting the dataset



#if we want to train our cnn based on this data the model could be biased to driving forward 
#

############through this course we are going to do many preprocessing techniques to ensure that our car can drive premary




###########EEEEEEEEEEEEEEEEEEEE

##################################################################

#balancing the data

#we must flatten our data distribution 
#do many preprocessing techniques to ensure the car will be able to drive good on the path
#remove_list=  []
#samples we want to remove in the model




#now we want balance the data

print('total data:', len(data))
remove_list = []
for j in range(num_bins):
        #the goal is to iterate through all the steering data belongs to each bin

  list_ = []
      #create a list and make the list an empty list

  for i in range(len(data['steering'])):
    if data['steering'][i] >= bins[j] and data['steering'][i] <= bins[j+1]:
      list_.append(i)
  list_ = shuffle(list_)
  list_ = list_[samples_per_bin:]
  remove_list.extend(list_)
 
print('removed:', len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print('remaining:', len(data))

hist, _ = np.histogram(data['steering'], (num_bins))
plt.bar(center, hist, width=0.05)
plt.plot((np.min(data['steering']), np.max(data['steering'])), (samples_per_bin, samples_per_bin))

########################################################################

#train and validation data split

#prepare the traing data
#return only the center left right and steering angle
def load_img_steering(datadir, df):
  image_path = []
  steering = []
  for i in range(len(data)):
    indexed_data = data.iloc[i]
    center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
    image_path.append(os.path.join(datadir, center.strip()))
    steering.append(float(indexed_data[3]))
    # left image append
    image_path.append(os.path.join(datadir,left.strip()))
    steering.append(float(indexed_data[3])+0.15)
    # right image append
    image_path.append(os.path.join(datadir,right.strip()))
    steering.append(float(indexed_data[3])-0.15)
  image_paths = np.asarray(image_path)
  steerings = np.asarray(steering)
  return image_paths, steerings

image_paths, steerings = load_img_steering(datadir + '/IMG', data)
X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size=0.2, random_state=6)
print('Training Samples: {}\nValid Samples: {}'.format(len(X_train), len(X_valid)))


#after the splitting data into traing and validation we want to make sure that both sets are centered around zerp
#below lines does this for us
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(y_train, bins=num_bins, width=0.05, color='blue')
axes[0].set_title('Training set')
axes[1].hist(y_valid, bins=num_bins, width=0.05, color='red')
axes[1].set_title('Validation set')


################################################################################


#behavioral cloning chapter 14 lecture 6

#prepare the data to be used inside the nuerual network

#crop the image and just allow the important parts of the image we decide the important part manully after drawing the image using matplotlib
#change the image into yuv frame
#yuv consist fo 3 channels components for the image
# gaussian blur  is useful for smoothing the image out with kernel size of (3,3)
#allows the nnt to extract more data since we are dealing with less noise
#resize the image_size into 200,66
#normanlize the image

#use map to apply the preprocessing to all the images



def img_preprocess(img):
    img = mpimg.imread(img)
    img = img[60:135,:,:]
    #slice the image from 60 to 135 since this is the important part
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img = cv2.GaussianBlur(img,  (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255
    return img
#image = image_paths[100]
#original_image = mpimg.imread(image)
#preprocessed_image = img_preprocess(original_image)
#
#fig, axs = plt.subplots(1, 2, figsize=(15, 10))
#fig.tight_layout()
#axs[0].imshow(original_image)
#axs[0].set_title('Original Image')
#axs[1].imshow(preprocessed_image)
#axs[1].set_title('Preprocessed Image')


X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))

plt.imshow(X_train[random.randint(0 , (len(X_train)-1))])
plt.axis('off')


###########################################################

##nvidia model 

#compare both nividia relu and elu acitvaition functions

#note draw the performance before applying the batch generator and after applying it
# google end to end deep learing for self driving car
#the use of relu may lead to dead relu  which means the data feed  to the next neuron with  relu is zero and the relu 
# we uses stride (2,2 ) since the images are larger and this allows speed up the process operation
#use elu activation functions to avoid the relu dead node phenomena
#subsample stride length of the kernel
#add drop out to reduce the propability of overfitting since we donot wnat the next layers to depend on specific neurons to learn from them
#note we use relu instead of sigmoid to avoid vanishing gradient
#dead relu when the neuron dies and only feeds zero

#dead relu occurs if the value passed to the relu is zero or less then it outputs zero, and the gradient of the relu in the positive region is one and 
#the gradient in the negative side is zero and since the gradient at this point is zero the weight of this node will not changed and this means the node will always 
#recieve negative values and outputs zero 
#elu in the negative region it returns negative valeu and also has gradient at the negative side

'''
###note run it to do comparison in performance


def nvidia_model_relu():
  model = Sequential()
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
  #subsample(x,y ) means as the kernel move it moves x pixels horizontial and y pixels vertical\
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
  model.add(Convolution2D(64, 3, 3, activation='elu'))
  
  model.add(Convolution2D(64, 3, 3, activation='elu'))
#   model.add(Dropout(0.5))
  
  
  model.add(Flatten())
  
  model.add(Dense(100, activation = 'relu'))
#   model.add(Dropout(0.5))
  
  model.add(Dense(50, activation = 'relu'))
#   model.add(Dropout(0.5))
  
  model.add(Dense(10, activation = 'relu'))
#   model.add(Dropout(0.5))

  model.add(Dense(1))
  #note the output contains single node which responsible of predicting the value of the steering angle
  optimizer = Adam(lr=1e-3)
  model.compile(loss='mse', optimizer=optimizer)
  #
  return model

model_relu = nvidia_model_relu()
print(model_relu.summary())
history_relu = model_relu.fit(X_train, y_train,
                                  
                                  epochs=30,
                                  validation_data=(X_valid, y_valid),
                                  batch_size = 128,
                                  verbose=1,
                                  shuffle = 1)
plt.plot(history_relu.history['loss'])
plt.plot(history_relu.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
model_relu.save('model.h5')

'''



#dead relu occurs if the value passed to the relu is zero or less then it outputs zero, and the gradient of the relu in the positive region is one and 
#the gradient in the negative side is zero and since the gradient at this point is zero the weight of this node will not changed and this means the node will always 
#recieve negative values and outputs zero 
#elu in the negative region it returns negative valeu and also has gradient at the negative side



def nvidia_model():
  model = Sequential()
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), input_shape=(66, 200, 3), activation='elu'))
  #subsample(x,y ) means as the kernel move it moves x pixels horizontial and y pixels vertical\
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='elu'))
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='elu'))
  model.add(Convolution2D(64, 3, 3, activation='elu'))
  
  model.add(Convolution2D(64, 3, 3, activation='elu'))
  model.add(Dropout(0.5))
  
  
  model.add(Flatten())
  
  model.add(Dense(100, activation = 'elu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(50, activation = 'elu'))
  model.add(Dropout(0.5))
  
  model.add(Dense(10, activation = 'elu'))
  model.add(Dropout(0.5))

  model.add(Dense(1))
  #note the output contains single node which responsible of predicting the value of the steering angle
  optimizer = Adam(lr=1e-3)
  model.compile(loss='mse', optimizer=optimizer)
  #
  return model

model = nvidia_model()
print(model.summary())
history = model.fit(X_train, y_train,
                                  
                                  epochs=30,
                                  validation_data=(X_valid, y_valid),
                                  batch_size = 128,
                                  verbose=1,
                                  shuffle = 1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('Epoch')
model.save('model.h5')

###############################################################################################

#Generator Augementation technique

#note dfter testing our model we have detected that the model was not perfect for our simulator and fails immediately in the second track
#this because the small datasets avaiable 
 
#data augmentation is the process of creating new datasets using your old datasets
#we have created our own data augmentation since it provide us with more customization and personlizing which important to us and which not important
# desing our own generator since we are interested in specific parts of the image


 #affine does transformantions with the object
    #translate percent Q argumer
    
    #scaling and zooming transfromations 
    
    
def zoom(image):
    #zoom the image so we focus more on the street rather than the car tip
    zoom = iaa.Affine(scale= (1,1.3))
    image = zoom.augment_image(image)   
    return image
image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
zoomed_image = zoom(original_image)
fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[1].imshow(zoomed_image)
axs[0].set_title('Origignal Image ' )
    

#pan image allow tanslation the image from the edges of hte road to the center
def pan(image):
    pan = iaa.Affine(translate_percent = {"x":(-0.1, 0.1), "y":(-0.1, 0.1)})
    image = pan.augment_image(image)
    return image
image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)
panned_image = pan(original_image)
fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[1].imshow(panned_image)
axs[0].set_title('Origignal Image ' )
    
 

#image random_
#making the image darker if the value if greater than one and lighter if its higher than one
def img_random_brightness(image):
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(image)
    return image

image = image_paths[random.randint(0, 1000)]
original_image = mpimg.imread(image)

brightness_altered_image = img_random_brightness(original_image)

fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[1].imshow(brightness_altered_image)
axs[0].set_title('Origignal Image ' )



#this is responsible of creating augmeneted image from the original one
def img_random_flip(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle

random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]

original_image = mpimg.imread(image)

flipped_image, flipped_steering_angle = img_random_flip(original_image, steerings)

fig, axs = plt.subplots(1,2, figsize = (15,10))
fig.tight_layout()
axs[0].imshow(original_image)
axs[0].set_title('Origignal Image ' )
axs[1].imshow(flipped_image)
axs[1].set_title('Flipped image ' )


#random augmentation
# return new augmented images with all the above process

def random_augment(image, steering_angle):
    image = mpimg.imread(image)    
    #define our syste
    #causes certian augmentation functions
    #note we are going to augment 50 of the new images
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = img_random_flip(image, steering_angle)
    return image, steering_angle
ncols =2
nrows = 10
fig, axis = plt.subplots(nrows, ncols, figsize = (15,50))

for i in range(10):
    randnum = random.randint(0, len(image_paths) -1)
    random_image = image_paths[randnum]
    random_steering = steerings[randnum]
    
    original_image = mpimg.imread(random_image)
    augmented_image, steering = random_augment(random_image, random_steering)
    
    axis[i][0].imshow(original_image)
    axis[i][0].set_title('original image')
    axis[i][1].imshow(augmented_image)
    axis[i][1].set_title('Augmented_ image')

######################################################
  
#fit generator

#batch generator is more effiecnet in fitting the model for cases of larger datasets
#its puts only small sets of the trainnig d

def batch_generator(image_paths, steering_ang, batch_size, istraining):
    #note istraning true if the data is training data
    #and false if validatate data we donot want our model to validate on augmented data
    #we want the model to validate on better data
  
  while True:
    #stops only with yield
    batch_img = []
    batch_steering = []
    
    for i in range(batch_size):
      random_index = random.randint(0, len(image_paths) - 1)
      
      if istraining:
        im, steering = random_augment(image_paths[random_index], steering_ang[random_index])
     
      else:
        im = mpimg.imread(image_paths[random_index])
        steering = steering_ang[random_index]
      
      im = img_preprocess(im)
      batch_img.append(im)
      batch_steering.append(steering)
    yield (np.asarray(batch_img), np.asarray(batch_steering))  

x_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
x_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

fig, axs = plt.subplots(1, 2, figsize=(15, 10))
fig.tight_layout()

axs[0].imshow(x_train_gen[0])
axs[0].set_title('Training Image')

axs[1].imshow(x_valid_gen[0])
axs[1].set_title('Validation Image')    
