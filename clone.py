### imports ###

import csv
import cv2
import numpy as np
import random
import sklearn
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.callbacks import EarlyStopping

# image preprocessing / augmentation
# separate module so the code can be used in drive.py as well
import ppc

### define some constants ###

# if steering angle is < this value, drop 70% of the data
min_steering_angle = 0.8

# steering angle adjustment for left / right camera
steering_angle_adjustment = 0.2

# how many samples per generator batch
batch_size = 32
# how many epochs to train
num_epochs = 10

#folder = "../windows_sim/record_lap2/"
folder = "../data/"

### define functions ###

def generator(data):
  while 1:
    for i in range(0, len(data), batch_size):
      batch_samples = data[i : i + batch_size]
      images = []
      measurements = []

      for line in batch_samples:
        (image, measurement) = augment_data(line)
        images.append(image)
        measurements.append(measurement)
      
      #print("got " + str(len(measurements)) + " measurements in batch")

      X_batch = np.array(images)
      y_batch = np.array(measurements)
      
      #print (X_batch.shape)
      #print (y_batch.shape)

      yield(X_batch, y_batch)



      
def augment_data(line):
  steering = float(line[3])

  # randomly choose the camera to take the image from
  camera = np.random.choice([0,1,2])

  # adjust the steering angle for left anf right cameras
  if camera == 1:
    steering += steering_angle_adjustment
  elif camera == 2:
    steering -= steering_angle_adjustment
    
  source_path = line[camera]
  
  # adjust file path depending on whether using udacity data or own data capture
  if "windows_sim" in source_path:
    filename = "/IMG/" + source_path.split('\\')[-1]
  else:
    filename = source_path.replace(' ', '')

  image = cv2.imread(folder + filename)

  # randomly flip images to reduce left steering bias
  # an alternative would be driving the track backwards, but this way
  # we don't need double amount of data
  if np.random.random() > 0.5:
    steering = -1 * steering
    image = cv2.flip(image, 1)

  # Do augmentation on the image - see code in ppc.py
  image = ppc.do_ppc(image)

  return (image, steering)



# This is just the basic model from the course video to check
# the pipeline is working correctly
def basic_model():
  model = Sequential()

  # normalisation:
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
  model.add(Flatten(input_shape=(160,320,3)))
  model.add(Dense(1))
  
  return model

# Real model based on the Nvidia paper at 
# https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def nvidia_model():
  model = Sequential()

  # Cropping:
  # remove top 70, bottom 25px, leave left / right as is
  model.add(Cropping2D(cropping=((70, 25), (0, 0)), dim_ordering='tf', input_shape=(160, 320, 3)))
  
  # Normalisation:
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(65, 320, 3), output_shape=(65, 320, 3)))
  
  # Convolutional layers:
  model.add(Conv2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
  model.add(Conv2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
  model.add(Conv2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
  # Dropout layer to prevent overfitting:
  model.add(Dropout(0.3))
  model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
  model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
  model.add(Dropout(0.2))
  
  # Flattening:
  model.add(Flatten())
  
  # RELU activation layers:
  model.add(Dense(1164, activation='relu'))
  model.add(Dense(100, activation='relu'))
  model.add(Dropout(0.3))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(10, activation='relu'))
  model.add(Dense(1, activation='tanh'))
  
  return model

### main ###
### read training data ###

# lines from the driving log CSV file:
lines = []

skip_one = 0

print("reading data from folder " + folder)

# Read the CSV driving logfiles
f = open(folder + '/driving_log.csv')
with f as csvfile:
  reader = csv.reader(csvfile)
  for line in reader:

    # skip header line
    if skip_one == 0:
      skip_one = 1
      continue

    # eliminate 70% of steering values too close to 0 in order to reduce bias 
    if random.randrange(10) < 7:
      if abs(float(line[3])) < min_steering_angle:
        continue

    lines.append(line)
f.close()

print("read " + str(len(lines)) + " lines from driving log")

# Sanity check
if len(lines) == 0:
  print ("0 useable measurements, giving up")
  quit()

### train ###  
  
# shuffle data, then split into training set and validation set
lines = sklearn.utils.shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print ("got " + str(len(train_samples)) + " training samples")
print ("got " + str(len(validation_samples)) + " validation samples")

# create the generators for training and validation sample batches
train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

# use the nvidia model
model = nvidia_model()

# use Adam optimizer for learning rate
model.compile(loss='mse', optimizer='adam')

# run model and save
model.fit_generator(train_generator, samples_per_epoch = len(train_samples), nb_epoch=num_epochs, \
  validation_data = validation_generator, nb_val_samples = len(validation_samples))
model.save('model.h5')
