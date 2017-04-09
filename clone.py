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

# TODO
# - [x] cropping in keras
# - [x] augment dataset by flipping images / measurements
# - [x] random brightness
# - [x] remove only 70% of the center steering
# - [ ] augment on the fly
# - [ ] train without dropout layers between the convolutional layers
# - [ ] try ELU instead of RELU
# - what else?
#
# only train on provided dataset!

min_steering_angle = 0.8
steering_angle_adjustment = 0.2

# how many samples per generator batch
batch_size = 32
# how many epochs to train
num_epochs = 100

# these two variables will be set later when we know how many training / validation samples
steps_per_epoch = 0 
validation_steps_per_epoch = 0



# image preprocessing / augmentation
# separate module to reuse pipeline in drive.py
import ppc

# lines from the driving log CSV file:
lines = []

path = "../windows_sim/"
folders = ["record_lap1", "record_lap2", "record_recovery", "record_problems"]

for folder in folders:
  print(folder)
  
  # Read the CSV driving logfiles
  f = open(path + folder + '/driving_log.csv')
  with f as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:

      # eliminate 70% of steering values too close to 0 in order to reduce bias
      if random.randrange(10) < 7:
        if abs(lines[3]) < min_steering_angle:
          continue

      lines.append(line)
  f.close()

  # remove header row
  # center,left,right,steering
  del lines[0]  

  print("read " + str(len(lines)) + " lines from driving log")

  
useable = len(lines)
print ("got " + str(useable) + " useable measurements")

# Sanity check
if useable == 0:
  print ("0 useable measurements, giving up")
  quit()

def generator(data):
  while 1:
    for i in range(0, len(data), batch_size)
      batch_samples = data[offset : offset + batch_size]
      images = []
      measurements = []

      for line in lines:
        (image, measurement) = augment_data(line)
        images.append(image)
        measurements.append(measurement)

      X_train = np.array(images)
      y_train = np.array(measurements)

      yield(X_train, y_train)

def augment_data(line):
  steering = line[3]

  # randomly choose the camera to take the image from
  camera = np.random.choice([0,1,2])

  # adjust the steering angle for left anf right cameras
  if camera == 1:
    steering += steering_angle_adjustment
  elif camera == 2:
    steering -= steering_angle_adjustment

  source_path = line[camera]
  filename = source_path.split('\\')[-1]

  image = cv2.imread(path + folder + '/IMG/' + filename)

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
  model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(320, 160, 3)))
  
  # Normalisation:
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
  
  model.add(Conv2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
  model.add(Conv2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
  model.add(Conv2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
  #model.add(Dropout(0.3))
  model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
  model.add(Conv2D(64,3,3,border_mode='valid', activation='relu', subsample=(1,1)))
  #model.add(Dropout(0.2))
  
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
  

lines = sklearn.utils.shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


train_generator = generator(train_samples)
validation_generator = generator(validation_sample)


# It should typically be equal to the number of unique samples if your dataset divided by the batch size.
steps_per_epoch = len(train_samples) / batch_size
validation_steps_per_epoch = len(validation_samples) / batch_size
  
model = nvidia_model()

model.compile(loss='mse', optimizer='adam')

# EarlyStopping callback - let keras monitor the loss function and stop training the model
# when it's not improving enough anymore. That way, I can set the epochs to a high
# value and not worry about it.
early_stopping = EarlyStopping(monitor='val_loss', patience=2, min_delta=0.002)

model.fit_generator(train_generator, steps_per_epoch, epochs=num_epochs, \
  callbacks=[early_stopping], validation_data=validation_generator, validation_steps=validation_steps_per_epoch)


model.save('model.h5')
