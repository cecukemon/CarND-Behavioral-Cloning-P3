import csv
import cv2
import numpy as np

# pretty printer
import pprint

# image preprocessing
# separate module to reuse pipeline in drive.py
import ppc

# recorded driving images
images = []
# recorded steering angle measurement
measurements = []

# List of folders containing the recordings (images + csv file)
#
# I separated the recordings into multiple folders for the following reasons:
# - train on a subset to quickly check model changes
# - add / remove training data with specific characteristics to see if
#   it improves the model or not
# - easier to manually correct training data if I made a mistake recodings
#   (I prefer RPG games to driving / racing games :D 
#
# lap1, lap2 are just regular laps
# recovery - recovering from sideline driving
# problems - training data for specific problem spots

#folders = ["record_lap1", "record_lap2", "record_recovery", "record_problems"]

folders = ["data"]

j = 0;
for folder in folders:
  print(folder)
  
  # lines from the driving log CSV file:
  lines = []
  
  # Read the CSV driving logfiles
  f = open('../windows_sim/' + folder + '/driving_log.csv')
  with f as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)
  f.close()
  print("read " + str(len(lines)) + " lines from driving log")

  for line in lines:
    for i in range(0,2):
      source_path = line[i]
      filename = source_path.split('\\')[-1]
    
      measurement = float(line[3])
    
      # eliminate steering values too close to 0 in order to reduce bias
      #if abs(measurement) <= 0.01:
      #if measurement <= 0.8 and j%2 == 0:
      if measurement == 0:
        continue
      
      # correct camera angle on left and right image
      if (i==1):
          measurement += 0.2
      elif(i==2):
          measurement += -0.2
    
      measurements.append(measurement)
      image = cv2.imread('../windows_sim/' + folder + '/IMG/' + filename)
      
      # Do preprocessing on the image - see code in ppc.py
      #image = ppc.do_ppc(image)
      
      images.append(image)
    j += 1
  print("got " + str(len(images)) + " images so far")

#pp = pprint.PrettyPrinter(indent=2)
#pp.pprint(images)
  
useable = len(measurements)
print ("got " + str(useable) + " useable measurements")

# Sanity checks - should have found more than 0 useable data sets
# and as many images as we have measurements
if useable == 0:
  print ("giving up")
  quit()

if len(images)!=len(measurements):
  print ("something went wrong")
  quit()


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D

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
  
  # Normalisation:
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
  
  model.add(Conv2D(24,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
  model.add(Conv2D(36,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
  model.add(Conv2D(48,5,5,border_mode='valid', activation='relu', subsample=(2,2)))
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
  

# small subsample for quick evaluation  
#X_train = np.array(images[:100])
#y_train = np.array(measurements[:100])
  
  
X_train = np.array(images)
y_train = np.array(measurements)
  
model = nvidia_model()

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')

