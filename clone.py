import csv
import cv2
import numpy as np

# folders containing the recordings (images + csv file)
# lap1, lap2 are just regular laps
# reverse is a lap driven in the other direction
# recenter 
#folders = ["record_lap1", "record_lap2", "record_lap_reverse", "record_recenter"]

folders = ["record_lap1", "record_recenter"]

# lines from the driving log CSV file:
lines = []
for folder in folders:
  print(folder)
  f = open('../windows_sim/' + folder + '/driving_log.csv')
  with f as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
      lines.append(line)
  f.close()

# driving recording images
images_tmp = []
images = []
# driving recording measurements (steering angle)
measurements_tmp = []
measurements = []

j = 0;
for folder in folders:
  for line in lines:
    for i in range(0,2):
      source_path = line[i]
      filename = source_path.split('\\')[-1]
    
      measurement = float(line[3])
    
      # eliminate steering values too close to 0
      # in order to reduce bias
      #if abs(measurement) <= 0.01:
      if measurement == 0.0:
        continue
      
      # correct camera angle on left and right image
      if (i==1):
          measurement += 0.2
      elif(i==2):
          measurement += -0.2
    
      measurements_tmp.append(measurement)
      image = cv2.imread('../windows_sim/' + folder + '/IMG/' + filename)
      images_tmp.append(image)
    j += 1

  # TODO color correction?
  #      also do that in drive.py

useable = len(measurements_tmp)
print ("got " + str(useable) + " useable measurements")

if useable == 0:
  print ("giving up")
  quit()

if len(images_tmp)!=len(measurements_tmp):
  print ("something went wrong")
  quit()


images = images_tmp
measurements = measurements_tmp

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Conv2D

def basic_model():
  model = Sequential()

  # normalisation:
  model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
  model.add(Flatten(input_shape=(160,320,3)))
  model.add(Dense(1))
  
  return model

# based on the Nvidia paper https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
def nvidia_model():
  model = Sequential()
  
  # normalisation:
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
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3)

model.save('model.h5')

