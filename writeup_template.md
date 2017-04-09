#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[mid]: ./examples/mid.jpg
[recovery_1]: ./examples/recovery_1.jpg
[recovery_2]: ./examples/reocovery_2.jpg
[recovery_3]: ./examples/recovery_3.jpg

## Rubric Points

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 with a video of a successful lap
* writeup_report.md summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is based on the architecture described in the following paper by NVIDIA: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

It is a convolution neural network.

The network starts by cropping the images - this would be part of the augmentation process, but I wanted to profit from GPU acceleration for this step, so I'm cropping in Keras and not with OpenCV.

Then, the network consisting of 5 convolutional layers with filter sizes 3x3 and 5x5; and depth between 24 and 64 (model.py lines 122-129). I added two dropout layers beetwen the convolutional layers to prevent overfitting.

The model includes RELU layers to introduce nonlinearity (code line 135-139) with another dropout layer, and the data is normalized in the model using a Keras lambda layer (code line 119). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 126,129,137). 

The model was trained and validated both on the Udacity data set and on data sets I collected myself. I had the most success with the Udacity data set, and the submitted model.h5 was trained solely on the Udacity data set. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 197).

####4. Appropriate training data

For the final training, I used the Udacity training data.

During development of the model, I used my own training data as well. I collected multiple laps driving in the middle of the lane, and one lap consisting only of "recovery" maneuvers, with the car driving from the road border back to the lane center. I also included a few recordings of specific problem points.

While it was very interesting seeing the trained model behavior change depending on the data used, in the end I used the Udacity data set because it gave me consistent and solid results.

###Model Architecture and Training Strategy

####1. Solution Design Approach

I used the model described in the Nvidia paper: https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

As it solved a similar problem as what I was trying to, I thought it would be a good fit.

My first step was to implement the convolutional neural network as described on page 5 of the paper, adding dropout layers to avoid overfitting and training on my own data. Since the data had strong peak of steering measurements around 0 degrees, I removed 70% of the steering measurements beetween -0.8 and 0.8 degrees. I also tried increasing the training data size by flipping the images, and by using all camera perspectives in the data set (left, center, right), correcting the steering angle for left and right.

Since the driving behavior wasn't satisfactory, I added augmentation on the data, adding some Gaussian blur and changing the color space from RGB to YUV. I also called the augmentation code from drive.py

These changes improved the driving behavior some, but at this point I still had problems with the car getting off-track pretty quickly.

I then rewrote the code to use generators for feeding the model, with a batch size of 32 images, and doing the augmentation on random images per batch. This ensures better training, as the model gets a more random set of images in every epoch.

For augmentation, I did the following:
- randomly flip image (and measurement)
- randomly adjust the brightness (per pixel)
- gaussian blur
- change color space from RGB to YUV
- crop the top 75 and bottom 20px of the image (this happens in the Keras model)

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

As described in "1. An appropriate model architecture has been employed"

####3. Creation of the Training Set & Training Process

Center lane driving to capture good behavior:

![center lane driving][mid]

Some pictures from a recovery (driving from left side of the road back to the middle)


![recovery start][recovery_1]
![recovery mid][recovery_2]
![recovery almost there!][recovery_3]

Image augmentation described in "Model Architecture and Training Strategy"

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 5-6 as the loss was not getting any lower after that. I used an adam optimizer so that manually training the learning rate wasn't necessary.
