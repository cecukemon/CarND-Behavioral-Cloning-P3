# image preprocessing

import cv2
import numpy as np

# randomly adjust image brightness
def brightness(image):

    image_adjusted = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)

    # random factor for brightness reduction - add constant so it's not completely dark
    bright = np.random.uniform() + 0.25

    # brightness reduction only gets applied to value channel:
    image_adjusted[:,:,2] = image_adjusted[:,:,2] * bright

    image_adjusted = cv2.cvtColor(image_adjusted,cv2.COLOR_HSV2RGB)
    return image_adjusted


# Apply gaussian blur and change image from RGV to YUV color space
def color(image):
  image_adjusted = cv2.GaussianBlur(image, (3,3),0)
  image_adjusted = cv2.cvtColor(image_adjusted, cv2.COLOR_RGB2YUV)
  return image_adjusted

def do_ppc(image):
  image1 = brightness(image)
  image2 = color(image1)
  
  return image2
  
