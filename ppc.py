# image preprocessing

import cv2
import numpy as np

# Crop region of interest from image
def crop(image):
  image = image[70:-25, :, :]
  return image

# Apply gaussian blur and change image from RGV to YUV color space
def color(image):
  image = cv2.GaussianBlur(image, (3,3), 0)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
  return image

def do_ppc(image):
  #image = crop(image)
  image = color(image)
  
  return image
  
