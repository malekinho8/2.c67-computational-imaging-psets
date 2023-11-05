# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import svd
import cv2
import matplotlib.pyplot as plt
def plot_image(input_image):    
    plt.figure()
    plt.imshow(input_image/np.amax(input_image))
    return

def plot_vector(input_vector):    
    plt.figure()
    plt.plot(input_vector)
    plt.ylim(0)
    return

#load the image
img = cv2.imread('image_compressing/MIT.png')
#change it into grey scale
img = np.sum(img,axis=-1)
print(img.shape)


#get the svd for the image
u,s,vh = np.linalg.svd(img,)

#Plot the diagnal values
plot_vector(s)

#Choose the remained diagnal dimensions 
nc = 2

#Calculate the compressed image
img_c = u[:,0:nc] @ np.diag(s[0:nc]) @ vh[0:nc,:]


plot_image(img)
plot_image(img_c)








