# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import cv2
import sys; sys.path.append('./')
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from numpy.fft import ifft2,fft2,fft,ifft,fftfreq, fftshift, ifftshift

# use latex for rendering plots
plt.rc('text',usetex=True)
plt.rc('font',family='serif',size=9)

def plot_image(input_image, clim=(0,1)):    
    plt.figure()
    plt.imshow(np.abs(input_image),cmap=plt.cm.hot)
    plt.colorbar()
    # plt.clim(*clim)

#In this problem, you can comment the rest when you work on the other problems.


#load the image
img_a = cv2.imread('ps2/pru_mono_homework.png')/255.0
img_b = cv2.imread('ps2/pru_mono_detail_1_homework.png')/255.0
img_c = cv2.imread('ps2/pru_mono_detail_2_homework.png')/255.0
# change it into the gray scale
img_a = np.mean(img_a,axis=-1)
img_b = np.mean(img_b,axis=-1)
img_c = np.mean(img_c,axis=-1)
print(img_a.shape)


#######################################
#problem (a)
#### Calculate the fft of the image
### Perform the 2D Fourier transform on the image
fftimg_a = fft2(img_a)
fftimg_b = fft2(img_b)
fftimg_c = fft2(img_c)

### shift the FFT so zero frequency is centered
fftimg_shifted_a = fftshift(fftimg_a)
fftimg_shifted_b = fftshift(fftimg_b)
fftimg_shifted_c = fftshift(fftimg_c)

### generate the frequency ticks
def get_shifted_freqs(img):
    """Gets the k_r and k_c values for a given image."""
    n_rows, n_cols = img.shape
    k_r = fftfreq(n_rows)
    k_c = fftfreq(n_cols)
    k_r_shifted = fftshift(k_r) * n_rows
    k_c_shifted = fftshift(k_c) * n_cols
    return k_r_shifted, k_c_shifted

k_r_a, k_c_a = get_shifted_freqs(img_a)
k_r_b, k_c_b = get_shifted_freqs(img_b)
k_r_c, k_c_c = get_shifted_freqs(img_c)

### plot the data
fig, axes = plt.subplots(3,2,figsize=(10,8))

ax0 = axes[0,0].imshow(img_a, cmap='gray')
axes[0,0].set_title('Image (a)')
axes[0,0].set_xticks([])
axes[0,0].set_yticks([])

ax1 = axes[0,1].imshow(np.log(np.abs(fftimg_shifted_a)+1), cmap='gray', extent=(k_c_a[0],k_c_a[-1],k_r_a[0],k_r_a[-1]))
axes[0,1].set_title(r'$ | F[k_r,k_c] | $')
cbar = fig.colorbar(ax1, ax=axes[0,1])
axes[0,1].set_xlabel('$k_c$')
axes[0,1].set_ylabel('$k_r$')

ax2 = axes[1,0].imshow(img_b, cmap='gray')
axes[1,0].set_title('Image (b)')
axes[1,0].set_xticks([])
axes[1,0].set_yticks([])

ax3 = axes[1,1].imshow(np.log(np.abs(fftimg_shifted_b)+1), cmap='gray', extent=(k_c_b[0],k_c_b[-1],k_r_b[0],k_r_b[-1]))
axes[1,1].set_title(r'$ | F[k_r,k_c] | $')
cbar2 = fig.colorbar(ax3,ax=axes[1,1])
axes[1,1].set_xlabel('$k_c$')
axes[1,1].set_ylabel('$k_r$')

ax4 = axes[2,0].imshow(img_c, cmap='gray')
axes[2,0].set_title('Image (c)')
axes[2,0].set_xticks([])
axes[2,0].set_yticks([])

ax5 = axes[2,1].imshow(np.log(np.abs(fftimg_shifted_c)+1), cmap='gray', extent=(k_c_c[0],k_c_c[-1],k_r_c[0],k_r_c[-1]))
axes[2,1].set_title(r'$ | F[k_r,k_c] | $')
cbar3 = fig.colorbar(ax5,ax=axes[2,1])
axes[2,1].set_xlabel('$k_c$')
axes[2,1].set_ylabel('$k_r$')

plt.tight_layout()
plt.show()


#######################################




# #######################################
# #problem (b)
# ####define your own 2D-kernel
kernel = np.array([[0,0,0],[1,1,1],[0,0,0]])

# ####Calculate the blurred image.
# blur_img = #####

# #### Calculate the fft of the blurred image
# fftblur_img = #####


# plot_image(blur_img)
# plot_image(fftblur_img)
# #######################################




# #######################################
# #problem (c)
# #### Calculate the fft of the kernel
# fftkernel = ####


# plt.figure()
# plt.imshow(fftkernel,cmap=plt.cm.hot)
# plt.colorbar()


# plot_image(fftkernel*fftimg)
# #######################################






