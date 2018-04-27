from __future__ import division
import cv2
import numpy as np
from numpy import exp, abs, angle
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter

def polar2z(mag,angle,size):
    new = []
    for i in range(0,size):
        new.append(mag[i] * exp( 1j * angle[i] ))
    return new

def fourier(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    mag = np.abs(fshift)
    angle = np.angle(fshift)
    return mag,angle

def image2back(data):
    f_ishift = np.fft.ifftshift(data)
    img_back = np.fft.ifft2(f_ishift)
    return np.abs(img_back)

img = cv2.imread('anime.jpg',0)
mag,angle = fourier(img)
magnitude_spectrum = 20*np.log(mag)

nmag = generic_filter(mag, np.mean, size=(3,3))
nangle = generic_filter(angle, np.mean, size=(2,2))

new1 = polar2z(nmag,angle,len(mag))
new2 = polar2z(mag,nangle,len(mag))

img_back = image2back(new1)
img_back2 = image2back(new2)

plt.figure(1)
plt.axis("off")
plt.gcf().canvas.set_window_title("Normal") 
plt.get_current_fig_manager().window.wm_geometry("-1400-800")
plt.imshow(img, cmap = 'gray')

plt.figure(2)
plt.axis("off")
plt.gcf().canvas.set_window_title("Magnitute alterada")
plt.get_current_fig_manager().window.wm_geometry("-640-800")
plt.imshow(img_back, cmap = 'gray')

plt.figure(3)
plt.axis("off")
plt.gcf().canvas.set_window_title("Phase alterada")
plt.get_current_fig_manager().window.wm_geometry("-0-800")
plt.imshow(img_back2, cmap = 'gray')
plt.show()