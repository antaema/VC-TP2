from __future__ import division
import cv2
import numpy as np
from numpy import exp, abs, angle
from matplotlib import pyplot as plt


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

def plotImage(n,img,img2,tex,tex2,pst):
    plt.figure(n)
    plt.get_current_fig_manager().window.wm_geometry(pst)
    plt.subplot(121),plt.imshow(img, cmap = 'gray')
    plt.title(tex), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(img2, cmap = 'gray')
    plt.title(tex2), plt.xticks([]), plt.yticks([])

img = cv2.imread('terra.jpg',0)
img2 = cv2.imread('marte.jpg',0)

mag,angle = fourier(img)
magnitude_spectrum = 20*np.log(mag)

mag2,angle2 = fourier(img2)

new1 = polar2z(mag,angle2,len(mag))
new2 = polar2z(mag2,angle,len(mag))

img_back = image2back(new1)
img_back2 = image2back(new2)

plotImage(1,img,img2,'Input Image 1', 'Input Image 2', "-900-300")
plotImage(2,img_back,img_back2,'Amp1 + Phase2','Amp2 + Phase1', "-200-300")
plt.show()