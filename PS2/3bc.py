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

img = cv2.imread('homo1.jpg')
img3 = cv2.imread('rosto2.jpg',0)
img2 = cv2.imread('homo2.jpg',0)
img4 = cv2.imread('rosto1.jpg',0)

mag,angle =fourier(img)
mag = mag * 2 
#magnitude_spectrum = 20*np.log(mag)
len1 = len(mag)

mag2,angle2 =fourier(img2)
angle2 = angle2 / 1.1
len2 = len(mag2)

mag3,angle3 =fourier(img3)
mag3 = mag3 * 2
len3 = len(mag3) 
#magnitude_spectrum = 20*np.log(mag)

mag4,angle4 =fourier(img4)
angle4 = angle4 / 1.1
len4 = len(mag4)

new1 = polar2z( mag, angle, len1)
new2 = polar2z( mag2, angle2, len2)
new3 = polar2z( mag3, angle3, len3)
new4 = polar2z( mag4, angle4, len4)


img_back = image2back(new1)
img_back2 = image2back(new2)
img_back3 = image2back(new3)
img_back4 = image2back(new4)

plotImage(1,img,img2,'Input Image 1', 'Input Image 2', "-900-300")
plotImage(2,img_back,img_back2,'Img1 with Amp * 2', 'Img2 with Angle / 1.1', "-200-300")
plotImage(3,img3,img4,'Input Image 3', 'Input Image 4', "-900-300")
plotImage(4,img_back3,img_back4,'Img1 with Amp * 2', 'Img2 with Angle / 1.1', "-200-300")

plt.show()