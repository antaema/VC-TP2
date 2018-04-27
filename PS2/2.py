from __future__ import division
import numpy as np
import cv2 
from matplotlib import pyplot as plt
import sys
sys.setrecursionlimit(10000)


#abre a imagem em grey scale
img = cv2.imread('anime.jpg',0)

#abre imagem colorida
img2 = cv2.imread('anime.jpg')
img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

#Aplica Log na Imagem
gaussian = cv2.GaussianBlur(img,(3,3),0)
log = cv2.Laplacian(gaussian,cv2.CV_64F)

sobely1 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
sobely2 = cv2.Sobel(img,cv2.CV_64F,0,2,ksize=5)
sobely3 = cv2.Sobel(img,cv2.CV_64F,0,3,ksize=5)
sobely4 = cv2.Sobel(img,cv2.CV_64F,0,4,ksize=5)

sobelx1 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
sobelx2 = cv2.Sobel(img,cv2.CV_64F,2,0,ksize=5)
sobelx3 = cv2.Sobel(img,cv2.CV_64F,3,0,ksize=5)
sobelx4 = cv2.Sobel(img,cv2.CV_64F,4,0,ksize=5)

sobel = sobelx1 + sobelx2 + sobelx3  + sobely1 + sobely2 + sobely3 

sobelg = cv2.Sobel(gaussian,cv2.CV_64F,1,1,ksize=1)
canny = cv2.Canny(img,100,200)

plt.figure(1)
plt.axis("off")
plt.gcf().canvas.set_window_title("Original") 
plt.get_current_fig_manager().window.wm_geometry("-1400-800")
plt.imshow(img2, cmap = 'gray')

plt.figure(2)
plt.axis("off")
plt.gcf().canvas.set_window_title("Metodo 1 LoG")
plt.get_current_fig_manager().window.wm_geometry("-640-800")
plt.imshow(log, cmap = 'gray')

plt.figure(3)
plt.axis("off")
plt.gcf().canvas.set_window_title("Metodo 2 Sum Sobel")
plt.get_current_fig_manager().window.wm_geometry("-0-800")
plt.imshow(sobel, cmap = 'gray')


plt.figure(4)
plt.axis("off")
plt.gcf().canvas.set_window_title("Metodo 3 Sobel Gaussiano dx,dy 1 ordem")
plt.get_current_fig_manager().window.wm_geometry("-0-0")
plt.imshow(sobelg, cmap = 'gray')

plt.figure(5)
plt.axis("off")
plt.gcf().canvas.set_window_title("Canny para comparacao ")
plt.get_current_fig_manager().window.wm_geometry("-1400-0")
plt.imshow(canny, cmap = 'gray')

plt.show()

cv2.waitKey(0)