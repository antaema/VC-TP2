from __future__ import division
import numpy as np
import cv2 
from matplotlib import pyplot as plt
import sys
sys.setrecursionlimit(10000)

def calc_lim(x,y,sig,img):
    rows ,cols = img.shape
    
    if y - sig >= 0:
        limIy = y - sig
    else:
        limIy = 0
    # mais um pois em python range vai ate < lim
    if y + sig < cols:
        limSy = y + sig + 1
    else:
        limSy = cols

    if x - sig >= 0:
        limIx = x - sig
    else:
        limIx = 0
    
    if x + sig < rows:
        limSx = x + sig +1
    else:
        limSx = rows

    return limIy,limSy,limIx,limSx

def sigma_filter(img,sig):
    rows ,cols = img.shape
    nimg = np.zeros((rows ,cols))
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    for y in range(0,cols):
        for x in range(0,rows):
            s = 0
            som = 0
            if img[y][x] - sig < 0:
                limIc = 0
            else:
                limIc = img[y][x] - sig

            if img[y][x] + sig > 255:
                limSc = 255
            else:
                limSc = img[y][x] + sig
            
            for i in range(limIc,limSc):
                s += hist[i]
                som += i * hist[i]
            
            nimg[y][x] = round(som/s)

            #Filtro diferene de sigma 
            # s = 0
            # som = 0
            # limIy,limSy,limIx,limSx = calc_lim(x,y,sig,img)
            # for j in range(limIy,limSy):
            #     for i in range(limIx,limSx):
            #         s += hist[img[j][i]]
            #         som += img[j][i] * hist[img[j][i]]
            # nimg[y][x] = round(som/s)
    return nimg

def histeql_grey(img,r):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    hist = np.power(hist,r)
    cdf = hist.cumsum()
    q = cdf.max()
    cdf_m = cdf * 255 / q
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img],cdf_m


# open image
img = cv2.imread('noise.png')
#converte imagem para grey scale
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
sig = raw_input("Digite o valor de sigma: ")
try:
    val = int(sig)
except:
    print "Valor invalido : Digite um numero inteiro positivo."
    sys.exit(0)
if val == 1:
    print "Valor invalido : Digite um numero diferente de 1."
    sys.exit(0)
sig = val

nimg = sigma_filter(img,sig)

ere = raw_input("Digite o valor de r: ")
try:
    val = float(ere)
except:
    print "Valor invalido : Digite um numero float[.]."
    sys.exit(0)

nimg = nimg.astype('int')

himg2,cdf_m = histeql_grey(img,val)
himg,cdf_m = histeql_grey(nimg,val)

plt.figure(1)
plt.axis("off")
plt.gcf().canvas.set_window_title("Normal") 
plt.get_current_fig_manager().window.wm_geometry("-1400-800")
plt.imshow(img, cmap = 'gray')

plt.figure(2)
plt.axis("off")
plt.gcf().canvas.set_window_title("Filtro Sigma")
plt.get_current_fig_manager().window.wm_geometry("-640-800")
plt.imshow(nimg, cmap = 'gray')

plt.figure(3)
plt.axis("off")
plt.gcf().canvas.set_window_title("Equalizado")
plt.get_current_fig_manager().window.wm_geometry("-0-800")
plt.imshow(himg, cmap = 'gray')

plt.figure(4)
plt.axis("off")
plt.gcf().canvas.set_window_title("Equalizado sem sigma")
plt.get_current_fig_manager().window.wm_geometry("-0-0")
plt.imshow(himg2, cmap = 'gray')

histnor,bins = np.histogram(img.flatten(),256,[0,256])
histeq,bins = np.histogram(himg.flatten(),256,[0,256])

plt.figure(5)
plt.gcf().canvas.set_window_title("Histograma normal")
plt.get_current_fig_manager().window.wm_geometry("-1400-0")
plt.plot(histnor,color = 'red')
plt.xlim([0,256])

plt.figure(6)
plt.gcf().canvas.set_window_title("Histograma equalizado")
plt.get_current_fig_manager().window.wm_geometry("-640-0")
plt.plot(histeq,color = 'green')
plt.xlim([0,256])

plt.figure(7)
plt.gcf().canvas.set_window_title("Histograma acumulativo")
plt.get_current_fig_manager().window.wm_geometry("-640-400")
plt.plot(cdf_m,color = 'blue')
plt.xlim([0,256])
plt.show()
# plt.get_current_fig_manager().window.wm_geometry("-640-800")
cv2.waitKey(0)