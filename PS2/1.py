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
            limIy,limSy,limIx,limSx = calc_lim(x,y,sig,img)
            s = 0
            som = 0
            for j in range(limIy,limSy):
                for i in range(limIx,limSx):
                    s += hist[img[j][i]]
                    som += img[j][i] * hist[img[j][i]]
            nimg[y][x] = round(som/s)
    return nimg

def histeql_grey(img,r):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    hist = np.power(hist,r)
    cdf = hist.cumsum()
    q = cdf.max()
    cdf_m = cdf * 255 / q
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]


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

sig = val

nimg = sigma_filter(img,sig)

ere = raw_input("Digite o valor de r: ")
try:
    val = float(ere)
except:
    print "Valor invalido : Digite um numero float[.]."
    sys.exit(0)

nimg = nimg.astype('int')
himg = histeql_grey(nimg,val)
himg2 = histeql_grey(img,val)
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
plt.show()
# plt.get_current_fig_manager().window.wm_geometry("-640-800")
cv2.waitKey(0)