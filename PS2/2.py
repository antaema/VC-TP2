from __future__ import division
import numpy as np
import cv2 
import math
from matplotlib import pyplot as plt



def calculate_entropy(total_size,hist):
    entropy = 0
    for i in range(0,len(hist)):
            if hist[i] > 0:
                entropy+=(hist[i]/total_size) * math.log((total_size/hist[i]),2)
    return entropy


def calculate_dataGrey(grey):
    mean_g , std_g = cv2.meanStdDev(grey)
    hist = cv2.calcHist([grey],[0],None,[256],[0,256])
    total_size = grey.shape[0] * grey.shape[1]
    entropy_g = calculate_entropy(total_size,hist)
    return mean_g, std_g, entropy_g

def normalize_two(f,g):
    f_mean = np.mean(f)
    f_std = np.std(f)
    g_mean = np.mean(g)
    g_std = np.std(g)
    alpha = (( g_std / f_std ) * f_mean) - g_mean
    beta = f_std / g_std

    print '[*] Alpha : ',
    print alpha
    print '[*] Beta: ',
    print beta

    g_new = []
    for i in range(0,len(g)):
        g_new.append(beta*(g[i]+alpha))
    
    return g_new

def calc_print(means,stds,entropys):
    mean_m = np.mean(means)
    mean_s = np.std(means)
    std_m = np.mean(stds)
    std_s = np.std(stds)
    entropy_m = np.mean(entropys)
    entropy_s = np.std(entropys)

    print " (*) Mean" 
    print '     mean:    ',
    print mean_m 
    print '     std:     ',
    print mean_s
    print '     l1-norm: ',
    print sum(abs(np.asarray(means)))
    

    print " (*) Std" 
    print '     mean:    ',
    print std_m
    print '     std:     ',
    print std_s
    print '     l1-norm: ',
    print sum(abs(np.asarray(stds)))

    print " (*) Entropy" 
    print '     mean:  ',
    print entropy_m
    print '     std:   ',
    print entropy_s
    print '     l1-norm:  ',
    print sum(abs(np.asarray(entropys)))

means=[]
stds=[]
entropys=[]

cap = cv2.VideoCapture('cabra.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean, std, entropy = calculate_dataGrey(grey)
        
        means.append(mean)
        stds.append(std)
        entropys.append(entropy)
    else:
        break
cap.release()

n_stds = normalize_two(means,stds)
n_entropys = normalize_two(means,entropys)


print "[*] Antes"
calc_print(means,stds,entropys)

print "\n\n[*] Depois"
calc_print(means,n_stds,n_entropys)


