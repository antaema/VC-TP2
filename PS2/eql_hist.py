from __future__ import division
import numpy as np
import cv2 
import math
from matplotlib import pyplot as plt

def hisEqulColor(img):
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    channels=cv2.split(ycrcb)
    cv2.equalizeHist(channels[0],channels[0])
    cv2.merge(channels,ycrcb)
    cv2.cvtColor(ycrcb,cv2.COLOR_YCR_CB2BGR,img)
    return img

def calculate_entropy(total_size,hist):
    entropy = 0
    for i in range(0,len(hist)):
            if hist[i] > 0:
                entropy+=(hist[i]/total_size) * math.log((total_size/hist[i]),2)
    return entropy

def histeql_grey(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    return cdf[img]

def calculate_dataGrey(grey):
    mean_g , std_g = cv2.meanStdDev(grey)
    hist = cv2.calcHist([grey],[0],None,[256],[0,256])
    total_size = grey.shape[0] * grey.shape[1]
    entropy_g = calculate_entropy(total_size,hist)
    return mean_g, std_g, entropy_g

def calculate_dataColor(color):
    mean_c , std_c = cv2.meanStdDev(color)
    mean_c = np.mean(mean_c)
    std_c = np.mean(std_c)
    histb = cv2.calcHist([color],[0],None,[256],[0,256])
    histg = cv2.calcHist([color],[1],None,[256],[0,256])
    histr = cv2.calcHist([color],[2],None,[256],[0,256])
    total_size = color.shape[0] * color.shape[1]
    entropy_b = calculate_entropy(total_size,histb)
    entropy_g = calculate_entropy(total_size,histg)
    entropy_r = calculate_entropy(total_size,histr)
    return mean_c, std_c, entropy_b, entropy_g, entropy_r 

def obtein_data(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_g, std_g, entropy_g = calculate_dataGrey(grey)
    lgrey = [ mean_g, std_g, entropy_g ]
    
    e_grey = histeql_grey(grey)
    mean_eg, std_eg, entropy_eg = calculate_dataGrey(e_grey)
    le_grey= [ mean_eg, std_eg, entropy_eg ]

    mean_c, std_c, entropy_b, entropy_g, entropy_r = calculate_dataColor(img)
    limg = [ mean_c, std_c, [ entropy_b, entropy_g, entropy_r ] ]
    
    e_img = hisEqulColor(img)
    mean_ec, std_ec, entropy_eb, entropy_eg, entropy_er = calculate_dataColor(img)
    le_img = [ mean_ec, std_ec, [ entropy_eb, entropy_eg, entropy_er ] ]
    return lgrey, le_grey, limg, le_img


means_grey=[]
means_egrey=[]
means_img=[]
means_eimg=[]

stds_grey=[]
stds_egrey=[]
stds_img=[]
stds_eimg=[]

entropys_grey=[]
entropys_egrey=[]
entropys_img=[]
entropys_eimg=[]

stds=[]
entropys = []
cap = cv2.VideoCapture('cabra.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if ret == True:
        lgrey, le_grey, limg, le_img = obtein_data(frame)
        
        means_grey.append(lgrey[0])
        stds_grey.append(lgrey[1])
        entropys_grey.append(lgrey[2])

        means_egrey.append(le_grey[0])
        stds_egrey.append(le_grey[1])
        entropys_egrey.append(le_grey[2])

        means_img.append(limg[0])
        stds_img.append(limg[1])
        entropys_img.append(limg[2])

        means_eimg.append(le_img[0])
        stds_eimg.append(le_img[1])
        entropys_eimg.append(le_img[2])
    else:
        break
cap.release()

print "[*] Antes"

print " (*) Grey" 
print '     Mean dif:  ',
print (max(means_grey) - min(means_grey)) 
print '     Std dif:  ',
print (max(stds_grey) - min(stds_grey)) 

print " (*) Color" 
print '     Mean dif:  ',
print (max(means_img) - min(means_img)) 
print '     Std dif:   ',
print (max(stds_img) - min(stds_img)) 

print "[*] Depois"

print " (*) Grey" 
print '     Mean dif: ',
print (max(means_egrey) - min(means_egrey)) 
print '     Std dif: ',
print (max(stds_egrey) - min(stds_egrey))

print " (*) Color" 
print '     Mean dif: ',
print (max(means_eimg) - min(means_eimg))
print '     Std dif: ',
print (max(stds_eimg) - min(stds_eimg)) 

cap.release()

