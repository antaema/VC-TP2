from __future__ import division
import cv2
import numpy as np
from numpy import exp, abs, angle
from matplotlib import pyplot as plt
from scipy.ndimage import generic_filter

def nothing(x):
    pass


img = cv2.imread('anime.jpg',0)


cv2.namedWindow('Normal')
cv2.createTrackbar('value','Normal',1,255,nothing)
cv2.moveWindow('Normal',100,100)
cv2.imshow('Normal',img)
cv2.waitKey(0)
cv2.destroyAllWindows()