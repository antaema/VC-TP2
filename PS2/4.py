from __future__ import division
import cv2
import numpy as np
import time
import sys
sys.setrecursionlimit(10000)

def nothing(x):
    pass

def normalise(v):
    norm=np.linalg.norm(v)
    
    if norm == 0: 
        return v

    v = v / norm
    return v
    
def calcPoints(plano):
    global value,normal, centroid
    xs = []
    ys = []
    zs = []
    points = []
    for z in range(0,256):
        for y in range (0,256):
            for x in range (0,256):
                if x + y  + z == plano[3]:
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)
                    points.append(np.array([ z,y,x]))
    return xs,ys,zs,points

def matmult(a,b):
    zip_b = zip(*b)
    # uncomment next line if python 3 : 
    # zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]

def generate_image(points,centroid):
    img = np.zeros((800,800,3), np.uint8)
    deslocamento = 400
    if len(points) > 1:
        find = True
        p = 1
        while find :
            axis1 = points[-p]
            axis1 = normalise( np.subtract(axis1,centroid))
            axis2 = [ 0, 0, 0]
            for i in range(0,len(points)):     
                axis2 = np.subtract(points[i],centroid) 
                if np.dot(axis2, axis1) == 0 and  any(points[i] != centroid):
                    axis2 = normalise( np.subtract(points[i],centroid))
                    find = False
                    break
            p += 1
        axis3 = normalise(np.cross(axis1,axis2))

        
        #axis3 = z
        matrix = np.array ([ axis1, 
                            axis2, 
                            axis3 
        ])
        
        for i in range(0,len(points)):
            new = np.matmul(matrix, points[i])
            x = int(round(new[0]) + deslocamento)
            y = int(round(new[1]) + deslocamento)
            img[x,y] = points[i]

    
    return img


img = np.zeros((800,800,3), np.uint8)
cv2.namedWindow('image')
cv2.createTrackbar('value','image',0,255,nothing)
cv2.moveWindow('image',100,100)
cv2.namedWindow('image2')
cv2.moveWindow('image2',950,100)
value = 0
oldv = 0  
normal = np.array([1, 1, 1])
normal = normalise(normal)
cv2.imshow('image',img)
cv2.imshow('image2',img)
while(1):
    #fecha com esc
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    value = cv2.getTrackbarPos('value','image')
    
    if oldv != value:
        oldv = value
        plano = [1, 1, 1, 3*value ] #eq  plano ax + by + cz = d -> d = u + u + u -> d = 3u
        xs,ys,zs,points = calcPoints(plano)
        centroid = np.array([np.mean(xs),np.mean(ys),np.mean(zs)])
        img = generate_image(np.asarray(points),centroid)
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img2)
        cv2.imshow('image',img)
        cv2.imshow('image2',s)
        

cv2.destroyAllWindows()