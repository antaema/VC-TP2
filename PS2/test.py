from __future__ import division
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import sys
sys.setrecursionlimit(10000)


def normalise(v):
    norm=np.linalg.norm(v)
    
    if norm == 0: 
        return v

    v = v / norm
    return v
    
def calcPoints(plano):
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
                if np.dot(points[i], axis1) == 0 and  any(points[i] != centroid):
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
            x = round(new[0]) + deslocamento
            y = round(new[1]) + deslocamento
            img[x,y] = points[i]
    return img

fig, ax = plt.subplots()
plt.axis("off")
axvalue = plt.axes([0.13, 0.1, 0.8, 0.03],)
svalue = Slider(axvalue, 'Value', 1, 255, valinit=0, valfmt='%0.0f')
img = np.zeros((800,800,3), np.uint8)
value = 0

def update(val):
    value = round(svalue.val)
    plano = [1, 1, 1, 3*value ] #eq  plano ax + by + cz = d -> d = u + u + u -> d = 3u
    xs,ys,zs,points = calcPoints(plano)
    centroid = np.array([np.mean(xs),np.mean(ys),np.mean(zs)])
    print centroid
    img = generate_image(np.asarray(points),centroid)
    plt.figure(2)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    fig.canvas.draw_idle()


svalue.on_changed(update)
plt.show()