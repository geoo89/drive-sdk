#!/bin/python

import numpy as np
import cv2
 
im1 = cv2.imread('sample_images/04.ppm')
im2 = cv2.imread('sample_images/08.ppm')
im3 = cv2.imread('sample_images/12.ppm')
med = cv2.imread('sample_images/12.ppm')
# here we overwrite the array 'med' to maintain the data type uint instead of float
np.median(np.array([im1, im2, im3]), 0, out=med)
cv2.imwrite('outmed.jpg', med)
im = cv2.subtract(im2, med)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray = cv2.GaussianBlur(imgray,(19,19),0)
ret,thresh = cv2.threshold(imgray,32,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im,contours,-1,(0,255,0),3)
print len(contours)
print contours
print hierarchy
for i in range(len(hierarchy[0])):
    # has a child:
    if hierarchy[0][i][2] >= 0:
        M = cv2.moments(contours[i])
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        print (centroid_x, centroid_y)
cv2.imwrite('out.jpg', im)
