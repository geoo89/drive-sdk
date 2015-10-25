import numpy as np
import cv2
 
im1 = cv2.imread('sample_images/04.ppm')
im2 = cv2.imread('sample_images/08.ppm')
im = cv2.subtract(im2, im1)
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
imgray = cv2.GaussianBlur(imgray,(19,19),0)
ret,thresh = cv2.threshold(imgray,32,255,0)
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im,contours,-1,(0,255,0),3)
print len(contours)
cv2.imwrite('out.jpg', im)