#!/usr/bin/python
import sysv_ipc
import numpy as np
import pylab
import cv2
from time import sleep

# attach to shared memory with key 192012003
S=sysv_ipc.SharedMemory(key=192012003)

im1 = None
for i in range(10):
    # read contents into numpy array
    im2 = np.frombuffer(S.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3)
    # im = cv2.imread('sample_images/donut.png')
    # im1 = cv2.imread('sample_images/04.ppm')
    # im2 = cv2.imread('sample_images/08.ppm')
    if im1 != None:
        im = cv2.subtract(im2, im1)
        # cv2.imwrite('out%s.jpg' % i, im)
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        imgray = cv2.GaussianBlur(imgray,(19,19),0)
        # imgray = cv2.medianBlur(imgray,5)
        ret,thresh = cv2.threshold(imgray,32,255,0)
        ret, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	#print contours
        #cv2.imwrite('outg%s.jpg' % i, imgray)
        #cv2.imwrite('outt%s.jpg' % i, thresh)
        cv2.drawContours(im,contours,-1,(0,255,0),3)
        #cv2.imwrite('outc%s.jpg' % i, im)
        M = cv2.moments(contours[0])
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        print (centroid_x, centroid_y)
        # print hierarchy
        cv2.imwrite('out%s.jpg' % i, im)

    im1 = im2
    sleep(1)

# finally, detach from memory again
S.detach()
