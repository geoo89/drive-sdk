from anki import *

import sysv_ipc
import numpy as np
import cv2
from time import sleep

RHO = "E6:D8:52:F1:D9:43"
BOSON = "D9:81:41:5C:D4:31"
KATAL = "D8:64:85:29:01:C0"
KOURAI = "EB:0D:D8:05:CA:1A"

car = Car(KOURAI)
if not car:
    print "Couldn't create Car object"
    raise SystemExit

err = car.connect()
if err:
    print "Couldn't connect, code", err
    raise SystemExit

status = car.set_speed(1200, 5000)
if status:
    print "Couldn't set speed, code",  status
    raise SystemExit

# attach to shared memory with key 192012003
S=sysv_ipc.SharedMemory(key=192012003)

ims = []
for i in range(3):
    # read contents into numpy array
    ims.append(np.frombuffer(S.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3))
    sleep(0.5)

med = np.array(ims[0])
np.median(np.array(ims), 0, out=med)
cv2.imwrite('outmed.jpg', med)

#while(True): 
for _ in range(100):
    car.set_speed(1200, 5000)
    # read contents into numpy array
    cur_im = np.frombuffer(S.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3)
    cur_im = cv2.cvtColor(cur_im,cv2.COLOR_BGR2RGB)
    im = cv2.subtract(cur_im, med)
    #cv2.imwrite('out%s.jpg' % i, im)
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray,(15,15),0)
    # imgray = cv2.medianBlur(imgray,5)
    ret,thresh = cv2.threshold(imgray,96,255,0)
    ret, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imwrite('outg%s.jpg' % i, imgray)
    #cv2.imwrite('outt%s.jpg' % i, thresh)
    #cv2.drawContours(im,contours,-1,(0,255,0),2)
    #cv2.imwrite('outc%s.jpg' % i, im)
    for i in range(len(hierarchy[0])):
        # has a child:
        if hierarchy[0][i][2] >= 0:
            M = cv2.moments(contours[i])
            centroid_x = int(M['m10']/M['m00'])
            centroid_y = int(M['m01']/M['m00'])
            print (centroid_x, centroid_y)
    #print hierarchy
    #cv2.imwrite('out%s.jpg' % i, im)
    sleep(0.2)


# finally, detach from memory again
S.detach()
res = car.stop()
sleep(1)
del car
