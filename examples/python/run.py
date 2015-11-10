from anki import *

import sysv_ipc
import numpy as np
import cv2
from time import sleep,clock

RHO = "E6:D8:52:F1:D9:43"
BOSON = "D9:81:41:5C:D4:31"
KATAL = "D8:64:85:29:01:C0"
KOURAI = "EB:0D:D8:05:CA:1A"
NUKE = "C5:34:5D:26:BE:53"
HADION = "D4:48:49:03:98:95"

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

sleep(1.5)


ims = []
for i in range(3):
    # obtain three different images
    ims.append(cv2.cvtColor(np.frombuffer(S.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3),cv2.COLOR_BGR2RGB))
    sleep(0.5)

# compute the median
med = np.array(ims[0])
np.median(np.array(ims), 0, out=med)
cv2.imwrite('outmed.jpg', med)

old_x = 0
old_y = 0
oldold_x = 0
oldold_y = 0
expected_x = 0
expected_y = 0

# our official position
xpos = 0
ypos = 0

# xposition of the starting line, and y position determining upper half of track
startx = 1696/2
starty = 720/2

# number of laps before quitting
NUM_LAPS = 20
# laps already driven
laps = 0

t_start = time.clock()
t_lapstart = t_start
laptimes = []

# number of times we didn't find any contour
bad = 0


while (True):
    car.set_speed(1200, 5000)
    # read image from camera
    cur_im = np.frombuffer(S.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3)
    cur_im = cv2.cvtColor(cur_im,cv2.COLOR_BGR2RGB)
    # subtract median images
    im = cv2.subtract(cur_im, med)
    #cv2.imwrite('out%s.jpg' % i, im)
    # convert to greyscale, blur and convert to B/W with threshold
    # TODO: adaptive threshold value depending on lighting
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray,(15,15),0)
    # imgray = cv2.medianBlur(imgray,5)
    ret,thresh = cv2.threshold(imgray,96,255,0)
    # compute contours. ret is an error value
    ret, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imwrite('outg%s.jpg' % i, imgray)
    #cv2.imwrite('outt%s.jpg' % i, thresh)
    #cv2.drawContours(im,contours,-1,(0,255,0),2)
    #cv2.imwrite('outc%s.jpg' % i, im)

    oldold_x = old_x
    oldold_y = old_y
    old_x = xpos
    old_y = ypos
    # TODO: take elapsed time into account and scale vector accordingly
    expected_x = old_x + (old_x - oldold_x)
    expected_y = old_y + (old_y - oldold_y)
    print ("Expected position: %4d, %4d" % (expected_x, expected_y))

    if hierarchy == None:
        # no contours found at all
        print("No car found")
        cv2.imwrite('out%s.jpg' % bad, im)
        bad += 1
        # use expected position if no car found
        xpos = expected_x
        ypos = expected_y
    else:
        # count cars with non-trivial homology
        homology_cars = 0
        # currently records centroids for contours with children, currently unused
        centroids = []
        for i in range(len(hierarchy[0])):
            if hierarchy[0][i][2] >= 0:
                # contour has a child:
                M = cv2.moments(contours[i])
                xpos = int(M['m10']/M['m00'])
                ypos = int(M['m01']/M['m00'])
                print("Actual position:   %4d, %4d" % (xpos, ypos))
                centroids.append((xpos, ypos))
                homology_cars += 1
        if homology_cars >= 2:
            # TODO: What happens if we find two cars?
            pass
        if homology_cars == 0:
            # TODO: find a good blob that might be our car instead of dead reckoning
            # TODO: use expected position and maybe convexity defects for this
            # use expected position if no homology car found
            xpos = expected_x
            ypos = expected_y
    #print hierarchy
    #cv2.imwrite('out%s.jpg' % i, im)
    #sleep(0.1)

    t_end = time.clock()
    dt = t_end - t_start
    print("Elapsed time:    %f seconds\n" % dt)
    t_start = t_end

    if ypos < starty and xpos >= startx and old_x < startx:
        #interpolate linearly to determine the actual time when crossing the line
        dx = xpos - old_x
        v = float(dx)/dt
        x_excess = xpos - startx
        t_excess = x_excess / v
        t_adjusted = t_end - t_excess
        laptime = t_adjusted - t_lapstart

        # in the top half of the track and just traversed from the left half to the right half
        print("Lap complete!    %f seconds\n" % laptime)
        if laps != 0:
            # don't record the incomplete lap
            laptimes.append(laptime)
        t_lapstart = t_adjusted
        laps += 1
        if laps == NUM_LAPS:
            # stop after 20 laps
            break


print(laptimes)

# finally, detach from memory again
S.detach()
res = car.stop()
sleep(1)
del car
