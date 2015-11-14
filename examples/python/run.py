from anki import *

import sysv_ipc
import numpy as np
import cv2
from time import sleep,time

# angle between two vectors, output in [0, pi]
def angle(a, b = (1,0)):
    return np.arccos(np.dot(a,b) / np.linalg.norm(a) * np.linalg.norm(b))

def dprint(text):
    #pass
    print(text)

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

# attach to shared memory with key 192012003
S=sysv_ipc.SharedMemory(key=192012003)

sleep(1.5)



status = car.set_speed(1200, 5000)
if status:
    print "Couldn't set speed, code",  status
    raise SystemExit


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
NUM_LAPS = 10
# laps already driven
laps = 0

t_start = time()
t_lapstart = t_start
laptimes = []

#velocities for the last few frames, first one is newest
NUM_VELS = 10
velocities = [(0,0)]*NUM_VELS
positions = [(0,0)]*NUM_VELS
is_curvy = [False]*NUM_VELS

# positions where track changes from curvy to straight or vice versa
landmarks = []

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
d    print ("Expected position: %4d, %4d" % (expected_x, expected_y))

    if hierarchy == None:
        # no contours found at all
        dprint("No car found")
        cv2.imwrite('out%s.jpg' % bad, im)
        bad = (bad + 1) % MAX_IMAGES
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
                dprint("Actual position:   %4d, %4d" % (xpos, ypos))
                centroids.append((xpos, ypos))
                homology_cars += 1
        if homology_cars >= 2:
            cv2.imwrite('out%s.jpg' % bad, im)
            bad = (bad + 1) % MAX_IMAGES
            # TODO: What happens if we find two cars?
            pass
        if homology_cars == 0:
            # TODO: find a good blob that might be our car instead of dead reckoning
            # TODO: use expected position and maybe convexity defects for this
            cv2.imwrite('out%s.jpg' % bad, im)
            bad = (bad + 1) % MAX_IMAGES
            # use expected position if no homology car found
            xpos = expected_x
            ypos = expected_y
    #print hierarchy
    #cv2.imwrite('out%s.jpg' % i, im)
    #sleep(0.1)


    # now x_pos and y_pos are the official positions
    velocities = [(xpos - old_x, ypos - old_y)] + velocities[0:NUM_VELS-1]
    positions = [(xpos, ypos)] + positions[0:NUM_VELS-1]
    # average the last two vectors, and the previous two vectors
    # and for each of them compute the angle of the resulting vector
    angle_new = angle(np.add(velocities[1], velocities[0]))
    angle_old = angle(np.add(velocities[3], velocities[2]))
    # compute the difference between the angles, centered around 0
    diff = (angle_new - angle_old + np.pi) % (2*np.pi) - np.pi

    if np.abs(diff) < 0.05:
        is_curvy = [False] + is_curvy[0:NUM_VELS-1]
        # straight
    else:
        is_curvy = [True] + is_curvy[0:NUM_VELS-1]
        # curvy

    dprint(angle_new, angle_old, is_curvy[0], positions[0])

    if (is_curvy[0] and is_curvy[1] and not is_curvy[2] and not is_curvy[3])
        or (if not is_curvy[0] and not is_curvy[1] and is_curvy[2] and is_curvy[3]):
        landmarks.append(positions[2])


    t_end = time()
    dt = t_end - t_start
    dprint("Elapsed time:    %f seconds\n" % dt)
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
    
    sleep(0.1)


print(laptimes)

# finally, detach from memory again
S.detach()
res = car.stop()
sleep(1)
del car
