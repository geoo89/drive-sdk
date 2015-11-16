from anki import *
from landmarks import *
from landmarks25 import *

import sysv_ipc
import numpy as np
import cv2
from time import sleep,time

# mode
mode = 0

BRIGHTNESS_THRESHOLD = 96

# angle between two vectors, output in [0, pi]
def angle(a, b = (1,0)):
    return np.arccos(np.dot(a,b) / np.linalg.norm(a) * np.linalg.norm(b))

def dprint(text):
    pass
    #print(text)

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


status = car.set_speed(900, 5000)
if status:
    print "Couldn't set speed, code",  status
    raise SystemExit


ims = []
for i in range(3):
    # obtain three different images
    ims.append(cv2.cvtColor(np.frombuffer(S.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3),cv2.COLOR_BGR2RGB))
    sleep(0.5)

MAX_IMAGES = 20

# compute the median
med = np.array(ims[0])
np.median(np.array(ims), 0, out=med)

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
NUM_VELS = 30
velocities = [(0,0)]*NUM_VELS
positions = [(0,0)]*NUM_VELS
# 0 is straight, 1 is curvy
is_curvy = [0]*NUM_VELS

# each landmark is a list of positions, in each list:
# first position is where track changes from curvy to straight or vice versa
# next are the positions visited before reaching the landmark
landmarks = []
certain_landmarks = []

# time in frames to look back to take the position as landmark position
# when we found a curvature change.
LPOS_INDEX = 5
# length of a landmark list
LANDMARK_LEN = NUM_VELS-LPOS_INDEX

# number of times we didn't find any contour
bad = 0

car.change_lane(-100, 100, -1000)

while (True):
    #car.set_speed(1200, 5000)

    # read image from camera
    cur_im = np.frombuffer(S.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3)
    cur_im = cv2.cvtColor(cur_im,cv2.COLOR_BGR2RGB)

    # subtract median image
    im = cv2.subtract(cur_im, med)
    # convert to greyscale, blur and convert to B/W with threshold
    # TODO: adaptive threshold value depending on lighting
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    imgray = cv2.GaussianBlur(imgray,(15,15),0)
    # imgray = cv2.medianBlur(imgray,5)
    ret,thresh = cv2.threshold(imgray,BRIGHTNESS_THRESHOLD,255,0)

    # compute contours. ret is an error value
    ret, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #cv2.imwrite('outg%s.jpg' % i, imgray)
    #cv2.imwrite('outt%s.jpg' % i, thresh)
    # Draw contours. Image is only saved if no suitable car found
    cv2.drawContours(im,contours,-1,(0,255,0),2)




    # compute expected position
    # TODO: take elapsed time into account and scale vector accordingly?
    # (improvement through that should be negligible)
    oldold_x = old_x
    oldold_y = old_y
    old_x = xpos
    old_y = ypos
    expected_x = old_x + (old_x - oldold_x)
    expected_y = old_y + (old_y - oldold_y)
    dprint ("Expected position: %4d, %4d" % (expected_x, expected_y))

    if hierarchy == None:
        # no contours found at all
        print("No car found, using prediction")
        cv2.imwrite('out%s.jpg' % bad, im)
        bad = (bad + 1) % MAX_IMAGES
        # use expected position if no car found
        xpos = expected_x
        ypos = expected_y
    else:
        # count cars with non-trivial homology
        homology_cars = 0
        # currently records centroids for contours with children, currently unused
        hom_centroids = []
        bad_centroids = []
        for i in range(len(hierarchy[0])):
            M = cv2.moments(contours[i])
            if M['m00'] != 0:
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])
                    if hierarchy[0][i][2] >= 0:
                        # contour has a child:
                        dprint("Actual position:   %4d, %4d" % (xpos, ypos))
                        hom_centroids.append((x, y))
                        homology_cars += 1
                        xpos = x
                        ypos = y
                    else:
                        bad_centroids.append((x, y))
                
        if homology_cars >= 2:
            # TODO: What happens if we find two cars?
            # This is rare, but take the one clostest to expect position?
            cv2.imwrite('out%s.jpg' % bad, im)
            bad = (bad + 1) % MAX_IMAGES
            
        if homology_cars == 0:
            # No homology car found
            dprint("No homology car found")
            # use expected position if no homology car found
            xpos = expected_x
            ypos = expected_y
            # find a good blob that might be our car instead of dead reckoning
            # using the one closest to expected position
            # TODO: maybe convexity defects or othre measures too?
            found = False
            # but if it's further off than 150 pixels from the prediction, don't use it
            if (xpos,ypos) == (0,0):
                best_dist = 9999
            else:
                best_dist = 150
            for c in bad_centroids:
                # distance from expected position
                dist = np.linalg.norm(np.array((c[0],c[1]))-np.array((expected_x, expected_y)))
                if dist < best_dist:
                    # take this one if it's close than all previous ones
                    xpos = c[0]
                    ypos = c[1]
                    best_dist = dist
                    found = True
            if found == False:
                print("No sensible alternative found, using prediction")
                cv2.imwrite('out%s.jpg' % bad, im)
                bad = (bad + 1) % MAX_IMAGES

    # now xpos and ypos are the official positions
    # and we can reasonably trust them to be correct

    velocities = [(xpos - old_x, ypos - old_y)] + velocities[0:NUM_VELS-1]
    positions = [(xpos, ypos)] + positions[0:NUM_VELS-1]

    if mode == 1:
    	# average the last two vectors, and the previous two vectors
    	# for each of them compute the angle of the resulting vector
    	# (we take two each to filter out noise)
    	angle_new = angle(np.add(velocities[1], velocities[0]))
    	angle_old = angle(np.add(velocities[3], velocities[2]))
    	# compute the difference between the angles, centered around 0
    	diff = (angle_new - angle_old + np.pi) % (2*np.pi) - np.pi

    	if np.abs(diff) < 0.15:
            # straight
            is_curvy = [0] + is_curvy[0:NUM_VELS-1]
    	else:
       	    # curvy
            is_curvy = [1] + is_curvy[0:NUM_VELS-1]

        # output angle change, curviness, position and number of bad 
        print((diff, is_curvy[0], positions[0], bad))

        # change from curvy to straight or vice versa
        if (sum(is_curvy[0:3]) >= 2 and sum(is_curvy[3:6]) <= 1)\
            or (sum(is_curvy[0:3]) <= 1 and sum(is_curvy[3:6]) >= 2):
            landmarks.append(positions[LPOS_INDEX:NUM_VELS])
        # change from curvy to straight or vice versa, strict criterion
        if (sum(is_curvy[0:3]) >= 3 and sum(is_curvy[3:6]) <= 0)\
            or (sum(is_curvy[0:3]) <= 0 and sum(is_curvy[3:6]) >= 3):
            certain_landmarks.append(positions[LPOS_INDEX:NUM_VELS])

    if mode == 0:
        if (positions[0][0] >= landmarks_data[2][0][0] and positions[1][0] <= landmarks_data[2][0][0]):
            car.set_speed(2700, 50000)    
        if (positions[0][0] >= landmarks_data[1][18][0] and positions[1][0] <= landmarks_data[1][18][0]):
            car.set_speed(1220, 50000)
        if (positions[0][0] <= landmarks_data[0][0][0] and positions[1][0] >= landmarks_data[0][0][0]):
            car.set_speed(1900, 50000)    
        if (positions[0][0] <= landmarks_data[3][24][0] and positions[1][0] >= landmarks_data[3][24][0]):
            car.set_speed(1220, 50000)
    
    #print(positions[0])
    
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
    
    


    sleep(0.03333)



print(laptimes)
#print(landmarks)
#print(certain_landmarks)
landmarks_final = calc_landmarks(landmarks, certain_landmarks)
print(landmarks_final)

# draw landmarks on the median image and save it.
for c in landmarks:
    cv2.circle(med, (int(c[0][0]), int(c[0][1])), 5, (255,0,127), -1)
for c in certain_landmarks:
    cv2.circle(med, (int(c[0][0]), int(c[0][1])), 5, (255,0,255), -1)
for c in landmarks_final:
    cv2.circle(med, (int(c[0][0]), int(c[0][1])), 5,   (0,0,255), -1)
cv2.imwrite('outmed.jpg', med)


# finally, detach from memory again, stop car, quit
S.detach()
res = car.stop()
sleep(1)
del car
