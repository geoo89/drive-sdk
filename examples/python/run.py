from anki import *
from landmarks import *
from landmarks25 import *
from landmarks_inv import *

import sysv_ipc
import numpy as np
import cv2
from time import sleep,time
import pickle

# modes -- would be much cleaner to use subclasses here...
DEMONSTRATION_SIMPLE = 0    # the normal track
COMPUTE_LANDMARKS = 1
LEARN = 2
COMPUTE_CURVATURE = 3
GET_MEDIAN_IMAGE = 4
EXPLOIT = 5

MODE = 2

RHO    = "E6:D8:52:F1:D9:43"    # red
BOSON  = "D9:81:41:5C:D4:31"    # blue?
KATAL  = "D8:64:85:29:01:C0"    # grey?
KOURAI = "EB:0D:D8:05:CA:1A"    # yellow
NUKE   = "C5:34:5D:26:BE:53"    # green
HADION = "D4:48:49:03:98:95"    # orange

CAR_NAME = KOURAI

# number of laps before quitting
NUM_LAPS = 100

BRIGHTNESS_THRESHOLD = 96

MAX_IMAGES = 20

#velocities for the last few frames, first one is newest
LEN_TRACEBACK = 30

# global variables
positions = [(0,0)]*LEN_TRACEBACK
med = None
sh_mem = None
car = None
# laps already driven
laps = 0
t_start = None
t_lapstart = None
laptimes = []
# number of times we didn't find any contour
bad = 0
nocommand_timer = 0
framecounter = 0

if MODE == COMPUTE_LANDMARKS:
    # each landmark is a list of positions, in each list:
    # first position is where track changes from curvy to straight or vice versa
    # next are the positions visited before reaching the landmark
    landmarks = []
    certain_landmarks = []
    # 0 is straight, 1 is curvy
    is_curvy = [0]*LEN_TRACEBACK

    # time in frames to look back to take the position as landmark position
    # when we found a curvature change.
    LPOS_INDEX = 5
    # length of a landmark list
    LANDMARK_LEN = LEN_TRACEBACK-LPOS_INDEX


if MODE == COMPUTE_CURVATURE:
    curv_list = []
    #curv_data = [] # TODO: initialize


if MODE == LEARN or MODE == EXPLOIT:
    with open('curves.dat', 'rb') as f:
        curv_data = pickle.load(f)

    # initial parameters
    NPARAMS = 4
    # -1600 = 2*800, maximum curvature is around 0.5
    params = np.array((0.167, 1.0/5000, -1600, 1600))
    params_base = params
    # maximum perturbation within a parameter
    param_ranges = np.array((0.005, 1.0/50000, 50, 120))
    temp = 1.0
    vcur = 900 # TODO

    MIN_DATA = 15
    # contains pairs:
    #   first element is the set of 4 parameters
    #   second element is the time taken with this policy
    # TODO: read to and from file
    data = []
    
    # position after the relevant curvature change that warranted
    # the previous decision. Until there don't make any new decisions
    nextpos = (0,0)
    index_dist = 0
    # don't make any decisions right now
    no_decision = False
    dnum = 0

# 25 pixel per frame at 30 fps
# --> speed 900 = 25*30 (750) pixels/second


# angle between two vectors, output in [0, pi]
def angle(a, b = (1,0)):
    return np.arccos(np.dot(a,b) / np.linalg.norm(a) * np.linalg.norm(b))


def dprint(text):
    pass
    #print(text)


def initialize(car_name):
    global car
    car = Car(car_name)
    if not car:
        print "Couldn't create Car object"
        raise SystemExit

    err = car.connect()
    if err:
        print "Couldn't connect, code", err
        raise SystemExit

    # attach to shared memory with key 192012003
    global sh_mem
    sh_mem = sysv_ipc.SharedMemory(key=192012003)

    sleep(0.5)
    # TODO: insert start upon certain time

    speed = 900
    if MODE == DEMONSTRATION_SIMPLE or MODE == LEARN:
        speed = 1220
    status = car.set_speed(speed, 5000)
    if status:
        print "Couldn't set speed, code",  status
        raise SystemExit

    global t_start
    t_start = time()
    global t_lapstart
    t_lapstart = t_start

    #car.change_lane(-100, 100, -1000)


def deinitialize():
    print(laptimes)
    global car

    # finally, detach from memory again, stop car, quit
    sh_mem.detach()
    res = car.stop()
    sleep(1)
    del car


def get_median_image():
    sleep(0.5)
    ims = []
    for i in range(3):
        # obtain three different images
        ims.append(cv2.cvtColor(np.frombuffer(sh_mem.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3),cv2.COLOR_BGR2RGB))
        sleep(0.5)

    # compute the median
    global med
    med = np.array(ims[0])
    np.median(np.array(ims), 0, out=med)
    cv2.imwrite('median.jpg', med)


def get_expected_pos():
    # compute expected position
    # TODO: take elapsed time into account and scale vector accordingly?
    # (improvement through that should be negligible)
    oldold_x, oldold_y = positions[1]
    old_x, old_y = positions[0]
    # could also use tuples and np.add/sub here
    expected_x = old_x + (old_x - oldold_x)
    expected_y = old_y + (old_y - oldold_y)
    dprint ("Expected position: %4d, %4d" % (expected_x, expected_y))
    return expected_x, expected_y


def position_from_camera(expected_x, expected_y):
    # read image from camera
    cur_im = np.frombuffer(sh_mem.read(1696*720*3),dtype=np.uint8).reshape(720,1696,3)
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

    global bad
    if hierarchy == None:
        # no contours found at all
        print("No car found, using prediction")
        # cv2.imwrite('out%s.jpg' % bad, im)
        bad = (bad + 1) % MAX_IMAGES
        # use expected position if no car found
        xpos = expected_x
        ypos = expected_y
    else:
        # count cars with non-trivial homology
        homology_cars = 0
        # centroids for contours with children
        hom_centroids = []
        # centroids for contours without children
        bad_centroids = []
        for i in range(len(hierarchy[0])):
            M = cv2.moments(contours[i])
            if M['m00'] != 0:
                    # compute the centroid
                    x = int(M['m10']/M['m00'])
                    y = int(M['m01']/M['m00'])
                    if hierarchy[0][i][2] >= 0:
                        # contour has a child:
                        dprint("Actual position:   %4d, %4d" % (x, y))
                        hom_centroids.append((x, y))
                        homology_cars += 1
                        # tentatively set this as our new position
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

    return xpos, ypos


def get_laptime():
    # xposition of the starting line, and y position determining upper half of track
    STARTX = 1696/2
    STARTY = 720/2

    xpos  = positions[0][0]
    ypos  = positions[0][1]
    old_x = positions[1][0]

    t_end = time()
    dt = t_end - t_start
    dprint("Elapsed time:    %f seconds\n" % dt)
    global t_start
    t_start = t_end
    global t_lapstart
    global laps

    if ypos < STARTY and xpos <= STARTX and old_x > STARTX:
        #interpolate linearly to determine the actual time when crossing the line
        dx = xpos - old_x
        v = float(dx)/dt
        x_excess = -(xpos - STARTX)
        t_excess = x_excess / v
        t_adjusted = t_end - t_excess
        laptime = t_adjusted - t_lapstart       

        # in the top half of the track and just traversed from the left half to the right half
        print("Lap complete!    %f seconds\n" % laptime)
        t_lapstart = t_adjusted
        laps += 1
        if laps > 1:
            # don't record the incomplete lap
            return laptime

    return None





def compute_landmarks():
    # average the last two vectors, and the previous two vectors
    # for each of them compute the angle of the resulting vector
    # (we take two each to filter out noise)
    angle_new = angle(np.subtract(positions[2], positions[0]))
    angle_old = angle(np.subtract(positions[4], positions[2]))
    # compute the difference between the angles, centered around 0
    diff = (angle_new - angle_old + np.pi) % (2*np.pi) - np.pi

    global is_curvy
    if np.abs(diff) < 0.15:
        # straight
        is_curvy = [0] + is_curvy[0:LEN_TRACEBACK-1]
    else:
        # curvy
        is_curvy = [1] + is_curvy[0:LEN_TRACEBACK-1]

    # output angle change, curviness, position and number of bad 
    print((diff, is_curvy[0], positions[0], bad))

    # change from curvy to straight or vice versa
    if (sum(is_curvy[0:3]) >= 2 and sum(is_curvy[3:6]) <= 1)\
        or (sum(is_curvy[0:3]) <= 1 and sum(is_curvy[3:6]) >= 2):
        landmarks.append(positions[LPOS_INDEX:LEN_TRACEBACK])
    # change from curvy to straight or vice versa, strict criterion
    if (sum(is_curvy[0:3]) >= 3 and sum(is_curvy[3:6]) <= 0)\
        or (sum(is_curvy[0:3]) <= 0 and sum(is_curvy[3:6]) >= 3):
        certain_landmarks.append(positions[LPOS_INDEX:LEN_TRACEBACK])    


def postprocess_landmarks():
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






def do_demonstration_simple_cw():
    if (positions[0][0] >= landmarks_data[2][0][0] and positions[1][0] <= landmarks_data[2][0][0]):
        car.set_speed(2700, 50000)
    if (positions[0][0] >= landmarks_data[1][18][0] and positions[1][0] <= landmarks_data[1][18][0]):
        car.set_speed(1220, 50000)
    if (positions[0][0] <= landmarks_data[0][0][0] and positions[1][0] >= landmarks_data[0][0][0]):
        car.set_speed(1900, 50000)
    if (positions[0][0] <= landmarks_data[3][24][0] and positions[1][0] >= landmarks_data[3][24][0]):
        car.set_speed(1220, 50000)


def do_demonstration_simple_ccw():
    if positions[1] != (0, 0):
        if (positions[0][0] <= landmarks_data_inv[2][0][0] and positions[1][0] >= landmarks_data_inv[2][0][0]):
            car.set_speed(2700, 5000)
        if (positions[0][0] <= landmarks_data_inv[3][18][0] and positions[1][0] >= landmarks_data_inv[3][18][0]):
            car.set_speed(1220, 5000)
        if (positions[0][0] >= landmarks_data_inv[1][0][0] and positions[1][0] <= landmarks_data_inv[1][0][0]):
            car.set_speed(1900, 5000)
        if (positions[0][0] >= landmarks_data_inv[0][13][0] and positions[1][0] <= landmarks_data_inv[0][13][0]):
            car.set_speed(1220, 5000)




def compute_curvature():
    # average the last two vectors, and the previous two vectors
    # for each of them compute the angle of the resulting vector
    # (we take two each to filter out noise)
    angle_new = angle(np.subtract(positions[2], positions[0]))
    angle_old = angle(np.subtract(positions[4], positions[2]))
    # compute the difference between the angles, centered around 0
    diff = (angle_new - angle_old + np.pi) % (2*np.pi) - np.pi

    curv_list.append((positions[0], diff))


def postprocess_curvature():
    # TODO: smoothen these values

    maxc = 0
    minc = 0

    i = 100
    startpos = np.array(curv_list[90][0])
    pos = np.array(curv_list[i][0])
    
    # we want to get one full loop of curvature data
    while np.linalg.norm(pos - startpos) > 20:
        maxc = max(maxc, curv_list[i][1])
        minc = min(minc, curv_list[i][1])
        i += 1
        pos = np.array(curv_list[i][0])
    relevant_curv = curv_list[90:i+1]
    # TODO: write this to a file
    print(relevant_curv)
    print(maxc, minc)

    maxabsc = max(abs(maxc), abs(minc))

    # draw landmarks on the median image and save it.
    for c in relevant_curv:
        curv = c[1]/maxabsc
        cv2.circle(med, (int(c[0][0]), int(c[0][1])), 5, (255+min(0,255*curv),0,255-max(0,255*curv), -1))

    cv2.imwrite('outcurv.jpg', med)


    with open('curves.dat', 'wb') as f:
        pickle.dump(relevant_curv, f)




def check_for_uturn():
    global nocommand_timer
    global no_decision
    if nocommand_timer != 0 or framecounter < 15:
        return
    
    minindex = 0
    mindist = 999999.0
    for i in range(len(curv_data)):
        dist = np.linalg.norm(curv_data[i][0] - np.array(positions[0]))
        if dist < mindist:
            mindist = dist
            minindex = i

    minindex2 = 0
    mindist2 = 999999.0
    for i in range(len(curv_data)):
        dist = np.linalg.norm(curv_data[i][0] - np.array(positions[12]))
        if dist < mindist2:
            mindist2 = dist
            minindex2 = i
    
    if (minindex - minindex2) % len(curv_data) > len(curv_data) // 2:
        car.uturn()
        print("uturn issued")
        nocommand_timer = 30
        no_decision = False
        
    





def get_next_curvature_change():
    global nextpos
    global index_dist
    cpos = np.array(positions[0])

    idx = 0
    for i in range(len(curv_data)):
        pos = np.array(curv_data[i][0])
        dist = np.linalg.norm(cpos - pos)
        if dist < 50:
            idx = i
            break

    ccurv = curv_data[idx][1]

    i = idx
    while True:
        curv = curv_data[i][1]
        if np.abs(ccurv - curv) > 0.2: # TODO: find constant
            while True:
                i = (i + 1) % len(curv_data)
                # still upwards trend in curvature
                if np.abs(curv - curv_data[i][1]) > 0.07:
                    curv = curv_data[i][1]
                else:
                    i -= 1
                    break
            pos = np.array(curv_data[i][0])
            dist = np.linalg.norm(cpos - pos)
            # compute a position soon after the relevant curvature change
            nextposid = (i+3)%len(curv_data)
            nextpos = np.array(curv_data[nextposid][0])
            index_dist = abs(idx - nextposid)
            # print(cpos, pos, nextpos, dist, ccurv, curv)
            return dist, ccurv, curv
        i = (i + 1) % len(curv_data)
        if i == idx:
            # TODO: error handling
            return None


def follow_policy():
    global vcur
    global no_decision
    global nocommand_timer
    global dnum
    xpos  = positions[0][0]
    ypos  = positions[0][1]
    pos = np.array((xpos, ypos))

    #if np.linalg.norm(nextpos - pos) < 20:
    #    no_decision = False
    
    if not no_decision and nocommand_timer == 0:
    
        dist, old_curv, new_curv = get_next_curvature_change()

        #vcur = TODO: compute here

        # definition of the actual policy
        vnew = int(params[2] * abs(new_curv) + params[3])
        if (1.2*dist / vcur) < params[0] + max(0, params[1]*(vcur - vnew)):
            # TODO: figure out the acceleration
            #no_decision = True
            car.set_speed(vnew, 50000)
            nocommand_timer = max(8, int(index_dist * 900.0 / vnew + 0.5))
            
            # debug stuff
            #print("Decision %d. oldcurv: %f, newcurv: %f" % (dnum, old_curv, new_curv))
            #print("(x,y): (%f,%f); vnew: %f; vcur: %f; dist: %f" % (xpos, ypos, vnew, vcur, dist))
            #print("1.2*d/v = %f < %f = p0 + max(0, p1*(vn-vc))\n" % (1.2*dist / vcur, params[0] + max(0, params[1]*(vcur - vnew))))
            #cv2.circle(med, (int(xpos), int(ypos)), 5, (255,0,255,-1))
            #cv2.imwrite('decision%d.jpg' % dnum, med)
            dnum += 1

            vcur = vnew


def update_policy(laptime):
    global params
    global params_base
    global data
    # NOTE: instead of every lap, we could only update every few laps

    data.append((params, laptime))
    print((params, laptime))
    
    # if we don't have enough data yet
    if len(data) < MIN_DATA:
        # just perturb a little
        params = np.random.normal(params_base, param_ranges*temp, NPARAMS)
    else:
        # do the linear regression here (explore)
        print(data)
        X = np.array([[1.0] + list(data[i][0]) for i in range(len(data))])
        y = np.array([data[i][1] for i in range(len(data))])
        print(X)
        print(y)
        a, stuff1, stuff2, stuff3 = np.linalg.lstsq(X, y)
        print(a)
        u0 = np.array([a[1], a[2], a[3], a[4], -1]) # normal vector
        u1 = np.array([-a[2], a[1], 0, 0, 0])
        u2 = np.array([0, -a[3], a[2], 0, 0])
        u3 = np.array([0, 0, -a[4], a[3], 0])
        u4 = np.array([0, 0, 0,     1, a[4]])
        
        A = np.transpose(np.array([u0, u1, u2, u3, u4]))
        q, r = np.linalg.qr(A)
        print(np.transpose(q))
        d = np.transpose(q)[4]
        print("reality check:")
        print(d)
        print(a[1] * d[0] +  a[2] * d[1] + a[3] * d[2] + a[4] * d[3] - d[4], -a[0])
        
        # now scale this guy proportional to its last component which is the time gain
        # achieved if we just add the normalized vector. 
        # So lower time = less gain = we want to move less
        dp = d[0:4]*(-1000*d[4])
        params_base += dp
        data = []
        print("Done linear regression, new params: " + str(params_base))
        
        






if __name__ == "__main__":

    initialize(CAR_NAME)
    
    if MODE == GET_MEDIAN_IMAGE:
        get_median_image()
    else:
        global med
        global framecounter
        med = cv2.imread('median.jpg')

        while (True):
            ex, ey = get_expected_pos()
            xpos, ypos = position_from_camera(ex, ey)

            # now xpos and ypos are the official positions
            # and we can reasonably trust them to be correct
            positions = [(xpos, ypos)] + positions[0:LEN_TRACEBACK-1]

            if MODE == COMPUTE_LANDMARKS:
                compute_landmarks()
            if MODE == DEMONSTRATION_SIMPLE:
                do_demonstration_simple_ccw()
            if MODE == LEARN or MODE == EXPLOIT:
                follow_policy()
                check_for_uturn()
            if MODE == COMPUTE_CURVATURE:
                compute_curvature()

            laptime = get_laptime()
            if laptime != None:
                laptimes.append(laptime)
                if MODE == LEARN:
                    update_policy(laptime)
            if laps == NUM_LAPS:
                break
            
            sleep(0.03333)
            if nocommand_timer != 0:
                nocommand_timer -= 1
            framecounter += 1

        if MODE == COMPUTE_LANDMARKS:
            postprocess_landmarks()
        if MODE == COMPUTE_CURVATURE:
            postprocess_curvature()

    deinitialize()
