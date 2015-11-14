import numpy as np

def calc_landmarks(landmarks, certain_landmarks):

    lm = dict()

    for l in certain_landmarks:
        found = False
        for e in lm:
            dist = np.linalg.norm(np.array(e)-np.array(l))
            if dist < 200:
                found = True
                lm[e].append(l)
        if found == False:
            lm[l] = [l]


    # certain landmarks are also in here, so they weigh double
    for l in landmarks:
        found = False
        for e in lm:
            dist = np.linalg.norm(np.array(e)-np.array(l))
            if dist < 200:
                found = True
                lm[e].append(l)
        if found == False:
            lm[l] = [l]

    #print(lm)

    landmarks_final = []
    for k,v in lm.items():
        if len(v) > 5:
            avg = np.mean(v, axis=0)
            landmarks_final.append(tuple(avg))

    return landmarks_final


if __name__ == "__main__":
    landmarks = [(0, 0), (0, 0), (0, 0), (317, 243), (343, 238), (366, 238), (1295, 246), (1325, 246), (1359, 247), (1388, 483), (1366, 489), (1340, 489), (403, 473), (375, 480), (350, 484), (308, 247), (334, 240), (359, 238), (1324, 246), (1354, 247), (1378, 249), (1377, 487), (1349, 489), (1320, 485), (404, 473), (374, 480), (347, 484), (306, 248), (330, 240), (359, 237)]
    certain_landmarks = [(343, 238), (1325, 246), (1366, 489), (334, 240), (1354, 247), (330, 240)]
    lf = calc_landmarks(landmarks, certain_landmarks)
    print(lf)