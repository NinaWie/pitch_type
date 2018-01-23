import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import json
%matplotlib inline

def from_json(file):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]
    with open(file, 'r') as inf:
        out = json.load(inf)

    liste = []
    for fr in out["frames"]:
        l_joints = []
        for j in joints_list[:12]:
            l_coo = []
            for xy in coordinates:
                l_coo.append(fr[j][xy])
            l_joints.append(l_coo)
        liste.append(l_joints)

    return np.array(liste)

class Node():
    def __init__(self, x1, y1, x2, y2):
        self.bbox = [x1, y1, x2, y2]
        l = abs(x1-x2)
        w = abs(y1-y2)
        self.area = l*w # (l+w)/float(l*w)
        self.center = np.array([(x1+x2)/2, (y1+y2)/2])
        self.children = []
        #self.area_diffs = []
        self.slopes = []
        self.dist = []
    def add_child(self, no):
        area_diff = abs(no.area-self.area)
        if area_diff<450:
            self.children.append(no)
            #self.area_diffs.append()
            self.slopes.append((self.bbox[1]-no.bbox[1])/(self.bbox[0]-no.bbox[0]))
            self.dist.append(np.linalg.norm(no.center-self.center))
    def favourite_child(self, no):
        self.fav_child=no

"""
# PREVIOUS VERSION
class Node():
    def __init__(self, x1, y1, x2, y2):
        self.bbox = [x1, y1, x2, y2]
        l = abs(x1-x2)
        w = abs(y1-y2)
        self.area = l*w # (l+w)/float(l*w)
        self.center = [(x1+x2)/2, (y1+y2)/2]
        self.children = []
    def add_child(self, no):
        self.children.append(no)
    def favourite_child(self, no):
        self.fav_child=no
"""



def get_candidates(im_tm1, im_t, im_tp1, min_area):
    delta_plus = cv2.absdiff(im_t, im_tm1)
    delta_0 = cv2.absdiff(im_tp1, im_tm1)
    delta_minus = cv2.absdiff(im_t,im_tp1)
    sp = cv2.meanStdDev(delta_plus)
    sm = cv2.meanStdDev(delta_minus)
    s0 = cv2.meanStdDev(delta_0)
    # print("E(d+):", sp, "\nE(d-):", sm, "\nE(d0):", s0)

    th = [
        sp[0][0, 0] + 3 * math.sqrt(sp[1][0, 0]),
        sm[0][0, 0] + 3 * math.sqrt(sm[1][0, 0]),
        s0[0][0, 0] + 3 * math.sqrt(s0[1][0, 0]),
    ]

    # OPENCV THRESHOLD

    ret, dbp = cv2.threshold(delta_plus, th[0], 255, cv2.THRESH_BINARY)
    ret, dbm = cv2.threshold(delta_minus, th[1], 255, cv2.THRESH_BINARY)
    ret, db0 = cv2.threshold(delta_0, th[2], 255, cv2.THRESH_BINARY)

    detect = cv2.bitwise_not(cv2.bitwise_and(cv2.bitwise_and(dbp, dbm),
                    cv2.bitwise_not(db0)))

    # CONNECTED BOX
    # The original `detect` image was suitable for display, but it is "inverted" and not suitable
    # for component detection; we need to invert it first.
    start = time.time()
    nd = cv2.bitwise_not(detect)
    # only stats is used, not num, labels, centroids
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(nd, ltype=cv2.CV_16U)
    # We set an arbitrary threshold to screen out smaller "components"
    # which may result simply from noise, or moving leaves, and other
    # elements not of interest.

    d = detect.copy()
    candidates = list()
    for stat in stats[1:]:
        area = stat[cv2.CC_STAT_AREA]
        if area < min_area:
            continue # Skip small objects (noise)

        lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
        rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])
        bottomLeftCornerOfText = (lt[0], lt[1] - 15)

        candidates.append((lt, rb, area))
    print("stelle")
    return candidates

def first_movement(cand_list, joints, ankle_move, fr):
    knees= joints[[7, 10],:]
    ankles = joints[[8,11],:]
    #print(knees, ankles, knees-ankles, np.mean(knees-ankles, axis=0))
    dist_ankle = np.linalg.norm(np.mean(knees-ankles, axis=0)) #//2
    #print("radius", dist_ankle)
    for k, cand in enumerate(cand_list):
        x1, y1 = cand[0]
        x2, y2 = cand[1]
        center = [(x1+x2)/2, (y1+y2)/2]
              #np.linalg.norm(cand.center - knees[0]),
            #np.linalg.norm(cand.center - knees[1]), np.linalg.norm(cand.center - ankles[1]),
                #                                     np.linalg.norm(cand.center - ankles[0]))
        #print(np.linalg.norm(cand.center - knees[0])<radius, np.linalg.norm(cand.center - knees[1])<radius)
        norms = np.array([np.linalg.norm(center - knees[0]), np.linalg.norm(center - knees[1]), np.linalg.norm(center - ankles[0]), np.linalg.norm(center - ankles[1])])
        #print("center", cand.center, "knees", knees[0],knees[1], "ankles", ankles, "norms", norms)
        if np.any(norms<dist_ankle):
            print("smaller radius", center)
            ankle_move.append(fr) #cand.center)
            break
    return ankle_move
        #if k==len(candidates_per_frame[-1])-1:
         #   ankle_move=[]
    #print(t, ankle_move)
def polyarea_bbox(bbox):
    x = bbox[:,0]
    y = bbox[:,1]
    # print("Pearson", stats.pearsonr(x,y))
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def poly_factor(bbox):
    x = bbox[:,0]
    y = bbox[:,1]
    # print("Pearson", stats.pearsonr(x,y))
    area = 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
    length = np.linalg.norm(bbox[0,:2]-bbox[2,:2])
    print(area, length)
    return length/(area+0.0000001)

def plot(im_t, candidates, frame_nr):
    #print("DETECTED", t-1, whiteness_values[-1], candidate_values[-1])
    plt.figure(figsize=(10, 10), edgecolor='r')
    # print(candidates[fom])
    img = np.tile(np.expand_dims(im_t.copy(), axis = 2), (1,1,3))
    #print(img.shape)
    #for jo in ankles:
     #   cv2.circle(img, (int(jo[0]), int(jo[1])), 8, [255,0,0], thickness=-1)
    #for kn in knees:
     #   cv2.circle(img, (int(kn[0]), int(kn[1])), 8, [255,0,0], thickness=-1)
    for can in candidates: # einfuegen falls alles plotten
        cv2.rectangle(img, can[0], can[1],[255,0,0], 4)
    #cv2.rectangle(img,tuple(balls[-1][:2]), tuple(balls[-1][2:]), [255,0,0], 4)
    plt.imshow(img, 'gray')
    plt.title("Detected FOM frame"+ str(frame_nr))
    plt.show()

def overlap(box1, box2):
    if box1[0]> box2[2] or box2[0]>box1[2] or box1[1]>box2[3] or box2[1]>box1[3]:
        return False
    else: return True

"""
def add_candidate_old(candidate, candidates_per_frame):
    # The first two elements of each `candidate` tuple are
    # the opposing corners of the bounding box.
    x1, y1 = candidate[0]
    x2, y2 = candidate[1]
    no = Node(x1, y1, x2, y2)
    candidates_per_frame[-1].append(no)
    if candidates_per_frame[-2]!=[]:
        for nodes_in in candidates_per_frame[-2]:
            nodes_in.add_child(no)
            # print("previous detection", nodes.bbox, "gets child", no.bbox)
    return candidates_per_frame
"""

def ball_detection_old(candidates_per_frame, balls, min_area):
    area_diff=[]
    nodes = []
    if len(balls)>0:
        bbox_last = balls[-1]
        area_last = abs((bbox_last[0]-bbox_last[2])*(bbox_last[1]-bbox_last[3]))
        for c in candidates_per_frame[-1]:
            if overlap(bbox_last, c.bbox):
                area_diff.append(np.inf)
            else:
                area_diff.append(abs(area_last - c.area))
            nodes.append(c)
    else:
        for cand in candidates_per_frame[-2]:
            if len(balls)>0:
                area_diff = [abs(balls[-1].area- c.area)]
            for c in cand.children:
                if overlap(cand.bbox, c.bbox):
                    area_diff.append(np.inf)
                else:
                    area_diff.append(abs(cand.area- c.area))
                nodes.append(c)
    if area_diff!=[] and np.min(area_diff)<min_area:
        if len(balls)>1 and len(np.where(np.array(area_diff)<min_area)[0])>1:
            d = []
            for j, n in enumerate(nodes):
                if area_diff[j]<min_area:
                    p1 =  np.array([(balls[0][0]+balls[0][2])/2, (balls[0][1]+balls[0][3])/2])
                    p2 =  np.array([(balls[1][0]+balls[1][2])/2, (balls[1][1]+balls[1][3])/2])
                    p3 =  np.array([(n.bbox[0]+n.bbox[2])/2, (n.bbox[1]+n.bbox[3])/2])
                    d.append(np.linalg.norm(np.cross(p2-p1, p1-p3))/np.linalg.norm(p2-p1))
                else: d.append(np.inf)
            balls.append(nodes[np.argmin(d)].bbox)
        else:
            balls.append(nodes[np.argmin(area_diff)].bbox) # ändern statt d area_diff
        print(balls)
    else:
        balls = []
    return balls

def add_candidate(candidate, candidates_per_frame):
    # The first two elements of each `candidate` tuple are
    # the opposing corners of the bounding box.
    x1, y1 = candidate[0]
    x2, y2 = candidate[1]
    no = Node(x1, y1, x2, y2)
    candidates_per_frame[-1].append(no)
    if candidates_per_frame[-2]!=[]:
        for nodes_in in candidates_per_frame[-2]:
            nodes_in.add_child(no)
            # print("previous detection", nodes.bbox, "gets child", no.bbox)
    return candidates_per_frame

def ball_detection(candidates_per_frame, balls, min_area):
    for cands3 in candidates_per_frame[-3]:
        for j, cands2 in enumerate(cands3.children): # counts through children (ebene 2)
            slope = cands3.slopes[j] # slope of 3 to 2
            dist = cands3.dist[j]
            print("j", j, slope, dist)
            for k, cands1 in enumerate(cands2.children):
                print("k", k, cands2.slopes[k], cands2.dist[k])
                if abs(slope-cands2.slopes[k]) < 0.1 and abs(dist-cands2.dist[k])<10:
                    return True
    return False


    # PARAMETERS
    def detect_ball(folder, joints_array=None, template = "%03d.jpg", min_area = 400, plotting=True, min_length_first=5, min_length_ball=3, every_x_frame=5):
        # length = len(os.listdir(folder))
        # images = [cv2.imread(folder+template %idx, cv2.IMREAD_GRAYSCALE) for idx in range(length)] #IMG_TEMPLATE.format(idx), )

        cap = cv2.VideoCapture(folder)
        images=[]
        start = time.time()

        candidates_per_frame = []
        location = []
        frame_indizes = []
        ankle_move=[]
        balls = []
        t=0
        # temp=0
        frame_before_close_wrist = False
        first_move_found = True
        while t<300:
            ret, frame = cap.read()

            if frame is None:
                break
            #if temp%every_x_frame !=0:
             #   temp+=1
            #  continue
            #temp+=1

            candidates_per_frame.append([])
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            """
            print(2)
            row,col= frame.shape
            mean = 0
            var = 500
            sigma = var**0.5
            print(3)
            gauss = np.random.normal(mean,sigma,(row,col))
            print(4)
            gaussframe = gauss.reshape(row,col) + frame
            gaussframe[gaussframe>255]=255
            gaussframe[gaussframe<0]=0
            frame = np.array(gaussframe.astype(np.uint8))
            """
        # for t in range(1, length-1):
            if t<11:
                images.append(frame)
                t+=1
                continue

            im_tm1 = images[-2]
            im_t = images[-1]
            im_tp1 = frame
            im_every_x_0 = images[0]
            im_every_x_1 = images[5]
            im_every_x_2 = images[10]

            # NORMAL CANDIDATES (THREE IN A ROW)
            candidates = get_candidates(im_tm1, im_t, im_tp1, min_area)
            #plt.figure(figsize=(10, 10), edgecolor='k')

            ### HERE INSERT RELEASE FRAME CLOSE TO WRIST
            for i, candidate in enumerate(candidates):
                if t>0:
                    candidates_per_frame = add_candidate(candidate, candidates_per_frame)
                ### HERE INSERT CLOSE TO WRIST
                ### HERE INSERT SHORTEST PATH CODE
                #save location of candidate and frame
                #location.append(center)
                frame_indizes.append(t-1)

            ### BALL DETECTION:
            l = np.array([len(candidates_per_frame[-i-1]) for i in range(3)])
            if np.all(l>0):
                if ball_detection(candidates_per_frame, balls, min_area):
                    print("release frame")
                    plot(im_t, candidates, t)
                    break
            """
            if len(candidates_per_frame[-2])>0:
                balls = ball_detection(candidates_per_frame, balls, min_area)
            else:
                balls = []

            # stop condition: if three consqcutive ball detection, stop
            if len(balls)==min_length_ball and poly_factor(np.array(balls))>1.5:# polyarea_bbox(np.array(balls))<min_area:
                print("release frame", t-min_length_ball)
                plot(im_t, candidates, t)
                break
            elif len(balls)==3:
                del balls[0]
            """

            if not first_move_found:
            # SHIFTED CANDIDATES:
                shifted_candidates = get_candidates(im_every_x_0, im_every_x_1, im_every_x_2, min_area)
                #plt.figure(figsize=(10, 10), edgecolor='k')

                ### FIRST MOVEMENT:
                if shifted_candidates!=[]:
                    ankle_move = first_movement(shifted_candidates, joints_array[t-every_x_frame], ankle_move, t)
                if len(ankle_move)>min_length_first-1 and t-ankle_move[-min_length_first]<10: #len(ankle_move)==3:
                    print("first movement frame: ", (ankle_move[-min_length_first]))
                    plot(im_t, shifted_candidates, t)
                    first_move_found=True

            if plotting and len(candidates)>0: #len(balls)>0: # ##
                plot(im_t, candidates, t)

            """
            if close_to_wrist:
                frame_before_close_wrist = True
            else:
                frame_before_close_wrist = False
            """
            t+=1
            images = np.roll(np.array(images), -1, axis=0)
            images[-1] = frame
        print("time for %s frames"%t, (time.time() - start) * 1000)

        return frame_indizes, location, candidates_per_frame

    # 40mph_1us_1.2f_170fps_40m_sun # 40mph_10us_6f_100fps_40m_cloudy # 40mph_10us_11f_100fps_noisy.avi


    example ="#9 RHP Ryan King (2)" #9 RHP Ryan King # #48 RHP Tom Flippin # 8 RHP Cole Johnson #15 Brandon Coborn # #10 Matt Glomb #26 RHP Tim Willites" (willites camera moves) #00 RHP Devin Smith
    BASE = "/Users/ninawiedemann/Desktop/UNI/Praktikum/high_quality_testing/pitcher/"+example+".mp4" #"data/Matt_Blais/" # für batter: pic davor, 03 streichen und (int(idx+1))
    joints_path = "/Volumes/Nina Backup/high_quality_outputs/"+example+".json"

    # joints_path = ""/Volumes/Nina Backup/"Nina's Pitch/40mph_10us_11f_100fps_noisy.json"

    #BASE = "/Volumes/Nina Backup/CENTERFIELD bsp videos/3d69a818-568e-4eef-9d63-24687477e7ee.mp4" # minarea 50
    #joints_path = "/Volumes/Nina Backup/outputs/new_videos/cf/490770_3d69a818-568e-4eef-9d63-24687477e7ee_pitcher.json"
    joints = from_json(joints_path)[:,:12,:]
    print(joints.shape)

    #for name in ["40mph_1us_1.2f_170fps_40m_sun.avi","40mph_10us_6f_100fps_40m_cloudy.avi", "40mph_10us_11f_100fps_noisy.avi"]:
    #    frame_indizes, location, candidates_per_frame = detect_ball("/Volumes/Nina Backup/Nina's Pitch/"+name, joints_array=None)
    #import sys
    #sys.exit()

    frame_indizes, location, candidates_per_frame = detect_ball(BASE, joints_array = joints, plotting=True, min_area=450) #400
