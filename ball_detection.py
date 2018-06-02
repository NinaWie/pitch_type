import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import json
import pandas as pd
from skvideo import io

from config import cfg

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

def get_slope_arctan(center1, center2):
    y_diff = (center1[1]-center2[1])
    slope = np.arctan(y_diff/(center1[0]-center2[0]))
    if y_diff < 0:
        slope+= np.pi
    return slope

def get_slope(center1, center2):
    y_diff = (center1[1]-center2[1])
    x_diff = (center1[0]-center2[0])
    return np.complex(x_diff, y_diff)/np.sqrt(x_diff**2 + y_diff**2)

class Node():
    def __init__(self, x1, y1, x2, y2):
        self.bbox = [x1, y1, x2, y2]
        self.l = abs(x1-x2)
        self.w = abs(y1-y2)
        self.angle = np.arctan(self.l/self.w)
        self.area = self.l*self.w # (l+w)/float(l*w)
        self.center = np.array([(x1+x2)/2, (y1+y2)/2])
        self.children = []
        self.area_diffs = []
        self.angle_diffs = []
        self.slopes = []
        self.dist = []
    def add_child(self, no):
        dist = np.linalg.norm(no.center-self.center)
        if dist>cfg.min_dist:
            self.children.append(no)
            self.area_diffs.append(abs(1-(no.area/self.area)))
            self.slopes.append(get_slope(self.center, no.center))
            self.dist.append(dist)
            self.angle_diffs.append(abs(self.angle-no.angle))
    def favourite_child(self, no):
        self.fav_child=no

def get_difference(im_tm1, im_t, im_tp1):
    """
    calculates difference image and applies threshold to detect significant motion
    parameters: three consecutive frames
    returns binary image of same size as input frames indicating significantly different pixels
    """
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

    detect = cv2.bitwise_and(cv2.bitwise_and(dbp, dbm), cv2.bitwise_not(db0))
    # nd = cv2.bitwise_not(detect)
    return detect


def get_candidates(nd, min_area):
    """
    find connected components in a binary difference image
    nd is a binary image indicating significant pixel changes
    min_area is the minimum area of pixels in nd that should be recognized as a connected region
    return list of candidates, each is a tuple (left_top_corner, right_bottom_corner, area)
    """
    # only stats is used, not num, labels, centroids
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(nd, ltype=cv2.CV_16U)
    # We set an arbitrary threshold to screen out smaller "components"
    # which may result simply from noise, or moving leaves, and other
    # elements not of interest.
    candidates = list()
    for stat in stats[1:]:
        area = stat[cv2.CC_STAT_AREA]
        if area < min_area:
            continue # Skip small objects (noise)

        lt = (stat[cv2.CC_STAT_LEFT], stat[cv2.CC_STAT_TOP])
        rb = (lt[0] + stat[cv2.CC_STAT_WIDTH], lt[1] + stat[cv2.CC_STAT_HEIGHT])

        candidates.append((lt, rb, area))
    return candidates


def first_movement(cand_list, joints, ankle_move, fr):
    knees= joints[[7, 10],:]
    ankles = joints[[8,11],:]
    #print(knees, ankles, knees-ankles, np.mean(knees-ankles, axis=0))
    dist_ankle = cfg.factor_knee_radius * np.linalg.norm(np.mean(knees-ankles, axis=0)) #//2
    #print("radius", dist_ankle)
    for k, cand in enumerate(cand_list):
        x1, y1 = cand[0]
        x2, y2 = cand[1]
        center = [(x1+x2)/2, (y1+y2)/2]
        norms = np.array([np.linalg.norm(center - knees[0]), np.linalg.norm(center - knees[1]), np.linalg.norm(center - ankles[0]), np.linalg.norm(center - ankles[1])])
        #print("center", cand.center, "knees", knees[0],knees[1], "ankles", ankles, "norms", norms)
        if np.any(norms<dist_ankle):
            # print("smaller radius", center)
            ankle_move.append(fr) #cand.center)
            break
    return ankle_move

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
    # cv2.circle(img, (690, 290),8, [255,0,0], thickness=-1)
    # cv2.circle(img, (50, 400), 8, [255,0,0], thickness=-1)
    for can in candidates: # einfuegen falls alles plotten
        cv2.rectangle(img, can[0], can[1],[255,0,0], 4)
    #cv2.rectangle(img,tuple(balls[-1][:2]), tuple(balls[-1][2:]), [255,0,0], 4)
    plt.imshow(img, 'gray')
    # plt.axis("off")
    plt.title("Detected FMO frame"+ str(frame_nr)) #+str(candidates))
    # plt.savefig("first_move_sequence/"+bsp[:-4]+"_"+str(frame_nr)) # [:400,200:700]
    # plt.savefig("/Users/ninawiedemann/Desktop/BA/fmo detection/connected_components", pad_inches=0)
    plt.show()
    # print(candidates)


def add_candidate(candidate, candidates_per_frame):
    # The first two elements of each `candidate` tuple are
    # the opposing corners of the bounding box.
    x1, y1 = candidate[0]
    x2, y2 = candidate[1]
    no = Node(x1, y1, x2, y2)
    #print("area_cand[3]:", candidate[2], "area node", no.area)
    candidates_per_frame[-1].append(no)
    if candidates_per_frame[-2]!=[]:
        for nodes_in in candidates_per_frame[-2]:
            nodes_in.add_child(no)
            # print("previous detection", nodes.bbox, "gets child", no.bbox)
    return candidates_per_frame

def ball_detection(candidates_per_frame, balls):
    if len(balls)==0:
        for cands3 in candidates_per_frame[-3]:
            for j, cands2 in enumerate(cands3.children): # counts through children (ebene 2)
                slope = cands3.slopes[j] # slope of 3 to 2
                dist = cands3.dist[j]
                area = cands3.area_diffs[j]
                angle = cands3.angle_diffs[j]
                # print("j", j, slope, dist, area, angle)
                for k, cands1 in enumerate(cands2.children):
                    #print("k", k, cands2.slopes[k], cands2.dist[k], cands2.area_diffs[k], cands2.angle_diffs[k])
                    #print("metric: ", abs(slope-cands2.slopes[k]), abs(1- dist/cands2.dist[k]), area, cands2.area_diffs[k])
                    metric = abs(slope-cands2.slopes[k]) + abs(1- dist/cands2.dist[k]) #+ area+cands2.area_diffs[k] + angle + cands2.angle_diffs[k]
                    if metric<cfg.metric_thresh:
                        balls = [cands3, cands2, cands1]
                        metric_thresh = metric
                    # if abs(slope-cands2.slopes[k]) < 0.1 and abs(dist-cands2.dist[k])<10:
    else:
        j = balls[-2].children.index(balls[-1])
        slope = balls[-2].slopes[j] # slope of 3 to 2
        dist = balls[-2].dist[j]
        area = balls[-2].area_diffs[j]
        angle = balls[-2].angle_diffs[j]
        new_ball = None
        for k, cands1 in enumerate(balls[-1].children):
            #print("k", k, balls[-1].slopes[k], balls[-1].dist[k], balls[-1].area_diffs[k], balls[-1].angle_diffs[k])
            #print("metric: ", abs(slope-balls[-1].slopes[k]), abs(1- dist/balls[-1].dist[k]), area, balls[-1].area_diffs[k])
            metric = abs(slope-balls[-1].slopes[k]) + abs(1- dist/balls[-1].dist[k]) # + area+balls[-1].area_diffs[k] + angle + balls[-1].angle_diffs[k]
            if metric<cfg.metric_thresh:
                new_ball = cands1
                metric_thresh = metric
        if new_ball is None:
            balls = []
        else:
            balls.append(new_ball)
            # if abs(slope-cands2.slopes[k]) < 0.1 and abs(dist-cands2.dist[k])<10:
    return balls

def _get_max_array(array_list):
    """
    returns union of the binary images in array_list
    """
    resultant_array = np.zeros(array_list[0].shape)
    for array in array_list:
        resultant_array = np.maximum(resultant_array, array)
    return resultant_array

def plot_trajectory(trajectory):
    # max_y = np.amax(trajectory[:,1])+100
    # max_x = np.amax(trajectory[:,0])+10
    plt.figure(figsize=(10, 5), edgecolor='r')
    plt.scatter(trajectory[:, 0], trajectory[:,1])
    plt.title("Ball trajectory", fontsize = 15)
    # plt.ylim(max_y,0)
    # plt.xlim(0,max_x)
    plt.gca().invert_yaxis()
    plt.show()

def plot_trajectory_on_video(vid_path, out_path, ball_trajectory, radius=10):
    cap = cv2.VideoCapture(vid_path)
    arr = []
    i=0
    count=0
    while True:
        ret, img = cap.read()
        if ret==False:
            break
        ori_img = img.copy()
        out_img = img.copy()
        if count<len(ball_trajectory) and ball_trajectory[count,2] ==i:
            scale = int(255/(count+1))
            for j in range(count+1):
                weight = count-j
                # print(j, weight*scale)
                center = tuple(ball_trajectory[j,:2].astype(int))
                img = cv2.circle(ori_img, center, int(radius),[255,weight*scale,weight*scale], thickness = 2)
                # print(j, weight, anti_weight)
            count+=1
        #plt.figure(figsize = (10,10))
        #plt.imshow(img[30:, 20:])
        #plt.show()
        i+=1
        # append multiple times to have slow motion like video
        for _ in range(5):
            arr.append(img[30:, 20:])
    io.vwrite(out_path, arr)

def detect_ball(folder, joints_array=None, min_area = 400, plotting=True, every_x_frame=1, roi=None, refine = False):
    """
    roi: region of interest if not the whole frame is relevant, format: list [top, bottom, left, right] with top<bottom
    """

    cap = cv2.VideoCapture(folder)

    location = []
    ankle_move=[]
    balls = []

    # Read first frame and put it in list every_x*2 +1 times so the difference images can be calculated
    length_lists = every_x_frame*2 +1
    ret, frame = cap.read()
    if roi is not None:
        frame = frame[roi[0]:roi[1], roi[2]:roi[3]]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    images = [frame for _ in range(length_lists)]
    motion_images = [np.zeros((len(frame), len(frame[0]))) for _ in range(length_lists)]
    candidates_per_frame = [[] for _ in range(length_lists)]
    # frame count
    t=1

    first_move_found = False
    ball_release_found = False

    # function returns
    ball_release = 0 # if not found
    first_move_frame = 0 # if not found
    ball_trajectory = []

    # Timing tests
    start = time.time()
    tocs = []
    tocs2 = []
    tocs3 = []
    balls_per_frame = []

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            frame = frame[roi[0]:roi[1], roi[2]:roi[3]]

        tic2 = time.time()

        candidates_per_frame.append([])

        tic = time.time()

        im_tm1 = images[-2*every_x_frame-1]
        im_t = images[-every_x_frame-1]
        im_tp1 = images[-1]

        diff_image = get_difference(im_tm1, im_t, im_tp1)

        if every_x_frame==1:
            cumulative_motion = _get_max_array(motion_images)
            final_frame = diff_image.astype(int) - cumulative_motion.astype(int)
            final_frame[final_frame < 0] = 0
        else:
            final_frame = diff_image

        candidates = get_candidates(final_frame.astype(np.uint8), min_area)

        balls_per_frame.append(len(candidates))

        for i, candidate in enumerate(candidates):
            if t>0:
                candidates_per_frame = add_candidate(candidate, candidates_per_frame)

        ### BALL DETECTION:
        triple = np.array([len(candidates_per_frame[-i-1]) for i in range(3)])
        if np.all(triple>0) and len(balls)==0:
            balls = ball_detection(candidates_per_frame, balls)

            if len(balls)==3 and not ball_release_found:
                ball_release_found = True
                ball_release = t-2 # first ball detection

                # print("ball release frame found", t)
                # ball_release = release_frame_real_time(balls, t) #, images = images)
        elif len(balls)>0: # already balls detected
            if len(candidates_per_frame)>0:
                new_balls = ball_detection(candidates_per_frame, balls)
            else:
                new_balls = []
            # no further detections: add to ball_trajectory list
            if len(new_balls)==0:
                # mph = trajectory_and_speed(balls, im_t, t-1)
                mean_slope = np.mean(np.array([get_slope(balls[i].center, balls[i+1].center) for i in range(len(balls)-1)]))
                if len(ball_trajectory)!=0:
                    mean_slope_previous = np.mean(np.array([get_slope(ball_trajectory[i], ball_trajectory[i+1]) for i in range(len(ball_trajectory)-1)]))
                else:
                    mean_slope_previous = mean_slope
                # print("slopes", mean_slope, mean_slope_previous)
                if abs(mean_slope-mean_slope_previous)<0.4:
                    for i, b in enumerate(balls):
                        frame_count = len(balls)-i
                        ball_trajectory.append([b.center[0], b.center[1], t-1-frame_count])

                balls = []
                # break
            else:
                balls = new_balls
                # print("new balls detected")
                # break
        else:
            balls=[]

        tocs2.append(time.time()-tic2)

        # FIRST MOVEMENT

        tic3 = time.time()

        if not first_move_found and joints_array is not None:
            if candidates!=[]:
                #old = np.array(ankle_move).copy()
                ankle_move = first_movement(candidates, joints_array[t-every_x_frame-1], ankle_move, t)
                #if len(ankle_move)>len(old):
                 #   plot(im_t, shifted_candidates, t)
            if len(ankle_move)>=cfg.min_length_first and t-ankle_move[-cfg.min_length_first]<cfg.max_frames_first_move: #len(ankle_move)==3:
                # print("first movement frame: ", (ankle_move[-min_length_first]))
                # plot(im_t, shifted_candidates, t)
                first_move_found = True
                first_move_frame = ankle_move[-cfg.min_length_first]

                if refine:
                    RADIUS_LOWER = min(cfg.refine_range, first_move_frame)
                    range_joints = joints_array[first_move_frame - RADIUS_LOWER: first_move_frame + cfg.refine_range]
                    # grad = np.gradient(range_joints, axis = 0) # OHNE GRADIENT; JUST HEIGHT OF LEG
                    mean_gradient = np.mean(range_joints[:, [7,8,10,11],1], axis = 1)
                    ### gradient plotting
                    # plt.plot(grad[:,:,1])
                    # plt.plot(mean_gradient, c="black")
                    # plt.title("black: mean height of knees and ankles")
                    # plt.show()
                    first_move_frame = first_move_frame - RADIUS_LOWER + np.argmin(mean_gradient)
                    # print("Refined first movement", first_move_frame)
                # break

        tocs3.append(time.time()-tic3)

        if plotting and len(candidates)>0: #len(balls)>0: # ##
            plot(im_t, candidates, t)

        t+=1
        images = np.roll(np.array(images), -1, axis=0)
        images[-1] = frame
        motion_images = np.roll(np.array(motion_images), -1, axis=0)
        motion_images[-1] = diff_image

        toc = time.time()
        tocs.append(toc-tic)

    ## FOR TIMING TESTS
    #print("average candidates", np.mean(balls_per_frame))
    #print("insgesamt", np.mean(tocs))
    #print("für ball", np.mean(tocs2))
    #print("first move", np.mean(tocs3))
    #print("time for %s frames"%t, (time.time() - start) * 1000)

    # calculate back to normal frame size
    if roi is not None:
        for b in ball_trajectory:
            b[0]+= roi[0]
            b[1]+= roi[2]
    return ball_release, np.array(ball_trajectory), first_move_frame, candidates_per_frame[length_lists+1:]
