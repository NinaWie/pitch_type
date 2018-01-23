import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import json
import argparse

from config import cfg

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
            y_diff = (self.center[1]-no.center[1])
            slope = np.arctan(y_diff/(self.center[0]-no.center[0]))
            if y_diff<0:
                slope+=np.pi
            self.slopes.append(slope)
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
    # thresholds: bigger than mean + 3* std
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
        # left top and bottom right corners saved as candidates
        candidates.append((lt, rb, area))
    return candidates


def first_movement(cand_list, joints, ankle_move, fr):
    """
    determines if a movement occured close to ankles and knees
    --> returns ankle_move list, adding frame number to the list if a detection was close to the legs
    cand_list: list of candidates in current frame
    joints: joint detections in current frame, array of size nr_joints* nr_coordinates
    ankle_move: list of previous frames with a movement clos to the leg
    fr: current frame index
    """
    knees= joints[[7, 10],:]
    ankles = joints[[8,11],:]
    dist_ankle = np.linalg.norm(np.mean(knees-ankles, axis=0))  # radius: determined by mean distance between knees and ankles
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
        #if k==len(candidates_per_frame[-1])-1:
         #   ankle_move=[]
    #print(t, ankle_move)

def plot(im_t, candidates, frame_nr):
    """
    plots a candidate detection as a red rectangle on the current image
    im_t: current image
    candidated: list of center points of candidates detected in this frame
    frame_nr: current frame index (only for plot title)
    """
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
    plt.title("Detected FMO frame"+ str(frame_nr))
    plt.show()


def add_candidate(candidate, candidates_per_frame):
    """
    Transforms candidate to a graph node
    --> adds this node firstly to list of candidates in the current frame
    --> adds this node also as a child to the candidates in the previous frame if the distance is large enough
    (threshold for distance which a FMO at least needs to move)
    candidate: one candidate of shape [lt, rb, area]
    candidates_per_frame: list of nodes of candidates in each frame
    """
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
    """
    evaluates if consecutive candidates are a ball -  applies metric comparing slopes and distances of candidates in consecutive frames
    candidates_per_frame: list containing the graph node references for each frame
    balls: list of ball trajectory, empty if no ball detection in previous frame
    metric_thresh: threshold when a trajectory is classified a ball
    """
    if len(balls)==0:
        for cands3 in candidates_per_frame[-3]:
            for j, cands2 in enumerate(cands3.children): # counts through children (level 2)
                slope = cands3.slopes[j] # slope of 3 to 2
                dist = cands3.dist[j]
                area = cands3.area_diffs[j]
                angle = cands3.angle_diffs[j]
                #print("j", j, slope, dist, area, angle)
                for k, cands1 in enumerate(cands2.children):
                    #print("k", k, cands2.slopes[k], cands2.dist[k], cands2.area_diffs[k], cands2.angle_diffs[k])
                    #print("metric: ", abs(slope-cands2.slopes[k]), abs(1- dist/cands2.dist[k]), area, cands2.area_diffs[k])
                    metric = abs(slope-cands2.slopes[k]) + abs(1- dist/cands2.dist[k]) #+ area+cands2.area_diffs[k] + angle + cands2.angle_diffs[k]
                    if metric<cfg.metric_thresh:
                        balls = [cands3, cands2, cands1]
                    # if abs(slope-cands2.slopes[k]) < 0.1 and abs(dist-cands2.dist[k])<10:
                        return balls
    else: # if balls were detected in previous frame, don't check all combination of nodes with children, but only the children of the last ball detection
        j = balls[-2].children.index(balls[-1])
        slope = balls[-2].slopes[j] # slope of 3 to 2
        dist = balls[-2].dist[j]
        area = balls[-2].area_diffs[j]
        angle = balls[-2].angle_diffs[j]
        for k, cands1 in enumerate(balls[-1].children):
            #print("k", k, balls[-1].slopes[k], balls[-1].dist[k], balls[-1].area_diffs[k], balls[-1].angle_diffs[k])
            #print("metric: ", abs(slope-balls[-1].slopes[k]), abs(1- dist/balls[-1].dist[k]), area, balls[-1].area_diffs[k])
            metric = abs(slope-balls[-1].slopes[k]) + abs(1- dist/balls[-1].dist[k]) # + area+balls[-1].area_diffs[k] + angle + balls[-1].angle_diffs[k]
            if metric<cfg.metric_thresh:
                balls.append(cands1)
            # if abs(slope-cands2.slopes[k]) < 0.1 and abs(dist-cands2.dist[k])<10:
                return balls
    return []

def distance_projected(p, p1,p2):
    """
    returns distance of p' (= p projected on line spanned by p1 and p2) from p1
    """
    v1 = p-p1
    v2 = p2-p1
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0) * np.linalg.norm(v1)

def _get_max_array(array_list):
    """
    returns union of the binary images in array_list
    """
    resultant_array = np.zeros(array_list[0].shape)
    for array in array_list:
        resultant_array = np.maximum(resultant_array, array)
    return resultant_array

def trajectory_and_speed(balls, im_t, t, fps = 30, plotting=True):
    """
    returns the release frame and speed of a ball trajectory
    input: balls: list of center coordinates of detected ball candidates in consecutive frames
    im_t: for plotting image at current frame is required
    t: current frame index
    factor_pixel_feet: distance in reality in feet * factor_pixel_feet = distance on image in pixel
    plotting: set true if trajectory should be plotted
    """
    #plot(im_t, candidates, t)
    trajectory = np.array([elem.center for elem in balls]).astype(int)
    # print("trajectory", trajectory.tolist())
    # distance
    dist_from_start = distance_projected(trajectory[0], np.array(eval(cfg.pitcher_mound_coordinates)),np.array(eval(cfg.batter_base_coordinates)))
    speed = np.mean([np.linalg.norm(trajectory[i]- trajectory[i+1]) for i in range(len(trajectory)-1)])
    frames_shifted = dist_from_start/speed + len(balls)-1 # frames because current frame is already after nth ball detection
    # print("frames from release frame (using distance from center of base projected)", frames_shifted)
    #plt.scatter(trajectory[:,0], trajectory[:,1])
    #plt.show()
    ball_release = round(t - frames_shifted)
    if plotting:
        plt.figure(figsize=(10, 10), edgecolor='r')
        img = np.tile(np.expand_dims(im_t.copy(), axis = 2), (1,1,3))
        # img = cv2.line(img, (110, 140),(690, 288), color = 2) # line from center of base to center of pitchers mound
        for i in range(len(trajectory)-1):
            img = cv2.line(img, tuple(trajectory[i]),tuple(trajectory[i+1]), color = 2)
        # img = cv2.line(img, tuple(trajectory[1]),tuple(trajectory[2]), color = 2)
        plt.imshow(img, 'gray')
        plt.title("Ball trajectory at frame "+str(t)+ ", speed in mph: "+ str(speed*fps* 0.681818 /cfg.factor_pixel_feet))
        plt.show()
        # print("SPEED in mph", speed*fps* 0.681818 /factor_pixel_feet) # 1 ft/s = 0.681818 mph
        print("")
    return ball_release

def detect_ball(folder, joints_array=None, min_area = 400, plotting=True, min_length_first=5, every_x_frame=3, roi=None):
    """
    folder: path to video file (e.g. test.mp4)
    min_area: minimum amount of pixels that a fast moving object must enclose - filtering out smaller ones (noise)
    plotting: set True if all candidate detections should be shown
    min_length_first:
    joints_array: for finding the pitcher's first movement, the joint array of size nr_frames*nr_joints*2 is required
    roi: region of interest if not the whole frame is relevant, format: list [top, bottom, left, right] with top<bottom
    """
    fps = 30
    cap = cv2.VideoCapture(folder)
    images=[]
    motion_images=[]
    start = time.time()

    candidates_per_frame = []
    location = []
    frame_indizes = []
    ankle_move=[]
    balls = []
    t=0 # frame count
    first_move_found = False
    ball_release_found = False

    # function returns
    ball_release = 0 # if not found
    first_move_frame = 0 # if not found
    ball_trajectory = []

    while True:
        ret, frame = cap.read()
        if frame is None:
            break

        candidates_per_frame.append([])
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            frame = frame[roi[0]:roi[1], roi[2]:roi[3]]

        if t>1:
            im_tm1 = images[-2]
            im_t = images[-1]
            im_tp1 = frame

            nd = get_difference(im_tm1, im_t, im_tp1)
            if t<11:
                motion_images.append(nd)
        if t<11:
            images.append(frame)
            t+=1
            continue
        im_every_x_0 = images[-2*every_x_frame]
        im_every_x_1 = images[-every_x_frame]
        im_every_x_2 = images[-1]

        # REMOVE NOISE FROM SHAKY CAMERA
        cumulative_motion = _get_max_array(motion_images)
        final_frame = nd.astype(int) - cumulative_motion.astype(int)
        final_frame[final_frame < 0] = 0
        candidates = get_candidates(final_frame.astype(np.uint8), min_area)

        for i, candidate in enumerate(candidates):
            if t>0:
                candidates_per_frame = add_candidate(candidate, candidates_per_frame)
            frame_indizes.append(t-1)

        ### BALL DETECTION - CANDIDATES:
        l = np.array([len(candidates_per_frame[-i-1]) for i in range(3)])
        if np.all(l>0) and len(balls)==0:
            balls = ball_detection(candidates_per_frame, balls)
            if len(balls)==3:
                est_ball_release = trajectory_and_speed(balls, im_t, t, plotting = False)
                if not ball_release_found:
                    ball_release_found = True
                    ball_release = est_ball_release
                    print("\nRELEASE FRAME AT ", ball_release, "\n")
                counter = 0
        elif len(balls)>0:
            if len(candidates_per_frame)>0:
                new_balls = ball_detection(candidates_per_frame, balls)
                if len(new_balls)==0 and counter>-1: # change from -1 to higher number to allow missing balls
                    # inbetween (problem: speed then calculated wrong because len(balls))
                    # problem 2: dist in metric - also there some number indicating this
                    # general problem: candidates not in balls[-1].children
                    _ = trajectory_and_speed(balls, im_t, t-1)
                    for b in balls:
                        ball_trajectory.append(b.center)
                    balls = []
                    # break
                else:
                    balls = new_balls
            else:
                _ = trajectory_and_speed(balls, im_t, t-1)
                for b in balls:
                    ball_trajectory.append(b.center)
                balls = []
                # break
        else:
            balls=[]

        # FIRST MOVEMENT --> SHIFTED CANDIDATES
        if not first_move_found and joints_array is not None:
            # SHIFTED CANDIDATES:
            detected_moves = get_difference(im_every_x_0, im_every_x_1, im_every_x_2)
            shifted_candidates = get_candidates(detected_moves, min_area)

            ### FIRST MOVEMENT:
            if shifted_candidates!=[]:
                ankle_move = first_movement(shifted_candidates, joints_array[t-every_x_frame], ankle_move, t)
            if len(ankle_move)>=min_length_first and t-ankle_move[-min_length_first]<cfg.max_frames_first_move: #len(ankle_move)==3:
                print("first movement frame: ", (ankle_move[-min_length_first]))
                plot(im_t, shifted_candidates, t)
                first_move_found = True
                first_move_frame = ankle_move[-min_length_first]
                if cfg.refine:
                    range_joints = joints_array[first_move_frame -cfg.refine_range: first_move_frame +cfg.refine_range]
                    grad = range_joints # np.gradient(range_joints, axis = 0) # OHNE GRADIENT; JUST HEIGHT OF LEG
                    mean_gradient = np.mean(grad[:, [7,8,10,11],1], axis = 1)
                    first_move_frame = first_move_frame - cfg.refine_range + np.argmin(mean_gradient)
                print("First movement (possibly refined)", first_move_frame)
                break

        if plotting and len(candidates)>0: #len(balls)>0: # ##
            plot(im_t, candidates, t)
        t+=1
        images = np.roll(np.array(images), -1, axis=0)
        images[-1] = frame
        motion_images = np.roll(np.array(motion_images), -1, axis=0)
        motion_images[-1] = nd
    # print("time for %s frames"%t, (time.time() - start) * 1000)
    return ball_release, np.array(ball_trajectory), first_move_frame

if __name__=="main":
    parser = argparse.ArgumentParser(description='Detect ball or first movement in video')
    parser.add_argument('video_file', type=str, help='path to video file')
    parser.add_argument('-min_area', default=400, type=int, help='mininum area of FMO candidates (ball or foot movement)')
    args = parser.parse_args()

    BASE = args.video_file
    frame_indizes, location, candidates_per_frame, first_move_frame = detect_ball(BASE, joints_array = None, plotting=False, min_area=args.min_area) #400
