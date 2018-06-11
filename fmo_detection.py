import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import json
import pandas as pd
from skvideo import io

from config_fmo import cfg

def from_json(file):
    """
    Yields and array of the joint trajectories of a json file
    """
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
    """
    old function for slope: slope defined as arctan
    not recommended as vectors in the opposite direction have the same slopes
    """
    y_diff = (center1[1]-center2[1])
    slope = np.arctan(y_diff/(center1[0]-center2[0]))
    if y_diff < 0:
        slope+= np.pi
    return slope

def get_slope(center1, center2):
    """
    slopes represented as normalized complex vectors
    """
    y_diff = (center1[1]-center2[1])
    x_diff = (center1[0]-center2[0])
    return np.complex(x_diff, y_diff)/np.sqrt(x_diff**2 + y_diff**2)

class Node():
    """
    Design graph with motion candidates as nodes
    """
    def __init__(self, x1, y1, x2, y2):
        self.bbox = [x1, y1, x2, y2] # bounding box
        self.l = abs(x1-x2) # length
        self.w = abs(y1-y2) # width
        self.angle = np.arctan(self.l/self.w) # angle of diagonals
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


def first_movement(cand_list, joints):
    """
    Checks whether there is a motion candidate that is sufficiently close to the ankles or knees
    If yes, the frame fr is added to the list of frames (ankle_move)
    :param cand_list: list of motion candidates detected in this frame
    :param joints: Array of joints detected by pose estimation in this frame
    """
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
            return True
    return False

def plot(im_t, candidates, frame_nr):
    """
    Plot motion candidates on image im_t
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
    """
    For each candidate, make a new Node object to put it in the Graph
    Decide for each candidate of the previous frame if it gets the new candidates as children
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

def confidence(s1, s2, d1, d2):
    """
    Confidence value C that indicated how likely a triple of motion candidates is a ball detection
    compares slopes s1 and s2 between the motion candidates, and distances d1 and d2
    """
    slope_similarity = 1 - 0.5*abs(s1-s2)
    distance_similarity = min(d1/float(d2), d2/float(d1))
    return slope_similarity + distance_similarity
    # abs(slope-cands2.slopes[k]) + abs(1- dist/cands2.dist[k]) #+ area+cands2.area_diffs[k] + angle + cands2.angle_diffs[k]

def ball_detection(candidates_per_frame, balls):
    """
    Determines for each connected (in the graph) triple of candidates whether the confidence value is
    sufficiently high
    :param candidates_per_frame: List of Node objects, each representing a motion candidate in a frame
    :balls: list of previous ball detections
    If balls is empty, each triple of candidates of the last three frames is evaluated,
    If balls is not empty, just the candidates of the current frame are checked (and added to the list if C is high enough)
    """
    metric_thresh = cfg.metric_thresh
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
                    confidence_value = confidence(slope, cands2.slopes[k], dist, cands2.dist[k])
                    if confidence_value>metric_thresh:
                        balls = [cands3, cands2, cands1]
                        # because we do not want the last one that overcomes the threshold, but the one with highest C
                        metric_thresh = confidence_value
    else: # if ball is already detected, just compare the new candidates to the last detected ball
        j = balls[-2].children.index(balls[-1])
        slope = balls[-2].slopes[j] # slope of 3 to 2
        dist = balls[-2].dist[j]
        area = balls[-2].area_diffs[j]
        angle = balls[-2].angle_diffs[j]
        new_ball = None
        for k, cands1 in enumerate(balls[-1].children):
            #print("k", k, balls[-1].slopes[k], balls[-1].dist[k], balls[-1].area_diffs[k], balls[-1].angle_diffs[k])
            #print("metric: ", abs(slope-balls[-1].slopes[k]), abs(1- dist/balls[-1].dist[k]), area, balls[-1].area_diffs[k])
            confidence_value = confidence(slope, balls[-1].slopes[k], dist, balls[-1].dist[k])
            # abs(slope-balls[-1].slopes[k]) + abs(1- dist/balls[-1].dist[k]) # + area+balls[-1].area_diffs[k] + angle + balls[-1].angle_diffs[k]
            if confidence_value>metric_thresh:
                new_ball = cands1
                # because we do not want the last one that overcomes the threshold, but the one with highest C
                metric_thresh = confidence_value
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
    """
    Scatter plot of the ball trajectory
    """
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
    """
    Mark each detected ball with a red circle in a video
    :param vid_path: path to the input video
    :param out_path: path where to save the video
    :param ball_trajectory: array of ball detections (as outputted by detect_ball function)
    :param radius: radius of the circle marking the ball
    """
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
    :param folder: path to the input video
    :param joints_array: for first movement, this must be an array of shape nr_frames x nr_joints x nr_coordinates
    (if only ball should be detected, set None)
    :param min_area: minium number of pixels to find connected components in the difference image
    :param plotting: if all detected candidates should be plotted on the frame
    :param roi: region of interest if not the whole frame is relevant, format: list [top, bottom, left, right] with top<bottom
    """
    if not os.path.exists(folder):
        print("Error: video path does not exist!")
        return 0

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
    # balls_per_frame = []

    while True:
        ret, frame = cap.read() # read frame
        if frame is None:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if roi is not None:
            frame = frame[roi[0]:roi[1], roi[2]:roi[3]]

        tic2 = time.time()

        candidates_per_frame.append([])

        tic = time.time()

        # three images for getting the difference image
        im_tm1 = images[-2*every_x_frame-1]
        im_t = images[-every_x_frame-1]
        im_tp1 = images[-1]

        diff_image = get_difference(im_tm1, im_t, im_tp1)

        if every_x_frame==1: # shakiness removal (only possible if every frame is taken into account)
            cumulative_motion = _get_max_array(motion_images)
            final_diff_image = diff_image.astype(int) - cumulative_motion.astype(int)
            final_diff_image[final_diff_image < 0] = 0
        else:
            final_diff_image = diff_image

        # get connected components from difference images
        candidates = get_candidates(final_diff_image.astype(np.uint8), min_area)

        # balls_per_frame.append(len(candidates)) # for timing tests

        # Add candidates to the graph for GBCV
        for i, candidate in enumerate(candidates):
            if t>0:
                candidates_per_frame = add_candidate(candidate, candidates_per_frame)

        ### BALL DETECTION:

        # check if there motion was detected in the last three frames
        triple = np.array([len(candidates_per_frame[-i-1]) for i in range(3)])
        # 1. case: No ball detected so far, and possible triple available
        if np.all(triple>0) and len(balls)==0:
            balls = ball_detection(candidates_per_frame, balls) # check confidence value

            if len(balls)==3 and not ball_release_found: # first ball detection
                ball_release_found = True
                ball_release = t-2 # first ball detection

        # 2. case: already balls detected
        elif len(balls)>0:
            # 2.1 possible new ball --> check metric
            if len(candidates_per_frame)>0:
                new_balls = ball_detection(candidates_per_frame, balls)
            # 2.2 no new candidates
            else:
                new_balls = []
            # --> if the confidence value was too low for all candidates: add the current ball list to the final ball_trajectory list
            if len(new_balls)==0:
                # Evaluate confidence value if there were balls detected earlier on
                mean_slope = np.mean(np.array([get_slope(balls[i].center, balls[i+1].center) for i in range(len(balls)-1)]))
                if len(ball_trajectory)!=0:
                    mean_slope_previous = np.mean(np.array([get_slope(ball_trajectory[i], ball_trajectory[i+1]) for i in range(len(ball_trajectory)-1)]))
                else:
                    mean_slope_previous = mean_slope
                # print("slopes", mean_slope, mean_slope_previous)
                if abs(mean_slope-mean_slope_previous)<0.4:
                    for i, b in enumerate(balls):
                        frame_count = len(balls)-i
                        # it might happen that the previous ball detection go until frame x, and then the ball is lost,
                        # but the next ball detection starts from frame x again. Thus, check if x already in the ball_trajectory
                        if len(ball_trajectory)==0 or t-1-frame_count not in np.asarray(ball_trajectory)[:,2]:
                            ball_trajectory.append([b.center[0], b.center[1], t-1-frame_count])

                balls = [] # new ball list
            else: # new ball found --> add to ball list, continue
                balls = new_balls
        # 3. case: no balls detected so far, and no suitable triple of nodes in the last three frames
        else:
            balls=[] # balls stay empty

        tocs2.append(time.time()-tic2)

        # FIRST MOVEMENT

        tic3 = time.time()

        if not first_move_found and joints_array is not None:
            if candidates!=[]:
                # check if any of the candidates is sufficiently close to knees or ankles
                motion_close_to_leg = first_movement(candidates, joints_array[t-every_x_frame-1])
                if motion_close_to_leg:
                    ankle_move.append(t) # add current frame index (t) to sequence if there was a motion close to the leg

            # check if sequence is long enough and fulfills the threshold requirements
            if len(ankle_move)>=cfg.min_length_first and t-ankle_move[-cfg.min_length_first]<cfg.max_frames_first_move: #len(ankle_move)==3:
                # print("first movement frame: ", (ankle_move[-min_length_first]))
                # plot(im_t, shifted_candidates, t)
                first_move_found = True
                first_move_frame = ankle_move[-cfg.min_length_first]

                if refine:
                    RADIUS_LOWER = min(cfg.refine_range, first_move_frame)
                    range_joints = joints_array[first_move_frame - RADIUS_LOWER: first_move_frame + cfg.refine_range]
                    mean_leg_position = np.mean(range_joints[:, [7,8,10,11],1], axis = 1)
                    ### gradient plotting
                    # plt.plot(grad[:,:,1])
                    # plt.plot(mean_gradient, c="black")
                    # plt.title("black: mean height of knees and ankles")
                    # plt.show()
                    first_move_frame = first_move_frame - RADIUS_LOWER + np.argmin(mean_leg_position)
                    # print("Refined first movement", first_move_frame)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run FMO-C on a video to find motion candidates')
    parser.add_argument('-min_area', default=400, type=int, help='Minimum area to find connected components from difference images')
    parser.add_argument('video_path', type=str, help='path to the video')
    args = parser.parse_args()

    ball_release, ball_trajectory, first_move_frame, candidates_per_frame = detect_ball(args.video_path, min_area=args.min_area)
