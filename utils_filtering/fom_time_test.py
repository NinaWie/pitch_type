import cv2
import math
from matplotlib import pyplot as plt
import numpy as np
import os
import time
import json
import pandas as pd

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

def get_slope(center1, center2):
    y_diff = (center1[1]-center2[1])
    slope = np.arctan(y_diff/(center1[0]-center2[0]))
    if y_diff < 0:
        slope+= np.pi
    return slope

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
        if dist>10:
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
        #if np.any(np.array(lt)==0) or np.any(np.array(rb)==0):
         #   continue
        bottomLeftCornerOfText = (lt[0], lt[1] - 15)

        candidates.append((lt, rb, area))
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
            # print("smaller radius", center)
            ankle_move.append(fr) #cand.center)
            break
    return ankle_move
        #if k==len(candidates_per_frame[-1])-1:
         #   ankle_move=[]
    #print(t, ankle_move)

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
    cv2.circle(img, (690, 290),8, [255,0,0], thickness=-1)
    cv2.circle(img, (50, 400), 8, [255,0,0], thickness=-1)
    for can in candidates: # einfuegen falls alles plotten
        cv2.rectangle(img, can[0], can[1],[255,0,0], 4)
    #cv2.rectangle(img,tuple(balls[-1][:2]), tuple(balls[-1][2:]), [255,0,0], 4)
    plt.imshow(img, 'gray')
    # plt.axis("off")
    plt.title("Detected FMO frame"+ str(frame_nr)) #+str(candidates))
    # plt.savefig("first_move_sequence/"+bsp[:-4]+"_"+str(frame_nr)) # [:400,200:700]
    # plt.savefig("/Users/ninawiedemann/Desktop/BA/fmo detection/connected_components", pad_inches=0)
    plt.show()
    print(candidates)


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

def ball_detection(candidates_per_frame, balls, metric_thresh =0.5):
    if len(balls)==0:
        for cands3 in candidates_per_frame[-3]:
            for j, cands2 in enumerate(cands3.children): # counts through children (ebene 2)
                slope = cands3.slopes[j] # slope of 3 to 2
                dist = cands3.dist[j]
                area = cands3.area_diffs[j]
                angle = cands3.angle_diffs[j]
                #print("j", j, slope, dist, area, angle)
                for k, cands1 in enumerate(cands2.children):
                    #print("k", k, cands2.slopes[k], cands2.dist[k], cands2.area_diffs[k], cands2.angle_diffs[k])
                    #print("metric: ", abs(slope-cands2.slopes[k]), abs(1- dist/cands2.dist[k]), area, cands2.area_diffs[k])
                    metric = abs(slope-cands2.slopes[k]) + abs(1- dist/cands2.dist[k]) #+ area+cands2.area_diffs[k] + angle + cands2.angle_diffs[k]
                    if metric<metric_thresh:
                        balls = [cands3, cands2, cands1]
                    # if abs(slope-cands2.slopes[k]) < 0.1 and abs(dist-cands2.dist[k])<10:
                        return balls
    else:
        j = balls[-2].children.index(balls[-1])
        slope = balls[-2].slopes[j] # slope of 3 to 2
        dist = balls[-2].dist[j]
        area = balls[-2].area_diffs[j]
        angle = balls[-2].angle_diffs[j]
        for k, cands1 in enumerate(balls[-1].children):
            #print("k", k, balls[-1].slopes[k], balls[-1].dist[k], balls[-1].area_diffs[k], balls[-1].angle_diffs[k])
            #print("metric: ", abs(slope-balls[-1].slopes[k]), abs(1- dist/balls[-1].dist[k]), area, balls[-1].area_diffs[k])
            metric = abs(slope-balls[-1].slopes[k]) + abs(1- dist/balls[-1].dist[k]) # + area+balls[-1].area_diffs[k] + angle + balls[-1].angle_diffs[k]
            if metric<metric_thresh:
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

def release_frame(balls, t, images=None):
    trajectory = np.array([elem.center for elem in balls]).astype(int)
    nr_balls = len(trajectory)
    dist_from_start = distance_projected(trajectory[0], np.array([110, 140]),np.array([690, 288]))
    speed = np.mean([np.linalg.norm(trajectory[i]- trajectory[i+1]) for i in range(nr_balls-1)])
    frames_shifted = int(round(dist_from_start/speed + len(balls)-1)) # frames because current frame is already after nth ball detection
    # print(dist_from_start, speed, frames_shifted)
    ball_release = t - frames_shifted
    if images is None:
        return ball_release

    plt.figure(figsize=(10,5))
    if frames_shifted >10:
        print("ATTENTION: image does not correspond to release frame, but to", frames_shifted-10,
              "frames after release frame - image was not saved in buffer anymore")
        frames_shifted = 10
    # box = np.array(balls[0].bbox) - 80
    img = images[-frames_shifted]# [80:180,80:200]
    #img = np.tile(np.expand_dims(im_t.copy(), axis = 2), (1,1,3))
    #cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]),[255,0,0], 1)
    plt.imshow(img)
    plt.title("release frame" +str(ball_release))
    plt.gray()
    plt.axis("off")
    plt.show()
        # plt.savefig("/Users/ninawiedemann/Desktop/BA/release_frame evaluation/"+str(ball_release))
    return ball_release

def trajectory_and_speed(trajectory, fps = 30, factor_pixel_feet=10, plotting=True):
    trajectory = np.asarray(trajectory)
    # print(trajectory)
    distances = []
    for i in range(len(trajectory)-1):
        frame_difference = trajectory[i+1, 2] - trajectory[i,2]
        dist_difference = np.linalg.norm(trajectory[i,:2]- trajectory[i+1, :2])
        distances.append(dist_difference/frame_difference)
    speed = np.mean(distances)
    if plotting:
        plt.figure(figsize=(10, 5), edgecolor='r')
        plt.scatter(trajectory[:, 0], trajectory[:,1])
        plt.title("speed in mph: "+ str(speed*fps* 0.681818 /factor_pixel_feet))
        plt.ylim(400,0)
        plt.xlim(0,800)
        plt.show()
    mph = speed*fps* 0.681818 /factor_pixel_feet
    return mph

def detect_ball(folder, joints_array=None, min_area = 400, plotting=True, min_length_first=5, every_x_frame=1, roi=None):
  """
  roi: region of interest if not the whole frame is relevant, format: list [top, bottom, left, right] with top<bottom
  """
  fps = 30
  factor_pixel_feet = 10
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
  tocs = []
  while t<200:
      ret, frame = cap.read()
      if t==0:
          print(frame.shape)
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
      tic = time.time()
      im_every_x_0 = images[-2*every_x_frame]
      im_every_x_1 = images[-every_x_frame]
      im_every_x_2 = images[-1]


      cumulative_motion = _get_max_array(motion_images)
      final_frame = nd.astype(int) - cumulative_motion.astype(int)
      final_frame[final_frame < 0] = 0
      # final_frame = nd

      candidates = get_candidates(final_frame.astype(np.uint8), min_area)

      # NORMAL CANDIDATES (THREE IN A ROW)
      # candidates = get_candidates(im_tm1, im_t, im_tp1, min_area)
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
      if np.all(l>0) and len(balls)==0:
          balls = ball_detection(candidates_per_frame, balls)
          if len(balls)==3 and not ball_release_found:
              ball_release_found = True
              ball_release = release_frame(balls, t) #, images = images)
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
              if abs(mean_slope-mean_slope_previous)<0.2:
                  for i, b in enumerate(balls):
                      frame_count = len(balls)-i
                      ball_trajectory.append([b.center[0], b.center[1], t-1-frame_count])

              balls = []
              # break
          else:
              balls = new_balls
              # break
      else:
          balls=[]

      # FIRST MOVEMENT --> SHIFtED CANDIDATES

      if not first_move_found and joints_array is not None:
      # SHIFTED CANDIDATES:
          detected_moves = get_difference(im_every_x_0, im_every_x_1, im_every_x_2)
          shifted_candidates = get_candidates(detected_moves, min_area)
          #plt.figure(figsize=(10, 10), edgecolor='k')

          ### FIRST MOVEMENT:
          if shifted_candidates!=[]:
              #old = np.array(ankle_move).copy()
              ankle_move = first_movement(shifted_candidates, joints_array[t-every_x_frame], ankle_move, t)
              #if len(ankle_move)>len(old):
               #   plot(im_t, shifted_candidates, t)
          if len(ankle_move)>=min_length_first and t-ankle_move[-min_length_first]<10: #len(ankle_move)==3:
              print("first movement frame: ", (ankle_move[-min_length_first]))
              plot(im_t, shifted_candidates, t)
              first_move_found = True
              first_move_frame = ankle_move[-min_length_first]
              range_joints = joints_array[first_move_frame -10: first_move_frame +10]
              grad = range_joints # np.gradient(range_joints, axis = 0) # OHNE GRADIENT; JUST HEIGHT OF LEG
              mean_gradient = np.mean(grad[:, [7,8,10,11],1], axis = 1)
              ### gradient plotting
              # plt.plot(grad[:,:,1])
              # plt.plot(mean_gradient, c="black")
              # plt.title("black: mean height of knees and ankles")
              # plt.show()
              first_move_frame = first_move_frame-10+np.argmin(mean_gradient)
              print("Refined first movement", first_move_frame)
              # break
      if plotting and len(candidates)>0: #len(balls)>0: # ##
          plot(im_t, candidates, t)
          #plt.imshow(cumulative_motion.astype(int))
          #plt.show()

      t+=1
      images = np.roll(np.array(images), -1, axis=0)
      images[-1] = frame
      motion_images = np.roll(np.array(motion_images), -1, axis=0)
      motion_images[-1] = nd
      toc = time.time()
      tocs.append(toc-tic)
  print(np.mean(tocs))
  print(t)
  # print("time for %s frames"%t, (time.time() - start) * 1000)
  return ball_release, np.array(ball_trajectory), first_move_frame

BASE = "/scratch/nvw224/videos/atl/2017-05-04/center field/490524-0de95d55-8dc1-4f0d-9e9d-d0672ff847b3.mp4"
joints = from_json("outputs/old/cf/490524-0de95d55-8dc1-4f0d-9e9d-d0672ff847b3_pitcher.json")
tic = time.time()
ball_release, ball_trajectory, first_move_frame = detect_ball(BASE, joints_array = joints, plotting=False, min_area=30) #400
toc = time.time()
print(ball_release, ball_trajectory, first_move_frame)
print((toc-tic))
