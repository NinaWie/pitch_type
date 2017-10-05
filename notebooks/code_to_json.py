import numpy as np
import json
import time

bsp = np.load("example.npy")

coordinates = ["x", "y"]
joints_list = ["right_shoulder", "left_shoulder", "right_elbow", "right_wrist","left_elbow", "left_wrist",
        "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle", "neck ",
        "right_eye", "right_ear","left_eye", "left_ear"]

def to_json(play, first_move, release):
    tic = time.time()
    frames, joints, xy = bsp.shape
    dic = {}
    dic["timestamp"] = time.time()
    dic["device"] = "?"
    dic["deployment"] = "?"
    dic["frames"] = []
    for i in range(frames):
        dic_joints = {}
        for j in range(joints):
            dic_xy = {}
            for k in range(xy):
                dic_xy[coordinates[k]] = bsp[i,j,k]
            dic_joints[joints_list[j]] = dic_xy
        dic_joints["events"]=[]
        if i==first_move:
            dic_joints["events"].append({"timestamp": time.time(), "name": "Pitcher's first movement","code": 1,
                                    "target_name": "Pitcher", "target_id": 1})
        if i ==release:
            dic_joints["events"].append({"timestamp": time.time(), "name": "Pitcher ball release","code": 2,
                                    "target_name": "Pitcher", "target_id": 1})
        dic["frames"].append(dic_joints)

    with open("test_json_format.json", 'w') as outfile:
        json.dump(dic, outfile, indent=10)
    print(time.time()-tic)


to_json(bsp, 2, 4)

def from_json(file):
    tic = time.time()
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

    print(time.time()-tic)
    return np.array(liste)

arr = from_json("test_json_format.json")

print(arr.shape)
