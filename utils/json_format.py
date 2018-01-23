import numpy as np
import json
import time



def to_json(play, events_dic):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]
    tic = time.time()
    frames, joints, xy = bsp.shape
    dic = {}
    dic["timestamp"] = int(round(time.time() * 1000))
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
        for j in events_dic.keys():
            if i==events_dic[j]:
                dic_joints["events"].append({"timestamp": int(round(time.time() * 1000)), "name": j,"code": 1,
                                    "target_name": "Pitcher", "target_id": 1})
        dic["frames"].append(dic_joints)

    with open("test_json_format.json", 'w') as outfile:
        json.dump(dic, outfile, indent=10)
    print(time.time()-tic)


to_json(bsp, {"release_frame": 4, "fist_move": 2})

def from_json(file):
    coordinates = ["x", "y"]
    joints_list = ["right_shoulder", "right_elbow", "right_wrist", "left_shoulder","left_elbow", "left_wrist",
            "right_hip", "right_knee", "right_ankle", "left_hip", "left_knee", "left_ankle",
            "right_eye", "right_ear","left_eye", "left_ear", "nose ", "neck"]
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
