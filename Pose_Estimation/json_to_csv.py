import pandas as pd
import numpy as np
import os
import json

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

def outcome_simplification(out):
    if pd.isnull(out):
        return np.nan
    if "Foul" in out or "Swinging strike" in out:
        return "hit"
    elif "Ball/Pitcher" in out or "Called strike" in out:
        return "no swing"
    elif "Hit into play" in out:
        return "run"
    else:
        return np.nan

cf = pd.read_csv("sv_data.csv")
path = "pitch_type/Pose_Estimation/sv_outputs/"
# path = "/Volumes/Nina Backup/finished_outputs/old_videos/cf/"
print("nr json files", len(os.listdir(path)))

df_pitcher = pd.DataFrame(columns=["Game", "Pitch Type", "balls", "prev_pitch_type", "First move","handedness", "Pitcher", "pitch_frame_index", "Pitching Position (P)", "all outcomes", "Play Outcome", "Data"]) #, index = np.unique(cf["Game"].values))
df_batter = pd.DataFrame(columns=["Game", "Play Outcome", "all outcomes", "Batter", "handedness", "balls", "Data"])

print("nr first cf", len(np.unique(cf["Game"].values)))
count = 0
for i, name in enumerate(np.unique(cf["Game"].values)):
    line = cf[cf["Game"]==name]
    f_p = path+name+"_pitcher.json"
    if  os.path.exists(f_p):
        data = from_json(f_p)
        # print(data.shape)
    else:
        # print("file not found")
        continue
    f_b = path+name+"_batter.json"
    if  os.path.exists(f_b):
        data_batter = from_json(f_b)
        # print(data.shape)
    else:
        # print("file not found")
        continue
    first_move = line["first_movement_frame_index"].values[0]
    if np.isnan(first_move) or first_move<5 or first_move>120:
        first_move = np.nan
    pitch_type = line["Pitch Type"].values[0]
    if pd.isnull(pitch_type) or pitch_type=="Unknown Pitch Type":
        pitch_type = np.nan
    position = line["Pitching Position (P)"].values[0]
    if pd.isnull(position) or (position!="Stretch" and position!="Windup"):
        position = np.nan
    release = line["pitch_frame_index"].values[0]
    if np.isnan(release) or release<70 or release>120:
        release = np.nan
    out = line["Play Outcome"].values[0]
    outcome = outcome_simplification(out)
    pitcher = line["Pitcher"].values[0]
    batter = line["Batter"].values[0]
    balls = line["balls"].values[0]
    pitcher_side =  line["Pitcher throws"].values[0]
    batter_side = line["Batter side"].values[0]
    prev_pichtype = None
    # print(name)
    df_pitcher.loc[count] = pd.Series({"Game": name, "balls":balls, "all outcomes":out, "Play Outcome": outcome, "prev_pitch_type": prev_pichtype, "Pitcher": pitcher, "handedness": pitcher_side,"Pitch Type": pitch_type, "Pitching Position (P)":position, "First move": first_move, "pitch_frame_index": release, "Data": data.tolist()})
    df_batter.loc[count] = pd.Series({"Game": name, "Batter": batter, "handedness": batter_side, "balls": balls,"all outcomes":out, "Play Outcome": outcome, "Data": data_batter.tolist()})
    count+=1

# already_there = df_pitcher["Game"].values
#
# metadata = pd.read_csv("pitch_type/Pose_Estimation/ATL_CF_metadata.csv")
# cf = pd.read_csv("csv_gameplay.csv", delimiter=";")
# print("nr second csv", len(np.unique(cf["play_id"].values)))
# for i, name in enumerate(np.unique(cf["play_id"].values)):
#     line = cf[cf["play_id"]==name]
#     line2 = metadata[metadata["play_id"]==name]
#     name = str(line["game_primary_key"].values[0])+"-"+ name
#     if name in already_there:
#         print("already there", name)
#         continue
#     f_p = path+ name +"_pitcher.json"
#     if  os.path.exists(f_p):
#         data = from_json(f_p)
#         # print(data.shape)
#     else:
#         # print("file not found")
#         continue
#     f_b = path+ name+"_batter.json"
#     if  os.path.exists(f_b):
#         data_batter = from_json(f_b)
#         # print(data.shape)
#     else:
#         # print("file not found")
#         continue
#     first_move = line2["first_movement_frame_index"].values[0]
#     if np.isnan(first_move) or first_move<5 or first_move>120:
#         #print("first move NAN")
#         first_move = np.nan
#     pitch_type = line["Pitch Type"].values[0]
#     if pd.isnull(pitch_type) or pitch_type=="Unknown Pitch Type":
#         #print("pitch type NAN")
#         pitch_type = np.nan
#     position = line["Pitching Position (P)"].values[0]
#     if pd.isnull(position) or (position!="Stretch" and position!="Windup"):
#         #print("position nan")
#         position = np.nan
#     release = line2["pitch_frame_index"].values[0]
#     if np.isnan(release) or release<70 or release>120:
#         #print("release nan")
#         release = np.nan
#     out = line["Play Outcome"].values[0]
#     outcome = outcome_simplification(out)
#     pitcher = line["Pitcher"].values[0]
#     batter = line["Batter"].values[0]
#     balls = line["pre_pitch_balls"].values[0]
#     pitcher_side =  line["Pitcher throws"].values[0]
#     batter_side = line["Batter side"].values[0]
#     prev_pichtype = line["pre_pitch_pitch_type"].values[0]
#     # print(name)
#     df_pitcher.loc[count] = pd.Series({"Game": name, "balls":balls, "prev_pitch_type": prev_pichtype, "all outcomes":out, "Play Outcome": outcome, "Pitcher": pitcher, "handedness": pitcher_side,"Pitch Type": pitch_type, "Pitching Position (P)":position, "First move": first_move, "pitch_frame_index": release, "Data": data.tolist()})
#     df_batter.loc[count] = pd.Series({"Game": name, "Batter": batter, "handedness": batter_side, "balls": balls,"all outcomes":out, "Play Outcome": outcome,  "Data": data_batter.tolist()})
#     count+=1



df_pitcher.to_csv("sv_pitcher.csv")
df_batter.to_csv("sv_batter.csv")


# retrieve
# load = pd.read_csv("df_pitcher.csv")
# print(len(np.unique(load["Game"].values)))
# new = np.array(eval(load[load["Game"]=="490251-000135df-5d5f-4b73-8bad-c7f32f207969"]["Data"].values[0]))
# print(new)
