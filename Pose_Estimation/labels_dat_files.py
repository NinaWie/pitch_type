import pandas as pd
import os
from os import listdir
import numpy as np
import json
import ast

dir = "/scratch/nvw224/videos_new/" # "/Volumes/Backup/umpeval/MIL/2017/" #
v = "CENTERFIELD"
savename = "ATL_CF_metadata.csv"

start = True
center_field_dic = {}
months = listdir(dir)
for month in months:
    days = listdir(os.path.join(dir, month))
    print(days)
    for day in days:
        nr = listdir(os.path.join(dir, month, day))[0]
        cf = listdir(os.path.join(dir, month, day, nr, v))
        for files in cf:
            if files[-4:]==".dat":
                name = files.split(".")[0]
                for i in open(os.path.join(dir, month, day, nr, v, files)).readlines():
                    datContent=ast.literal_eval(i)
                if start:
                    d = {}
                    for k in list(datContent.keys()):
                        d[k] = [datContent[k]]
                    d["play_id"] = [name]
                    start = False
                    print(d)
                else:
                    for k in list(d.keys()):
                        if k == "play_id":
                            d[k].append(name)
                        else:
                            try:
                                d[k].append(datContent[k])
                            except KeyError:
                                d[k].append(np.nan)
                                print(k)
                                continue

df = pd.DataFrame(data=d)
new_df = df.rename(columns = {"pitch_type": "Pitch Type"})
new_df.to_csv(savename)

# TO CONVERT TO CSV AFTERWARDS:
#with open("/scratch/nvw224/pitch_type/Pose_Estimation/release_frame_boston.json", "w") as outfile:
#    json.dump(center_field_dic, outfile)
# a = list(center_field_dic.keys())
# b = np.zeros(len(a))
# i=0
# for play in a:
#     b[i]=center_field_dic[play]
#     i+=1
#
# d = {"play_id":a, column:b}
