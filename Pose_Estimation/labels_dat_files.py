
import os
from os import listdir
import numpy as np

dir = "/scratch/nvw224/videos_new"
center_field_dic = {}
months = listdir(dir)
for month in months:
    days = listdir(os.path.join(dir, month))
    print(days)
    for day in days:
        nr = listdir(os.path.join(dir, month, day))[0]
        cf = listdir(os.path.join(dir, month, day, nr, "CENTERFIELD"))
        for files in cf:
            if files[-4:]=".dat":
                name = files.split(".")[0]
                for i in open(f+".dat").readlines():
                    datContent=ast.literal_eval(i)
                frame_index = datContent["pitch_frame_index"]
                center_field_dic[name]=float(frame_index)

with open("/scratch/nvw224/pitch_type/Pose_Estimation/release_frame_new_cf.json", "w") as outfile:
    json.dump(center_field_dic, outfile)
