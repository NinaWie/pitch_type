from os import listdir
import json
import os

path = "SPECIFY"
output_path = "SPECIFY"
# path_batter_runs = "/Volumes/Nina Backup/low_quality_testing/batter_runs/"

with open("play_ids.json", "r") as outfile:
    files = json.load(outfile)

result_list = []
# files = []
for vid in files:
    # if "video" in vid or "labels" in vid:
    #     continue
    # else:
    #     files.append(vid[:-5])
    for m in listdir(path): # iterate through months
        date_path = os.path.join(path,m)
        for day in listdir(date_path): # iterate through days
            day_path = os.path.join(data_path, day)
            number = os.listdir(day_path)[0] # game id (six length number)

            center_path = os.path.join(day_path, number, "CENTERFIELD")
            for video in listdir(center_path):
                if video[:-4]==vid: # if video play id is the same as the first play id
                    print("found correct video file at:", center_path+video)
                    result_list.append(center_path+video) # append video and dat files
                    result_list.append(center_path+video+".dat")

from shutil import copyfile

for p in result_list:
    # print(p, output_pah+p.split("/")[-1])
    copyfile(p, output_path+p.split("/")[-1]) # comment in to copy files to a different path
