# Pose estimation

In this folder, files for pose estimation, player localization and filtering of the joint trajectories are provided. For pose estimation itself, a pre-trained model from the work of [@pose_estimation] is used. All files in this directory that are not explained in this documentation belong to the code of [@pose_estimation].

## Script for pose estimation

To test pose estimation (No own code!), use the script [pose_estimation_script.py](pose_estimation_script.py)

Input one video and the script outputs images with plotted skeletons (without player localization, filtering etc), and a json file with the joint coordinates output, which is a list of shape #frames x #detected_persons x #joints x 2 (x and y coordinate)

Usage: 

python pose_estimation_script.py [input_file] [output_folder] [-number_frames]

* input_file: path to the input video, must be a video file (.m4v, .mp4, .avi, etc.)
* output_folder: does not need to exist yet
* number_frames: if only a certain number of frames should be processed, specify the number

## Process videos to joint trajectories

In my framework, the scripts read a video frame by frame and use the pose estimation network to yield the skeletons of all detected persons. Then the target person is localized and the region of interest for the next frame is calculated. After processing all frames, the output trajectories are smoothed and interpolated. 

More in detail, the following steps are executed:

* Pose estimation on each frame --> list of detected people
* Localize the target player (start position required) --> joint coordinates for each frame for the target person
* swap right and left joints because sometimes in unusual positions they are suddenly swapped
* interpolate missing values
* smooth the trajectories with a lowpass filter
* save the output in a standardized json file for each video, containing a dictionary with 2D coordinates per joint per frame

Usage: 

joint_trajectories.py [-h] input_dir output_did center

Pose Estimation Baseball

positional arguments:

* input_dir: folder with video files to be processed
* output_dir: folder where to store the json files with the output coordinates (does not need to exist before)
* center: specify what kind of file is used for specifying the center of the target person: either path_to_json_dictionary.json, or datPitcher, or datBatter

Examples:

```bash
python joint_trajectories.py demo_data/ out_demo/ datPitcher 
python joint_trajectories.py hq_videos/ out_hq/ center_dics.json 
```

Note: The "center" parameter might be confusing: It refers to the center of the hips of the target person in the first frame, which is required for localizing the target from all detected persons. However, I have tested pose estimation with two different kind of input videos: The database of MLBAM, containing ten thousands of videos of plays, and 30 high quality videos on the other hand. For MLBAM videos, the starting position is giving in a .dat file for each video. If these videos are used, then dependent on whether the target player is the Pitcher or the Batter, the center argument must be datPitcher or datBatter. Then for each video the belonging dat file is used. If the high quality videos are used though, no .dat files are available, so I just manually made a json file containing the starting point for each video file in a dictionary. In this case, the argument must be the file to the json path, which can be found in train_data/center_dic.

### Files that are imported in the joint trajectories script:

* The pose estimation is done using the function handle_one(img) in the file pose_estimation_script.py
* Localization, smoothing and interpolating functions can be found in [data_processing.py](data_processing.py)

### Run for all videos:

To process all files from the Atlanta stadium (by date), use [from_video_to_joints](from_video_to_joints.py). The script iterates through all folders of the video data and processes each video. The outputs are saved as json files per video separate folders for center field and side view videos.

Finally, the [json_to_csv](json_to_csv.py) file is used to take all json files of one folder and save them in a csv file instead (better for training models later than loading json files individually every time)


## Player localization (visualization)

A [notebook](player_localization.ipynb) called player_localization demonstrates the approach for player localization, comparing the results on a video in which the approach works to one in which the target player is los.


## Filtering (visualization

A [notebook](visualization_pose_estimation.ipynb) called visualization_pose_estimation is used to color videos compare different methods for filling in missing values and smoothing/filtering the joint trajectories.

\pagebreak