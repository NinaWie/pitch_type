# Pose estimation

For all versions, you first need to download the model: `cd model; sh get_model.sh`

Simple test version of pose estimation: input one video and outputs images with plotted skeletons: [here](pose_estimation_script.py) - without player localization, filtering etc

usage: 

python pose_estimation_script.py path_to_input_video output_directory -number_of_frames

output_directory: does not need to exist yet
path_to_input_video: must be a video file (.m4v, .mp4, .avi, etc.)
number_of_frames: if only a some frames should be processed

### Process videos to joint trajectories

In this folder, most files just belong to the original Pose estimation network. In our framework, the scripts read a video frame by frame and use the pose estimation network in to output the skeletons of all detected persons. Then the target person is localized and the region of interest for the next frame is calculated. After processing all frames, the output trajectories are smoothed and interpolated. 

Following steps are executed:
* Pose estimation on each frame --> list of detected people
* Localize the target player (start position required) --> joint coordinates for each frame for the target person
* swap right and left joints because sometimes in unusual positions they are suddenly swapped
* interpolate missing values
* filter the trajectories with a lowpass filter
* save the output in a standardized json file for each video, containing a dictionary with 2D coordinates per joint per frame

Two different files for either testing a few videos or all available videos:

* To process all files from the Atlanta stadium (by date), use [folder script](from_video_to_joints.py)
* For some example files in one folder, use [file script](joint_trajectories.py)

usage: joint_trajectories.py [-h] DIR DIR center

Pose Estimation Baseball

positional arguments:
  DIR         folder with video files to be processed
  DIR         folder where to store the json files with the output coordinates
  center      specify what kind of file is used for specifying the center of
              the target person: either path_to_json_dictionary.json, or
              datPitcher, or datBatter

optional arguments:
  -h, --help  show this help message and exit

optional arguments:
  -h, --help  show this help message and exit

Examples:

```bash
python joint_trajectories.py demo_data/ out_demo/ datPitcher 
python joint_trajectories.py high_quality_videos/ out_hq/ center_dics.json 
```

Note: The "center" parameter might be confusing: It refers to the center of the hips of the target person in the first frame, which is required for localizing the target from all detected persons. However, we have tested pose estimation with two different kind of input videos: The database of MLBAM, containing ten thousands of videos of plays, and 30 high quality videos on the other hand. For MLBAM videos, the starting position is giving in a .dat file for each video. If these videos are used, then dependent on whether the target player is the Pitcher or the Batter, the center argument must be datPitcher or datBatter. Then for each video the belonging dat file is used. If the high quality videos are used though, no .dat files are available, so we just manually made a json file containing the starting point for each video file in a dictionary. In this case, the argument must be the file to the json path, which can be found in train_data/center_dic.

### Other files in this folder:
* The pose estimation is done using the function handle_one(img) in the file pose_estimation_script.py
* Localization, smoothing and interpolating functions can be found in [data_processing](data_processing.py)
* A [notebook](visualization_pose_estimation.ipynb) is used to color videos compare different methods for filling in missing values and smoothing/filtering the joint trajectories.
* json_to_csv is used to take all output json files of one folder and save them in a csv file instead (better for training models later than loading sons individually every time)
