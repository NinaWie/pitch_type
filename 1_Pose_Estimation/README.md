# Pose estimation

In this folder, files for pose estimation, player localization and filtering of the joint trajectories are provided. For pose estimation itself, a pre-trained model from the work of [@pose_estimation] is used. All files in this directory that are not explained in this documentation belong to the code of [@pose_estimation].

## Script for pose estimation

To test pose estimation (not my own code), use the script [*pose_estimation_script.py*](pose_estimation_script.py)

Input one video and the script outputs images with plotted skeletons (without player localization, filtering etc), and a json file with the joint coordinates output, which is a list of shape #frames x #detected_persons x #joints x 2 (x and y coordinate)

Usage: 

python pose_estimation_script.py [input_file] [output_folder] [-number_frames]

* input_file: path to the input video, must be a video file (.m4v, .mp4, .avi, etc.)
* output_folder: does not need to exist yet
* number_frames: if only a certain number of frames should be processed, specify the number

Example:

```bash
cd 1_Pose_estimation
python pose_estimation_script.py "../demo_data/example_1.mp4" 
"../demo_data/demo_outputs/example_1_output"
```

(This command would process the video *example_1.mp4* in the *demo_data* folder. The outputs are saved in a new folder called *example_1_output* in *demo_data/demo_outputs*. Note that the output is worse than usually because no ROI is used)

## Process videos to joint trajectories

In my framework, a video is read frame by frame and the pose estimation network yields the skeletons of all detected persons. Then the target person is localized and the region of interest for the next frame is calculated. After processing all frames, the output trajectories are smoothed and interpolated. 

More in detail, the following steps are executed:

* Pose estimation on each frame --> a list of detected people
* Localize the target player (start position required) --> joint coordinates for each frame for the target person
* swap right and left joints because sometimes in unusual positions they are suddenly swapped
* interpolate missing values
* smooth the trajectories with a lowpass filter
* save the output in a standardized json file for each video, containing a dictionary with 2D coordinates per joint per frame

Usage: 

joint_trajectories.py [-h] input_dir output_dir center

Arguments:

* input_dir: folder with video files to be processed
* output_dir: folder where to store the json files with the output coordinates (does not need to exist before)
* center: specify what kind of file is used for specifying the center of the target person: Possible arguments:
	* "../train_data/center_dics.json" for high quality videos (json file with starting position of the target player - either pitcher or batter is filmed in these videos) 
	* "datPitcher" for all other kind of videos, if you want to get the joint trajectories for the pitcher
	* "datBatter" for all other kind of videos, if the batter is the target person

Examples:

```bash
cd 1_Pose_estimation
python joint_trajectories.py ../demo_data/ ../demo_data/demo_outputs 
datPitcher 

python joint_trajectories.py ../demo_data/ ../demo_data/demo_outputs 
datBatter 

python joint_trajectories.py 
../train_data/high_quality_videos/batter/
../demo_data/demo_outputs ../train_data/center_dics.json 

python joint_trajectories.py 
../train_data/high_quality_videos/pitcher/
../demo_data/demo_outputs ../train_data/center_dics.json 
```

Note: The "center" parameter might be confusing: It refers to the center of the hips of the target person in the first frame, which is required for localizing the target from all detected persons. However, I have tested pose estimation with two different kind of input videos: The database of MLBAM, containing ten thousands of videos of plays, and 30 high quality videos (downloaded from YouTube). For MLBAM videos, the starting position is given in a *.dat* file for each video (in the same directories). If these videos are used, then dependent on whether the target player is the Pitcher or the Batter, the center argument must be datPitcher or datBatter. Then for each video, the belonging dat file is used to get the start position of the target player. However, for the high quality videos of course no metadata is available. Thus, I just manually labeled the starting position for each video, and put the coordinates in a json file. In this case, the argument must be the path to the json file, which is [*../train_data/center_dic*](../train_data/center_dic).

### Utils for joint trajectories

* The pose estimation is done using the function *handle_one(img)* in the file [*pose_estimation_script.py*](pose_estimation_script.py)
* Localization, smoothing and interpolating functions can be found in [*data_processing.py*](data_processing.py)

### Convert json files to csv

The [*json_to_csv.py*](json_to_csv.py) file is used to take all json files of one folder and save them in a csv file instead (better for training models later than loading json files individually every time). In the file, a csv file with metadata and the folder with json files must be specified. Here, all videos were processed already. If you process them again, you can take the output files *cf_pitcher.csv* and *cf_batter.csv* as metadata.


## Player localization (visualization)

A notebook called [*player_localization.ipynb*](player_localization.ipynb) demonstrates the approach for player localization. The IoU approach can be run on a video in which the approach works, in comparison to one in which the target player is lost.


## Filtering (visualization)

A notebook called [*visualization_pose_estimation*](visualization_pose_estimation.ipynb) is used to compare different methods for filling in missing values and smoothing the joint trajectories. The data provided is all processed already. However, for the visualization in this notebook, there is one file with raw data in the *demo_data* folder (*example_1_raw.mp4*)

The (filtered) pose estimation can be plotted on a video. Unfortunately, Quick Time Player on OSX is not able to display the output videos. Use VLC or Elmedia Video Player, or Windows Media Player.

\pagebreak