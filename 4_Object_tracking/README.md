
# Object tracking

## Ball speed

Ball tracking is tested on videos from [a public database](http://ze-video.mlb.com/video/mlbam/2016/10/01/umpeval/video/449253/), filming just the ball trajectory from pitcher to batter (less than one second long).

Experiments are run in a three step process: Firstly the 2D trajectory is estimated with FMO-C, and the outputs are saved in json files in the folder [*ball_speed_from_2D/449253*](ball_speed_from_2D/449253) (because the game ID of the processed data is 449253). Secondly, the 2D trajectories are transformed into 3D and the speed is estimated. I did not implement this myself but used code from Prof. Dietrich. Thirdly, the speed outputs are plotted.

### 2D trajectory

The notebook [*ball_speed.ipynb*](ball_speed.ipynb) is used to detect the ball in all videos and save the trajectories as json files. For ball detection, the script [*fmo_detection.py*](../fmo_detection.py) is used. The 2D trajectories are saved in one file for each one-second-video (dictionary with the results for all three cameras). These json files are saved in [*ball_speed_from_2D/449253*](ball_speed_from_2D/449253). See the notebook for further information.

### 3D coordinates and speed

For this step, C++ code by Prof. Dietrich is used. All code can be found in the folder [*ball_speed_from_2D*](ball_speed_from_2D), which also includes a README with explanations. Unfortunately, the code can only be executed in Windows. To process all videos and output a report containing the speed, simply execute the batch file [*449253_from_camera_A*](ball_speed_from_2D/449253_from_camera_A) or [*449253_from_camera_B*](ball_speed_from_2D/449253_from_camera_B) depending on the camera. The output is a "report" csv file that is saved in the folder [*ball_speed_from_2D/449253*](ball_speed_from_2D/449253).

### Evaluation

To evaluate the results and plot the error distribution, see the notebook [*ball_speed.ipynb*](ball_speed.ipynb) again.

## Bat and glove detection

The bat is detected by a combination of Faster R-CNN and FMO-C. All relevant code and explanations are in the notebook [*bat_detection.ipynb*](bat_detection.ipynb). The notebook uses the Faster R-CNN implementation and models from the Tensorflow Object Detection API [@tensorflowAPI]. The code from the API is stored in the folder [*models*](models).

For this unit, I use the high quality videos in the *train_data* folder in the main directory. There are 44 videos available. I ran Faster R-CNN and FMO-C for each of them and stored the outputs as json file in the folder [*outputs*](outputs). In each *_fasterrcnn.json* file, the bounding boxes of bat and glove for each frame are saved. In the *_fmoc.json* files, the motion candidates of FMO-C are stored. 

In the notebook [*bat_detection.ipynb*](bat_detection.ipynb), FMO-C and Faster R-CNN are merged, tip and base coordinates of the bat are be derived with the wrist position, and the outputs are visualized. They can also be plotted on a video which is saved in the [*outputs*](outputs) as well.

### Run FMO-C and Faster R-CNN on all videos

For the experiments though, it is unhandy to run the notebook for each video to get the detection rates. Thus, I created two other files that can process all video at once: Firstly, in [*bat_detection.py*](bat_detection.py) FMO-C and Faster R-CNN are applied on all frames of all videos. The outputs are saved in json files as described above. 

In order to run FMO-C and Faster R-CNN for all videos, run

```bash
cd 4_Object_tracking
python bat_detection.py
```

### Detection rates

Secondly, [*bat_experiments.py*](bat_experiments.py) contains the same code as in the notebook for loading the Faster R-CNN and FMO-C results, merging them and displaying the detection rates. 


```bash
cd 4_Object_tracking
python bat_experiments.py
```

prints the results. Since the Faster R-CNN and FMO-C are only evaluated on the frames during the swing, the swing frames for each video had to be found manually. The start and end frame of the swing for each video is saved in [*swing_frames_bat.json*](swing_frames_bat.json). This is loaded in both the notebook and in [*bat_experiments.py*](bat_experiments.py) in order to calculate the detection rates.

\pagebreak

# Bibliography

