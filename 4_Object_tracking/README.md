
# Object tracking

## Ball speed

Ball tracking is tested on videos from [a public database](http://ze-video.mlb.com/video/mlbam/2016/10/01/umpeval/video/449253/), filming just the ball trajectory from pitcher to batter (<1 second).

Using the Ball speed [notebook](Ball speed.ipynb) for ball detection and speed, firstly the for each videos the ball trajectory is retrieved, running the script [fmo_detection.py](../fmo_detection.py).

I saved the ball trajectories of each camera for each one-second-video in json files, that can be found in the same folder as the data: [../train_data/pitchfx](../train_data/pitchfx)

The trajectories were then transferred to the 3D domain and speed was estimated with linear regression CODE EINFUEGEN. The outputs are saved in the folder [outputs](outputs) as reports (in csv format) for both side view cameras.



## Bat and glove detection

The bat is detected by a combination of Faster R-CNN and FMO-C. All relevant code is in the Bat Detection [notebook](Bat Detection.ipynb). The notebook uses the Faster R-CNN implementation and models from the Tensorflow Object Detection API [@tensorflowAPI]. The code from the API is all found in the folder [models](models).

As data for bat detection I use the high quality videos in the train_data folder in the main directory. The output of bat and glove detection is stored as a json file in [outputs](outputs) and can also be plotted on a video which is saved in the same folder.

\pagebreak

# Bibliography

