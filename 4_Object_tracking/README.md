# Object tracking:

## Ball speed:

Ball tracking can be tested best on 20 second videos from [a public database](http://ze-video.mlb.com/video/mlbam/2016/10/01/umpeval/video/449253/), filming just the ball trajectory from pitcher to batter.

Using the [notebook](Ball speed.ipynb) for Ball detection and speed, firstly the for each videos ball trajectory is calculated (from the script [fmo_detection.py](../fmo_detection.py) in the main directory).

Running this on the videos from the database above, I saved the ball trajectories of each camera in json files, that can be found [here](https://drive.google.com/drive/folders/12jk-r-lehDzxwWm3v3N8RC1YwHUz3il4?usp=sharing)

The trajectories were than transferred to the 3D domain and speed was estimated with linear regression CODE EINFUEGEN. The outputs are saved in the folder [outputs](outputs) as reports (in csv format) for both side view cameras.



## Bat and glove detection

The bat is found as a combination of Faster R-CNN and FMO-C. All relevant code is in the Bat Detection [notebook](Bat Detection.ipynb). The notebook uses the Faster R-CNN implementation and models from the Tensorflow Object Detection API [@tensorflowAPI]. The code from the API is all found in the folder [models](models).

As data for bat detection I use the high quality videos in the train_data folder in the main directory. The output of bat and glove detection is stored as a json file in [outputs](outputs) and can also be plotted in a video which is saved in the same folder.

\pagebreak

