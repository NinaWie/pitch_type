# Object tracking:

## Ball speed:

Ball tracking can be tested best on 20 second videos from [this](http://ze-video.mlb.com/video/mlbam/2016/10/01/umpeval/video/449253/) database, filming just the ball trajectory from pitcher to batter.

Using the notebook for Ball detection, simply a file path to a video is input and the ball trajectory is returned from the detect_fom Function.

Running this on the videos from the database above, I saved the ball trajectories of each camera in json files, that can be found [here](https://drive.google.com/drive/folders/12jk-r-lehDzxwWm3v3N8RC1YwHUz3il4?usp=sharing)

## Bat trajectory

faster R-CNN is run separately, with the code from [here](https://github.com/rbgirshick/py-faster-rcnn)

faster R-CNN and everything else can be found in the jupyter notebook on bat tracking. As data we use the high quality videos, for which I saved the joint trajectories in json files on Google Drive [download](https://drive.google.com/drive/folders/1eD2ElaV43lkqEGQNd6EGUt8rPV6TKN9B?usp=sharing)

