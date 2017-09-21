# pitch_type

* Pitch type classification from videos of baseball games

### 1. Find hyperparameters for RNN 
* [hyperparameters](hyperparameter_finding)
* Trained differen convolutional and recurrent NN models
* Tested hyperparameters systematically with csv file and with genetic programming (but tended to make only fully connected layers)
* Best accuracy 64% on all data

### 2. Explore data: 
* [see plots of data here](notebooks/Evaluation.ipynb)
* Coordinate trajectories
* Investigate data by plotting mean and different examples of joints by pitch type
* Testing different interpolation/smoothing
* Filling in missing values does not work properly
* Tried different filters (Kalmann, Gaussian, Cubic and linear interpolation) to fill in the values and smoothen the curve
* Problem: Sometimes the outliers are actually the right values
* Used LSTM to learn coordinate trajectories to fill in missing values, works but not plotted yet
* ML coord fill in [RNN here](videos_to_joints/coord_fill_in.py) 

### 3. Process videos to joint coordinates
* see [Pose estimation](Pose_Estimation)
* Aim: in the end one system for real time inferences from videos
* stitched together Estelle's preprocessing and my model
* Problem with tensorflow pytorch compatability
* Evaluated times for different steps

### 4. RNN for getting pitch type directly from video
* maybe information loss when using coordinates and not videos
* trained LSTM on video arrays directly
* see this [folder](video_to_pitchtype_directly)
* see [notebook](notebooks/On_videos.ipynb) for exploration
* Maybe loss of information when getting the coordinates - therefore trained LSTM directly on videos
* up to 50% Acc
* possible todo: try 3D conv net
