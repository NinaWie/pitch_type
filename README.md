# pitch_type

* Pitch type classification from videos of baseball games

### 1. Find hyperparameters for RNN [hyperparameters](hyperparameter finding)
* Trained differen convolutional and recurrent NN models
* Tested hyperparameters systematically with csv file and with genetic programming (but tended to make only fully connected layers)
* Best accuracy 64% on all data

### 2. Explore data: [Evaluation.ipynb](notebooks exploration/Evaluation.ipynb)
* Coordinate trajectories
* Investigate data by plotting mean and different examples of joints by pitch type

### 3. Testing different interpolation/smoothing
* Filling in missing values does not work properly
* Tried different filters (Kalmann, Gaussian, Cubic and linear interpolation) to fill in the values and smoothen the curve
* Problem: Sometimes the outliers are actually the right values
* Used LSTM to learn coordinate trajectories to fill in missing values, works but not plotted yet

### 4. One script from video to pitch type/other inferences
* Aim: in the end one system for real time inferences from videos
* stitched together Estelle's preprocessing and my model
* Problem with tensorflow pytorch compatability
* Evaluated times for different steps

### IN ON VIDEOS notebook:
* Maybe loss of information when getting the coordinates - therefore trained LSTM directly on videos
* up to 50% Acc
* possible todo: try 3D conv net
