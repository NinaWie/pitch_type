# Lego Tracker

* Baseball analysis system
![Modules for baseball analysis](assets/data_preprocessing.png){#fig:data_preprocessing}

## 1. Player tracking:
* Instructions on running pose estimation, localization and smoothing in [pose estimation readme](Pose_Estimation/README.md)

## 2. Testing
* Usage of ALL TEST FUNCTIONS is demonstrated in [this notebook](demo.ipynb)
* Event detection: Test functions for release frame, batter movement and pitcher's first move in [detect events](detect_events.py)
* Movement classification: use [test file](test.py) with approriate model from [models](saved_models) and your input data



# Other folders/files:
* [hyperparameters](hyperparameter_finding)
  * Trained differen convolutional and recurrent NN models
  * Tested hyperparameters systematically with csv file and with genetic programming (but tended to make only fully connected layers)
* [old versions](old_versions)
  * backup of old versions of other files, mostly useless now
* [coord_fill_in](coord_fill_in_train.py)
  * Used LSTM to learn coordinate trajectories to fill in missing values, works but not plotted yet
  * ML coord fill in [RNN here](data_preprocessing/coord_fill_in.py)
