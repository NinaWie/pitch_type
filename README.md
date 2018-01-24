# Lego Tracker

Baseball analysis system

[Modules for baseball analysis](assets/data_preprocessing.png)
{#fig: data_preprocessing}

![Modules for baseball analysis](assets/data_preprocessing.png)
{#fig: data_preprocessing}

## 1. Player tracking:
* Instructions on running pose estimation, localization and smoothing in [pose estimation readme](Pose_Estimation/README.md)

## 2. Testing
* Usage of ALL TEST FUNCTIONS is demonstrated in [this notebook](demo.ipynb)
* Event detection: Test functions for release frame, batter movement and pitcher's first move in [detect events](detect_events.py)
* Movement classification: use [test file](test.py) with approriate model from [models](saved_models) and your input data
* Fast moving object detection: [this file](fom_detection.py) contains a script and function taking a video as input, outputting pitcher's first movement frame index, release frame index and ball trajectory and speed
  * hyperparameters can be set in [config](config.py)
  * for usage type $ python fom_detection.py --help

## 3. Training
* Data can partly be found in [this folder](train_data), but large files need to be downloaded from Google Drive
* [run detect events](run_events.py) and [run classify movement](run_thread.py) are classes to train a model in tensorflow - used by all train files - input is just data, labels, hyperparameters and a selection which model should be used
* Event detection: see train files (e.g. [release](release_frame_train.py)
* Movement classification in [this](classify_movement.py) file
* For usage (arguments) type $ python classify_movement.py --help
* Testing functions in the train files refer to testing on specific data, e.g. high quality videos to check for visualization - for own data please use testing functions from 2)
* change hyperparameters in [config](config.py)

# 4. Utils:
* Single files with helper functions can be found in [utils](utils)
* [Tools](tools.py) contains functions for calculating accuracy, extending data, balancing data - used in almost all train files
* [Preprocessing](data_preprocess.py) offers functions to read data from old csv files, from folders containing json files or new csv files

## Other folders/files:
* old_data_train_test: Old data before smoothing and new localization (from Estelle's cf_data.csv and sv_data.csv files) --> training and testing is still possible, change hyperparameters in [config](config.py)
* [hyperparameters](hyperparameter_finding)
  * Trained differen convolutional and recurrent NN models
  * Tested hyperparameters systematically with csv file and with genetic programming (but tended to make only fully connected layers)
* [old versions](old_versions)
  * backup of old versions of other files, mostly useless now
* [coord_fill_in](coord_fill_in_train.py)
  * Used LSTM to learn coordinate trajectories to fill in missing values, works but not plotted yet
  * ML coord fill in [RNN here](data_preprocessing/coord_fill_in.py)
