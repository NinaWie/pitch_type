# Movement classification

This folder is only the last stage of the processing pipeline from videos to motion classes. So first with the files in the folder 1_Pose_estimation the video must be processed, such that joint trajectories for ONE player are outputted. For movement classification, these trajectories are input to a NN called MC-CNN that can solve different classification problems.


## 3. Training
* Data for training is available on Google Drive called for example cf_pitcher.csv, specifying viewpoint of the camera and target player. (https://drive.google.com/open?id=1EQlLGq6XTSws9hSxtHVp3T6Z-0DoI_1D)
* Download the files and store them in the train_data folder
* Then the classify_movement.py file can be be executed. Tasks can be specified as arguments. Examples:

$ python classify_movement.py models/pitch_type_model -label="Pitch Type" -view="cf"
$ python classify_movement.py models/position_model -label="Pitching Position (P)" -view="sv"

This will train the network for the number of epochs specified in the file. To reduce the number of classes for the pitch type (3 superclasses) or to reduce the number of players, change parameters in the training function in classify_movement.py.

* [run detect events](run_events.py) and [run classify movement](run_thread.py) are classes to train a model in tensorflow - used by all train files - input is just data, labels, hyperparameters and a selection which model should be used
* Event detection: see train files (e.g. [release](release_frame_train.py)
* Movement classification in [this](classify_movement.py) file
* For usage (arguments) type $ python classify_movement.py --help
* Testing functions in the train files refer to testing on specific data, e.g. high quality videos to check for visualization - for own data please use testing functions from 2)
* change hyperparameters in [config](config.py)


## 2. Testing
* Usage of ALL TEST FUNCTIONS is demonstrated in [this notebook](demo.ipynb)
* Event detection: Test functions for release frame, batter movement and pitcher's first move in [detect events](detect_events.py)
* Movement classification: use [test file](test.py) with approriate model from [models](saved_models) and your input data
* Fast moving object detection: [this file](fom_detection.py) contains a script and function taking a video as input, outputting pitcher's first movement frame index, release frame index and ball trajectory and speed
  * hyperparameters can be set in [config](config.py)
  * for usage type $ python fom_detection.py --help

