# Movement classification

This folder is only the last stage of the processing pipeline from videos to motion classes. So first with the files in the folder 1_Pose_estimation the video must be processed, such that joint trajectories for ONE player are outputted. For movement classification, these trajectories are input to a NN called MC-CNN that can solve different classification problems.


## Training
* Data for training is available on Google Drive called for example cf_pitcher.csv, specifying viewpoint of the camera and target player. [Here](https://drive.google.com/open?id=1EQlLGq6XTSws9hSxtHVp3T6Z-0DoI_1D) these csv files can be downloaded.
* Download the files and store them in the train_data folder
* Then the classify_movement file [here](classify_movement.py) can be be executed. It uses [this](run_thread.py) file for building a tensor flow graph and training. Tasks can be specified as arguments. For usage (arguments) type 

Usage:classify_movement.py [-h] [-training TRAINING] [-label LABEL]
                            [-view VIEW]
                            save_path

positional arguments:
  save_path           indicates path to save the model if training, or path to the model for restoring if testing

optional arguments:
  -h, --help          show this help message and exit
  -training TRAINING  if training, set True, if testing, set False
  -label LABEL        Pitch Type, Play Outcome or Pitching Position (P)
                      possible so far
  -view VIEW          either cf (center field) or sv (side view)

Examples:

$ python classify_movement.py models/pitch_type_model -label="Pitch Type" -view="cf"
$ python classify_movement.py models/position_model -label="Pitching Position (P)" -view="sv"

This will train the network for the number of epochs specified in the file. 

Note:

* To train in 10 fold cross validation, change line 12 in the classify movement file to "from run_10fold import Runner" instead of "from run_thread import Runner"
* To reduce the number of classes for the pitch type (3 superclasses) or to reduce the number of players, change parameters in the training function in classify_movement.py.
* Testing functions in the train files refer to testing on specific data, e.g. high quality videos to check for visualization - for own data please use testing functions from 2)
* change hyperparameters in [config](config.py)
* [run detect events](run_events.py) and [run classify movement](run_thread.py) are classes to train a model in tensorflow - used by all train files - input is just data, labels, hyperparameters and a selection which model should be used

## Testing
* Use [test file](test.py) with approriate model from [models](saved_models) and your input data

## Other folders/files:
* old_data_train_test: Old data before smoothing and new localization (from Estelle's cf_data.csv and sv_data.csv files) --> training and testing is still possible, change hyperparameters in [config](config.py)
