# Movement classification

This folder is the last stage of the processing pipeline from videos to motion classes. So first with the files in the folder 1_Pose_estimation the video must be processed, such that joint trajectories for ONE player are outputted. For movement classification, these trajectories are input to a CNN called MC-CNN that can solve different classification problems.

For training and testing, the cf_pitcher.csv and cf_batter.csv files in the train_data folder are used. 5% are used as validation data, while a network is trained (again splitting into train and test set) on the other 95%.

## Training
* Data for training is available on Google Drive called for example cf_pitcher.csv, specifying viewpoint of the camera and target player. [Here](https://drive.google.com/open?id=1EQlLGq6XTSws9hSxtHVp3T6Z-0DoI_1D) these csv files can be downloaded.
* Download the files and store them in the train_data folder
* Then the classify_movement file [here](classify_movement.py) can be be executed. It uses [this](run_thread.py) file for building a tensor flow graph and training. Tasks can be specified as arguments. For usage (arguments) type 

Usage: classify_movement.py [-h] [-training TRAINING] [-label LABEL]
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

```bash
python classify_movement.py models/pitch_type_model -label="Pitch Type" -view="cf"
python classify_movement.py models/position_model -label="Pitching Position (P)" -view="sv"
```
This will train the network for the number of epochs specified in the file. 

### Ten-fold-cross validation:

To train in 10 fold cross validation, change line 18 in the classify movement file to "from run_10fold import Runner" instead of "from run_thread import Runner"

### Parameters:

All parameters are set in the [config](config.py) file.

* Specify if the number of classes for the pitch type (3 superclasses) should be reduced, or if the data should be restricted to 5 players or to one pitching position
* Hyperparameters for the CNN architecture can be set (number of layers, learning rate, etc.)

## Testing

### Validation data:

Test directly on data from the same csv file (only 95% of this file is taken for training, the other 5% serve as validation data). Make sure that the same configuration (number of players and position included, labels as super classes etc) as used in training is set in the config file.

### Own data:

Use the test file in the main directory to input your own data as a numpy array (no labels required, but can be added for comparison). Use [test file](test.py) with approriate model from [models](saved_models) in main folder with your input data.

### Available models:

* pitch_type: All players and positions included, 10 classes, should have around 55% test accuracy
* position: All included, should have around 96% accuracy
* play_outcome: Must be used with cf_batter.csv file (batter trajectories), should have around 98% test accuracy
