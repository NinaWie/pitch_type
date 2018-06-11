
# Movement classification

The 2_Movement_classifictaion folder contains files to recognize actions on the field. Basically, it is the last stage of the processing pipeline from videos to motion classes. So first, a video must be processed with the files in the folder 1_Pose_estimation, such that joint trajectories for ONE player are outputted. For movement classification, these trajectories are input to a CNN called MC-CNN that can solve different classification problems.

Here, pose estimation was already run on all available videos and saved in csv files. Thus, for training and testing MC-CNN, the cf_pitcher.csv and cf_batter.csv files in the train_data folder are used. 5% are used as validation data, while a network is trained (again splitting into train and test set) on the other 95%. For the saved models, the saved indices [test_indices.npy](test_indices.npy) were used as test data.

## Training

To train a CNN (or other networks) to classify motion, run the file [classify_movement.py](classify_movement.py). It uses the run-, model-, and utils-files from the main directory to create a tensor flow graph and start training. Tasks and data can be specified as arguments. 

Usage: classify_movement.py [-h] [-training TRAINING] [-label LABEL]
                            [-view VIEW]
                            save_path

Arguments:

*  save_path: indicates path to save the model if training, or path to the model for restoring if testing
* - training: if training, set True, if testing, set False (default: True)
* - label: "Pitch Type", "Play Outcome" or "Pitching Position (P)" are possible so far
* - view: either "cf" (center field) or "sv" (side view)

Examples:

```bash
python classify_movement.py saved_models/pitch_type
 -label="Pitch Type" -view="cf"

python classify_movement.py saved_models/pitching_position
 -label="Pitching Position (P)" -view="sv"
```

### Ten-fold-cross validation:

To train in 10 fold cross validation, change line 18 in the classify movement file to "from run_10fold import Runner" instead of "from run_thread import Runner"

### Parameters:

All parameters are set in the [config](config.py) file.

* For other ANN architectures, change the required field
* Specify if the number of classes for the pitch type (3 superclasses) should be reduced, or if the data should be restricted to 5 players or to one pitching position
* Hyperparameters for the CNN architecture can be set (number of layers, learning rate, number of epochs etc.)

## Testing

### Validation data:

Test directly on data from the same csv file (only 95% of this file is taken for training, the other 5% serve as validation data). Make sure that the same configuration (number of players and position included, labels as super classes etc) as used in training is set in the config file.

Run

```bash
python classify_movement.py saved_models/pitch_type
 -label="Pitch Type" -view="cf" -training=False
```
to test a trained model on the rows of the csv corresponding to the saved test_indices.

### Own data:

Use the test file in the main directory to input your own data as a numpy array (no labels required, but can be added for comparison). Use [test file](../test.py) with the approriate model from [saved_models](saved_models) in main folder with your input data.

### Available models:

* pitch_type: All players and positions included, 10 classes, should have around 55% test accuracy
* pitching_position: All included, should have around 96% accuracy
* play_outcome: Must be used with cf_batter.csv file (batter trajectories), should have around 98% test accuracy

\pagebreak
