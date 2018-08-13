
# Movement classification

This unit of the framework contains files to recognize actions on the field. Basically, it is the last stage of the processing pipeline from videos to motion classes. So first, a video must be processed with the files in the folder *1_Pose_estimation*, such that joint trajectories for **one** player are outputted. For movement classification, these trajectories are used as input to a CNN called MC-CNN that can solve different classification problems.

Here, pose estimation was already run on all available videos and saved in csv files. Thus, for training and testing MC-CNN, the *cf_pitcher.csv* and *cf_batter.csv* files in the *train_data* folder serve as input data and ground truth labels. Since pose estimation is too inaccurate for side-view data, only the joint trajectories of center-field videos are used. 5% are taken as validation data, and the network is trained on the other 95% (again split into training and testing data). For the saved models, the indices saved in [*test_indices.npy*](test_indices.npy) were used as test data (indices refer to the rows in the csv files).

## Training

Run the file [*classify_movement.py*](classify_movement.py) in order to train a CNN (or other networks) to classify motion. It uses the *run*-, *model*-, and *utils*-files from the main directory to create a Tensorflow graph and train a model. Tasks and data can be specified as arguments.

Usage: classify_movement.py [-h] save_path [-training TRAINING] [-label LABEL]

Arguments:

*  save_path: indicates path to save the model if training, or path to the model for restoring if testing
* -training: if training, set True, if testing, set False (default: True)
* -label: "Pitch Type", "Play Outcome" or "Pitching Position (P)" are possible so far

Examples training:

```bash
cd 2_Movement_classification
python classify_movement.py ../saved_models/pitch_type_new
 -label="Pitch Type"

python classify_movement.py ../saved_models/pitching_position_new
 -label="Pitching Position (P)"
```

### Ten-fold-cross validation

To train in 10 fold cross validation, change line 18 in *classify_movement.py* from *"from run_thread import Runner"* to *"from run_10fold import Runner"*. Then, training will be executed ten times, each time on different 90% of the data. The mean accuracies are saved in the file *ten_fold_results.json* in the end.

### Parameters

All parameters are set in the [*config.py*](config.py) file.

* Specify if the number of classes for the pitch type (3 superclasses) should be reduced, or if the data should be restricted to 5 players or to one pitching position
* Hyperparameters for the CNN architecture can be set (number of layers, learning rate, number of epochs etc.)
* For completely different ANN architectures, change line 49 in *classify_movement.py*. Available ANNs:
	* "adjustable conv1d": default, MC-CNN - kernel and filter sizes can be changed in in *config.py*
	* "rnn": LSTM with number of stacked cells and hidden units as specified in the *config.py* file
	* "conv1d big": larger CNN with batch normalization (see *model.py* file in main directory)


## Testing

### Validation data

Running *classify_movement.py*, tests are automatically run on data from the same csv file (only 95% of this file is taken for training, the other 5% for validation). Make sure that the same configuration (number of players and position included, labels sorted into super classes) as for training is set in the [*config.py*](config.py) file. 

Example:

```bash
cd 2_Movement_classification
python classify_movement.py saved_models/pitch_type
 -label="Pitch Type" -training=False
```

(This tests the pre-trained model on the rows of the csv corresponding to the saved *test_indices*, on the task of classifying the pitch type)

### Own data

Use the test file in the main directory to input your own data as a numpy array (no labels required, but can be added for comparison). Use [*test.py*](../test.py) with the approriate model from [*saved_models*](../saved_models) in main folder with your input data.

### Available models

The models saved in the folder [*saved_models*](../saved_models) in the main directory are all trained as explained above: 5% of the csv file were excluded from training and are saved in *train_indices.npy* to serve as validation data. The saved models can thus be tested on this data. The accuracies should be similar to the ones in the table in section 6.6.2 of my thesis.

* pitching_position: 3 classes, trained on all data in *cf_pitcher.csv* for which position labels are available
* play_outcome: Takes data from the *cf_batter.csv* file (batter trajectories)
* pitch_type: All players and positions included (10 classes), with *cf_pitcher.csv*
* pitch_type_5players: Only the data of the five players with most data is used. Since these five pitchers do not use all pitch types, only 7 classes are included
* pitch_type_superclasses: The pitch type classification task is simplified, by sorting the pitch types into superclasses: Fastballs, Breaking Balls and Changeups.
* pitch_type_super_5: Both restrictions from above together: Only the five players with most data and only three classes.

\pagebreak
