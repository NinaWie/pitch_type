# Event detection

This folder contains test and train files for event detection. Four events can be distinguished: The pitcher's first movement, ball release, the batter's lifting of his leg and the batter's first step towards first base.

Test functions for just a single video for each event are in [detect_event](detect_event.py). These functions are demonstrated in a demo [notebook](demo.ipynb), that visualizes the results for all events for videos in the demo_data [folder](demo_data).

## Pitcher's first movement

See the related [notebook](Pitcher's first movement.ipynb). Outputs of the experiments are saved in [outputs/first_move_result_dic](outputs/first_move_result_dic).

## Ball release frame

Three different methods to find the ball release frame are presented: Using the joint trajectories to find the moment the arm is highest above the shoulders, train a CNN to learn to recognize the pitcher's stance in images, and using FMO-C to detect the ball and calculate back to ball release.

### Higher - Shoulders Release:

See explanations in the related notebook [HS_release.ipynb](HS_release.ipynb)

### Stance recognition in images:

Training and testing can be done by running [release_from_image_train.py](release_from_image_train.py).

Usage: release_from_image_train.py [-h] [-training TRAINING]
                                   [-model_save_path MODEL_SAVE_PATH]

Train ANN to find batter first step

optional arguments:
  -h, --help            show this help message and exit
  -training		if training, set True, if testing, set False
  -model_save_path	if training, path to save model, it testing, path to
                        restore model

In this file, the training and testing data is directly split by specifying the dates of games from which the videos are selected.

### Release from ball detection:

See explanations in the related notebook [Release frame from ball detection.ipynb](Release frame from ball detection.ipynb)

## Batter's lifting of the leg:

The code is a simple function in the [detect_event](detect_event.py)file. It is used in the Batter's movement notebook, where experiments are run and results are visualized.

## Batter's first step

For the batter's first step, a LSTM is trained. It learns to find the frame index of the batter's first step, given the joint trajectories of the batter as input. 

### Training

Use the [batter_first_move_train.py](batter_first_move_train.py) file to train the LSTM. 

Usage: batter_first_move_train.py [-h] [-training TRAINING]
                                  [-model_save_path MODEL_SAVE_PATH]
                                  [-data_path DATA_PATH]
optional arguments:
  -h, --help            show this help message and exit
  -training TRAINING    if training, set True, if testing, set False
  -model_save_path MODEL_SAVE_PATH
                        if training, path to save model, it testing, path to
                        restore model
  -data_path DATA_PATH  path to data for training and testing

Example:

```bash
python batter_first_move_train.py -model_save_path="../saved_models/batter_first_step_new" -training="True" -data_path="../train_data/batter_runs"
```

### Testing

Testing is done with the same script as training. In train_data/batter_runs, the data is distinguished between train and test data directly. Thus, if the argument training=False is passed, directly the correct data is processed and outputs are saved in the [outputs](outputs) folder in a json file called batter_first_move_test_outputs.

example:

```bash
python batter_first_move_train.py -model_save_path="../saved_models/batter_first_step" -training="False" -data_path="../train_data/batter_runs"
```

Experiments and visualization can be found in the Batter's movement notebook.

\pagebreak