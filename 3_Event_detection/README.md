# Event detection

This folder contains testing- and training files for event detection. Four events can be distinguished: The pitcher's first movement, ball release, the batter's lifting of his leg and the batter's first step towards first base. Test functions for each of these videos can be found in [*detect_event.py*](detect_event.py). The functions basically return the frame index of an event, given the input data (video frames, joint trajectories etc.). For the experiments on more data though, other notebooks or files are used which will be explained in the following.

## Pitcher's first movement

Experiments are conducted on one game-date (one folder of videos) with different configurations. See the related notebook [*pitcher_first_movement.ipynb*](pitcher_first_movement.ipynb) for further explanations. Outputs of the experiments are saved in [*outputs/first_move_result_dic*](outputs/first_move_result_dic).

## Ball release frame

Three different methods to find the ball release frame are presented: Using the joint trajectories to find the moment the arm is highest above the shoulders, training a CNN to learn to recognize the pitcher's stance in images, and running FMO-C to detect the ball.

### Arm higher than shoulders

The moment when the arm of the pitcher is highest above his shoulders is taken as the ball release frame. See explanations and visualizations in the related notebook [*HS_release.ipynb*](HS_release.ipynb).

### Stance recognition in images

Training and testing is both implemented in [*release_from_image_train.py*](release_from_image_train.py). The data is split by specifying some game-dates as test data, and other games as training data. Then, a CNN is trained to distinguish between positive frames (ball release frame) and negative frames (other frames). In the tests, the network is applied on each frame and the one with the highest output is selected as the ball release frame. The results are visualized as box plots of the error distribution.

Usage: 

release_from_image_train.py [-h] [-training][-model_save_path]


Optional arguments:

* -training: Set to *"False"* if you want to test the model
* -model_save_path: if training: path to save model, if testing: path to
                        restore model

A pre-trained model is stored in [*saved_models/release_model*](../saved_models/release_model) in the main directory. 

Training example:

```bash
cd 3_Event_detection
python release_from_image_train.py -training=True, 
-model_save_path="../saved_models/release_model_new"
```

Testing example: (outputs a boxplot with error)

```bash
cd 3_Event_detection
python release_from_image_train.py -training=False, 
-model_save_path="../saved_models/release_model"
```

### Release from ball detection

The ball is detected with FMO-C. Then, with a rough ball speed approximation (2D) and with the distance to the pitcher, it can be concluded when ball release must have occurred. See explanations and tests in the related notebook [*release_ball_detection.ipynb*](release_ball_detection.ipynb).

## Batter's lifting of the leg

A simple thresholding and maximum approach is employed to find the moment the batter lifts his leg, with his joint trajectories as input. The code is simply a function in the [*detect_event.py*](detect_event.py) file. The experiments are demonstrated in the notebook [*batter_movement.ipynb*](batter_movement.ipynb), where the outputs are visualized together with the batter's first step.

## Batter's first step

For the batter's first step, a LSTM is trained. It learns to find the frame index of the batter's first step, given the joint trajectories of the batter as input. 

Use the [*batter_first_move_train.py*](batter_first_move_train.py) file to train and test the LSTM. In train_data/batter_runs, the data is directly distinguished between train and test data. Thus, if the argument *training=False* is passed, the corresponding data is processed. Outputs of the test labels are saved in a json file in [*outputs/batter_first_move_test_outputs.json*](outputs/batter_first_move_test_outputs.json).


Usage: 

batter_first_move_train.py [-h] [-training][-model_save_path][-data_path]

* -training: Set to *"False"* if you want to test the model (default: "True" = training)
* -model_save_path: if training: path to save model, if testing: path to restore model
* -data_path: path to data for training and testing

Training example:

```bash
cd 3_Event_detection
python batter_first_move_train.py -training="True"
-model_save_path="../saved_models/batter_first_step_new"  
-data_path="../train_data/batter_runs"
```

Testing example:

A pre-trained model is stored as [*saved_models/batter_first_step*](../saved_models/batter_first_step) in the main directory. It can be tested on the test data with:

```bash
cd 3_Event_detection
python batter_first_move_train.py -training="False"
-model_save_path="../saved_models/batter_first_step"
-data_path="../train_data/batter_runs"
```

Experiments and visualization of the results of a trained model can be found in the notebook [*batter_movement.ipynb*](batter_movement.ipynb) again. 

\pagebreak