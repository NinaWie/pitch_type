# Event detection

This folder contains test and train files for event detection. 

All testing functions can be found [here](detect_event.py)

## Training functions are separate files, with topics in title:
* Batter first step here: [batter_first_step](batter_first_move_train.py)
* Ball release frame and Pitcher's first movement can both be determined running [this](fom_detection.py) script on a video
* Release frame can also be determined by a neural network (not in paper), either training on joint trajectories with [release_frame_train](release_frame_train.py) file or training on single images [here](release_from_image_train.py)

## Jupiter notebooks are mostly used for tests and visualizations
* First movement outputs are evaluated in the corresponding notebook
* Data from first movement testing is available as a json [file](all_first_move_tests.json) 

