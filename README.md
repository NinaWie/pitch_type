\pagebreak

![Framework overview](assets/system_overview.png)
{#fig: system_overview}

![Modules](assets/game_analysis_system.png)
{#fig: game_analysis_system}

# Overview (Main directory)

The proposed framework consists of three main modules: player tracking, object tracking and event detection. Different methods are combined for each module. The codes is also sorted into these three modules, where player tracking is further divided into pose estimation and movement classification.

## Dependencies:
To create a temporary environment in anaconda:

```bash
conda env create -f environment.yml
source activate baseball_analysis
```
Otherwise see requirements.txt file - can be installed in anaconda with 

```bash
conda install --yes --file requirements.txt
```
or
```bash
while read requirement; do conda install --yes requirement; done < requirements.txt
```


## Train data

All data required for training is contained in the [train_data](train_data) folder:

### Video data:
* The major video database consists of videos of a center-field and a side-view videos, one for each play (6 seconds length). The videos are sorted by data are stored in [train_data/atl](train_data/atl).
* For bat detection, high quality videos from YouTube are used. They are stored in [train_data/high_quality_videos](train_data/high_quality_videos).
* For ball speed, shorter videos that show only the ball trajectory (<1 second)from a public database are used. There are two side view and one center field camera. Stored in: [train_data/pitchfx](train_data/pitchfx).

### Pose estimation output:
* Joint trajectories for pitcher and batter in center field and side view videos are saved as csv files, one for each view (cf for center-field, sv for side-view) and each player (pitcher and batter): cf_pitcher.csv, cf_batter.csv, sv_pitcher.csv and sv_batter.csv
* Joint trajectories for high quality videos can be found in the folder [train_data/batter_hq_joints](train_data/batter_hq_joints)

### Metadata
* Speed labels for side view videos (for ball release frame evaluation) as json file
* Manually labeled (with gradient approach) data for the batters first step: [train_data/labels_first_batter_test](train_data/labels_first_batter_test) and [train_data/labels_first_batter_train](train_data/labels_first_batter_train)

## FMO-detection

Fast Moving Object Detection finds moving objects by thresholding difference images and searching for connecting components. Here, it is used to track ball and bat and to find the pitcher's first movement. The algorithm is build on the work in "The World of Fast Moving Objects"[@Rozumnyi2017]. 

As FMO is used for both object tracking and event detection, the script and relevant functions are stored in the main folder, in [fmo_detection.py](fmo_detection.py). The script takes a video as input, and outputs pitcher's first movement frame index (if joint trajectories were inputted as well), ball trajectory and the motion candidates for each frame. 

In order to run FMO detection on a video directly, run

fmo_detection [-h] [-min_area MIN_AREA] [video_path VIDEO_PATH]

### Parameters

Hyperparameters can be set in the [config_fmo.py](config_fmo.py) file.

## Neural Network related files

### Training

For both movement classification and for event detection ANNs are trained. Since the general code for training an ANN is the same, scripts are contained here in the main folder. The [model](model.py) file contains different ANN models, from LSTMs to one-dimensional CNNs, while the run-files are used to start training.

* run_thread.py is used for classification tasks, where classes are represented as one-hot-vectors
* run_10fold.py is used for ten fold cross validation in the experiments
* run_events.py is used to train a network to find the frame index of an event, such as the moment of the batter's first step

Runner classes can be executed as Threads.

### Testing

In the [test.py](test.py) file, any model can be loaded and data (saved as a numpy array) can be loaded and the labels are predicted. If labels are available, they can also be loaded and the accuracy is displayed. The function is mainly used in the the different modules (movement classification and event detection), but can also be executed directly with:

test.py [-h] [-labels LABELS] data_path model_path

data_path: Path to a numpy array of the data (joint trajectories)
model_path: Path to a saved model of a Neural Network to restore the model
-labels: Path to a numpy array with same length as the data, with labels for each data point

For example:

```bash
python test.py train_data/data_array.npy saved_models/pitch_type 
-labels=train_data/labels_array.npy
```

## Utils:

### Utils for data processing

In [utils.py](utils.py) functions for all kind of tasks can be found, for example calculating the accuracy per class, getting data from the csv files, shifting joint trajectories by a range of frames, etc. The functions are saved in a class Tools, so it is clearly visible if a function is saved in utils.

### Other utils of previous versions or used for filtering:

Helper files and filtering files are saved in the folder [utils_filtering](utils_filtering). For example, filtering functions used to smooth the joint trajectories are saved here. 

Conversions between csv to json files, saving videos as jumpy arrays, converting numpy arrays into a json file and similar can be found as well. In the experiments, only filtering is used though.

\pagebreak
