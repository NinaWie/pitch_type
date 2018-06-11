import tensorflow as tf
import os
import inspect

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# WORKING DIRECTORY
#path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
#path_of_parent_dir = os.path.abspath(os.path.join(path, os.pardir))
#print("main dir", path_of_parent_dir)
#flags.DEFINE_string('main_directory', path_of_parent_dir, 'parent directory of current subdirectory, required to load data etc')

# PARAMETERS: FILTERING OF TRAINING DATA:
flags.DEFINE_integer('nr_frames', 160, "number of frames that play should be cut to")

flags.DEFINE_boolean('five_players', False, 'set true if the network should only be trained on the five players with most data')
flags.DEFINE_boolean('super_classes', False, 'set true if the pitch type labels should be sorted into three superclasses (Fastballs, Breaking Balls and Changeups) instead of 10 classes')
flags.DEFINE_string('position', "", 'for training to recognize pitch types, the data can be filtered to include only one Pitching position. Set this variable to Windup or Stretch if the data should be filtered this way')
# NETWORK CONSTANTS
flags.DEFINE_integer('epochs', 40, "Number of epochs to train")
flags.DEFINE_integer('batch_size', 40, "Batch size")
flags.DEFINE_integer('batches_per_epoch', 100, "Number of batches in each epoch")
flags.DEFINE_float('dropout', 0, "Dropout rate if included in model")
flags.DEFINE_float('learning_rate', 0.0005, "learning rate")
flags.DEFINE_integer('first_filters', 128, "number of filters in first convolutional layer")
flags.DEFINE_integer('first_kernel', 5, "kernel size in first convolutional layer")
flags.DEFINE_integer('second_conv_filter', 128, "number of filters in second convolutional layer")
flags.DEFINE_integer('second_conv_kernel', 9, "kernel size in second convolutional layer")
flags.DEFINE_integer('first_hidden_dense', 128, "number of neurons in first forward layer")
flags.DEFINE_integer('second_hidden_dense', 0, "number of neurons in second forward layer")


# for LSTM:

flags.DEFINE_integer('layers_lstm', 4, "number of stacked lstm cells")
flags.DEFINE_integer('hidden_lstm', 128, "number of hidden layers in one lstm cell")



# # OLD DATA TRAIN TEST
# flags.DEFINE_boolean('head_out', True, 'true if only lower and upper body, but not the head, should be used')
# flags.DEFINE_boolean('align', False, 'align frames by release frame')
# flags.DEFINE_string('filter_position', " ", 'set to windup if only windup pitches shall be included, stretch if only stretch')
# flags.DEFINE_string('data_path', "train_data/cf_data.csv", 'path for arrays and labels in one csv file')
# flags.DEFINE_boolean('superclasses', False, 'if classes should be sorted in superclasses Fastball, Breaking ball and Changup')
# flags.DEFINE_integer('min_class_members', 10, 'classes with less than x data examples are excluded')

cfg = tf.app.flags.FLAGS
