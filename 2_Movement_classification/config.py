import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# NETWORK CONSTANTS
flags.DEFINE_integer('batch_size', 40, "Batch size")
flags.DEFINE_integer('batches_per_epoch', 100, "Number of batches in each epoch")
flags.DEFINE_float('dropout', 0.4, "Dropout rate if included in model")
flags.DEFINE_float('learning_rate', 0.0005, "learning rate")
flags.DEFINE_integer('first_filters', 128, "number of filters in first convolutional layer)
flags.DEFINE_integer('first_kernel', 5, "kernel size in first convolutional layer)
flags.DEFINE_integer('second_conv_filter', 128, "number of filters in second convolutional layer)
flags.DEFINE_integer('second_conv_kernel', 9, "kernel size in second convolutional layer)
flags.DEFINE_integer('first_hidden_dense', 128, "number of neurons in first forward layer)
flags.DEFINE_integer('second_hidden_dense', 0, "number of neurons in second forward layer)


# for LSTM:

flags.DEFINE_integer('layers_lstm', 4, "number of stacked lstm cells)
flags.DEFINE_integer('hidden_lstm', 128, "number of hidden layers in one lstm cell)



# OLD DATA TRAIN TEST
flags.DEFINE_boolean('head_out', True, 'true if only lower and upper body, but not the head, should be used')
flags.DEFINE_boolean('align', False, 'align frames by release frame')
flags.DEFINE_string('filter_position', " ", 'set to windup if only windup pitches shall be included, stretch if only stretch')
flags.DEFINE_string('data_path', "train_data/cf_data.csv", 'path for arrays and labels in one csv file')
flags.DEFINE_boolean('superclasses', False, 'if classes should be sorted in superclasses Fastball, Breaking ball and Changup')
flags.DEFINE_integer('min_class_members', 10, 'classes with less than x data examples are excluded')
