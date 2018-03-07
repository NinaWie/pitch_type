import tensorflow as tf

flags = tf.app.flags


############################
#    hyper parameters      #
############################

# FMO detection
flags.DEFINE_float('metric_thresh', 0.5, 'threshold below which slopes and distance metric is classified a ball trajectory')
flags.DEFINE_integer('min_dist', 10, 'minimum distance (in pixel) a fast moving object must have travelled to be added to graph')
flags.DEFINE_float('factor_pixel_feet', 0.5, 'distance in reality in feet * factor_pixel_feet = distance on image in pixel (required to calculate speed)')
flags.DEFINE_string('pitcher_mound_coordinates', '[110, 140]', 'xy coordinate of pitchers mound center (to calculate distance of ball from pitcher)')
flags.DEFINE_string('batter_base_coordinates', '[690, 288]', 'xy coordinate of batter base center (to calculate distance of ball from pitcher)')
flags.DEFINE_integer('max_frames_first_move', 10, 'if first movement has started at frame i, it must be maximal i+10 when a sequence is classified as the first move')
flags.DEFINE_boolean('refine', True, 'set False if first movement should not be refined as the moment the leg is highest')
flags.DEFINE_integer('refine_range', 10, 'radius around predicted first movement where it can be refined')




# # FROM CAPSNET
#
# # For separate margin loss
# flags.DEFINE_float('m_plus', 0.9, 'the parameter of m plus')
# flags.DEFINE_float('m_minus', 0.1, 'the parameter of m minus')
# flags.DEFINE_float('lambda_val', 0.5, 'down weight of the loss for absent digit classes')
#
# # for training
# flags.DEFINE_integer('batch_size', 128, 'batch size')
# flags.DEFINE_integer('epoch', 50, 'epoch')
# flags.DEFINE_integer('iter_routing', 3, 'number of iterations in routing algorithm')
# flags.DEFINE_boolean('mask_with_y', True, 'use the true label to mask out target capsule or not')
#
# flags.DEFINE_float('stddev', 0.01, 'stddev for W initializer')
# flags.DEFINE_float('regularization_scale', 0.392, 'regularization coefficient for reconstruction loss, default to 0.0005*784=0.392')
#
#
# ############################
# #   environment setting    #
# ############################
# flags.DEFINE_string('dataset', 'data/mnist', 'the path for dataset')
# flags.DEFINE_boolean('is_training', True, 'train or predict phase')
# flags.DEFINE_integer('num_threads', 8, 'number of threads of enqueueing exampls')
# flags.DEFINE_string('logdir', 'logdir', 'logs directory')
# flags.DEFINE_integer('train_sum_freq', 50, 'the frequency of saving train summary(step)')
# flags.DEFINE_integer('test_sum_freq', 500, 'the frequency of saving test summary(step)')
# flags.DEFINE_integer('save_freq', 3, 'the frequency of saving model(epoch)')
# flags.DEFINE_string('results', 'results', 'path for saving results')
#
# ############################
# #   distributed setting    #
# ############################
# flags.DEFINE_integer('num_gpu', 2, 'number of gpus for distributed training')
# flags.DEFINE_integer('batch_size_per_gpu', 128, 'batch size on 1 gpu')
# flags.DEFINE_integer('thread_per_gpu', 4, 'Number of preprocessing threads per tower.')

cfg = tf.app.flags.FLAGS
# tf.logging.set_verbosity(tf.logging.INFO)
