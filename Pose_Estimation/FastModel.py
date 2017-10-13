# Keras/TF dependencies
import keras
from keras.models import load_model, Sequential, Model
from keras.layers import Input, Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
import tensorflow as tf
from keras import backend as K

# Common deps
import numpy as np
import cv2
import util

from config_reader import config_reader
param_, model_ = config_reader()
PYTORCH_WEIGHTS_PATH = model_['pytorch_model']
TENSORFLOW_WEIGHTS_PATH = model_['tensorflow_model']
USE_MODEL = model_['use_model']
USE_GPU = param_['use_gpu']
TORCH_CUDA = lambda x: x.cuda() if USE_GPU else x

def relu(x):
    return Activation('relu')(x)

def conv(x, nf, ks, name):
    x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
    return x1

def pooling(x, ks, st, name):
    x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
    return x

def block_def(layer_type, args, repeats, autopool, names):
    return [layer_type, args, repeats, autopool, names, None]

VGG_BLOCK_DEF = [
    block_def('conv', (64, 3), 2, True, ['conv1_1', 'conv1_2']),
    block_def('conv', (128, 3), 2, True, ['conv2_1', 'conv2_2']),
    block_def('conv', (256, 3), 4, True, ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4']),
    block_def('conv', (512, 3), 2, False, ['conv4_1', 'conv4_2']),
    block_def('conv', (256, 3), 1, False, ['conv4_3_CPM']),
    block_def('conv', (128, 3), 1, False, ['conv4_4_CPM'])
]

STAGE1_ONE_BLOCK_DEF = [
    block_def('conv', (128, 3), 3, False, ['conv5_1_CPM_L1', 'conv5_2_CPM_L1', 'conv5_3_CPM_L1']),
    block_def('conv', (512, 1), 1, False, ['conv5_4_CPM_L1']),
    block_def('just_conv', (38, 1), 1, False, ['conv5_5_CPM_L1'])
]

STAGE1_TWO_BLOCK_DEF = [
    block_def('conv', (128, 3), 3, False, ['conv5_1_CPM_L2', 'conv5_2_CPM_L2', 'conv5_3_CPM_L2']),
    block_def('conv', (512, 1), 1, False, ['conv5_4_CPM_L2']),
    block_def('just_conv', (19, 1), 1, False, ['conv5_5_CPM_L2'])
]

def DEFINE_HEATMAP_PAF_BLOCKS(btype='heatmap'): # or paf
    first_block = STAGE1_ONE_BLOCK_DEF if btype is 'heatmap' else STAGE1_TWO_BLOCK_DEF
    output_size, branch_label = (38, 1) if btype is 'heatmap' else (19, 2)

    blocks = [first_block]
    for stage in range(2, 7):
        first_five_names = ['Mconv%d_stage%d_L%d' % (ii + 1, stage, branch_label) for ii in range(5)]

        oneblock = [
            block_def('conv', (128, 7), 5, False, first_five_names),
            block_def('conv', (128, 1), 1, False, ['Mconv6_stage%d_L%d' % (stage, branch_label)]),
            block_def('just_conv', (output_size, 1), 1, False, ['Mconv7_stage%d_L%d' % (stage, branch_label)]),
        ]
        blocks.append(oneblock)

    return blocks

HEATMAP_BLOCK_DEFS = DEFINE_HEATMAP_PAF_BLOCKS('heatmap')
PAF_BLOCK_DEFS = DEFINE_HEATMAP_PAF_BLOCKS('paf')

BUILDER_ID = 0

def build_block(inout, blockdef):
    global BUILDER_ID

    for batchdef in blockdef:
        layer_type, args, repeats, pool, names, _ = batchdef
        for ii in range(repeats):
            layer_name = names[ii]
            if layer_type is 'conv':
                inout = conv(inout, args[0], args[1], layer_name)
                batchdef[-1] = inout # keep node for later
                inout = relu(inout)
            elif layer_type is 'just_conv':
                inout = conv(inout, args[0], args[1], layer_name)
                batchdef[-1] = inout # again, keep node for later
            else:
                raise Exception('OTHER TYPES UNAVAILABLE')
        if pool:
            inout = pooling(inout, 2, 2, 'pooling_auto_%d' % BUILDER_ID)
            BUILDER_ID += 1
    return inout

class FastModel:
    """
    TensorFlow model credited to Michal F.
    (https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)
    """

    def __init__(self):
        self.session = tf.Session()

        # Hyper-parameters
        input_shape = (None,None,3)
        img_input = Input(shape=input_shape)

        # output resize operations
        # TODO: probably better of batch resizing at the end instead of doing them discretely
        self.raw_heatmap = tf.placeholder(tf.float32, shape=(None, None, None, 19))
        self.raw_paf = tf.placeholder(tf.float32, shape=(None, None, None, 38))
        self.resize_size = tf.placeholder(tf.int32, shape=(2))

        self.resize_heatmap = tf.transpose(tf.image.resize_images(self.raw_heatmap, self.resize_size, align_corners=True), perm=[0, 3, 1, 2])
        self.resize_paf = tf.transpose(tf.image.resize_images(self.raw_paf, self.resize_size, align_corners=True), perm=[0, 3, 1, 2])

        self.model = self.build_pose_estimation_model(img_input, VGG_BLOCK_DEF, HEATMAP_BLOCK_DEFS, PAF_BLOCK_DEFS)
        self.model.load_weights(TENSORFLOW_WEIGHTS_PATH)

        self.model = self.compress_model(img_input)

    def build_pose_estimation_model(self, img_input, vgg_block_def, heatmap_block_defs, paf_block_defs):
        stages = 6

        # VGG
        with tf.name_scope('VggConvLayer'):
            stage0_out = build_block(img_input, vgg_block_def)

        running_node = stage0_out
        stage_iter = 0
        for sn in range(1, stages + 1):
            with tf.name_scope('DualLayer%d' % (sn)):
                stageT_branch1_out = build_block(running_node, heatmap_block_defs[stage_iter])
                stageT_branch2_out = build_block(running_node, paf_block_defs[stage_iter])
                stage_iter += 1
                if (sn < stages):
                    running_node = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

        return Model(img_input, [stageT_branch1_out, stageT_branch2_out])

    def compress_block(self, in_block):
        out_block = []
        # return out_block
        return in_block

    def compress_model(self, img_input):
        cmp_vgg_block_def = self.compress_block(VGG_BLOCK_DEF)
        cmp_heatmap_block_defs = [self.compress_block(bdef) for bdef in HEATMAP_BLOCK_DEFS]
        cmp_paf_block_defs = [self.compress_block(bdef) for bdef in PAF_BLOCK_DEFS]

        cmp_model = self.build_pose_estimation_model(img_input, cmp_vgg_block_def, cmp_heatmap_block_defs, cmp_paf_block_defs)
        cmp_model.load_weights(TENSORFLOW_WEIGHTS_PATH)

        return cmp_model

    def evaluate(self, oriImg, scale=1.0):
        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
        input_img = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,0,1,2))/256 - 0.5;

        output1, output2 = self.model.predict(input_img)

        # Replicating bilinear upsampling to heatmaps procedure.
        resize_dict = {
            self.resize_size: [oriImg.shape[0], oriImg.shape[1]],
            self.raw_heatmap: output2,
            self.raw_paf: output1,
        }

        heatmap, paf = self.session.run([self.resize_heatmap, self.resize_paf], feed_dict=resize_dict)
        heatmap, paf = heatmap[0], paf[0]

        return (output1, output2), (heatmap, paf)