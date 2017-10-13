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
import numpy.linalg as la
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

def layer_def(layer_type, args, repeats, autopool, names):
    return (layer_type, args, repeats, autopool, names)

VGG_BLOCK_DEF = [
    layer_def('conv', (64, 3), 2, True, ['conv1_1', 'conv1_2']),
    layer_def('conv', (128, 3), 2, True, ['conv2_1', 'conv2_2']),
    layer_def('conv', (256, 3), 4, True, ['conv3_1', 'conv3_2', 'conv3_3', 'conv3_4']),
    layer_def('conv', (512, 3), 2, False, ['conv4_1', 'conv4_2']),
    layer_def('conv', (256, 3), 1, False, ['conv4_3_CPM']),
    layer_def('conv', (128, 3), 1, False, ['conv4_4_CPM'])
]

STAGE1_ONE_BLOCK_DEF = [
    layer_def('conv', (128, 3), 3, False, ['conv5_1_CPM_L1', 'conv5_2_CPM_L1', 'conv5_3_CPM_L1']),
    layer_def('conv', (512, 1), 1, False, ['conv5_4_CPM_L1']),
    layer_def('just_conv', (38, 1), 1, False, ['conv5_5_CPM_L1'])
]

STAGE1_TWO_BLOCK_DEF = [
    layer_def('conv', (128, 3), 3, False, ['conv5_1_CPM_L2', 'conv5_2_CPM_L2', 'conv5_3_CPM_L2']),
    layer_def('conv', (512, 1), 1, False, ['conv5_4_CPM_L2']),
    layer_def('just_conv', (19, 1), 1, False, ['conv5_5_CPM_L2'])
]

def DEFINE_HEATMAP_PAF_BLOCKS(btype='heatmap'): # or paf
    first_block = STAGE1_ONE_BLOCK_DEF if btype is 'heatmap' else STAGE1_TWO_BLOCK_DEF
    output_size, branch_label = (38, 1) if btype is 'heatmap' else (19, 2)

    blocks = [first_block]
    for stage in range(2, 7):
        first_five_names = ['Mconv%d_stage%d_L%d' % (ii + 1, stage, branch_label) for ii in range(5)]

        oneblock = [
            layer_def('conv', (128, 7), 5, False, first_five_names),
            layer_def('conv', (128, 1), 1, False, ['Mconv6_stage%d_L%d' % (stage, branch_label)]),
            layer_def('just_conv', (output_size, 1), 1, False, ['Mconv7_stage%d_L%d' % (stage, branch_label)]),
        ]
        blocks.append(oneblock)

    return blocks

HEATMAP_BLOCK_DEFS = DEFINE_HEATMAP_PAF_BLOCKS('heatmap')
PAF_BLOCK_DEFS = DEFINE_HEATMAP_PAF_BLOCKS('paf')

BUILDER_ID = 0

def build_block(inout, blockdef):
    global BUILDER_ID

    for batchdef in blockdef:
        layer_type, args, repeats, pool, names = batchdef
        for ii in range(repeats):
            layer_name = names[ii]
            if layer_type is 'conv':
                inout = conv(inout, args[0], args[1], layer_name)
                inout = relu(inout)
            elif layer_type is 'just_conv':
                inout = conv(inout, args[0], args[1], layer_name)
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

    def find_keras_node(self, with_name, some_model):
        for layer in some_model.layers:
            if layer.name is with_name:
                return layer
        return None
        # raise Exception('NO LAYER WITH NAME "%s" FOUND' % (with_name))

    def decompose_weights(self, weights, keep_ratio = 0.5):
        fx, fy, d1, d2 = weights.shape
        flattened = np.reshape(weights, (fx * fy * d1, d2))

        U, S, V = la.svd(flattened)
        keep_ranks = int(keep_ratio * len(S))

        smat = np.zeros((U.shape[1], len(S)))
        smat[:len(S), :] = np.diag(S)

        # print weights.shape
        # print U.shape, S.shape, V.shape, keep_ranks

        Amat = U.copy()[:, :]
        Amat = np.dot(Amat, smat.copy()[:, :keep_ranks])
        Amat = np.reshape(Amat, (fx, fy, d1, keep_ranks))
        Bmat = V.copy()[:keep_ranks, :]
        Bmat = np.reshape(Bmat, (1, 1, keep_ranks, d2))

        return Amat, Bmat, (U, smat, V)

    onetime = False
    def compress_block(self, in_block, printout=True):
        if printout: print '============== COMPRESS BLOCK =============='

        # Unravels any repeats for finer control on layer definitions.
        out_block = []
        for bb, batchdef in enumerate(in_block):
            layer_type, args, repeats, autopool, names = batchdef
            for ii in range(repeats):
                layer_name = names[ii]
                should_pool = autopool and (ii is repeats - 1)

                if bb is 0 and ii is 0 and layer_type is 'conv':
                # if bb is 0 and ii is 0 and layer_type is 'conv' and not self.onetime:
                    self.onetime = True
                    """
                    Try compressing only first conv layers of blocks.
                    """
                    kerasnode = self.find_keras_node(layer_name, self.model)
                    weightmat, biasvect = kerasnode.get_weights()
                    Amat, Bmat, (U, S, V) = self.decompose_weights(weightmat)
                    # FIXME: should_pool will be incorrect for final layers of blocks...
                    out_block.append(layer_def(layer_type, (Bmat.shape[2], args[1], [Amat, np.zeros(Amat.shape[-1])]), 1, should_pool, ['compressed_A_' + layer_name]))
                    out_block.append(layer_def(layer_type, (args[0], 1, [Bmat, biasvect]), 1, should_pool, ['compressed_B_' + layer_name]))

                    if printout:
                        print '| [%d]:' % (ii + 1), names[ii], '(shape: %s)' % (str(weightmat.shape))
                        print '|     * USV:', U.shape, S.shape, V.shape
                        print '|     * A-B:', Amat.shape, Bmat.shape
                else:
                    out_block.append(layer_def(layer_type, args, 1, should_pool, [layer_name]))
        return out_block

    # def transfer_weights_to_compressed(self):

    def compress_model(self, img_input):

        # Build the structure of the compressed model.
        cmp_vgg_block_def = self.compress_block(VGG_BLOCK_DEF)
        cmp_heatmap_block_defs = [self.compress_block(bdef) for bdef in HEATMAP_BLOCK_DEFS]
        cmp_paf_block_defs = [self.compress_block(bdef) for bdef in PAF_BLOCK_DEFS]

        cmp_model = self.build_pose_estimation_model(img_input, cmp_vgg_block_def, cmp_heatmap_block_defs, cmp_paf_block_defs)

        # Given the block defs of the compressed model, transfer weights from original model.
        print '============== WEIGHT TRANSFER =============='
        count_transferred = 0
        for layer in self.model.layers:
            if layer.get_weights():
                node = self.find_keras_node(layer.name, cmp_model)
                if node:
                    weights = layer.get_weights()
                    node.set_weights(weights)
                    count_transferred += 1

        count_transferred_comp = 0
        names_index = 4
        args_index = 1
        for block in [cmp_vgg_block_def] + cmp_heatmap_block_defs + cmp_paf_block_defs:
            for batchdef in block:
                layer_name = batchdef[names_index][0]
                if 'compressed' in layer_name:
                    _, _, weights = batchdef[args_index]
                    node = self.find_keras_node(layer_name, cmp_model)
                    node.set_weights(weights)
                    count_transferred_comp += 1
        print '| Transferred Original %d weights.' % count_transferred
        print '| Set Compressed       %d weights.' % count_transferred_comp

        cmp_model.summary()
        self.model.summary()

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