"""
@created: 9/24/2017
@author : Ulzee An
"""

# PyTorch dependencies
import torch
import torch as T
import torch.nn as nn
from torch.autograd import Variable

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


class TensorFlowModel:
    """
    TensorFlow model credited to Michal F.
    (https://github.com/michalfaber/keras_Realtime_Multi-Person_Pose_Estimation)
    """

    def __init__(self, load_weights=True, compressed_model=None):
        self.session = tf.Session()
        K.set_session(self.session)

        def relu(x):
            return Activation('relu')(x)

        def conv(x, nf, ks, name):
            x1 = Conv2D(nf, (ks, ks), padding='same', name=name)(x)
            return x1

        def pooling(x, ks, st, name):
            x = MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)
            return x

        def vgg_block(x):

            # Block 1
            x = conv(x, 64, 3, "conv1_1")
            x = relu(x)
            x = conv(x, 64, 3, "conv1_2")
            x = relu(x)
            x = pooling(x, 2, 2, "pool1_1")

            # Block 2
            x = conv(x, 128, 3, "conv2_1")
            x = relu(x)
            x = conv(x, 128, 3, "conv2_2")
            x = relu(x)
            x = pooling(x, 2, 2, "pool2_1")

            # Block 3
            x = conv(x, 256, 3, "conv3_1")
            x = relu(x)
            x = conv(x, 256, 3, "conv3_2")
            x = relu(x)
            x = conv(x, 256, 3, "conv3_3")
            x = relu(x)
            x = conv(x, 256, 3, "conv3_4")
            x = relu(x)
            x = pooling(x, 2, 2, "pool3_1")

            # Block 4
            x = conv(x, 512, 3, "conv4_1")
            x = relu(x)
            x = conv(x, 512, 3, "conv4_2")
            x = relu(x)

            # Additional non vgg layers
            x = conv(x, 256, 3, "conv4_3_CPM")
            x = relu(x)
            x = conv(x, 128, 3, "conv4_4_CPM")
            x = relu(x)

            return x

        def stage1_block(x, num_p, branch):

            # Block 1
            x = conv(x, 128, 3, "conv5_1_CPM_L%d" % branch)
            x = relu(x)
            x = conv(x, 128, 3, "conv5_2_CPM_L%d" % branch)
            x = relu(x)
            x = conv(x, 128, 3, "conv5_3_CPM_L%d" % branch)
            x = relu(x)
            x = conv(x, 512, 1, "conv5_4_CPM_L%d" % branch)
            x = relu(x)
            x = conv(x, num_p, 1, "conv5_5_CPM_L%d" % branch)

            return x

        def stageT_block(x, num_p, stage, branch, prefix='Heatmap'):

            # Block 1
            with tf.name_scope('%sBlock' % (prefix)):
                x = conv(x, 128, 7, "Mconv1_stage%d_L%d" % (stage, branch))
                x = relu(x)
                x = conv(x, 128, 7, "Mconv2_stage%d_L%d" % (stage, branch))
                x = relu(x)
                x = conv(x, 128, 7, "Mconv3_stage%d_L%d" % (stage, branch))
                x = relu(x)
                x = conv(x, 128, 7, "Mconv4_stage%d_L%d" % (stage, branch))
                x = relu(x)
                x = conv(x, 128, 7, "Mconv5_stage%d_L%d" % (stage, branch))
                x = relu(x)
                x = conv(x, 128, 1, "Mconv6_stage%d_L%d" % (stage, branch))
                x = relu(x)
                x = conv(x, num_p, 1, "Mconv7_stage%d_L%d" % (stage, branch))

            return x

        # Hyper-parameters
        input_shape = (None,None,3)
        img_input = Input(shape=input_shape)

        stages = 6
        np_branch1 = 38
        np_branch2 = 19

        # output resize operations
        # TODO: probably better of batch resizing at the end instead of doing them discretely
        self.raw_heatmap = tf.placeholder(tf.float32, shape=(None, None, None, 19))
        self.raw_paf = tf.placeholder(tf.float32, shape=(None, None, None, 38))
        self.resize_size = tf.placeholder(tf.int32, shape=(2))

        self.resize_heatmap = tf.transpose(tf.image.resize_images(self.raw_heatmap, self.resize_size, align_corners=True), perm=[0, 3, 1, 2])
        self.resize_paf = tf.transpose(tf.image.resize_images(self.raw_paf, self.resize_size, align_corners=True), perm=[0, 3, 1, 2])

        if compressed_model:
            print '| Loading compressed model:', compressed_model
            self.model = load_model(compressed_model)

            compressed_weights = compressed_model.replace('.h5', '_w.h5')
            self.model.load_weights(compressed_weights)
            return

        # VGG
        with tf.name_scope('VggConvLayer'):
            stage0_out = vgg_block(img_input)

        # stage 1
        with tf.name_scope('DualLayer%d' % (1)):
            stage1_branch1_out = stage1_block(stage0_out, np_branch1, 1)
            stage1_branch2_out = stage1_block(stage0_out, np_branch2, 2)
            x = Concatenate()([stage1_branch1_out, stage1_branch2_out, stage0_out])

        # stage t >= 2
        for sn in range(2, stages + 1):
            with tf.name_scope('DualLayer%d' % (sn)):
                stageT_branch1_out = stageT_block(x, np_branch1, sn, 1, prefix='Heat')
                stageT_branch2_out = stageT_block(x, np_branch2, sn, 2, prefix='PAF')
                if (sn < stages):
                    x = Concatenate()([stageT_branch1_out, stageT_branch2_out, stage0_out])

        self.model = Model(img_input, [stageT_branch1_out, stageT_branch2_out])
        if load_weights:
            self.model.load_weights(TENSORFLOW_WEIGHTS_PATH)

        test_writer = tf.summary.FileWriter('logs/test', self.session.graph)

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

class PyTorchModel(nn.Module):
    """
    Refactoring code from Functions.py - original PyTorch model credited to Estelle A. and Nina W.
    """

    def __init__(self):
        super(PyTorchModel, self).__init__()

        torch.set_num_threads(torch.get_num_threads())
        blocks = {}

        block0  = [{'conv1_1':[3,64,3,1,1]},{'conv1_2':[64,64,3,1,1]},{'pool1_stage1':[2,2,0]},{'conv2_1':[64,128,3,1,1]},{'conv2_2':[128,128,3,1,1]},{'pool2_stage1':[2,2,0]},{'conv3_1':[128,256,3,1,1]},{'conv3_2':[256,256,3,1,1]},{'conv3_3':[256,256,3,1,1]},{'conv3_4':[256,256,3,1,1]},{'pool3_stage1':[2,2,0]},{'conv4_1':[256,512,3,1,1]},{'conv4_2':[512,512,3,1,1]},{'conv4_3_CPM':[512,256,3,1,1]},{'conv4_4_CPM':[256,128,3,1,1]}]

        blocks['block1_1']  = [{'conv5_1_CPM_L1':[128,128,3,1,1]},{'conv5_2_CPM_L1':[128,128,3,1,1]},{'conv5_3_CPM_L1':[128,128,3,1,1]},{'conv5_4_CPM_L1':[128,512,1,1,0]},{'conv5_5_CPM_L1':[512,38,1,1,0]}]

        blocks['block1_2']  = [{'conv5_1_CPM_L2':[128,128,3,1,1]},{'conv5_2_CPM_L2':[128,128,3,1,1]},{'conv5_3_CPM_L2':[128,128,3,1,1]},{'conv5_4_CPM_L2':[128,512,1,1,0]},{'conv5_5_CPM_L2':[512,19,1,1,0]}]

        for i in range(2,7):
            blocks['block%d_1'%i]  = [{'Mconv1_stage%d_L1'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L1'%i:[128,128,7,1,3]},
        {'Mconv5_stage%d_L1'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L1'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L1'%i:[128,38,1,1,0]}]
            blocks['block%d_2'%i]  = [{'Mconv1_stage%d_L2'%i:[185,128,7,1,3]},{'Mconv2_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv3_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv4_stage%d_L2'%i:[128,128,7,1,3]},
        {'Mconv5_stage%d_L2'%i:[128,128,7,1,3]},{'Mconv6_stage%d_L2'%i:[128,128,1,1,0]},{'Mconv7_stage%d_L2'%i:[128,19,1,1,0]}]

        def make_layers(cfg_dict):
            layers = []
            for i in range(len(cfg_dict)-1):
                one_ = cfg_dict[i]
                for k,v in one_.iteritems():
                    if 'pool' in k:
                        layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
                    else:
                        conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                        layers += [conv2d, nn.ReLU(inplace=True)]
            one_ = cfg_dict[-1].keys()
            k = one_[0]
            v = cfg_dict[-1][k]
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
            layers += [conv2d]
            return nn.Sequential(*layers)

        layers = []
        for i in range(len(block0)):
            one_ = block0[i]
            for k,v in one_.iteritems():
                if 'pool' in k:
                    layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2] )]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride = v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]

        model_dict = {}
        model_dict['block0']=nn.Sequential(*layers)

        for k,v in blocks.iteritems():
            model_dict[k] = make_layers(v)

        self.model0   = model_dict['block0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

        # load precomputed weights
        self.load_state_dict(torch.load(PYTORCH_WEIGHTS_PATH))

        if USE_GPU: self.cuda()
        self.float()
        self.eval()

    def forward (self, x):
        """
        Function internally used by PyTorch.
        This function defines how PyTorch will evaluate an example for this model.
        """

        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2  = torch.cat([out1_1,out1_2,out1],1)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3   = torch.cat([out2_1,out2_2,out1],1)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4   = torch.cat([out3_1,out3_2,out1],1)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5   = torch.cat([out4_1,out4_2,out1],1)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6   = torch.cat([out5_1,out5_2,out1],1)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)

        return out6_1,out6_2

    def evaluate(self, oriImg, scale=1.0):
        """
        Calculate the heatmap and paf given a single image.
        """

        imageToTest = cv2.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model_['stride'], model_['padValue'])
        imageToTest_padded = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5

        feed = TORCH_CUDA(Variable(T.from_numpy(imageToTest_padded)))

        output1, output2 = self(feed)

        heatmap = TORCH_CUDA(nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])))(output2)
        paf = TORCH_CUDA(nn.UpsamplingBilinear2d((oriImg.shape[0], oriImg.shape[1])))(output1)

        return (output1, output2), (heatmap[0].data.cpu().numpy(), paf[0].data.cpu().numpy())

AvailableModels = { 'tensorflow': TensorFlowModel, 'pytorch': PyTorchModel }
