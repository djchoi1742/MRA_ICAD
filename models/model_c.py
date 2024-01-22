import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import sklearn.metrics
from tensorflow.keras import backend as K
import numpy as np
import cv2
import pandas as pd
import itertools
import sys, re
sys.path.append('/workspace/bitbucket/MRA')

from models.blocks import _conv2d, _pool2d, _deconv2d, _convlstm2d, _conv3d, _pool3d, _deconv3d, _avgpool2d
from models.blocks import se_block_2d, se_block_3d


class Model03:  # only segmentation
    def __init__(self, input_size, f_num, is_training, t=True, **kwargs):

        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)
            concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

            print(decodes[j])

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        self.cam_layer_name = 'time_distributed_8'
        self.model = Model(inputs=self.images, outputs=seg_probs)

        print(self.model.get_layer(self.cam_layer_name).output)


class Model04:  # Spider U-Net Classification branch
    def __init__(self, input_size, f_num, is_training, t=True, add_conv=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images
        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        concats, decodes = {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        if add_conv:
            seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
            bot_gap = GlobalAveragePooling3D()(seg_conv)
        else:
            bot_gap = GlobalAveragePooling3D()(decodes[b_num])

        # bot_gap = GlobalAveragePooling3D()(bot_conv)
        cls_logits = layers.Dense(1, activation=None)(bot_gap)
        cls_probs = tf.nn.sigmoid(cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8

        self.model = Model(inputs=self.images, outputs=cls_probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output, self.model.output])


class Model05:  # 2D Classification
    def __init__(self, input_size, f_num, is_training, t=False, add_conv=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 3

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        concats, decodes = {}, {}
        decodes[b_num] = _conv2d(downs[reps], f_num[-1], 3, 1, t, is_training=is_training)

        if add_conv:
            seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
            bot_gap = GlobalAveragePooling2D()(seg_conv)
        else:
            bot_gap = GlobalAveragePooling2D()(decodes[b_num])

        # bot_gap = GlobalAveragePooling2D()(decodes[b_num])
        cls_logits = layers.Dense(1, activation=None)(bot_gap)
        cls_probs = tf.nn.sigmoid(cls_logits)

        self.cam_layer_name = 'max_pooling2d_2'
        self.model = Model(inputs=self.images, outputs=cls_probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output])


class Model06:  # 3D Classification
    def __init__(self, input_size, f_num, is_training, add_conv=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv3d(downs[i-1], f_num[i - 1], 3, 1, is_training=is_training)
            encodes[i] = _conv3d(encodes[i], f_num[i - 1], 3, 1, is_training=is_training)
            downs[i] = _pool3d(encodes[i])

        concats, decodes = {}, {}
        decodes[b_num] = _conv3d(downs[reps], f_num[-1], 3, 1, is_training=is_training)

        if add_conv:
            seg_conv = _conv3d(decodes[b_num], f_num[-1], 3, 1, is_training=is_training)
            bot_gap = GlobalAveragePooling3D()(seg_conv)
        else:
            bot_gap = GlobalAveragePooling3D()(decodes[b_num])

        # bot_gap = GlobalAveragePooling3D()(decodes[b_num])
        cls_logits = layers.Dense(1, activation=None)(bot_gap)
        cls_probs = tf.nn.sigmoid(cls_logits)

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=cls_probs)

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output])


class Model07:  # 2D U-Net
    def __init__(self, input_size, class_num, f_num, is_training, t=False):
        self.images = layers.Input(input_size)
        assert len(input_size) == 3

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        concats, decodes = {}, {}
        decodes[b_num] = _conv2d(downs[reps], f_num[-1], 3, 1, t, is_training=is_training)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            concats[j] = concatenate([encodes[j], _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)], axis=-1)
            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        self.model = Model(inputs=self.images, outputs=seg_probs)


class Model08:  # 2D U-Net + branch
    def __init__(self, input_size, class_num, f_num, is_training, t=False, add_conv=False, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 3

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        concats, decodes = {}, {}
        decodes[b_num] = _conv2d(downs[reps], f_num[-1], 3, 1, t, is_training=is_training)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            concats[j] = concatenate([encodes[j], _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)], axis=-1)
            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        if add_conv:
            seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
            bot_gap = GlobalAveragePooling2D()(seg_conv)
        else:
            bot_gap = GlobalAveragePooling2D()(decodes[b_num])

        cls_logits = tf.squeeze(layers.Dense(1, activation=None)(bot_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(cls_logits)

        self.cam_layer_name = 'batch_normalization_6'
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        # if mtl_mode:
        #     self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
        #     self.model.params = self.model.trainable_variables + [self.log_vars]
        # else:
        #     self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model09:  # Spider U-Net
    def __init__(self, input_size, class_num, f_num, is_training, t=True, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        concats, decodes = {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            concats[j] = concatenate([encodes[j], _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)], axis=-1)
            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        self.model = Model(inputs=self.images, outputs=seg_probs)


def Model10(input_size, class_num, f_num, is_training):  # Spider U-Net (default)
    inputs = Input(input_size)
    t = True if len(input_size) == 4 else False

    f_num = [*map(int, re.split(',', f_num))]
    # encoder
    encode_1 = _conv2d(_conv2d(inputs, f_num[0], 3, 1, t, is_training=is_training),
                       f_num[0], 3, 1, t, is_training=is_training)
    down_1 = _pool2d(encode_1, t)
    encode_2 = _conv2d(_conv2d(down_1, f_num[1], 3, 1, t, is_training=is_training),
                       f_num[1], 3, 1, t, is_training=is_training)
    down_2 = _pool2d(encode_2, t)
    encode_3 = _conv2d(_conv2d(down_2, f_num[2], 3, 1, t, is_training=is_training),
                       f_num[2], 3, 1, t, is_training=is_training)
    down_3 = _pool2d(encode_3, t)
    encode_4 = _conv2d(_conv2d(down_3, f_num[3], 3, 1, t, is_training=is_training),
                       f_num[3], 3, 1, t, is_training=is_training)
    down_4 = _pool2d(encode_4, t)

    bot_conv = _convlstm2d(down_4, f_num[4], 3, 1)

    # decoder
    concat_4 = concatenate([encode_4, _deconv2d(bot_conv, f_num[3], 2, 2, t)], axis=-1)
    decode_4 = _conv2d(_conv2d(concat_4, f_num[3], 3, 1, t, is_training=is_training),
                       f_num[3], 3, 1, t, is_training=is_training)
    concat_3 = concatenate([encode_3, _deconv2d(decode_4, f_num[2], 2, 2, t)], axis=-1)
    decode_3 = _conv2d(_conv2d(concat_3, f_num[2], 3, 1, t, is_training=is_training),
                       f_num[2], 3, 1, t, is_training=is_training)
    concat_2 = concatenate([encode_2, _deconv2d(decode_3, f_num[1], 2, 2, t)], axis=-1)
    decode_2 = _conv2d(_conv2d(concat_2, f_num[1], 3, 1, t, is_training=is_training),
                       f_num[1], 3, 1, t, is_training=is_training)
    concat_1 = concatenate([encode_1, _deconv2d(decode_2, f_num[0], 2, 2, t)], axis=-1)
    decode_1 = _conv2d(_conv2d(concat_1, f_num[0], 3, 1, t, is_training=is_training),
                       f_num[0], 3, 1, t, is_training=is_training)

    top_conv = _conv2d(decode_1, 32, 3, 1, t, is_training=is_training)
    logits = _conv2d(top_conv, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)


class Model11:  # Spider U-Net Classification branch
    def __init__(self, input_size, f_num, is_training, t=True, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images
        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        bot_conv = _convlstm2d(downs[reps], f_num[-1], 3, 1)
        bot_gap = TimeDistributed(GlobalAveragePooling2D(), name='bot_gap')(bot_conv)
        cls_logits = tf.squeeze(TimeDistributed(layers.Dense(1, activation=None))(bot_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8

        self.model = Model(inputs=self.images, outputs=cls_probs)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output, self.model.output])


class Model15:  # Spider U-Net (3 blocks) + branch
    def __init__(self, input_size, class_num, f_num, is_training, t=True, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        concats, decodes = {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            concats[j] = concatenate([encodes[j], _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)], axis=-1)
            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        # bot_gap = GlobalAveragePooling3D()(decodes[b_num])
        # cls_logits = layers.Dense(1, activation=None)(bot_gap)
        # cls_probs = tf.nn.sigmoid(cls_logits)

        bot_gap = TimeDistributed(GlobalAveragePooling2D(), name='bot_gap')(decodes[b_num])
        cls_logits = tf.squeeze(TimeDistributed(layers.Dense(1, activation=None))(bot_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8

        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])

        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)


class Model16:  # 3D U-Net
    def __init__(self, input_size, class_num, f_num, is_training, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv3d(downs[i - 1], f_num[i - 1], 3, 1, is_training=is_training)
            encodes[i] = _conv3d(encodes[i], f_num[i - 1], 3, 1, is_training=is_training)
            downs[i] = _pool3d(encodes[i])

        concats, decodes = {}, {}
        decodes[b_num] = _conv3d(downs[reps], f_num[-1], 3, 1, is_training=is_training)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            concats[j] = concatenate([encodes[j], _deconv3d(decodes[j + 1], f_num[j - 1], 2, 2)], axis=-1)
            decodes[j] = _conv3d(concats[j], f_num[j - 1], 3, 1, is_training=is_training)
            decodes[j] = _conv3d(decodes[j], f_num[j - 1], 3, 1, is_training=is_training)

        top_conv = _conv3d(decodes[1], 32, 3, 1, is_training=is_training)
        seg_probs = _conv3d(top_conv, class_num, 3, 1, af='sigmoid', bn=False, is_training=is_training)
        self.model = Model(inputs=self.images, outputs=seg_probs)


class Model17:  # 3D U-Net + Branch
    def __init__(self, input_size, class_num, f_num, is_training, add_conv=False, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv3d(downs[i-1], f_num[i - 1], 3, 1, is_training=is_training)
            encodes[i] = _conv3d(encodes[i], f_num[i - 1], 3, 1, is_training=is_training)
            downs[i] = _pool3d(encodes[i])

        concats, decodes = {}, {}
        decodes[b_num] = _conv3d(downs[reps], f_num[-1], 3, 1, is_training=is_training)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            concats[j] = concatenate([encodes[j], _deconv3d(decodes[j + 1], f_num[j - 1], 2, 2)], axis=-1)
            decodes[j] = _conv3d(concats[j], f_num[j - 1], 3, 1, is_training=is_training)
            decodes[j] = _conv3d(decodes[j], f_num[j - 1], 3, 1, is_training=is_training)

        top_conv = _conv3d(decodes[1], 32, 3, 1, is_training=is_training)
        seg_probs = _conv3d(top_conv, class_num, 3, 1, af='sigmoid', bn=False, is_training=is_training)

        if add_conv:
            seg_conv = _conv3d(decodes[b_num], f_num[-1], 3, 1, is_training=is_training)
            bot_gap = GlobalAveragePooling3D()(seg_conv)
        else:
            bot_gap = GlobalAveragePooling3D()(decodes[b_num])

        cls_logits = layers.Dense(1, activation=None)(bot_gap)
        cls_probs = tf.nn.sigmoid(cls_logits)

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)

        print(self.model.get_layer(self.cam_layer_name).output)

        self.dense_kernel = [v for v in self.model.weights if v.name == 'dense/kernel:0'][0]
        self.dense_bias = [v for v in self.model.weights if v.name == 'dense/bias:0'][0]


class Model18:  # Spider U-Net (3 blocks) + branch (each_ste=False)
    def __init__(self, input_size, class_num, f_num, is_training, t=True, add_conv=False, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        concats, decodes = {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            concats[j] = concatenate([encodes[j], _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)], axis=-1)
            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        if add_conv:
            seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
            bot_gap = GlobalAveragePooling3D()(seg_conv)
        else:
            bot_gap = GlobalAveragePooling3D()(decodes[b_num])
        cls_logits = layers.Dense(1, activation=None)(bot_gap)
        cls_probs = tf.nn.sigmoid(cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8

        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])

        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)


class Model19:  # 2D U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, class_num, f_num, is_training, use_ic=True, use_se=False, t=False,
                 mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 3

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i-1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}

        decodes[b_num] = _conv2d(downs[reps], f_num[-1], 3, 1, t, is_training=is_training)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)

            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = GlobalAveragePooling2D()(inner_conv)
        inner_cls_logits = layers.Dense(1, activation=None)(inner_gap)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model20:  # Spider U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, class_num, f_num, is_training, use_ic=True, use_se=False, t=True,
                 mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)

            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = GlobalAveragePooling3D()(inner_conv)
        inner_cls_logits = layers.Dense(1, activation=None)(inner_gap)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8

        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)),
                                        name='mtl_weight', trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)

        print(self.model.get_layer(self.cam_layer_name).output)


class Model202:  # Spider U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, class_num, f_num, is_training, use_ic=True, use_se=False, t=True,
                 mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)

            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        cls_det_conv = _conv2d(inner_conv, 1, 3, 1, t, is_training=is_training)
        cls_logits = GlobalAveragePooling3D()(cls_det_conv)
        cls_probs = tf.nn.sigmoid(cls_logits)

        det_logits = tf.reduce_mean(AveragePooling3D(pool_size=(8, 2, 2), padding='SAME')(cls_det_conv), axis=1)
        det_scores = tf.nn.sigmoid(det_logits)

        cls_scores = tf.reshape(cls_probs, (-1, 1, 1, 1))
        det_probs = det_scores * cls_scores

        # cls_det_conv = _conv2d(inner_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)
        # cls_probs = GlobalAveragePooling3D()(cls_det_conv)
        # cls_probs = tf.nn.sigmoid(cls_logits)

        # inner_cls_logits = layers.Dense(1, activation=None)(inner_gap)
        # cls_probs = tf.nn.sigmoid(inner_cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8

        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)),
                                        name='mtl_weight', trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)

        print(self.model.get_layer(self.cam_layer_name).output)


class Model203:  # Spider U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, class_num, f_num, is_training, use_ic=True, use_se=False, t=True,
                 mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)

            if use_se:
                deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
                encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, class_num, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)

        print(self.model.get_layer(self.cam_layer_name).output)


class Model21:  # 3D U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, class_num, f_num, is_training, use_ic=True, use_se=False,
                 mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:  # [1, 2, 3]
            encodes[i] = _conv3d(downs[i-1], f_num[i - 1], 3, 1, is_training=is_training)
            encodes[i] = _conv3d(encodes[i], f_num[i - 1], 3, 1, is_training=is_training)
            downs[i] = _pool3d(encodes[i])

        deconvs, concats, decodes = {}, {}, {}

        decodes[b_num] = _conv3d(downs[reps], f_num[-1], 3, 1, is_training=is_training)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv3d(decodes[j + 1], f_num[j - 1], 2, 2)

            if use_se:
                deconv_se = se_block_3d(_conv3d(deconvs[j], f_num[j - 1], 1, 1, is_training=is_training))
                encode_conv = _conv3d(encodes[j], f_num[j - 1], 3, 1, is_training=is_training)
                concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            else:
                concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv3d(concats[j], f_num[j - 1], 3, 1, is_training=is_training)
            decodes[j] = _conv3d(decodes[j], f_num[j - 1], 3, 1, is_training=is_training)

        top_conv = _conv3d(decodes[1], 32, 3, 1, is_training=is_training)
        seg_probs = _conv3d(top_conv, class_num, 3, 1, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv3d(decodes[b_num], f_num[-1], 3, 1, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv3d(inner_concat, f_num[-1], 3, 1, is_training=is_training)
        else:
            inner_conv = _conv3d(seg_conv, f_num[-1], 3, 1, is_training=is_training)

        inner_gap = GlobalAveragePooling3D()(inner_conv)
        inner_cls_logits = layers.Dense(1, activation=None)(inner_gap)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        self.cam_layer_name = 'batch_normalization_6'
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1]])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


if __name__ == '__main__':
    nn = Model03(input_size=[8, 256, 256, 1], class_num=1, f_num='64,112,160,208', use_ic=False, use_se=False,
                 is_training=True, mtl_mode=True)

    # print(nn.cam_model.outputs)
    nn.model.summary()
    # guided_model = built_guided_model(nn.model)

    # gb = guided_backprop(
    # guided_model, img,  'time_distributed_8')
    # ggc = deprocess_image( guided_grad_cam(gb, gradcam_heatmap))

    # for layer in nn.model.layers:
    #     print(layer.name)

