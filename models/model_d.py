import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
import numpy as np
import itertools
import sys, re
sys.path.append('/workspace/bitbucket/MRA')

from models.blocks import _conv2d, _pool2d, _deconv2d, _convlstm2d, _avgpool2d, _conv3d, _pool3d, _deconv3d
from models.blocks import _last_conv2d, _loc_conv2d, _loc_conv3d, se_block_2d, se_block_3d


def se_block(input_x, reduction_ratio=16):
    squeeze = tf.reduce_mean(input_x, axis=[2, 3], keepdims=True)  # global average pooling
    excitation = layers.Dense(units=squeeze.shape[-1] // reduction_ratio,
                              kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              activation='relu')(squeeze)
    excitation = layers.Dense(units=squeeze.shape[-1],
                              kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              activation='sigmoid')(excitation)
    return excitation


class Model203:  # Spider U-Net + Branch + Inner-connected module + Attention skip-layer fusion
    def __init__(self, input_size, class_num, f_num, is_training, use_ic=False, use_se=False, t=True,
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


class Model22:  # Backbone: 2D U-Net
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=False, mtl_mode=False, **kwargs):

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
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = GlobalAveragePooling2D()(inner_conv)
        inner_cls_logits = layers.Dense(1, activation=None)(inner_gap)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 1
        # det_conv = _conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 2
        det_conv = _loc_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 3

        det_gap = AveragePooling2D(pool_size=(16, 16), padding='SAME')(det_conv)
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_gap, det_scores])

        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model221:  # Backbone: 2D U-Net
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=False, mtl_mode=False, **kwargs):

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

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = GlobalAveragePooling2D()(inner_conv)
        inner_cls_logits = layers.Dense(1, activation=None)(inner_gap)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        det_conv = _loc_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 3

        det_gap = AveragePooling2D(pool_size=(16, 16), padding='SAME')(det_conv)
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=[cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1],
                                              det_gap, det_scores])

        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model232:  # detection path in segmentation path, no segmentation mask
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=True, mtl_mode=False, **kwargs):
        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        encodes, downs = {}, {}
        downs[0] = self.images

        encode_idx = [v + 1 for v in range(reps)]
        for i in encode_idx:
            encodes[i] = _conv2d(downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            encodes[i] = _conv2d(encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            downs[i] = _pool2d(encodes[i], t)

        deconvs, concats, decodes = {}, {}, {}
        decodes[b_num] = _convlstm2d(downs[reps], f_num[-1], 3, 1)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:
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

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        # det_conv = _conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 0, 4
        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, act=False, is_training=is_training)  # exp009, serial 2
        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, act=True, is_training=is_training)  # exp009, serial 3
        det_conv = _loc_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 5

        det_gap = AveragePooling3D(pool_size=(1, 16, 16), padding='SAME')(det_conv)
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 8, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1],
                                              det_gap, det_scores])

        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model242:  # each_ste=True, separated cls & det
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=True, mtl_mode=False, **kwargs):
        self.use_ic = use_ic
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
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        # det_conv = _conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 3
        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 1
        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, act=False, is_training=is_training)  # exp009, serial 2
        det_conv = _loc_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 5

        det_gap = AveragePooling3D(pool_size=(1, 16, 16), padding='SAME')(det_conv)
        det_probs = tf.nn.sigmoid(det_gap)

        self.cam_layer_name = 'time_distributed_8'
        self.model = Model(inputs=self.images, outputs=[seg_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], det_gap])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model25:  # FCN-RNN
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=True, mtl_mode=False,
                 det_size=16, **kwargs):
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
        decodes[b_num] = _conv2d(downs[reps], f_num[-1], 3, 1, t, is_training=is_training)

        decode_idx = list(reversed(encode_idx))
        for j in decode_idx:  # [3, 2, 1]
            deconvs[j] = _deconv2d(decodes[j + 1], f_num[j - 1], 2, 2, t)
            concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)
            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

            print(decodes[j])

        top_conv = _convlstm2d(decodes[1], 32, 3, 1)
        # top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        det_conv = _loc_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 25

        d_s = det_conv.shape[2] // det_size
        det_gap = AveragePooling3D(pool_size=(1, d_s, d_s), padding='SAME')(det_conv)  # exp009 serial 0

        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 8, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_gap, det_scores])
        self.check_model = keras.Model(inputs=self.images, outputs=det_scores)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model26:  # 3D U-Net + Detection branch
    def __init__(self, input_size, f_num, is_training, det_size=16, use_ic=False, use_se=False,
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
            encodes[i] = _conv3d(downs[i - 1], f_num[i - 1], 3, 1, is_training=is_training)
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
        seg_probs = _conv3d(top_conv, 1, 3, 1, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv3d(decodes[b_num], f_num[-1], 3, 1, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv3d(inner_concat, f_num[-1], 3, 1, is_training=is_training)
        else:
            inner_conv = _conv3d(seg_conv, f_num[-1], 3, 1, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        # det_conv = _last_conv3d(top_conv, 1, 3, 1, is_training=is_training)  # exp009, serial 2
        # det_conv = _last_conv3d(top_conv, 1, 3, 1, act=False, is_training=is_training)  # exp009, serial 3
        det_conv = _loc_conv3d(top_conv, 1, 3, 1, is_training=is_training)  # exp009, serial 4

        det_gap = tf.reduce_mean(AveragePooling3D(pool_size=(8, 16, 16), padding='SAME')(det_conv), axis=1)
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_gap, det_scores])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model262:  # 3D U-Net + Detection branch
    def __init__(self, input_size, f_num, is_training, det_size=16, seq_len=8, use_ic=False, use_se=False,
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
            encodes[i] = _conv3d(downs[i - 1], f_num[i - 1], 3, 1, is_training=is_training)
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
        seg_probs = _conv3d(top_conv, 1, 3, 1, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv3d(decodes[b_num], f_num[-1], 3, 1, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv3d(inner_concat, f_num[-1], 3, 1, is_training=is_training)
        else:
            inner_conv = _conv3d(seg_conv, f_num[-1], 3, 1, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(seq_len, activation=None)(inner_gap), axis=1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        det_conv = _loc_conv3d(top_conv, 1, 3, 1, is_training=is_training)
        det_gap = AveragePooling3D(pool_size=(1, 16, 16), padding='SAME')(det_conv)
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, seq_len, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_gap, det_scores])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model263:  # 3D U-Net + Detection branch
    def __init__(self, input_size, f_num, is_training, det_size=16, seq_len=8, use_ic=False, use_se=False,
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
            encodes[i] = _conv3d(downs[i - 1], f_num[i - 1], 3, 1, is_training=is_training)
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
        seg_probs = _conv3d(top_conv, 1, 3, 1, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv3d(decodes[b_num], f_num[-1], 3, 1, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv3d(inner_concat, f_num[-1], 3, 1, is_training=is_training)
        else:
            inner_conv = _conv3d(seg_conv, f_num[-1], 3, 1, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(seq_len, activation=None)(inner_gap), axis=1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        det_conv = _loc_conv3d(top_conv, 1, 3, 1, is_training=is_training)
        det_gap = AveragePooling3D(pool_size=(1, 16, 16), padding='SAME')(det_conv)
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, seq_len, 1, 1, 1))
        # import pdb; pdb.set_trace()
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_gap, det_scores])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model264:  # 3D U-Net + Detection branch
    def __init__(self, input_size, f_num, is_training, det_size=16, seq_len=8, use_ic=False, use_se=False,
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
            encodes[i] = _conv3d(downs[i - 1], f_num[i - 1], 3, 1, is_training=is_training)
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
        # seg_probs = _conv3d(top_conv, 1, 3, 1, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv3d(decodes[b_num], f_num[-1], 3, 1, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv3d(inner_concat, f_num[-1], 3, 1, is_training=is_training)
        else:
            inner_conv = _conv3d(seg_conv, f_num[-1], 3, 1, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(seq_len, activation=None)(inner_gap), axis=1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        det_conv = _loc_conv3d(top_conv, 1, 3, 1, is_training=is_training)
        det_gap = AveragePooling3D(pool_size=(1, 16, 16), padding='SAME')(det_conv)
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, seq_len, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'batch_normalization_6'  # or batch_normalization_6
        self.model = Model(inputs=self.images, outputs=[cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1],
                                              det_gap, det_scores])

        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model28:  # each_ste=True
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=True, mtl_mode=False,
                 det_size=16, **kwargs):
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
            # if use_se:
            #     deconv_se = se_block_2d(_conv2d(deconvs[j], f_num[j - 1], 1, 1, t, is_training=is_training))
            #     encode_conv = _conv2d(encodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            #     concats[j] = concatenate([encode_conv * deconv_se, deconvs[j]], axis=-1)
            # else:
            concats[j] = concatenate([encodes[j], deconvs[j]], axis=-1)

            decodes[j] = _conv2d(concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            decodes[j] = _conv2d(decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)

            print(decodes[j])

        top_conv = _conv2d(decodes[1], 32, 3, 1, t, is_training=is_training)
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        # det_conv = _conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 0~3,5, 13
        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, act=True, is_training=is_training)  # exp009, serial 4, 23
        # det_conv = _last_conv2d(det_conv, 1, 3, 1, t, act=False, is_training=is_training)  # add serial 24~25: conv

        det_conv = _loc_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 25
        # det_conv = _conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # serial 19~20,22: conv-relu-bn
        # det_conv = _last_conv2d(det_conv, 1, 3, 1, t, act=False, is_training=is_training)  # add serial 21: conv

        d_s = det_conv.shape[2] // det_size
        det_gap = AveragePooling3D(pool_size=(1, d_s, d_s), padding='SAME')(det_conv)  # exp009 serial 0

        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, act=False, is_training=is_training)  # exp009, serial 13~18
        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 8, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_gap, det_scores])
        self.check_model = keras.Model(inputs=self.images, outputs=det_scores)
        print(self.model.get_layer(self.cam_layer_name).output)

        if False:
            idx = 0
            for weight in self.model.trainable_variables:
                print(idx, weight.name, weight.shape)
                idx += 1


class Model282:  # each_ste=True, separated cls & det
    def __init__(self, input_size, f_num, is_training, use_ic=False, use_se=False, t=True, mtl_mode=False, **kwargs):
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
        seg_probs = _conv2d(top_conv, 1, 3, 1, t, af='sigmoid', bn=False, is_training=is_training)

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        if use_ic:
            inner_concat = concatenate([decodes[b_num], seg_conv], axis=-1)
            inner_conv = _conv2d(inner_concat, f_num[-1], 3, 1, t, is_training=is_training)
        else:
            inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 4,5
        # det_conv = _last_conv2d(top_conv, 1, 3, 1, t, act=False, is_training=is_training)  # exp009, serial 6
        # det_conv = _conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # serial 7: conv-relu-bn
        det_conv = _loc_conv2d(top_conv, 1, 3, 1, t, is_training=is_training)  # exp009, serial 8

        det_gap = AveragePooling3D(pool_size=(1, 16, 16), padding='SAME')(det_conv)
        det_scores = tf.nn.sigmoid(det_gap)

        det_probs = det_scores

        self.cam_layer_name = 'time_distributed_8'
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_gap, det_scores])
        self.seg_model = keras.Model(inputs=self.images, outputs=decodes)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model32:  # each_ste=True
    def __init__(self, input_size, f_num, is_training, t=True, mtl_mode=False, det_size=16, **kwargs):
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

        img_seg = tf.concat([self.images, seg_probs], axis=-1)

        det_encodes, det_downs = {}, {}
        det_downs[0] = img_seg
        for i in encode_idx:  # [1, 2, 3]
            det_encodes[i] = _conv2d(det_downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            det_encodes[i] = _conv2d(det_encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            det_downs[i] = _pool2d(det_encodes[i], t)

        det_deconvs, det_concats, det_decodes = {}, {}, {}
        det_decodes[b_num] = _convlstm2d(det_downs[reps], f_num[-1], 3, 1)

        for j in decode_idx:  # [3, 2, 1]
            det_deconvs[j] = _deconv2d(det_decodes[j + 1], f_num[j - 1], 2, 2, t)
            det_concats[j] = concatenate([det_encodes[j], det_deconvs[j]], axis=-1)

            det_decodes[j] = _conv2d(det_concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            det_decodes[j] = _conv2d(det_decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            print(det_decodes[j])

        seg_conv = _conv2d(det_decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        det_top_conv = _conv2d(det_decodes[1], 32, 3, 1, t, is_training=is_training)
        det_conv = _loc_conv2d(det_top_conv, 1, 3, 1, t, is_training=is_training)

        d_s = det_conv.shape[2] // det_size
        det_gap = AveragePooling3D(pool_size=(1, d_s, d_s), padding='SAME')(det_conv)

        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 8, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=[seg_probs, cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], self.model.output[2],
                                              det_gap, det_scores])
        self.check_model = keras.Model(inputs=self.images, outputs=det_scores)
        print(self.model.get_layer(self.cam_layer_name).output)


class Model33:  # each_ste=True
    def __init__(self, input_size, f_num, is_training, t=True, mtl_mode=False, det_size=16, **kwargs):
        self.images = layers.Input(input_size)
        self.pred_masks = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]
        b_num = len(f_num)
        reps = b_num - 1

        img_seg = tf.concat([self.images, self.pred_masks], axis=-1)
        encode_idx = [v + 1 for v in range(reps)]
        decode_idx = list(reversed(encode_idx))

        det_encodes, det_downs = {}, {}
        det_downs[0] = img_seg
        for i in encode_idx:  # [1, 2, 3]
            det_encodes[i] = _conv2d(det_downs[i - 1], f_num[i - 1], 3, 1, t, is_training=is_training)
            det_encodes[i] = _conv2d(det_encodes[i], f_num[i - 1], 3, 1, t, is_training=is_training)
            det_downs[i] = _pool2d(det_encodes[i], t)

        det_deconvs, det_concats, det_decodes = {}, {}, {}
        det_decodes[b_num] = _convlstm2d(det_downs[reps], f_num[-1], 3, 1)

        for j in decode_idx:  # [3, 2, 1]
            det_deconvs[j] = _deconv2d(det_decodes[j + 1], f_num[j - 1], 2, 2, t)
            det_concats[j] = concatenate([det_encodes[j], det_deconvs[j]], axis=-1)

            det_decodes[j] = _conv2d(det_concats[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            det_decodes[j] = _conv2d(det_decodes[j], f_num[j - 1], 3, 1, t, is_training=is_training)
            print(det_decodes[j])

        seg_conv = _conv2d(det_decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)
        inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        det_top_conv = _conv2d(det_decodes[1], 32, 3, 1, t, is_training=is_training)
        det_conv = _loc_conv2d(det_top_conv, 1, 3, 1, t, is_training=is_training)

        d_s = det_conv.shape[2] // det_size
        det_gap = AveragePooling3D(pool_size=(1, d_s, d_s), padding='SAME')(det_conv)

        det_scores = tf.nn.sigmoid(det_gap)

        cls_scores = tf.reshape(cls_probs, (-1, 8, 1, 1, 1))
        det_probs = det_scores * cls_scores

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=[self.images, self.pred_masks], outputs=[cls_probs, det_probs])

        if mtl_mode:
            self.log_vars = tf.Variable(initial_value=tf.zeros(len(self.model.outputs)), trainable=True)
            self.model.params = self.model.trainable_variables + [self.log_vars]
        else:
            self.model.params = self.model.trainable_variables.copy()

        self.cam_model = keras.Model(inputs=[self.images, self.pred_masks],
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,
                                              self.model.output[0], self.model.output[1], det_gap, det_scores])

        print(self.model.get_layer(self.cam_layer_name).output)


if __name__ == '__main__':
    nn = Model28(input_size=[8, 256, 256, 1], class_num=1, f_num='64,112,160,208', det_size=16, is_training=True)
    print(nn.model.summary())
    # print(nn.model.output)

    if False:
        import os
        import pandas as pd

        weight_csv = pd.DataFrame({'INDEX': pd.Series(dtype=int), 'NAME': pd.Series(dtype=object),
                                   'SHAPE': pd.Series(dtype=object), 'MEAN': pd.Series(dtype=float)})

        weights_list = nn.model.weights
        idx = 0
        for w in weights_list:
            weight_csv.loc[idx] = idx, w.name, w.shape, tf.reduce_mean(w).numpy()
            idx += 1

        weight_csv.to_csv(os.path.join('/workspace/MRA/models/', 'Model28_weights.csv'),  index=False)
