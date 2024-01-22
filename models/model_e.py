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


def _plain_block(x, f_num, k_size, stride, padding='same', af='relu', bn=True, init='he_normal', is_training=True):
    x = Conv3D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)
    if bn:
        return BatchNormalization(trainable=is_training)(x)
    return x


def _se_block(x, reduction_ratio=16):
    squeeze = tf.reduce_mean(x, axis=[1, 2, 3], keepdims=True)
    excitation = layers.Dense(units=squeeze.shape[-1] // reduction_ratio,
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              activation='relu')(squeeze)
    excitation = layers.Dense(units=squeeze.shape[-1],
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              activation='sigmoid')(excitation)
    out = x * excitation
    return out


def _identity_block(x, f_num, k_size, stride, use_se=True, is_training=True):
    out = _plain_block(x, f_num=f_num, k_size=k_size, stride=stride, is_training=is_training)
    out = _plain_block(out, f_num=f_num, k_size=k_size, stride=stride, is_training=is_training)
    if use_se:
        out = _se_block(out)

    x = Conv3D(f_num, 1, 1)(x)
    x_add = x + out

    x_add_relu = tf.nn.relu(x_add)
    return x_add_relu


def _projection_block(x, f_num, k_size, stride1=1, stride2=2, use_se=True, is_training=True):
    out1 = _plain_block(x, f_num=f_num, k_size=k_size, stride=stride2, is_training=is_training)
    out2 = _plain_block(x, f_num=f_num, k_size=k_size, stride=stride2, is_training=is_training)
    out2 = _plain_block(out2, f_num=f_num, k_size=k_size, stride=stride1, is_training=is_training)
    if use_se:
        out2 = _se_block(out2)

    x_add = out1 + out2
    x_add_relu = tf.nn.relu(x_add)
    return x_add_relu


class Model30:  # Stenosis detection for ICA
    def __init__(self, input_size, class_num, f_num, is_training, t=True, use_se=True, **kwargs):

        self.images = layers.Input(input_size)
        assert len(input_size) == 4

        f_num = [*map(int, re.split(',', f_num))]

        plain_x = _plain_block(self.images, f_num=f_num[0], k_size=6, stride=2)
        plain_pool = MaxPooling3D(pool_size=2, strides=2)(plain_x)

        out = _identity_block(plain_pool, f_num=f_num[1], k_size=3, stride=1, use_se=use_se, is_training=is_training)
        out = _identity_block(out, f_num=f_num[1], k_size=3, stride=1, use_se=use_se, is_training=is_training)
        out = _projection_block(out, f_num=f_num[2], k_size=3, stride1=1, stride2=2, use_se=use_se,
                                is_training=is_training)

        out = _identity_block(out, f_num=f_num[2], k_size=3, stride=1, use_se=use_se, is_training=is_training)
        out = _projection_block(out, f_num=f_num[3], k_size=3, stride1=1, stride2=2, use_se=use_se,
                                is_training=is_training)

        out = _identity_block(out, f_num=f_num[3], k_size=3, stride=1, use_se=use_se, is_training=is_training)
        out = _identity_block(out, f_num=f_num[3], k_size=3, stride=1, use_se=use_se, is_training=is_training)

        out = AveragePooling3D(out.shape[1])(out)
        out = Flatten()(out)

        logits = Dense(class_num, activation=None)(out)
        cls_probs = tf.nn.sigmoid(logits)

        self.model = Model(inputs=self.images, outputs=cls_probs)


class Model31:  # each_ste=True
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

        seg_conv = _conv2d(decodes[b_num], f_num[-1], 3, 1, t, is_training=is_training)

        inner_conv = _conv2d(seg_conv, f_num[-1], 3, 1, t, is_training=is_training)

        inner_gap = tf.reduce_mean(AveragePooling3D(pool_size=(1, 32, 32), padding='SAME')(inner_conv), axis=[2, 3])
        inner_cls_logits = tf.reduce_mean(layers.Dense(1, activation=None)(inner_gap), axis=-1)
        cls_probs = tf.nn.sigmoid(inner_cls_logits)

        self.cam_layer_name = 'time_distributed_8'  # bidirectional or time_distributed_8
        self.model = Model(inputs=self.images, outputs=cls_probs)

        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer(self.cam_layer_name).output,  cls_probs])
        print(self.model.get_layer(self.cam_layer_name).output)


if __name__ == '__main__':
    nn = Model31(input_size=[8, 256, 256, 1], class_num=1, f_num='64,112,160,208', use_se=True, is_training=True)
    # nn = Model30(input_size=[64, 64, 64, 1], f_num='8,16,32,64', is_training=True, use_ic=False)
    print(nn.model.input, nn.model.output)
    pass
