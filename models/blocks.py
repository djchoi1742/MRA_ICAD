import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


def _conv3d(x, f_num, k_size, stride, padding='same', af='relu', bn=True, init='he_normal', is_training=True):
    x = Conv3D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)
    if bn:
        return BatchNormalization(trainable=is_training)(x)
    return x


def _last_conv3d(x, f_num, k_size, stride, padding='same', init='he_normal', act=True, is_training=True):
    x = Conv3D(f_num, k_size, stride, padding=padding, activation=None, kernel_initializer=init)(x)
    x = BatchNormalization(trainable=is_training)(x)
    if act:
        # x = BatchNormalization(trainable=is_training)(x)
        x = tf.nn.relu(x)
    return x


def _pool3d(x, padding='same'):
    return MaxPooling3D(padding=padding)(x)


def _deconv3d(x, f_num, k_size, stride, padding='same', af='relu', init='he_normal'):
    return Conv3DTranspose(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)


def _last_conv2d(x, f_num, k_size, stride, add_time_axis, init='he_normal', act=True, is_training=True):
    if add_time_axis:
        x = TimeDistributed(Conv2D(f_num, k_size, stride, padding='same', activation=None, kernel_initializer=init))(x)
    else:
        x = Conv2D(f_num, k_size, stride, padding='same', activation=None, kernel_initializer=init)(x)

    # apply bn:  Model28 serial 15~17, Model282 serial 6, Model232 serial 4
    # x = BatchNormalization(trainable=is_training)(x)  #  removed Model28 serial 18, 20
    if act:
        x = BatchNormalization(trainable=is_training)(x)
        x = tf.nn.relu(x)
    return x


def _loc_conv2d(x, f_num, k_size, stride, add_time_axis, init='he_normal', is_training=True):
    if add_time_axis:
        x = TimeDistributed(Conv2D(f_num, k_size, stride, padding='same', activation=None, kernel_initializer=init))(x)
        x = BatchNormalization(trainable=is_training)(x)
        x = tf.nn.relu(x)
        x = TimeDistributed(Conv2D(f_num, k_size, stride, padding='same', activation=None, kernel_initializer=init))(x)

    else:
        x = Conv2D(f_num, k_size, stride, padding='same', activation=None, kernel_initializer=init)(x)
        x = BatchNormalization(trainable=is_training)(x)
        x = tf.nn.relu(x)
        x = Conv2D(f_num, k_size, stride, padding='same', activation=None, kernel_initializer=init)(x)

    return x


def _loc_conv3d(x, f_num, k_size, stride, padding='same', init='he_normal', is_training=True):
    x = Conv3D(f_num, k_size, stride, padding=padding, activation=None, kernel_initializer=init)(x)
    x = BatchNormalization(trainable=is_training)(x)
    x = tf.nn.relu(x)
    x = Conv3D(f_num, k_size, stride, padding=padding, activation=None, kernel_initializer=init)(x)
    return x


def _conv2d(x, f_num, k_size, stride, add_time_axis, padding='same', af='relu', bn=True,
            init='he_normal', is_training=True):
    if add_time_axis:
        x = TimeDistributed(Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init))(x)
    else:
        x = Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)

    if bn:
        return BatchNormalization(trainable=is_training)(x)
    return x


def _pool2d(x, add_time_axis, pool_size=(2, 2), padding='same'):
    if add_time_axis:
        return TimeDistributed(MaxPooling2D(pool_size=pool_size, padding=padding))(x)
    else:
        return MaxPooling2D(padding=padding)(x)


def _avgpool2d(x, add_time_axis, pool_size=(2, 2), padding='same'):
    if add_time_axis:
        return TimeDistributed(AveragePooling2D(pool_size=pool_size, padding=padding))(x)
    else:
        return MaxPooling2D(padding=padding)(x)


def _deconv2d(x, f_num, k_size, stride, add_time_axis, padding='same', af='relu', init='he_normal'):
    if add_time_axis:
        return TimeDistributed(Conv2DTranspose(f_num, k_size, stride, padding=padding,
                                               activation=af, kernel_initializer=init))(x)
    else:
        return Conv2DTranspose(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)


def _convlstm2d(x, f_num, k_size, stride, padding='same', return_seq=True, merge_mode='ave', is_bi=True):
    if is_bi:
        return Bidirectional(ConvLSTM2D(f_num, k_size, stride, padding=padding, return_sequences=return_seq),
                             merge_mode=merge_mode)(x)
    else:
        return ConvLSTM2D(f_num, k_size, stride, padding=padding, return_sequences=return_seq)(x)


def convlstm2d(x, f_num, k_size, stride, padding='same', return_seq=True, merge_mode='ave', is_bi=True,
               name='conv_lstm'):
    if is_bi:
        return Bidirectional(ConvLSTM2D(f_num, k_size, stride, padding=padding, return_sequences=return_seq),
                             merge_mode=merge_mode, name=name)(x)
    else:
        return ConvLSTM2D(f_num, k_size, stride, padding=padding, return_sequences=return_seq)(x)


def se_block_2d(input_x, reduction_ratio=16):
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


def se_block_3d(input_x, reduction_ratio=16):
    squeeze = tf.reduce_mean(input_x, axis=[1, 2, 3], keepdims=True)  # global average pooling
    excitation = layers.Dense(units=squeeze.shape[-1] // reduction_ratio,
                              kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              activation='relu')(squeeze)
    excitation = layers.Dense(units=squeeze.shape[-1],
                              kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                              kernel_initializer=tf.keras.initializers.he_normal(),
                              activation='sigmoid')(excitation)
    return excitation


