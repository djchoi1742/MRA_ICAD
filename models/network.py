import tensorflow as tf
import numpy as np
import sys, re
from tensorflow import keras
from tensorflow.keras import layers, models


def _conv_layer(input_x, num_outputs, is_time=True, kernel_size=3, stride=1, act_fn='relu',
               last_layer=False, is_train=False):
    layer = layers.Conv2D(filters=num_outputs, kernel_size=kernel_size, strides=(stride, stride),
                          padding='same', activation=act_fn, kernel_initializer='he_normal')
    out = layers.TimeDistributed(layer=layer)(input_x) if is_time else layer(input_x)
    out = layers.BatchNormalization(trainable=is_train)(out) if not last_layer else out
    print(out)
    return out


def conv_layer(input_x, num_outputs, rep_num=2, is_time=True, kernel_size=3, stride=1, act_fn='relu',
                   last_layer=False, is_train=False):
    out = input_x
    for i in range(rep_num):
        layer = layers.Conv2D(filters=num_outputs, kernel_size=kernel_size, strides=(stride, stride),
                              padding='same', activation=act_fn, kernel_initializer='he_normal')
        out = layers.TimeDistributed(layer=layer)(out) if is_time else layer(out)
        out = layers.BatchNormalization(trainable=is_train)(out) if not last_layer else out
    print(out)
    return out


def pool_layer(input_x, is_time, padding='same'):
    layer = layers.MaxPooling2D(padding=padding)
    out = layers.TimeDistributed(layer=layer)(input_x) if is_time else layer(input_x)
    print(out)
    return out


def deconv_layer(input_x, num_outputs, is_time=True, kernel_size=2, stride=2):
    layer = layers.Conv2DTranspose(filters=num_outputs, kernel_size=kernel_size, strides=(stride, stride),
                                   padding='same', activation='relu', kernel_initializer='he_normal')
    out = layers.TimeDistributed(layer=layer)(input_x) if is_time else layer(input_x)
    print(out)
    return out


def conv_lstm_layer(input_x, num_outputs, kernel_size, stride, padding='same', return_seq=True,
                    merge_mode='ave', is_bi=True):
    layer = layers.ConvLSTM2D(filters=num_outputs, kernel_size=kernel_size, strides=(stride, stride),
                             padding=padding, return_sequences=return_seq)
    out = layers.Bidirectional(layer=layer, merge_mode=merge_mode)(input_x) if is_bi else layer(input_x)
    print(out)
    return out


class UnetLSTM:
    def __init__(self):
        # self.images = tf.placeholder(tf.float32, shape=[None, 10, 512, 512, 1])
        # self.is_train = tf.placeholder(tf.bool, shape=None)
        # self.is_time = tf.placeholder(tf.bool, shape=None)
        self.images = keras.Input(shape=(10, 512, 512, 1))

        self.is_train = True
        self.is_time = True
        self.class_num = 2

        f_size = [64, 128, 256, 512, 1024]
        f0, f1, f2, f3, f4 = f_size

        image_x = self.images
        enc1 = conv_layer(input_x=image_x, num_outputs=f0, rep_num=2, is_time=self.is_time, is_train=self.is_train)
        down1 = pool_layer(input_x=enc1, is_time=self.is_time)

        enc2 = conv_layer(input_x=down1, num_outputs=f1, rep_num=2, is_time=self.is_time, is_train=self.is_train)
        down2 = pool_layer(input_x=enc2, is_time=self.is_time)

        enc3 = conv_layer(input_x=down2, num_outputs=f2, rep_num=2, is_time=self.is_time, is_train=self.is_train)
        down3 = pool_layer(input_x=enc3, is_time=self.is_time)

        enc4 = conv_layer(input_x=down3, num_outputs=f3, rep_num=2, is_time=self.is_time, is_train=self.is_train)
        down4 = pool_layer(input_x=enc4, is_time=self.is_time)

        bot_conv = conv_lstm_layer(down4, num_outputs=f4, kernel_size=3, stride=1)

        concat4 = layers.concatenate([enc4, deconv_layer(bot_conv, f3, self.is_time)], axis=-1)
        dec4 = conv_layer(input_x=concat4, num_outputs=f3, rep_num=2, is_time=self.is_time, is_train=self.is_train)

        concat3 = layers.concatenate([enc3, deconv_layer(dec4, f2, self.is_time)], axis=-1)
        dec3 = conv_layer(input_x=concat3, num_outputs=f2, rep_num=2, is_time=self.is_time, is_train=self.is_train)

        concat2 = layers.concatenate([enc2, deconv_layer(dec3, f1, self.is_time)], axis=-1)
        dec2 = conv_layer(input_x=concat2, num_outputs=f1, rep_num=2, is_time=self.is_time, is_train=self.is_train)

        concat1 = layers.concatenate([enc1, deconv_layer(dec2, f0, self.is_time)], axis=-1)
        dec1 = conv_layer(input_x=concat1, num_outputs=f0, rep_num=2, is_time=self.is_time, is_train=self.is_train)

        top_conv = conv_layer(input_x=dec1, num_outputs=32, rep_num=1, is_time=self.is_time, is_train=self.is_train)

        self.logits = conv_layer(input_x=top_conv, num_outputs=self.class_num, rep_num=1,
                                 is_time=self.is_time, act_fn='softmax', last_layer=True)


        # import pdb; pdb.set_trace()


if __name__== '__main__':
    model = UnetLSTM()
    nn = keras.Model(inputs=model.images, outputs=model.logits)

    import pdb; pdb.set_trace()
