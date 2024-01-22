import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.layers import *
from tensorflow.keras.models import *


# CNN-LSTM reference
def _group_norm(x, G=32, eps=1e-5, scope='group_norm'):
    with tf.variable_scope(scope):
        N, H, W, C = x.get_shape().as_list()
        G = min(G, C)

        x = tf.reshape(x, [N, H, W, G, C // G])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, C], initializer=tf.constant_initializer(1.0))
        beta = tf.get_variable('beta', [1, 1, 1, C], initializer=tf.constant_initializer(0.0))

        x = tf.reshape(x, [N, H, W, C]) * gamma + beta

    return x


def _conv3d(x, f_num, k_size, stride, padding='same', af='relu', bn=True, init='he_normal', is_training=True):
    x = Conv3D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)
    if bn:
        return BatchNormalization(trainable=is_training)(x)
    return x


def _pool3d(x, padding='same'):
    return MaxPooling3D(padding=padding)(x)


def _deconv3d(x, f_num, k_size, stride, padding='same', af='relu', init='he_normal'):
    return Conv3DTranspose(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)


def _conv2d(x, f_num, k_size, stride, add_time_axis, padding='same', af='relu', bn=True, init='he_normal', is_training=True):
    if add_time_axis:
        x = TimeDistributed(Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init))(x)
    else:
        x = Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)

    if bn:
        return BatchNormalization(trainable=is_training)(x)
    return x


def _pool2d(x, add_time_axis, padding='same'):
    if add_time_axis:
        return TimeDistributed(MaxPooling2D(padding=padding))(x)
    else:
        return MaxPooling2D(padding=padding)(x)


def _deconv2d(x, f_num, k_size, stride, add_time_axis, padding='same', af='relu', init='he_normal'):
    if add_time_axis:
        return TimeDistributed(Conv2DTranspose(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init))(x)
    else:
        return Conv2DTranspose(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)


def _convlstm2d(x, f_num, k_size, stride, padding='same', return_seq=True, merge_mode='ave', is_bi=True):
    if is_bi:
        return Bidirectional(ConvLSTM2D(f_num, k_size, stride, padding=padding, return_sequences=return_seq),
                             merge_mode=merge_mode)(x)
    else:
        return ConvLSTM2D(f_num, k_size, stride, padding=padding, return_sequences=return_seq)(x)


def _att_conv2d(x, g, f_num, add_time_axis):
    x_dim = x.get_shape()
    g_dim = g.get_shape()

    if add_time_axis:
        # x_dim : b, t, wx, hx, c
        # g_dim : b, t, wg, hg, c
        x_flat = tf.reshape(x, [-1, x_dim[2], x_dim[3], x_dim[4]])  # b*t, wx, hx ,c
        x_rshp = tf.reshape(tf.image.resize(x_flat, [g_dim[2], g_dim[3]]), [-1, x_dim[1], g_dim[2], g_dim[3], x_dim[4]])  # b, t, wg, hg, c

        x_rsz = TimeDistributed(Conv2D(f_num, 1, 1))(x_rshp)  # b, t, wg, hg, c
        g = TimeDistributed(Conv2D(f_num, 1, 1))(g)  # b, t, wg, hg, c
        sigma_1 = Activation('relu')(Add()([g, x_rsz]))  # b, t, wg, hg, c
        sigma_2 = Activation('sigmoid')(TimeDistributed(Conv2D(1, 1, 1))(sigma_1))  # b, t, wg, hg, 1

        s2_dim = sigma_2.get_shape()
        print(s2_dim)
        sigma_2 = tf.reshape(sigma_2, [-1, s2_dim[2], s2_dim[3], s2_dim[4]])  # b*t, wg, hg, 1
        sigma_2 = tf.reshape(tf.image.resize(sigma_2, [x_dim[2], x_dim[3]]), [-1, s2_dim[1], x_dim[2], x_dim[3], s2_dim[4]])  # b, t, wx, hx, 1

    else:
        x_rsz = Conv2D(f_num, 1, 1)(tf.image.resize(x, [g_dim[1], g_dim[2]]))  # 64,64,f_num
        g = Conv2D(f_num, 1, 1)(g)  # 64,64,f_num
        sigma_1 = Activation('relu')(Add()([g, x_rsz]))  # 64,64,f_num
        sigma_2 = tf.image.resize(Activation('sigmoid')(Conv2D(1, 1, 1)(sigma_1)), [x_dim[1], x_dim[2]])  # 64,64,1

    return Multiply()([sigma_2, x])


def _res_conv2d(x, f_num, k_size, stride, padding='same', af='relu', bn=True, init='he_normal', is_training=True):
    # projection / identity path
    in_dim = x.get_shape().as_list()
    if in_dim[-1] != f_num:
        proj = Conv2D(f_num, 1, 1, padding=padding)(x)
    else:
        proj = x

    # residual path
    y = Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)
    if bn:
        y = BatchNormalization(trainable=is_training)(y)
    y = Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(y)
    if bn:
        y = BatchNormalization(trainable=is_training)(y)

    return Add()([proj, y])


def _rec_conv2d(x, f_num, k_size, stride, rec_iter=2, padding='same', af='relu', bn=True, init='he_normal', is_training=True):
    y = Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)
    if bn:
        y = BatchNormalization(trainable=is_training)(y)
    for i in range(rec_iter):
        y = Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(Add()([x, y]))
        if bn:
            y = BatchNormalization(trainable=is_training)(y)

    return y


def _r2_conv2d(x, f_num, k_size, stride, rec_iter=2, padding='same', af='relu', bn=True, init='he_normal', is_training=True):
    # proj path
    in_dim = x.get_shape().as_list()
    if in_dim[-1] != f_num:
        x = Conv2D(f_num, 1, 1, padding=padding)(x)

    # recurrent part
    y = _rec_conv2d(x, f_num, k_size, stride, rec_iter, padding, af, bn, init, is_training)
    y = _rec_conv2d(y, f_num, k_size, stride, rec_iter, padding, af, bn, init, is_training)

    return Add()([x, y])


def unet_3d(input_size, class_num, f_num, is_training):
    inputs = Input(input_size)

    encode_1 = _conv3d(_conv3d(inputs, f_num[0], 3, 1, is_training=is_training), f_num[0], 3, 1, is_training=is_training)
    down_1 = _pool3d(encode_1)
    encode_2 = _conv3d(_conv3d(down_1, f_num[1], 3, 1, is_training=is_training), f_num[1], 3, 1, is_training=is_training)
    down_2 = _pool3d(encode_2)
    encode_3 = _conv3d(_conv3d(down_2, f_num[2], 3, 1, is_training=is_training), f_num[2], 3, 1, is_training=is_training)
    down_3 = _pool3d(encode_3)
    encode_4 = _conv3d(_conv3d(down_3, f_num[3], 3, 1, is_training=is_training), f_num[3], 3, 1, is_training=is_training)
    down_4 = _pool3d(encode_4)

    bot_conv = _conv3d(down_4, f_num[4], 3, 1, is_training=is_training)

    concat_4 = concatenate([encode_4, _deconv3d(bot_conv, f_num[3], 2, 2)], axis=-1)
    decode_4 = _conv3d(_conv3d(concat_4, f_num[3], 3, 1, is_training=is_training), f_num[3], 3, 1, is_training=is_training)
    concat_3 = concatenate([encode_3, _deconv3d(decode_4, f_num[3], 2, 2)], axis=-1)
    decode_3 = _conv3d(_conv3d(concat_3, f_num[2], 3, 1, is_training=is_training), f_num[2], 3, 1, is_training=is_training)
    concat_2 = concatenate([encode_2, _deconv3d(decode_3, f_num[3], 2, 2)], axis=-1)
    decode_2 = _conv3d(_conv3d(concat_2, f_num[1], 3, 1, is_training=is_training), f_num[1], 3, 1, is_training=is_training)
    concat_1 = concatenate([encode_1, _deconv3d(decode_2, f_num[3], 2, 2)], axis=-1)
    decode_1 = _conv3d(_conv3d(concat_1, f_num[0], 3, 1, is_training=is_training), f_num[0], 3, 1, is_training=is_training)

    top_conv = _conv3d(decode_1, 32, 3, 1, is_training=is_training)
    logits = _conv3d(top_conv, class_num, 3, 1, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)


def wide_unet(input_size, class_num, f_num, is_training):
    inputs = Input(input_size)
    if len(input_size) == 4:
        t = True
    else:
        t = False

    # encoder
    encode_1 = _conv2d(_conv2d(inputs, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)
    down_1 = _pool2d(encode_1, t)
    encode_2 = _conv2d(_conv2d(down_1, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    down_2 = _pool2d(encode_2, t)
    encode_3 = _conv2d(_conv2d(down_2, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    down_3 = _pool2d(encode_3, t)
    encode_4 = _conv2d(_conv2d(down_3, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    down_4 = _pool2d(encode_4, t)

    bot_conv = _conv2d(down_4, f_num[4], 3, 1, t, is_training=is_training)

    # decoder
    concat_4 = concatenate([encode_4, _deconv2d(bot_conv, f_num[3], 2, 2, t)], axis=-1)
    decode_4 = _conv2d(_conv2d(concat_4, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    concat_3 = concatenate([encode_3, _deconv2d(decode_4, f_num[2], 2, 2, t)], axis=-1)
    decode_3 = _conv2d(_conv2d(concat_3, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    concat_2 = concatenate([encode_2, _deconv2d(decode_3, f_num[1], 2, 2, t)], axis=-1)
    decode_2 = _conv2d(_conv2d(concat_2, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    concat_1 = concatenate([encode_1, _deconv2d(decode_2, f_num[0], 2, 2, t)], axis=-1)
    decode_1 = _conv2d(_conv2d(concat_1, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)

    top_conv = _conv2d(decode_1, 32, 3, 1, t, is_training=is_training)
    logits = _conv2d(top_conv, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)


# LSTM_UNET
def lstm_unet(input_size, class_num, f_num, is_training):
    inputs = Input(input_size)
    if len(input_size) == 4:
        t = True
    else:
        t = False

    # encoder
    encode_1 = _conv2d(_conv2d(inputs, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)
    down_1 = _pool2d(encode_1, t)
    encode_2 = _conv2d(_conv2d(down_1, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    down_2 = _pool2d(encode_2, t)
    encode_3 = _conv2d(_conv2d(down_2, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    down_3 = _pool2d(encode_3, t)
    encode_4 = _conv2d(_conv2d(down_3, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    down_4 = _pool2d(encode_4, t)

    bot_conv = _convlstm2d(down_4, f_num[4], 3, 1)

    # decoder
    concat_4 = concatenate([encode_4, _deconv2d(bot_conv, f_num[3], 2, 2, t)], axis=-1)
    decode_4 = _conv2d(_conv2d(concat_4, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    concat_3 = concatenate([encode_3, _deconv2d(decode_4, f_num[2], 2, 2, t)], axis=-1)
    decode_3 = _conv2d(_conv2d(concat_3, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    concat_2 = concatenate([encode_2, _deconv2d(decode_3, f_num[1], 2, 2, t)], axis=-1)
    decode_2 = _conv2d(_conv2d(concat_2, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    concat_1 = concatenate([encode_1, _deconv2d(decode_2, f_num[0], 2, 2, t)], axis=-1)
    decode_1 = _conv2d(_conv2d(concat_1, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)

    top_conv = _conv2d(decode_1, 32, 3, 1, t, is_training=is_training)
    logits = _conv2d(top_conv, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)



def att_unet(input_size, class_num, f_num, is_training):
    """
    input_size (tuple) : input image shape without batch axis - e.g. (w, h, c) or (t, w, h, c)
    f_num (list) : list of length 5 for u-net filter numbers - e.g. [64, 128, 256, 512, 1024]
    """
    inputs = Input(input_size)
    if len(input_size) == 4:
        t = True
    else:
        t = False

    # encoder
    encode_1 = _conv2d(_conv2d(inputs, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)
    down_1 = _pool2d(encode_1, t)
    encode_2 = _conv2d(_conv2d(down_1, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    down_2 = _pool2d(encode_2, t)
    encode_3 = _conv2d(_conv2d(down_2, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    down_3 = _pool2d(encode_3, t)
    encode_4 = _conv2d(_conv2d(down_3, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    down_4 = _pool2d(encode_4, t)

    encode_5 = _conv2d(down_4, f_num[4], 3, 1, t, is_training=is_training)

    # decoder
    concat_4 = concatenate([_att_conv2d(encode_4, encode_5, f_num[3] // 2, t), _deconv2d(encode_5, f_num[3], 2, 2, t)], axis=-1)
    decode_4 = _conv2d(_conv2d(concat_4, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    concat_3 = concatenate([_att_conv2d(encode_3, decode_4, f_num[2] // 2, t), _deconv2d(decode_4, f_num[2], 2, 2, t)], axis=-1)
    decode_3 = _conv2d(_conv2d(concat_3, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    concat_2 = concatenate([_att_conv2d(encode_2, decode_3, f_num[1] // 2, t), _deconv2d(decode_3, f_num[1], 2, 2, t)], axis=-1)
    decode_2 = _conv2d(_conv2d(concat_2, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    concat_1 = concatenate([_att_conv2d(encode_1, decode_2, f_num[0] // 2, t), _deconv2d(decode_2, f_num[0], 2, 2, t)], axis=-1)
    decode_1 = _conv2d(_conv2d(concat_1, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)

    decode_0 = _conv2d(decode_1, 32, 3, 1, t, is_training=is_training)
    logits = _conv2d(decode_0, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)


def r2_unet(input_size, class_num, f_num, is_training):
    inputs = Input(input_size)
    if len(input_size) == 4:
        t = True
    else:
        t = False

    # encoder
    encode_1 = _r2_conv2d(inputs, f_num[0], 3, 1, bn=True)
    down_1 = _pool2d(encode_1, t)
    encode_2 = _r2_conv2d(down_1, f_num[1], 3, 1, bn=True)
    down_2 = _pool2d(encode_2, t)
    encode_3 = _r2_conv2d(down_2, f_num[2], 3, 1, bn=True)
    down_3 = _pool2d(encode_3, t)
    encode_4 = _r2_conv2d(down_3, f_num[3], 3, 1, bn=True)
    down_4 = _pool2d(encode_4, t)

    encode_5 = _r2_conv2d(down_4, f_num[4], 3, 1, t, is_training=is_training)

    # decoder
    concat_4 = concatenate([encode_4, _deconv2d(encode_5, f_num[3], 2, 2, t)], axis=-1)
    decode_4 = _r2_conv2d(concat_4, f_num[3], 3, 1, bn=True)
    concat_3 = concatenate([encode_3, _deconv2d(decode_4, f_num[2], 2, 2, t)], axis=-1)
    decode_3 = _r2_conv2d(concat_3, f_num[2], 3, 1, bn=True)
    concat_2 = concatenate([encode_2, _deconv2d(decode_3, f_num[1], 2, 2, t)], axis=-1)
    decode_2 = _r2_conv2d(concat_2, f_num[1], 3, 1, bn=True)
    concat_1 = concatenate([encode_1, _deconv2d(decode_2, f_num[0], 2, 2, t)], axis=-1)
    decode_1 = _r2_conv2d(concat_1, f_num[0], 3, 1, bn=True)

    decode_0 = _conv2d(decode_1, 32, 3, 1, t, is_training=is_training)
    logits = _conv2d(decode_0, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)


def fcn_rnn(input_size, class_num, f_num, is_training):
    inputs = Input(input_size)
    if len(input_size) == 4:
        t = True
    else:
        t = False

    # encoder
    encode_1 = _conv2d(_conv2d(inputs, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)
    down_1 = _pool2d(encode_1, t)
    encode_2 = _conv2d(_conv2d(down_1, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    down_2 = _pool2d(encode_2, t)
    encode_3 = _conv2d(_conv2d(down_2, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    down_3 = _pool2d(encode_3, t)
    encode_4 = _conv2d(_conv2d(down_3, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    down_4 = _pool2d(encode_4, t)

    bot_conv = _conv2d(down_4, f_num[4], 3, 1, t, is_training=is_training)

    # decoder
    concat_4 = concatenate([encode_4, _deconv2d(bot_conv, f_num[3], 2, 2, t)], axis=-1)
    decode_4 = _conv2d(_conv2d(concat_4, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    concat_3 = concatenate([encode_3, _deconv2d(decode_4, f_num[2], 2, 2, t)], axis=-1)
    decode_3 = _conv2d(_conv2d(concat_3, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    concat_2 = concatenate([encode_2, _deconv2d(decode_3, f_num[1], 2, 2, t)], axis=-1)
    decode_2 = _conv2d(_conv2d(concat_2, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    concat_1 = concatenate([encode_1, _deconv2d(decode_2, f_num[0], 2, 2, t)], axis=-1)
    decode_1 = _conv2d(_conv2d(concat_1, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)

    top_conv = _convlstm2d(decode_1, 64, 3, 1)
    logits = _conv2d(top_conv, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)


def lstm_unet_64(input_size, class_num, f_num, is_training):
    inputs = Input(input_size)
    if len(input_size) == 4:
        t = True
    else:
        t = False

    # encoder
    encode_1 = _conv2d(_conv2d(inputs, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)
    down_1 = _pool2d(encode_1, t)
    encode_2 = _conv2d(_conv2d(down_1, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    down_2 = _pool2d(encode_2, t)
    encode_3 = _conv2d(_conv2d(down_2, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    down_3 = _pool2d(encode_3, t)
    encode_4 = _conv2d(_conv2d(down_3, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    down_4 = _pool2d(encode_4, t)

    bot_conv = _convlstm2d(down_4, 64, 3, 1)

    # decoder
    concat_4 = concatenate([encode_4, _deconv2d(bot_conv, f_num[3], 2, 2, t)], axis=-1)
    decode_4 = _conv2d(_conv2d(concat_4, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    concat_3 = concatenate([encode_3, _deconv2d(decode_4, f_num[2], 2, 2, t)], axis=-1)
    decode_3 = _conv2d(_conv2d(concat_3, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    concat_2 = concatenate([encode_2, _deconv2d(decode_3, f_num[1], 2, 2, t)], axis=-1)
    decode_2 = _conv2d(_conv2d(concat_2, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    concat_1 = concatenate([encode_1, _deconv2d(decode_2, f_num[0], 2, 2, t)], axis=-1)
    decode_1 = _conv2d(_conv2d(concat_1, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)

    top_conv = _conv2d(decode_1, 32, 3, 1, t, is_training=is_training)
    logits = _conv2d(top_conv, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)


def lstm_att_unet(input_size, class_num, f_num, is_training):
    inputs = Input(input_size)
    if len(input_size) == 4:
        t = True
    else:
        t = False

    # encoder
    encode_1 = _conv2d(_conv2d(inputs, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)
    down_1 = _pool2d(encode_1, t)
    encode_2 = _conv2d(_conv2d(down_1, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    down_2 = _pool2d(encode_2, t)
    encode_3 = _conv2d(_conv2d(down_2, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    down_3 = _pool2d(encode_3, t)
    encode_4 = _conv2d(_conv2d(down_3, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    down_4 = _pool2d(encode_4, t)

    encode_5 = _convlstm2d(down_4, f_num[4], 3, 1)

    # decoder
    concat_4 = concatenate([_att_conv2d(encode_4, encode_5, f_num[3] // 2, t), _deconv2d(encode_5, f_num[3], 2, 2, t)], axis=-1)
    decode_4 = _conv2d(_conv2d(concat_4, f_num[3], 3, 1, t, is_training=is_training), f_num[3], 3, 1, t, is_training=is_training)
    concat_3 = concatenate([_att_conv2d(encode_3, decode_4, f_num[2] // 2, t), _deconv2d(decode_4, f_num[2], 2, 2, t)], axis=-1)
    decode_3 = _conv2d(_conv2d(concat_3, f_num[2], 3, 1, t, is_training=is_training), f_num[2], 3, 1, t, is_training=is_training)
    concat_2 = concatenate([_att_conv2d(encode_2, decode_3, f_num[1] // 2, t), _deconv2d(decode_3, f_num[1], 2, 2, t)], axis=-1)
    decode_2 = _conv2d(_conv2d(concat_2, f_num[1], 3, 1, t, is_training=is_training), f_num[1], 3, 1, t, is_training=is_training)
    concat_1 = concatenate([_att_conv2d(encode_1, decode_2, f_num[0] // 2, t), _deconv2d(decode_2, f_num[0], 2, 2, t)], axis=-1)
    decode_1 = _conv2d(_conv2d(concat_1, f_num[0], 3, 1, t, is_training=is_training), f_num[0], 3, 1, t, is_training=is_training)

    decode_0 = _conv2d(decode_1, 32, 3, 1, t, is_training=is_training)
    logits = _conv2d(decode_0, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    return Model(inputs=inputs, outputs=logits)


class Unet(tf.keras.Model):
    def __init__(self, class_num, f_list=[32, 64, 128, 256, 512], del_skip_connection=False):
        super(Unet, self).__init__()
        self.del_skip_connection = del_skip_connection

        self.encode_1 = layers.TimeDistributed(layers.Conv2D(f_list[0], 3, 1, padding='same', kernel_initializer='he_normal'))  # b,t,256,256,64
        self.down_1 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 128,128,64
        self.encode_2 = layers.TimeDistributed(layers.Conv2D(f_list[1], 3, 1, padding='same', kernel_initializer='he_normal'))  # 128,128,128
        self.down_2 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 64,64,128
        self.encode_3 = layers.TimeDistributed(layers.Conv2D(f_list[2], 3, 1, padding='same', kernel_initializer='he_normal'))  # 64,64,256
        self.down_3 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 32,32,256
        self.encode_4 = layers.TimeDistributed(layers.Conv2D(f_list[3], 3, 1, padding='same', kernel_initializer='he_normal'))  # 32,32,512
        self.down_4 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 16,16,512

        self.encode_5 = layers.TimeDistributed(layers.Conv2D(f_list[4], 3, 1, padding='same', kernel_initializer='he_normal'))  # 32,32,512

        self.up_4 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[3], 2, 2, padding='same', kernel_initializer='he_normal'))  # 32,32,512
        self.decode_4 = layers.TimeDistributed(layers.Conv2D(f_list[3], 3, 1, padding='same', kernel_initializer='he_normal'))  # 16,16,512
        self.up_3 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[2], 2, 2, padding='same', kernel_initializer='he_normal'))  # 64,64,256
        self.decode_3 = layers.TimeDistributed(layers.Conv2D(f_list[2], 3, 1, padding='same', kernel_initializer='he_normal'))  # 32,32,256
        self.up_2 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[1], 2, 2, padding='same', kernel_initializer='he_normal'))  # 128,128,128
        self.decode_2 = layers.TimeDistributed(layers.Conv2D(f_list[1], 3, 1, padding='same', kernel_initializer='he_normal'))  # 64,64,128
        self.up_1 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[0], 2, 2, padding='same', kernel_initializer='he_normal'))  # 256,256,64
        self.decode_1 = layers.TimeDistributed(layers.Conv2D(f_list[0], 3, 1, padding='same', kernel_initializer='he_normal'))  # 128,128,64
        self.finetune_layer = layers.TimeDistributed(layers.Conv2D(128, 1, 1, padding='same', kernel_initializer='he_normal'))  # 128,128,128
        self.logits_layer = layers.TimeDistributed(layers.Conv2D(class_num, 1, 1, padding='same', kernel_initializer='he_normal'))  # 256,256,class_num

    def call(self, x_seq):
        # 2 - Encoder Graphs
        en_1 = self.encode_1(x_seq)
        en_2 = self.encode_2(self.down_1(en_1))
        en_3 = self.encode_3(self.down_2(en_2))
        en_4 = self.encode_4(self.down_3(en_3))
        en_5 = self.encode_5(self.down_4(en_4))  # [batch,time,16,16,1024]

        if self.del_skip_connection:
            de_4 = self.decode_4(self.up_4(en_5))
            de_3 = self.decode_3(self.up_3(de_4))
            de_2 = self.decode_2(self.up_2(de_3))
            de_1 = self.decode_1(self.up_1(de_2))
            logits = self.logits_layer(de_1)
        else:
            de_4 = self.decode_4(tf.concat([en_4, self.up_4(en_5)], axis=-1))
            de_3 = self.decode_3(tf.concat([en_3, self.up_3(de_4)], axis=-1))
            de_2 = self.decode_2(tf.concat([en_2, self.up_2(de_3)], axis=-1))
            de_1 = self.decode_1(tf.concat([en_1, self.up_1(de_2)], axis=-1))

            finetune = self.finetune_layer(de_1)
            logits = self.logits_layer(finetune)  # batch,time,128,128,2

        softmax = tf.nn.softmax(logits, axis=-1)  # batch,time,128,128,2
        return softmax


class UnetLSTM(tf.keras.Model):
    def __init__(self, class_num, f_list=[32, 64, 128, 256, 512], del_skip_connection=False):
        super(UnetLSTM, self).__init__()
        self.del_skip_connection = del_skip_connection

        self.encode_1 = layers.TimeDistributed(layers.Conv2D(f_list[0], 3, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # b,t,256,256,64
        self.down_1 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 128,128,64
        self.encode_2 = layers.TimeDistributed(layers.Conv2D(f_list[1], 3, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # 128,128,128
        self.down_2 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 64,64,128
        self.encode_3 = layers.TimeDistributed(layers.Conv2D(f_list[2], 3, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # 64,64,256
        self.down_3 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 32,32,256
        self.encode_4 = layers.TimeDistributed(layers.Conv2D(f_list[3], 3, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # 32,32,512
        self.down_4 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 16,16,512

        self.lstm_layer = layers.Bidirectional(layers.ConvLSTM2D(f_list[4], 3, 1, padding='same', return_sequences=True), merge_mode='ave')

        self.up_4 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[3], 2, 2, padding='same', kernel_initializer='he_normal', activation='relu'))  # 32,32,512
        self.decode_4 = layers.TimeDistributed(layers.Conv2D(f_list[3], 3, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # 16,16,512
        self.up_3 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[2], 2, 2, padding='same', kernel_initializer='he_normal', activation='relu'))  # 64,64,256
        self.decode_3 = layers.TimeDistributed(layers.Conv2D(f_list[2], 3, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # 32,32,256
        self.up_2 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[1], 2, 2, padding='same', kernel_initializer='he_normal', activation='relu'))  # 128,128,128
        self.decode_2 = layers.TimeDistributed(layers.Conv2D(f_list[1], 3, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # 64,64,128
        self.up_1 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[0], 2, 2, padding='same', kernel_initializer='he_normal', activation='relu'))  # 256,256,64
        self.decode_1 = layers.TimeDistributed(layers.Conv2D(f_list[0], 3, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # 128,128,64
        self.finetune_layer = layers.TimeDistributed(layers.Conv2D(128, 1, 1, padding='same', kernel_initializer='he_normal', activation='relu'))  # 128,128,128
        self.logits_layer = layers.TimeDistributed(layers.Conv2D(class_num, 1, 1, padding='same', kernel_initializer='he_normal'))  # 256,256,class_num

    def call(self, x_seq):
        # 2 - Encoder Graphs
        en_1 = self.encode_1(x_seq)
        en_2 = self.encode_2(self.down_1(en_1))
        en_3 = self.encode_3(self.down_2(en_2))
        en_4 = self.encode_4(self.down_3(en_3))
        lstm = self.lstm_layer(self.down_4(en_4))  # [batch,time,16,16,1024]

        if self.del_skip_connection:
            de_4 = self.decode_4(self.up_4(lstm))
            de_3 = self.decode_3(self.up_3(de_4))
            de_2 = self.decode_2(self.up_2(de_3))
            de_1 = self.decode_1(self.up_1(de_2))
        else:
            de_4 = self.decode_4(tf.concat([en_4, self.up_4(lstm)], axis=-1))
            de_3 = self.decode_3(tf.concat([en_3, self.up_3(de_4)], axis=-1))
            de_2 = self.decode_2(tf.concat([en_2, self.up_2(de_3)], axis=-1))
            de_1 = self.decode_1(tf.concat([en_1, self.up_1(de_2)], axis=-1))

        finetune = self.finetune_layer(de_1)
        logits = self.logits_layer(finetune)  # b,t,w,h
        # sparse_logits = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1, output_type=tf.int32)  # b,t,w,h

        return tf.nn.softmax(logits, axis=-1)


class UnetLSTM_Nto1(tf.keras.Model):
    """
    Description
        LSTM U-Net
    Args
    """
    def __init__(self, class_num, f_list=[32, 64, 128, 256, 512], del_skip_connection=False):
        super(UnetLSTM_Nto1, self).__init__()
        self.del_skip_connection = del_skip_connection

        self.encode_1 = layers.TimeDistributed(layers.Conv2D(f_list[0], 3, 1, padding='same', kernel_initializer='he_normal'))  # b,t,256,256,64
        self.down_1 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 128,128,64
        self.encode_2 = layers.TimeDistributed(layers.Conv2D(f_list[1], 3, 1, padding='same', kernel_initializer='he_normal'))  # 128,128,128
        self.down_2 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 64,64,128
        self.encode_3 = layers.TimeDistributed(layers.Conv2D(f_list[2], 3, 1, padding='same', kernel_initializer='he_normal'))  # 64,64,256
        self.down_3 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 32,32,256
        self.encode_4 = layers.TimeDistributed(layers.Conv2D(f_list[3], 3, 1, padding='same', kernel_initializer='he_normal'))  # 32,32,512
        self.down_4 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 16,16,512

        self.lstm_layer = layers.Bidirectional(layers.ConvLSTM2D(f_list[4], 3, 1, padding='same', return_sequences=False), merge_mode='ave')

        self.up_4 = layers.Conv2DTranspose(f_list[3], 2, 2, padding='same', kernel_initializer='he_normal')  # 32,32,512
        self.decode_4 = layers.Conv2D(f_list[3], 3, 1, padding='same', kernel_initializer='he_normal')  # 16,16,512
        self.up_3 = layers.Conv2DTranspose(f_list[2], 2, 2, padding='same', kernel_initializer='he_normal')  # 64,64,256
        self.decode_3 = layers.Conv2D(f_list[2], 3, 1, padding='same', kernel_initializer='he_normal')  # 32,32,256
        self.up_2 = layers.Conv2DTranspose(f_list[1], 2, 2, padding='same', kernel_initializer='he_normal')  # 128,128,128
        self.decode_2 = layers.Conv2D(f_list[1], 3, 1, padding='same', kernel_initializer='he_normal')  # 64,64,128
        self.up_1 = layers.Conv2DTranspose(f_list[0], 2, 2, padding='same', kernel_initializer='he_normal')  # 256,256,64
        self.decode_1 = layers.Conv2D(f_list[0], 3, 1, padding='same', kernel_initializer='he_normal')  # 128,128,64
        self.finetune_layer = layers.Conv2D(128, 1, 1, padding='same', kernel_initializer='he_normal')  # 128,128,128
        self.logits_layer = layers.Conv2D(class_num, 1, 1, padding='same', kernel_initializer='he_normal')  # 256,256,class_num

    def call(self, x_seq):
        # 2 - Encoder Graphs
        en_1 = self.encode_1(x_seq)
        en_2 = self.encode_2(self.down_1(en_1))
        en_3 = self.encode_3(self.down_2(en_2))
        en_4 = self.encode_4(self.down_3(en_3))
        lstm = self.lstm_layer(self.down_4(en_4))  # [batch,time,16,16,1024]

        avg_en_1 = layers.Average()(tf.unstack(en_1, axis=1))
        avg_en_2 = layers.Average()(tf.unstack(en_2, axis=1))
        avg_en_3 = layers.Average()(tf.unstack(en_3, axis=1))
        avg_en_4 = layers.Average()(tf.unstack(en_4, axis=1))

        if self.del_skip_connection:
            de_4 = self.decode_4(self.up_4(lstm))
            de_3 = self.decode_3(self.up_3(de_4))
            de_2 = self.decode_2(self.up_2(de_3))
            de_1 = self.decode_1(self.up_1(de_2))
            logits = self.logits_layer(de_1)
        else:
            de_4 = self.decode_4(tf.concat([avg_en_4, self.up_4(lstm)], axis=-1))
            de_3 = self.decode_3(tf.concat([avg_en_3, self.up_3(de_4)], axis=-1))
            de_2 = self.decode_2(tf.concat([avg_en_2, self.up_2(de_3)], axis=-1))
            de_1 = self.decode_1(tf.concat([avg_en_1, self.up_1(de_2)], axis=-1))

            finetune = self.finetune_layer(de_1)
            logits = self.logits_layer(finetune)  # batch,time,128,128,2

        softmax = tf.nn.softmax(logits, axis=-1)  # batch,time,128,128,2
        # mask = tf.argmax(softmax, axis=-1)  # b,t,128,128,1

        return softmax


class __UnetLSTMLegacy(tf.keras.Model):
    def __init__(self, class_num, f_list=[32, 64, 128, 256, 512], del_skip_connection=False):
        super(__UnetLSTMLegacy, self).__init__()
        self.del_skip_connection = del_skip_connection

        self.encode_1 = layers.TimeDistributed(layers.Conv2D(f_list[0], 3, 1, padding='same', kernel_initializer='he_normal'))  # b,t,256,256,64
        self.down_1 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 128,128,64
        self.encode_2 = layers.TimeDistributed(layers.Conv2D(f_list[1], 3, 1, padding='same', kernel_initializer='he_normal'))  # 128,128,128
        self.down_2 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 64,64,128
        self.encode_3 = layers.TimeDistributed(layers.Conv2D(f_list[2], 3, 1, padding='same', kernel_initializer='he_normal'))  # 64,64,256
        self.down_3 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 32,32,256
        self.encode_4 = layers.TimeDistributed(layers.Conv2D(f_list[3], 3, 1, padding='same', kernel_initializer='he_normal'))  # 32,32,512
        self.down_4 = layers.TimeDistributed(layers.MaxPooling2D(padding='same'))  # 16,16,512

        self.lstm_layer = layers.Bidirectional(layers.ConvLSTM2D(f_list[4], 3, 1, padding='same', return_sequences=True), merge_mode='ave')

        self.up_4 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[3], 2, 2, padding='same', kernel_initializer='he_normal'))  # 32,32,512
        self.decode_4 = layers.TimeDistributed(layers.Conv2D(f_list[3], 3, 1, padding='same', kernel_initializer='he_normal'))  # 16,16,512
        self.up_3 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[2], 2, 2, padding='same', kernel_initializer='he_normal'))  # 64,64,256
        self.decode_3 = layers.TimeDistributed(layers.Conv2D(f_list[2], 3, 1, padding='same', kernel_initializer='he_normal'))  # 32,32,256
        self.up_2 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[1], 2, 2, padding='same', kernel_initializer='he_normal'))  # 128,128,128
        self.decode_2 = layers.TimeDistributed(layers.Conv2D(f_list[1], 3, 1, padding='same', kernel_initializer='he_normal'))  # 64,64,128
        self.up_1 = layers.TimeDistributed(layers.Conv2DTranspose(f_list[0], 2, 2, padding='same', kernel_initializer='he_normal'))  # 256,256,64
        self.decode_1 = layers.TimeDistributed(layers.Conv2D(f_list[0], 3, 1, padding='same', kernel_initializer='he_normal'))  # 128,128,64
        self.logits_layer = layers.TimeDistributed(layers.Conv2D(class_num, 1, 1, padding='same', kernel_initializer='he_normal'))  # 256,256,class_num

    def call(self, x_seq):
        # 2 - Encoder Graphs
        en_1 = self.encode_1(x_seq)
        en_2 = self.encode_2(self.down_1(en_1))
        en_3 = self.encode_3(self.down_2(en_2))
        en_4 = self.encode_4(self.down_3(en_3))
        lstm = self.lstm_layer(self.down_4(en_4))  # [batch,time,16,16,1024]

        if self.del_skip_connection:
            de_4 = self.decode_4(self.up_4(lstm))
            de_3 = self.decode_3(self.up_3(de_4))
            de_2 = self.decode_2(self.up_2(de_3))
            de_1 = self.decode_1(self.up_1(de_2))
            logits = self.logits_layer(de_1)
        else:
            de_4 = self.decode_4(tf.concat([en_4, self.up_4(lstm)], axis=-1))
            de_3 = self.decode_3(tf.concat([en_3, self.up_3(de_4)], axis=-1))
            de_2 = self.decode_2(tf.concat([en_2, self.up_2(de_3)], axis=-1))
            de_1 = self.decode_1(tf.concat([en_1, self.up_1(de_2)], axis=-1))
            logits = self.logits_layer(de_1)  # batch,time,128,128,2

        softmax = tf.nn.softmax(logits, axis=-1)  # batch,time,128,128,2
        # mask = tf.argmax(softmax, axis=-1)  # b,t,128,128,1

        return softmax


# Build model.
# model = get_model(width=128, height=128, depth=64)
# model.summary()


if __name__ == '__main__':

    # gpu = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation
    # tf.config.experimental.set_visible_devices(gpu[0], 'GPU')

    # network = lstm_unet(input_size=[5, 256, 256, 1], class_num=2,
    #                     f_num=[64, 128, 256, 512, 1024], is_training=True)

    network = unet_3d(input_size=[16, 256, 256, 1], class_num=2,
                      f_num=[64, 128, 256, 512, 1024], is_training=True)
    network.summary()



# x_ = keras.Input(shape=(256, 256, 1))
# y_ = unet_basic(x_, 2)
# x_ = keras.Input(shape=(1, 256, 256, 1))
# y_ = unet_lstm(x_, 2, del_skip_connection=False)
#
# model = keras.Model(x_, y_)
# model.summary()
# keras.utils.plot_model(model, 'D:\\mra\\architecture\\lstm_unet.png', show_shapes=False, show_layer_names=False)


# in_dims = keras.Input(shape=(10, 256, 256, 1))
# out_dims = unet_lstm(in_dims, 2)
# out_dims = lstm_unet(in_dims, 2)
# model = keras.Model(in_dims, out_dims)
# model.summary()


# in_dims = (3, 256, 256, 1)  # for custom model call
# neural_network = fcn_rnn(in_dims, 2, [64, 128, 256, 512, 1024], True)
# neural_network.build(in_dims)
# neural_network.summary()
# keras.utils.plot_model(neural_network, 'D:\\mra\\architecture\\FCNRNN.png', show_shapes=True, show_layer_names=False)