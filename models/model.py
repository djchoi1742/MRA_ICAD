import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import *
from tensorflow import keras


def _conv2d(x, f_num, k_size, stride, add_time_axis, padding='same', af='relu', bn=True,
            init='he_normal', is_training=True):
    if add_time_axis:
        x = layers.TimeDistributed(layers.Conv2D(f_num, k_size, stride, padding=padding,
                                                 activation=af, kernel_initializer=init))(x)
    else:
        x = layers.Conv2D(f_num, k_size, stride, padding=padding, activation=af, kernel_initializer=init)(x)

    if bn:
        return layers.BatchNormalization(trainable=is_training)(x)

    return x


def _pool2d(x, add_time_axis, padding='same'):
    if add_time_axis:
        return layers.TimeDistributed(layers.MaxPooling2D(padding=padding))(x)
    else:
        return layers.MaxPooling2D(padding=padding)(x)


def _deconv2d(x, f_num, k_size, stride, add_time_axis, padding='same', af='relu', init='he_normal'):
    if add_time_axis:
        return layers.TimeDistributed(layers.Conv2DTranspose(f_num, k_size, stride,
                                                             padding=padding, activation=af,
                                                             kernel_initializer=init))(x)
    else:
        return layers.Conv2DTranspose(f_num, k_size, stride, padding=padding, activation=af,
                                      kernel_initializer=init)(x)


def _convlstm2d(x, f_num, k_size, stride, padding='same', return_seq=True, merge_mode='ave', is_bi=True):
    if is_bi:
        return layers.Bidirectional(layers.ConvLSTM2D(f_num, k_size, stride, padding=padding,
                                                      return_sequences=return_seq),
                                    merge_mode=merge_mode)(x)
    else:
        return layers.ConvLSTM2D(f_num, k_size, stride, padding=padding, return_sequences=return_seq)(x)


def lstm_unet(input_size, class_num, f_num, is_training):
    inputs = layers.Input(input_size)
    if len(input_size) == 4:
        t = True
    else:
        t = False

    f0, f1, f2, f3, f4 = f_num
    # encoder
    encode_1 = _conv2d(_conv2d(inputs, f0, 3, 1, t, is_training=is_training),
                       f0, 3, 1, t, is_training=is_training)
    down_1 = _pool2d(encode_1, t)

    encode_2 = _conv2d(_conv2d(down_1, f1, 3, 1, t, is_training=is_training),
                       f1, 3, 1, t, is_training=is_training)
    down_2 = _pool2d(encode_2, t)

    encode_3 = _conv2d(_conv2d(down_2, f2, 3, 1, t, is_training=is_training),
                       f2, 3, 1, t, is_training=is_training)
    down_3 = _pool2d(encode_3, t)

    encode_4 = _conv2d(_conv2d(down_3, f3, 3, 1, t, is_training=is_training),
                       f3, 3, 1, t, is_training=is_training)
    down_4 = _pool2d(encode_4, t)

    bot_conv = _convlstm2d(down_4, f4, 3, 1)

    # decoder
    concat_4 = layers.concatenate([encode_4, _deconv2d(bot_conv, f3, 2, 2, t)], axis=-1)
    decode_4 = _conv2d(_conv2d(concat_4, f3, 3, 1, t, is_training=is_training),
                       f3, 3, 1, t, is_training=is_training)

    concat_3 = layers.concatenate([encode_3, _deconv2d(decode_4, f2, 2, 2, t)], axis=-1)
    decode_3 = _conv2d(_conv2d(concat_3, f2, 3, 1, t, is_training=is_training),
                       f2, 3, 1, t, is_training=is_training)

    concat_2 = layers.concatenate([encode_2, _deconv2d(decode_3, f1, 2, 2, t)], axis=-1)
    decode_2 = _conv2d(_conv2d(concat_2, f1, 3, 1, t, is_training=is_training),
                       f1, 3, 1, t, is_training=is_training)

    concat_1 = layers.concatenate([encode_1, _deconv2d(decode_2, f0, 2, 2, t)], axis=-1)
    decode_1 = _conv2d(_conv2d(concat_1, f0, 3, 1, t, is_training=is_training),
                       f0, 3, 1, t, is_training=is_training)

    top_conv = _conv2d(decode_1, 32, 3, 1, t, is_training=is_training)
    logits = _conv2d(top_conv, class_num, 3, 1, t, af='softmax', bn=False, is_training=is_training)

    output = keras.Model(inputs=inputs, outputs=logits)
    return output



if __name__ == '__main__':
    import os

    # import pydot
    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation
    tf.config.experimental.set_visible_devices(gpu[0], 'GPU')

    # InferenceModel01(growth_k=32, image_size=256, block_rep='4,6,6', theta=0.5, use_se=False)

    if False:
        network = lstm_unet(input_size=[5, 256, 256, 1], class_num=1,
                            f_num=[64, 128, 256, 512, 1024], is_training=True)
        network.summary()
        network_save_path = '/workspace/MRA/png'
        if not os.path.exists(network_save_path): os.makedirs(network_save_path)
        network_save_png = os.path.join(network_save_path, 'LSTM_Unet.png')
        keras.utils.plot_model(network, network_save_png, show_shapes=True, show_layer_names=False)
