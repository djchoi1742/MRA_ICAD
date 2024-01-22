import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import *
from tensorflow import keras
import sys, re
import tensorflow.keras.backend as backend

sys.path.append('/workspace/bitbucket/MRA')

from models.model_ref import _conv2d, _pool2d, _convlstm2d


def design_scope(class_name):
    model_scope = re.sub('Inference', '', class_name)
    classifier_scope = re.sub('Model', 'Classifier', model_scope)
    return model_scope, classifier_scope


def bottleneck_layer(input_x, growth_k, is_training, use_dropout=False, dropout_rate=0.2):
    out = layers.BatchNormalization(trainable=is_training)(input_x)
    out = layers.Activation('relu')(out)
    out = layers.Conv2D(filters=4*growth_k, kernel_size=1, strides=1, padding='same', activation=None,
                        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                        kernel_initializer=tf.keras.initializers.he_normal())(out)
    out = layers.BatchNormalization(trainable=is_training)(out)
    out = layers.Activation('relu')(out)
    out = layers.Conv2D(filters=growth_k, kernel_size=3, strides=1, padding='same', activation=None,
                        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                        kernel_initializer=tf.keras.initializers.he_normal())(out)
    if use_dropout:
        out = layers.Dropout(rate=dropout_rate, is_training=is_training)(out)

    return out


def bottleneck_layer_3d(input_x, growth_k, is_training, use_dropout=False, dropout_rate=0.2):
    out = layers.BatchNormalization(trainable=is_training)(input_x)
    out = layers.Activation('relu')(out)
    out = layers.Conv3D(filters=4*growth_k, kernel_size=(1, 1, 1), strides=1, padding='same', activation=None,
                        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                        kernel_initializer=tf.keras.initializers.he_normal())(out)
    out = layers.BatchNormalization(trainable=is_training)(out)
    out = layers.Activation('relu')(out)
    out = layers.Conv3D(filters=growth_k, kernel_size=(4, 4, 1), strides=1, padding='same', activation=None,
                        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                        kernel_initializer=tf.keras.initializers.he_normal())(out)
    if use_dropout:
        out = layers.Dropout(rate=dropout_rate, is_training=is_training)(out)

    return out


def dense_block_3d(input_x, layer_name, rep, growth_k, is_training, use_dropout=False,
                   use_se=False, reduction_ratio=16):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer_3d(input_x, growth_k, is_training, use_dropout)
        layers_concat.append(x)

        for i in range(rep - 1):
            x = tf.concat(layers_concat, axis=4)
            x = bottleneck_layer_3d(x, growth_k, is_training, use_dropout)
            layers_concat.append(x)
        x = tf.concat(layers_concat, axis=4)

        if use_se:
            excitation = se_block_3d(x, reduction_ratio)
            x = x * excitation
    return x


def se_block(input_x, reduction_ratio=16):
    squeeze = tf.reduce_mean(input_x, axis=[1, 2], keepdims=True)  # global average pooling
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


def dense_block(input_x, layer_name, rep, growth_k, is_training, use_dropout=False,
                use_se=False, reduction_ratio=16):
    with tf.name_scope(layer_name):
        layers_concat = list()
        layers_concat.append(input_x)

        x = bottleneck_layer(input_x, growth_k, is_training, use_dropout)
        layers_concat.append(x)

        for i in range(rep - 1):
            x = tf.concat(layers_concat, axis=3)
            x = bottleneck_layer(x, growth_k, is_training, use_dropout)
            layers_concat.append(x)
        x = tf.concat(layers_concat, axis=3)

        if use_se:
            excitation = se_block(x, reduction_ratio)
            x = x * excitation
    return x


def transition_layer_3d(input_x, layer_name, is_training, theta=0.5, reduction_ratio=16, last_layer=False):
    with tf.name_scope(layer_name):
        in_channel = input_x.shape[-1]
        out = layers.BatchNormalization(trainable=is_training)(input_x)
        out = layers.Activation('relu')(out)
        out = layers.Conv3D(filters=int(in_channel*theta), kernel_size=(1, 1, 1), strides=(1, 1, 1),
                            padding='same', activation=None,
                            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                            kernel_initializer=tf.keras.initializers.he_normal())(out)

        if last_layer is False:
            excitation = se_block_3d(out, reduction_ratio)
            se_out = out * excitation
            avg_pool = layers.AvgPool3D(pool_size=(2, 2, 1), strides=(2, 2, 1),
                                        padding='same')(se_out)
            print(avg_pool)
        else:
            avg_pool = out

    return avg_pool


def transition_layer(input_x, layer_name, is_training, theta=0.5, reduction_ratio=16, last_layer=False):
    with tf.name_scope(layer_name):
        in_channel = input_x.shape[-1]
        out = layers.BatchNormalization(trainable=is_training)(input_x)
        out = layers.Activation('relu')(out)
        out = layers.Conv2D(filters=int(in_channel*theta), kernel_size=1, strides=1,
                            padding='same', activation=None,
                            kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                            kernel_initializer=tf.keras.initializers.he_normal())(out)

        if last_layer is False:
            excitation = se_block(out, reduction_ratio)
            se_out = out * excitation
            avg_pool = layers.AvgPool2D(pool_size=(2, 2), strides=2, padding='same')(se_out)
            print(avg_pool)
        else:
            avg_pool = out

    return avg_pool


class InferenceModel01:
    def __init__(self, input_size, growth_k=32, image_size=256, is_training=False, theta=0.5,
                 block_rep='3,3,3,3', k_p='1,1,1,1', use_se=False, **kwargs):
        # super(InferenceModel01, self).__init__()
        self.model_scope, _ = design_scope(class_name=type(self).__name__)

        block_rep_list = list(map(int, re.split(',', block_rep)))
        block_num = len(block_rep_list)
        k_p_list = list(map(int, re.split(',', k_p)))
        self.is_training = is_training
        self.class_num = 1

        self.img_h, self.img_w, self.img_c = image_size, image_size, 1
        self.images = layers.Input(input_size)

        first_conv = layers.Conv2D(filters=2*growth_k, kernel_size=3, strides=2,
                                   padding='same', activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   )(self.images)
        first_conv = layers.BatchNormalization(trainable=self.is_training)(first_conv)
        first_conv = layers.Activation('relu', name='conv1/relu')(first_conv)
        first_pool = layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(first_conv)

        dsb = first_pool
        for i in range(0, block_num-1):
            dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(i+1), rep=block_rep_list[i],
                              growth_k=k_p_list[i]*growth_k, use_se=use_se, is_training=self.is_training)
            dsb = transition_layer(input_x=dsb, layer_name='Transition'+str(i+1),
                                   theta=theta, is_training=self.is_training)

        self.last_dsb = dense_block(input_x=dsb, layer_name=self.model_scope + '_DB'+str(block_num),
                                    rep=block_rep_list[-1], growth_k=k_p_list[-1]*growth_k,
                                    use_se=use_se, is_training=self.is_training)

        bn_relu = layers.BatchNormalization(trainable=self.is_training)(self.last_dsb)
        self.bn_relu = layers.Activation('relu', name='last_bn_relu')(bn_relu)

        self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)

        flatten = tf.reduce_mean(self.last_pool, axis=[1, 2], keepdims=False)
        self.fc = layers.Dense(units=flatten.shape[-1], activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                               kernel_initializer=tf.keras.initializers.he_normal()
                               )(flatten)  # activation='relu'

        self.logits = layers.Dense(units=self.class_num, activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   kernel_initializer=tf.keras.initializers.he_normal()
                                   )(self.fc)
        self.prob = layers.Activation('sigmoid')(self.logits)
        self.model = keras.Model(inputs=self.images,
                                 outputs=[self.bn_relu, self.logits])
        # self.cam = keras.Model(inputs=self.images, outputs=self.bn_relu)

    def grad_cam(self, input_x):
        with tf.GradientTape() as tape:
            cam_layer, logits = self.model(input_x)
            probs = tf.nn.sigmoid(logits)
            # cam = self.cam(img)

        grads = tape.gradient(probs, cam_layer)
        local1 = tf.reduce_mean(grads * cam_layer, axis=-1, keepdims=True)
        local = tf.image.resize(images=local1, size=[self.img_h, self.img_w])
        # print('local: ', local.shape)
        return local


class InferenceModel02:
    def __init__(self, seq_len, img_h, img_w, img_c, f_num, is_training):
        self.class_num = 1
        input_size = [seq_len, img_h, img_w, img_c]
        self.images = layers.Input(input_size)
        self.seq_len, self.img_h, self.img_w, self.img_c = seq_len, img_h, img_w, img_c
        self.cam_w, self.cam_h, self.cam_d, self.cam_f = 16, 16, 1, 256

        t = True if len(input_size) == 4 else False

        encode_1 = _conv2d(_conv2d(self.images, f_num[0], 3, 1, t, is_training=is_training),
                           f_num[0], 3, 1, t, is_training=is_training)
        down_1 = _pool2d(encode_1, t)
        encode_2 = _conv2d(_conv2d(down_1, f_num[1], 3, 1, t, is_training=is_training),
                           f_num[1], 3, 1, t, is_training=is_training)
        down_2 = _pool2d(encode_2, t)
        encode_3 = _conv2d(_conv2d(down_2, f_num[2], 3, 1, t, is_training=is_training),
                           f_num[2], 3, 1, t, is_training=is_training)
        down_3 = _pool2d(encode_3, t)
        # encode_4 = _conv2d(_conv2d(down_3, f_num[3], 3, 1, t, is_training=is_training),
        #                    f_num[3], 3, 1, t, is_training=is_training)
        # down_4 = _pool2d(encode_4, t)

        # cam = tf.reduce_mean(down_3, axis=-1, keepdims=True, name='cam')
        # cam_split = tf.split(cam, self.seq_len, axis=1)
        # cam_split = [*map(lambda x: tf.squeeze(x, axis=[1,-1]), cam_split)]

        # self.local = tf.expand_dims(tf.stack(cam_split, axis=1), axis=-1)

        bot_conv = _convlstm2d(x=down_3, f_num=f_num[-1], k_size=3, stride=1)
        self.local = tf.reduce_mean(bot_conv, axis=-1, keepdims=False)

        self.logsum = tf.math.reduce_logsumexp(self.local, axis=[2, 3], keepdims=False, name='logsum')

        self.logits = tf.reduce_mean(self.logsum, axis=1)
        self.probs = tf.nn.sigmoid(self.logits)

        self.model = keras.Model(inputs=self.images, outputs=[self.local, self.probs])
        # self.cam_h, self.cam_w = self.local.shape[-2:]


class InferenceModel03:
    def __init__(self, img_w=128, img_h=128, img_d=16, img_c=1, **kwargs):
        self.class_num = 1
        self.img_w, self.img_h, self.depth, self.img_c = img_w, img_h, img_d, img_c
        self.images = keras.Input((img_w, img_h, img_d, img_c))
        self.cam_w, self.cam_h, self.cam_d, self.cam_f = 16, 16, 1, 256

        x = layers.Conv3D(filters=64, kernel_size=3, padding='same', activation='relu')(self.images)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        x = layers.Conv3D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
        x = layers.MaxPool3D(pool_size=2)(x)
        cam = layers.BatchNormalization(name='cam')(x)

        x = layers.GlobalAveragePooling3D()(cam)
        x = layers.Dense(units=512, activation="relu")(x)

        self.probs = layers.Dense(units=1, activation="sigmoid")(x)

        self.model = keras.Model(self.images, [self.probs], name="3d_cnn")

        self.cam_model = keras.Model([self.model.inputs], [self.model.get_layer('cam').output, self.model.output])

        # self.cam_model = keras.Model(self.images, [cam], name="3d_cnn")


class InferenceModel04:
    def __init__(self, image_size=256, depth=16, is_training=False, growth_k=32, theta=0.5,
                 block_rep='4,6,6', use_se=False, **kwargs):
        # super(InferenceModel01, self).__init__()
        self.model_scope, _ = design_scope(class_name=type(self).__name__)

        block_rep_list = list(map(int, re.split(',', block_rep)))
        block_num = len(block_rep_list)

        self.is_training = is_training
        self.class_num = 1

        self.img_h, self.img_w, self.img_d, self.img_c = image_size, image_size, depth, 1
        self.images = layers.Input((self.img_h, self.img_w, self.img_d, self.img_c))

        first_conv = layers.Conv3D(filters=2*growth_k, kernel_size=(4, 4, 1), strides=(2, 2, 1),
                                   padding='same', activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   )(self.images)
        first_batch = layers.BatchNormalization(trainable=self.is_training)(first_conv)
        first_relu = layers.Activation('relu', name='conv1/relu')(first_batch)
        first_pool = layers.MaxPooling3D(pool_size=(4, 4, 1), strides=2, padding='same')(first_relu)

        dsb = first_pool
        for i in range(0, block_num-1):
            dsb = dense_block_3d(input_x=dsb, layer_name=self.model_scope + '_DB'+str(i+1), rep=block_rep_list[i],
                                 growth_k=growth_k, use_se=use_se, is_training=self.is_training)
            print('%d DB: ' % (i+1), dsb)
            dsb = transition_layer_3d(input_x=dsb, layer_name='Transition'+str(i+1),
                                      theta=theta, is_training=self.is_training)
            print('%d Trans: ' % (i+1), dsb)

        self.last_dsb = dense_block_3d(input_x=dsb, layer_name=self.model_scope + '_DB'+str(block_num),
                                       rep=block_rep_list[-1], growth_k=growth_k,
                                       use_se=use_se, is_training=self.is_training)

        last_bn = layers.BatchNormalization(trainable=self.is_training)(self.last_dsb)
        self.bn_relu = layers.Activation('relu', name='last_bn_relu')(last_bn)

        self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2, 3], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)
        flatten = tf.reduce_mean(self.last_pool, axis=[1, 2, 3], keepdims=False)
        self.fc = layers.Dense(units=flatten.shape[-1], activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                               kernel_initializer=tf.keras.initializers.he_normal()
                               )(flatten)  # activation='relu'

        self.logits = layers.Dense(units=self.class_num, activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   kernel_initializer=tf.keras.initializers.he_normal()
                                   )(self.fc)
        self.prob = layers.Activation('sigmoid')(self.logits)
        self.model = keras.Model(inputs=self.images, outputs=self.prob)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer('last_bn_relu').output, self.model.output])


class InferenceModel05:
    def __init__(self, image_size=256, depth=1, is_training=False, growth_k=32, theta=0.5,
                 block_rep='4,6,6', use_se=False, **kwargs):
        # super(InferenceModel01, self).__init__()
        self.model_scope, _ = design_scope(class_name=type(self).__name__)

        block_rep_list = list(map(int, re.split(',', block_rep)))
        block_num = len(block_rep_list)

        self.is_training = is_training
        self.class_num = 1

        self.img_h, self.img_w, self.img_d, self.img_c = image_size, image_size, depth, 1
        self.images = layers.Input((self.img_h, self.img_w, self.img_d, self.img_c))

        first_conv = layers.Conv3D(filters=2*growth_k, kernel_size=(4, 4, 1), strides=(2, 2, 1),
                                   padding='same', activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   kernel_initializer=tf.keras.initializers.he_normal(),
                                   )(self.images)
        first_batch = layers.BatchNormalization(trainable=self.is_training)(first_conv)
        first_relu = layers.Activation('relu', name='conv1/relu')(first_batch)
        first_pool = layers.MaxPooling3D(pool_size=(4, 4, 1), strides=(2, 2, 1),
                                         padding='same')(first_relu)

        dsb = first_pool
        for i in range(0, block_num-1):
            dsb = dense_block_3d(input_x=dsb, layer_name=self.model_scope + '_DB'+str(i+1), rep=block_rep_list[i],
                                 growth_k=growth_k, use_se=use_se, is_training=self.is_training)
            print('%d DB: ' % (i+1), dsb)
            dsb = transition_layer_3d(input_x=dsb, layer_name='Transition'+str(i+1),
                                      theta=theta, is_training=self.is_training)
            print('%d Trans: ' % (i+1), dsb)

        self.last_dsb = dense_block_3d(input_x=dsb, layer_name=self.model_scope + '_DB'+str(block_num),
                                       rep=block_rep_list[-1], growth_k=growth_k,
                                       use_se=use_se, is_training=self.is_training)

        last_bn = layers.BatchNormalization(trainable=self.is_training)(self.last_dsb)
        self.bn_relu = layers.Activation('relu', name='last_bn_relu')(last_bn)

        self.last_pool = tf.reduce_mean(self.bn_relu, axis=[1, 2, 3], keepdims=True)  # global average pooling
        print('last_pool: ', self.last_pool)
        flatten = tf.reduce_mean(self.last_pool, axis=[1, 2, 3], keepdims=False)
        self.fc = layers.Dense(units=flatten.shape[-1], activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                               kernel_initializer=tf.keras.initializers.he_normal()
                               )(flatten)  # activation='relu'

        self.logits = layers.Dense(units=self.class_num, activation=None,
                                   kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                                   kernel_initializer=tf.keras.initializers.he_normal()
                                   )(self.fc)
        self.prob = layers.Activation('sigmoid')(self.logits)
        self.model = keras.Model(inputs=self.images, outputs=self.prob)
        self.cam_model = keras.Model(inputs=self.images,
                                     outputs=[self.model.get_layer('last_bn_relu').output, self.model.output])


def focal_loss_sigmoid(y_true, y_pred, alpha=0.05, gamma=2.):
    # y_pred = tf.nn.sigmoid(y_pred)
    fcl_loss = -y_true*(1-alpha)*((1-y_pred)**gamma)*tf.math.log(y_pred)-\
               (1-y_true)*alpha*(y_pred**gamma)*tf.math.log(1-y_pred)
    return tf.reduce_mean(fcl_loss)


class FocalLossSigmoid(keras.losses.Loss):
    def call(self, y_true, y_pred, alpha=0.05, gamma=2.):
        y_pred = tf.squeeze(tf.nn.sigmoid(y_pred))

        fcl_loss = -y_true*(1-alpha)*((1-y_pred)**gamma)*tf.math.log(y_pred)-\
                   (1-y_true)*alpha*(y_pred**gamma)*tf.math.log(1-y_pred)
        return tf.reduce_mean(fcl_loss)


if __name__ == '__main__':
    import os, logging
    import numpy as np
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    logging.disable(logging.WARNING)

    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation
    # tf.config.experimental.set_visible_devices(gpu[0], 'GPU')
    # infer = InferenceModel01(input_size=[256, 256, 1], growth_k=32,
    #                          image_size=256, block_rep='4,6,6', theta=0.5, use_se=False)

    # infer = InferenceModel03(128, 128, 16, 1, f_num=[64, 128, 256, 512], is_training=True)
    infer = InferenceModel05(is_training=True, use_se=True)
    network_save_png = '/workspace/MRA/exp001/Model05.png'

    keras.utils.plot_model(infer.model, network_save_png, show_shapes=True, show_layer_names=False)

    if False:
        layer_name = 'block3_conv1'
        model = tf.keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
        submodel = tf.keras.models.Model([model.inputs[0]], [model.get_layer(layer_name).output])

        input_img_data = np.random.random((4, 224, 224, 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128.

        input_img_data = tf.Variable(tf.cast(input_img_data, tf.float32))

        # Iterate gradient ascents
        for _ in range(100):
            with tf.GradientTape() as tape:
                outputs = submodel(input_img_data)
                loss_value = tf.reduce_mean(outputs[:, :, :, 0])

            import pdb; pdb.set_trace()
            grads = tape.gradient(loss_value, input_img_data)
            normalized_grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)
            input_img_data.assign_add(normalized_grads * 1.)

    # img = tf.random.normal([8, 256, 256, 1])
    # infer.grad_cam(img)
    # prob = tf.nn.sigmoid(infer.model.outputs)