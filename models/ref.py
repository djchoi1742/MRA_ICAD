from typing import Text
import tensorflow as tf
from tensorflow.keras import layers


def _get_name(prefix, suffix, sep="/"):
    return "{}{}{}".format(prefix, sep, suffix)


def _transition_block(x: tf.Tensor, reduction: float, name: Text, dropout: float = 0.0):
    """"""
    x = layers.BatchNormalization(axis=3, epsilon=1e-5, name=_get_name(name, 'bn'))(x)
    x = layers.ReLU(name=_get_name(name, 'relu'))(x)
    x = layers.Conv2D(filters=int(x.shape[-1] * reduction),
                      kernel_size=(1,1), strides=(1,1), padding='valid', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                      kernel_initializer=tf.keras.initializers.he_normal(),
                      name=_get_name(name, 'conv'))(x)
    if dropout > 0:
        x = layers.Dropout(rate=dropout, name=_get_name(name, 'dropout'))
    x = layers.AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name=_get_name(name, 'avgpool'))(x)
    return x


def _final_transition_block(x: tf.Tensor, name: Text):
    x = layers.BatchNormalization(axis=3, epsilon=1e-5, name=_get_name(name, 'bn'))(x)
    x = layers.ReLU(name=_get_name(name, 'relu'))(x)
    x = layers.GlobalAvgPool2D(name=_get_name(name, 'avgpool'))(x)
    return x


def _conv_block(x: tf.Tensor, growth_rate: int, name: Text, dropout: float=0.0):
    x_out = layers.BatchNormalization(axis=3, epsilon=1e-5, name=_get_name(name, 'bn_0'))(x)
    x_out = layers.ReLU(name=_get_name(name, 'relu_0'))(x_out)
    x_out = layers.Conv2D(filters=4*growth_rate, kernel_size=(1,1), strides=(1,1),
                          padding='valid', use_bias=False, kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
                          kernel_initializer=tf.keras.initializers.he_normal(),
                          name=_get_name(name, 'conv_1'))(x_out)
    if dropout > 0:
        x = layers.Dropout(rate=dropout, name=_get_name(name, 'dropout_1'))
    x_out = layers.BatchNormalization(axis=3, epsilon=1e-5, name=_get_name(name, 'bn_1'))(x_out)
    x_out = layers.ReLU(name=_get_name(name, 'relu_1'))(x_out)

    x_out = tf.keras.layers.ZeroPadding2D(
        padding=((1, 1), (1, 1)), name=_get_name(name, "pad")
    )(x_out)
    x_out = tf.keras.layers.Conv2D(
        filters=growth_rate,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="valid",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        kernel_initializer=tf.keras.initializers.he_normal(),
        name=_get_name(name, "conv_2"),
    )(x_out)
    if dropout > 0:
        x = tf.keras.layers.Dropout(rate=dropout, name=_get_name(name, "dropout_2"))
    x_out = tf.keras.layers.Concatenate(axis=3, name=_get_name(name, "concat"))(
        [x, x_out]
    )
    return x_out


def _dense_block(
    x: tf.Tensor, growth_rate: int, num_per_block: int, name: Text, dropout: float = 0.0
):
    """"""
    for i in range(num_per_block):
        x = _conv_block(
            x, growth_rate, name="{}/block_{}".format(name, i + 1), dropout=dropout
        )
    return x


def _densenet_224x224(
    depth: int = 121,
    growth_rate: int = 32,
    reduction: float = 0.5,
    dropout: float = 0.0,
    num_classes: int = 1,
) -> tf.keras.Model:

    if depth not in {121, 161, 169, 201}:
        raise ValueError("invalid `depth` specified: {}.".format(depth))

    # if num_classes not in {10, 1000}:
    #     raise ValueError("invalid `num_classes` specified: {}.".format(num_classes))

    stages = [4, 6, 6]

    if False:
        stages = [6, 12, 48, 32]  # depth == 201
        if depth == 121:
            stages = [6, 12, 24, 16]
        elif depth == 161:
            stages = [6, 12, 36, 24]
        elif depth == 169:
            stages = [6, 12, 32, 32]

    # input layer
    inputs = tf.keras.Input(shape=(256, 256, 1), name="input")

    # initial conv layer
    x = tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="init/pad_1")(
        inputs
    )
    x = tf.keras.layers.Conv2D(
        filters=growth_rate * 2,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="valid",
        use_bias=False,
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        kernel_initializer=tf.keras.initializers.he_normal(),
        name="init/conv",
    )(x)
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1e-5, name="init/bn")(x)
    x = tf.keras.layers.ReLU(name="init/relu")(x)
    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="init/pad_2")(x)
    x = tf.keras.layers.MaxPool2D(
        pool_size=(3, 3), strides=(2, 2), padding="valid", name="input/maxpool"
    )(x)

    # intermediate blocks
    for i, num_per_block in enumerate(stages[:-1]):
        x = _dense_block(
            x, growth_rate, num_per_block, "dense_{}".format(i + 1), dropout=dropout
        )
        x = _transition_block(x, reduction, "trans_{}".format(i + 1), dropout)

    # final block
    x = _dense_block(x, growth_rate, stages[-1], "dense_4", dropout=dropout)
    x = _final_transition_block(x, "trans_4")

    # output layer
    outputs = tf.keras.layers.Dense(
        units=num_classes,
        kernel_regularizer=tf.keras.regularizers.l2(l=1e-4),
        name="output",
        activation=None
    )(x)

    model = tf.keras.Model(
        inputs=inputs, outputs=outputs, name="densenet_32x32_{}".format(num_classes)
    )

    model.summary()

    return model


def densenet121(
    growth_rate: int = 32,
    reduction: float = 0.5,
    dropout: float = 0.0,
    num_classes: int = 1,
) -> tf.keras.Model:
    return _densenet_224x224(
        depth=121,
        growth_rate=growth_rate,
        reduction=reduction,
        dropout=dropout,
        num_classes=num_classes,
    )


if __name__ == '__main__':
    model = densenet121()
    print('inputs: ', model.inputs)
    print('outputs: ', model.outputs)
