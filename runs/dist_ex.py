import numpy as np
import os
import warnings
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

import inspect
print(inspect.getfile(inspect.currentframe()))


fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# 하나의 차원을 배열에 추가 -> 새로운 shape == (28, 28, 1)
# 이렇게 하는 이유는 우리의 모델에서 첫 번째 층이 합성곱 층이고
# 합성곱 층은 4D 입력을 요구하기 때문입니다.
# (batch_size, height, width, channels).
# batch_size 차원은 나중에 추가할 것입니다.

train_images = train_images[..., None]
test_images = test_images[..., None]

# 이미지를 [0, 1] 범위로 변경하기.
train_images = train_images / np.float32(255)
test_images = test_images / np.float32(255)

strategy = tf.distribute.MirroredStrategy()
print('# of Devices: {}'.format(strategy.num_replicas_in_sync))

BUFFER_SIZE = len(train_images)

BATCH_SIZE_PER_REPLICA = 1
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

EPOCHS = 10


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')])
    return model


def compute_loss(labels, predictions):
    per_example_loss = loss_object(labels, predictions)

    return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)


@tf.function
def distributed_train_step(dataset_inputs):

    per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))

    losses = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    return losses


def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = compute_loss(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_accuracy.update_state(labels, predictions)
    return loss


def test_step(inputs):
    images, labels = inputs

    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss.update_state(t_loss)
    test_accuracy.update_state(labels, predictions)


def distributed_test_step(dataset_inputs):

    return strategy.run(test_step, args=(dataset_inputs,))


with strategy.scope():
    # @tf.function
    # def distributed_train_step(dataset_inputs):
    #     per_replica_losses = strategy.run(train_step,
    #                                       args=(dataset_inputs,))
    #     return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
    #                            axis=None)

    # tf.data.Iterator.get_next_as_optional()
    train_dataset = tf.data.Dataset.from_tensor_slices((
        train_images, train_labels)).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE)
    train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(GLOBAL_BATCH_SIZE)
    test_dist_dataset = strategy.experimental_distribute_dataset(test_dataset)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # Evaluation metrics
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

    model = create_model()
    optimizer = tf.keras.optimizers.Adam()
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    checkpoint_dir = '/workspace/bitbucket/Strategy/training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

for epoch in range(EPOCHS):
    # 훈련 루프
    total_loss = 0.0
    num_batches = 0

    for x in train_dist_dataset:
        total_loss += distributed_train_step(x)

        num_batches += 1

    train_loss = total_loss / num_batches

    # 테스트 루프
    for x in test_dist_dataset:
        distributed_test_step(x)

    if epoch % 2 == 0:
        checkpoint.save(checkpoint_prefix)

    output = 'Epoch:%d Loss:%.4f Accuracy:%.4f Val-Loss:%.4f Val-Accuracy:%.4f' % (
        epoch + 1, train_loss, train_accuracy.result(), test_loss.result(), test_accuracy.result()
    )

    print(output)

    test_loss.reset_states()
    train_accuracy.reset_states()
    test_accuracy.reset_states()





