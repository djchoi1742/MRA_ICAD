import tensorflow as tf
import numpy as np
x = []


@tf.function  # Wrap the function with tf.function
def create_variable():
    if not x:
        x.append(tf.Variable(1.))
    return x[0]


@tf.function
def replica_fn(inputs):
    img, mask, lbl = inputs
    if img.shape[0] == 0:
    # if len(inputs) == 0:
        return tf.constant(1.0)
    else:
        return tf.constant(2.0)

strategy = tf.distribute.MirroredStrategy(["GPU:0", "GPU:1"])
# tensor_input = tf.constant(3.0)

# result = strategy.run(replica_fn, args=(tensor_input,))

images = [np.ones((8, 256, 256, 1)), np.ones((8, 256, 256, 1)), np.ones((8, 256, 256, 1))]
masks = [np.zeros((8, 256, 256, 1)), np.zeros((8, 256, 256, 1)), np.zeros((8, 256, 256, 1))]
labels = np.array([0, 1, 2])
with strategy.scope():
    dataset = tf.data.Dataset.from_tensor_slices((images, masks, labels))
    dataset = dataset.batch(2, drop_remainder=False)
    dataset = strategy.experimental_distribute_dataset(dataset)

step = 0
for x in dataset:

    result = strategy.run(replica_fn, args=(x,))
    print(strategy.reduce(tf.distribute.ReduceOp.SUM, result, axis=None))
    # import pdb; pdb.set_trace()
    # print(step, result)
    # print(tf.convert_to_tensor(result))
    step += 1



