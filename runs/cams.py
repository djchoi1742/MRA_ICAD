import sys
import tensorflow as tf
import numpy as np
import skimage.transform


sys.path.append('/workspace/bitbucket/MRA')


def gen_grad_cam_2d(cam_layer, loss, tape, infer):
    grads = tape.gradient(loss, cam_layer)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    batch_size = cam_layer.shape[0]
    img_size = infer.model.input.shape[1]
    heatmaps = np.zeros((batch_size, img_size, img_size), dtype=np.float32)

    for batch in range(batch_size):  # batch size
        cam_batch = np.zeros(cam_layer.shape[1:3], dtype=np.float32)
        for index, w in enumerate(weights[batch]):  # each weights of batch
            cam_batch += w * cam_layer[batch, :, :, index]
        cam_resize = skimage.transform.resize(cam_batch, (img_size, img_size))
        cam_resize = np.maximum(0, cam_resize)
        heatmaps[batch, :, :] = cam_resize

    heatmaps = np.expand_dims(heatmaps, axis=-1)

    return heatmaps


def gen_grad_cam_lstm(cam_layer, loss, tape, infer):
    grads = tape.gradient(loss, cam_layer)
    weights = tf.reduce_mean(grads, axis=(1, 2, 3))
    cam = np.zeros(cam_layer.shape[0:4], dtype=np.float32)
    batch_size = cam_layer.shape[0]
    seq_len, img_size = infer.model.input.shape[1:3]
    heatmaps = np.zeros((batch_size, seq_len, img_size, img_size), dtype=np.float32)

    for batch in range(batch_size):  # batch size
        for index, w in enumerate(weights[batch]):  # each weights of batch
            cam[batch, :, :, :] += w * cam_layer[batch, :, :, :, index]

        cam_resize = skimage.transform.resize(cam[batch, :, :, :], (seq_len, img_size, img_size))
        cam_resize = np.maximum(0, cam_resize)
        heatmaps[batch, :, :, :] = cam_resize

    heatmaps = np.expand_dims(heatmaps, axis=-1)

    return heatmaps


def gen_cam_lstm(cam_layer, grad, infer):
    weights = tf.reduce_mean(grad, axis=(1, 2, 3))
    cam = np.zeros(cam_layer.shape[0:4], dtype=np.float32)
    batch_size = cam_layer.shape[0]
    seq_len, img_size = infer.model.input.shape[1:3]
    heatmaps = np.zeros((batch_size, seq_len, img_size, img_size), dtype=np.float32)

    for batch in range(batch_size):  # batch size
        for index, w in enumerate(weights[batch]):  # each weights of batch
            cam[batch, :, :, :] += w * cam_layer[batch, :, :, :, index]

        cam_resize = skimage.transform.resize(cam[batch, :, :, :], (seq_len, img_size, img_size))
        cam_resize = np.maximum(0, cam_resize)
        heatmaps[batch, :, :, :] = cam_resize

    heatmaps = np.expand_dims(heatmaps, axis=-1)

    return heatmaps


def gen_grad_cam_3d(cam_layer, loss, tape, infer):
    grads = tape.gradient(loss, cam_layer)
    weights = tf.reduce_sum(grads, axis=(1, 2, 3))
    batch_size = cam_layer.shape[0]
    seq_len, img_size = infer.model.input.shape[1:3]
    heatmaps = np.zeros((batch_size, seq_len, img_size, img_size), dtype=np.float32)

    for batch in range(batch_size):  # batch size
        cam_batch = np.zeros(cam_layer.shape[1:4], dtype=np.float32)
        for index, w in enumerate(weights[batch]):  # each weights of batch
            cam_batch += w * cam_layer[batch, :, :, :, index]
        cam_batch = np.squeeze(cam_batch)
        cam_resize = skimage.transform.resize(cam_batch, (img_size, img_size), preserve_range=True)
        cam_resize = np.maximum(0, cam_resize)
        # cam_resize_st = (cam_resize - np.mean(cam_resize)) / np.std(cam_resize)
        for d in range(seq_len):
            heatmaps[batch, d, :, :] = cam_resize

    heatmaps = np.expand_dims(heatmaps, axis=-1)

    return heatmaps


