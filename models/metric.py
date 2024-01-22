import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import *
import sklearn.metrics
from tensorflow.keras import backend as K
import numpy as np
import cv2
import pandas as pd
import re
import sys
from itertools import compress

sys.path.append('/workspace/bitbucket/MRA')


@tf.function
def weighted_dice_score_loss(y_true, y_pred, smooth=1.0, pos_weight=0.90, gamma=2.0, each=False):
    y_true, y_pred = tf.cast(y_true[..., 0], tf.float32), tf.cast(y_pred[..., 0], tf.float32)
    y_true_bg, y_pred_bg = 1. - y_true, 1. - y_pred

    numer_fg = 2. * tf.reduce_sum(y_true * y_pred, axis=[-1, -2])
    denom_fg = tf.reduce_sum(y_true + y_pred, axis=[-1, -2]) + smooth
    numer_bg = 2. * tf.reduce_sum(y_true_bg * y_pred_bg, axis=[-1, -2])
    denom_bg = tf.reduce_sum(y_true_bg + y_pred_bg, axis=[-1, -2]) + smooth

    fg_loss = pos_weight * (1 - numer_fg / denom_fg)
    bg_loss = (1. - pos_weight) * (1 - numer_bg / denom_bg)

    if each:
        return tf.reduce_mean(fg_loss + bg_loss ** gamma, axis=-1)
    else:
        return tf.reduce_mean(fg_loss + bg_loss ** gamma)


@tf.function
def focal_loss_sigmoid(y_true, y_pred, alpha=0.05, gamma=0.):  # gamma=2.
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    smooth = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, smooth, 1.0 - smooth)

    fcl_loss = -y_true*(1-alpha)*((1-y_pred)**gamma)*tf.math.log(y_pred) - \
               (1-y_true)*alpha*(y_pred**gamma)*tf.math.log(1-y_pred)

    return tf.reduce_mean(fcl_loss)


@tf.function
def object_loss(y_true, y_pred, alpha=0.01, gamma=2., init_w=0.1):  # previous: alpha=0.01, gamma=2.
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    smooth = keras.backend.epsilon()
    y_pred = keras.backend.clip(y_pred, smooth, 1.0 - smooth)

    # object_true = tf.reshape(tf.reduce_sum(y_true, axis=(1, 2, 3)), (-1, 1, 1, 1))
    pos_loss = - (1 - alpha) * y_true * ((1-y_pred) ** gamma) * tf.math.log(y_pred)
    neg_loss = - alpha * (1 - y_true) * (y_pred ** gamma) * tf.math.log(1 - y_pred)

    total_loss = (pos_loss + neg_loss)  # * object_true

    # return tf.reduce_sum(pos_loss + neg_loss ** gamma)  # previous
    return init_w * tf.reduce_sum(total_loss)


@tf.function
def weighted_object_loss(y_true, y_pred, alpha=0.01, gamma=2.):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    batch_size, seq_length = y_true.shape[0:2]

    y_true_sum = tf.reduce_sum(y_true, axis=(2, 3))
    y_true_weight = tf.where(y_true_sum != 0, 1 / y_true_sum, 0)

    pos_loss = - (1 - alpha) * y_true * ((1-y_pred) ** gamma) * tf.math.log(y_pred)

    pos_loss *= tf.reshape(y_true_weight, (batch_size, seq_length, 1, 1, 1))
    neg_loss = - alpha * (1 - y_true) * (y_pred ** gamma) * tf.math.log(1 - y_pred)

    total_loss = (pos_loss + neg_loss)  # * object_true

    # return tf.reduce_sum(pos_loss + neg_loss ** gamma)  # previous
    return tf.reduce_sum(total_loss)


@tf.function
def weighted_object_loss_3d(y_true, y_pred, alpha=0.01, gamma=2.):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)

    y_true_sum = tf.reduce_sum(y_true, axis=(1, 2, 3))
    y_true_weight = tf.map_fn(lambda x: 1 / x if x != 0 else 0, y_true_sum)  # lesion-weight: 1/x

    pos_loss = - (1 - alpha) * y_true * ((1-y_pred) ** gamma) * tf.math.log(y_pred)
    pos_loss *= tf.reshape(y_true_weight, (-1, 1, 1, 1))
    neg_loss = - alpha * (1 - y_true) * (y_pred ** gamma) * tf.math.log(1 - y_pred)

    total_loss = (pos_loss + neg_loss)  # * object_true

    # return tf.reduce_sum(pos_loss + neg_loss ** gamma)  # previous
    return tf.reduce_sum(total_loss)


def dcs_2d(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)  # (n, t, h, w, c)
    y_pred = tf.where(y_pred < 0.5, 0.0, 1.0)

    numer = tf.cast(2 * tf.reduce_sum(y_true * y_pred, axis=[-3, -2]), tf.float32)  # reduce (h, w)
    denom = tf.cast(1 + tf.reduce_sum(y_true + y_pred, axis=[-3, -2]), tf.float32)  # reduce (h, w)

    return numer / denom


def dcs_3d(y_true, y_pred):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)  # (n, t, h, w, c)
    y_pred = tf.where(y_pred < 0.5, 0.0, 1.0)

    numer = tf.cast(2 * tf.reduce_sum(y_true * y_pred, axis=[-4, -3, -2]), tf.float32)  # reduce (h, w)
    denom = tf.cast(1 + tf.reduce_sum(y_true + y_pred, axis=[-4, -3, -2]), tf.float32)  # reduce (h, w)

    return numer / denom


def iou_2d(y_true, y_pred):  # consider 3d input
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)  # (n, t, h, w, c)
    y_pred = tf.where(y_pred < 0.5, 0.0, 1.0)

    inter = tf.cast(tf.math.count_nonzero(y_true * y_pred, axis=[-3, -2]), tf.float32)
    union = tf.cast(tf.math.count_nonzero(y_true + y_pred, axis=[-3, -2]) + 1, tf.float32)

    return inter / union


def iou_3d(y_true, y_pred):  # consider 3d input
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)  # (n, t, h, w, c)
    y_pred = tf.where(y_pred < 0.5, 0.0, 1.0)

    inter = tf.cast(tf.math.count_nonzero(y_true * y_pred, axis=[-4, -3, -2]), tf.float32)
    union = tf.cast(tf.math.count_nonzero(y_true + y_pred, axis=[-4, -3, -2]) + 1, tf.float32)

    return inter / union


def detect_iou(y_true, y_pred, add=False):
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)
    y_pred = tf.where(y_pred < 0.9, 0.0, 1.0)

    sum_y_true = tf.map_fn(tf.reduce_sum, y_true)

    ious = []
    recs, pcss = [], []
    for idx in range(y_true.shape[0]):
        if sum_y_true[idx] != 0:
            y_true_batch, y_pred_batch = y_true[idx, :, :, :, :], y_pred[idx, :, :, :, :]
            inter = tf.cast(tf.math.count_nonzero(y_true_batch * y_pred_batch), tf.float32)
            union = tf.cast(tf.math.count_nonzero(y_true_batch + y_pred_batch) + 1, tf.float32)
            each_iou = inter / union

            if add:
                ious.append(each_iou.numpy())
                rec = inter / tf.reduce_sum(y_true_batch)
                pcs = inter / tf.reduce_sum(y_pred_batch)
                recs.append(rec.numpy())
                pcss.append(pcs.numpy())
            else:
                ious.append(each_iou)
    if add:
        return ious, recs, pcss
    else:
        return ious


def mtl_vars(input_model):  # input_model: Model
    log_vars = tf.Variable(initial_value=tf.zeros(len(input_model.outputs)), trainable=True)
    input_model.params = input_model.model.trainable_variables + [log_vars]
    return log_vars, input_model.parmas


def multi_task_loss(loss_vars, log_vars):
    loss = 0

    for i in range(len(loss_vars)):
        task_weight = tf.math.exp(-log_vars[i])
        loss += tf.reduce_sum(task_weight * loss_vars[i] + log_vars[i])

    return tf.reduce_mean(loss)


def naive_sum_loss(losses, seg_lambda=1.0, cls_lambda=None, det_lambda=None):
    if len(losses) == 2:
        if cls_lambda is not None:
            loss_weight = tf.Variable([seg_lambda, cls_lambda])
        elif det_lambda is not None:
            loss_weight = tf.Variable([seg_lambda, det_lambda])
        else:
            raise ValueError
    elif len(losses) == 3:
        if cls_lambda is not None and det_lambda is not None:
            loss_weight = tf.Variable([seg_lambda, cls_lambda, det_lambda])
        else:
            raise ValueError
    else:
        raise ValueError

    loss_batch = tf.reduce_sum(tf.Variable(losses) * loss_weight)

    return loss_batch


def object_accuracy(y_true, y_pred):
    height, width = 256, 256
    acc = float((1 - np.sum(abs(y_true - y_pred)) /
                 (height * width * int(y_true.shape[0]))) * 100)
    return acc


def iou(y_true, y_pred):  # consider 3d input
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)  # (n, t, h, w, c)
    y_pred = tf.where(y_pred < 0.5, 0.0, 1.0)

    inter = tf.cast(tf.math.count_nonzero(y_true * y_pred), tf.float32)
    union = tf.cast(tf.math.count_nonzero(y_true + y_pred) + 1, tf.float32)

    return tf.reduce_mean(inter / union)


def each_dcs(y_true, y_pred):  # per slice
    y_true, y_pred = tf.cast(y_true, tf.float32), tf.cast(y_pred, tf.float32)  # (h, w, c)
    y_pred = tf.where(y_pred < 0.5, 0.0, 1.0)

    numer = tf.cast(2 * tf.reduce_sum(y_true * y_pred, axis=[0, 1]), tf.float32)
    denom = tf.cast(1 + tf.reduce_sum(y_true + y_pred, axis=[0, 1]), tf.float32)

    return tf.reduce_mean(numer / denom)


def calculate_auc(y, x):
    if len(pd.Series(y).value_counts()) == 1:  # all label 0 or 1
        auc_value = 0.0000
    else:
        try:
            fpr, tpr, _ = sklearn.metrics.roc_curve(y, x, drop_intermediate=False)
            auc_value = sklearn.metrics.auc(fpr, tpr)
        except:
            auc_value = 0.0000
    return auc_value


def set_jafroc_2d(dets, pred_dets, pred_scores, name):
    name = name.numpy()
    batch_size, seq_len = pred_dets.shape[0:2]

    each_ns1, each_ss1, rts1, wts1, scs1 = [], [], [], [], []
    each_ns2, each_ss2, rts2, wts2, scs2 = [], [], [], [], []

    # each_ns1, each_ss1, rts1, scs1, wts1 = [], [], [], [], []
    # each_ns2, each_ss2, rts2, scs2, wts2 = [], [], [], [], []

    for i in range(batch_size):
        patient_id, start_idx = name_idx(name[i])

        each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1)])
        each_det = dets.numpy()[i, :, :, 0]
        each_pred_det = pred_dets.numpy()[i, :, :, 0]
        # each_score_det = pred_scores.numpy()[i, :, :, 0]
        each_ste = np.sum(each_det).astype(int)

        if each_ste == 0:
            normal_max = np.max(each_pred_det)
            each_ss1.extend([each_ste])
            each_ns1.extend([each_name])
            rts1.extend([normal_max])
            # scs1.extend([np.max(each_score_det)])
            wts1.extend([0])

            each_ss2.extend([each_ste])
            each_ns2.extend([each_name])
            rts2.extend([normal_max])
            # scs2.extend([np.max(each_score_det)])
            wts2.extend([0])

        else:
            each_ss1.extend(np.repeat(1, each_ste).tolist())
            each_ns1.extend(np.repeat(each_name, each_ste).tolist())

            each_ss2.extend(np.repeat(1, each_ste).tolist())
            each_ns2.extend(np.repeat(each_name, each_ste).tolist())

            each_gt_idx = np.where(each_det == 1)
            each_gt_prob = each_pred_det[each_gt_idx].tolist()
            # each_gt_score = each_score_det[each_gt_idx].tolist()

            rts1.extend(each_gt_prob)
            # scs1.extend(each_gt_score)
            wts1.extend(np.repeat(1 / each_ste, each_ste).tolist())

            rts2.extend(each_gt_prob)
            # scs2.extend(each_gt_score)
            wts2.extend(np.repeat(1 / each_ste, each_ste).tolist())

            # the highest noise in abnormal case (HN)
            each_ss1.extend([0])
            each_ns1.extend([each_name])

            each_hn_idx = np.where(each_det == 0)
            each_hn_prob = np.max(each_pred_det[each_hn_idx].tolist())
            # each_hn_score = np.max(each_score_det[each_hn_idx].tolist())

            rts1.extend([each_hn_prob])
            # scs1.extend([each_hn_score])
            wts1.extend([0])

    return each_ns1, each_ss1, rts1, scs1, wts1, each_ns2, each_ss2, rts2, scs2, wts2


def set_jafroc_seq(dets, pred_dets, pred_scores, name):
    name = name.numpy()
    batch_size, seq_len = pred_dets.shape[0:2]

    each_ns1, each_ss1, rts1, wts1, scs1 = [], [], [], [], []
    each_ns2, each_ss2, rts2, wts2, scs2 = [], [], [], [], []

    for i in range(batch_size):
        patient_id, start_idx = name_idx(name[i])
        for j in range(seq_len):
            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])
            each_det = dets.numpy()[i, j, :, :, 0]
            each_pred_det = pred_dets.numpy()[i, j, :, :, 0]
            # each_score_det = pred_scores.numpy()[i, j, :, :, 0]
            each_ste = np.sum(each_det).astype(int)

            each_gt_idx = np.where(each_det == 1)
            each_hn_idx = np.where(each_det == 0)

            if each_ste == 0:
                each_ss1.extend([each_ste])
                each_ns1.extend([each_name])
                rts1.extend([np.max(each_pred_det)])
                wts1.extend([0])

                each_ss2.extend([each_ste])
                each_ns2.extend([each_name])
                rts2.extend([np.max(each_pred_det)])
                wts2.extend([0])

            else:
                each_ss1.extend(np.repeat(1, each_ste).tolist())
                each_ns1.extend(np.repeat(each_name, each_ste).tolist())

                each_ss2.extend(np.repeat(1, each_ste).tolist())
                each_ns2.extend(np.repeat(each_name, each_ste).tolist())

                # each_gt_idx = np.where(each_det == 1)
                each_gt_prob = each_pred_det[each_gt_idx].tolist()

                rts1.extend(each_gt_prob)
                wts1.extend(np.repeat(1 / each_ste, each_ste).tolist())

                rts2.extend(each_gt_prob)
                wts2.extend(np.repeat(1 / each_ste, each_ste).tolist())

                # the highest noise in abnormal case (HN)
                each_ss1.extend([0])
                each_ns1.extend([each_name])

                # each_hn_idx = np.where(each_det == 0)
                each_hn_prob = np.max(each_pred_det[each_hn_idx].tolist())

                rts1.extend([each_hn_prob])
                wts1.extend([0])

            if pred_scores is not None:
                each_score_det = pred_scores.numpy()[i, j, :, :, 0]

                if each_ste == 0:
                    scs1.extend([np.max(each_score_det)])
                    scs2.extend([np.max(each_score_det)])
                else:
                    each_gt_score = each_score_det[each_gt_idx].tolist()

                    scs1.extend(each_gt_score)
                    scs2.extend(each_gt_score)

                    # the highest noise in abnormal case (HN)
                    each_hn_score = np.max(each_score_det[each_hn_idx].tolist())
                    scs1.extend([each_hn_score])

    return each_ns1, each_ss1, rts1, scs1, wts1, each_ns2, each_ss2, rts2, scs2, wts2


def set_jafroc_seq_np(dets, pred_dets, pred_scores, name):
    batch_size, seq_len = pred_dets.shape[0:2]

    each_ns1, each_ss1, rts1, wts1, scs1 = [], [], [], [], []
    each_ns2, each_ss2, rts2, wts2, scs2 = [], [], [], [], []

    for i in range(batch_size):
        patient_id, start_idx = name_idx(name[i])
        for j in range(seq_len):

            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])
            each_det = dets[i, j, :, :, 0]
            each_pred_det = pred_dets[i, j, :, :, 0]
            # each_score_det = pred_scores.numpy()[i, j, :, :, 0]
            each_ste = np.sum(each_det).astype(int)

            each_gt_idx = np.where(each_det == 1)
            each_hn_idx = np.where(each_det == 0)

            if each_ste == 0:
                each_ss1.extend([each_ste])
                each_ns1.extend([each_name])
                rts1.extend([np.max(each_pred_det)])
                wts1.extend([0])

                each_ss2.extend([each_ste])
                each_ns2.extend([each_name])
                rts2.extend([np.max(each_pred_det)])
                wts2.extend([0])

            else:
                each_ss1.extend(np.repeat(1, each_ste).tolist())
                each_ns1.extend(np.repeat(each_name, each_ste).tolist())

                each_ss2.extend(np.repeat(1, each_ste).tolist())
                each_ns2.extend(np.repeat(each_name, each_ste).tolist())

                # each_gt_idx = np.where(each_det == 1)
                each_gt_prob = each_pred_det[each_gt_idx].tolist()

                rts1.extend(each_gt_prob)
                wts1.extend(np.repeat(1 / each_ste, each_ste).tolist())

                rts2.extend(each_gt_prob)
                wts2.extend(np.repeat(1 / each_ste, each_ste).tolist())

                # the highest noise in abnormal case (HN)
                each_ss1.extend([0])
                each_ns1.extend([each_name])

                # each_hn_idx = np.where(each_det == 0)
                each_hn_prob = np.max(each_pred_det[each_hn_idx].tolist())

                rts1.extend([each_hn_prob])
                wts1.extend([0])

            if pred_scores is not None:
                each_score_det = pred_scores[i, j, :, :, 0]

                if each_ste == 0:
                    scs1.extend([np.max(each_score_det)])
                    scs2.extend([np.max(each_score_det)])
                else:
                    each_gt_score = each_score_det[each_gt_idx].tolist()

                    scs1.extend(each_gt_score)
                    scs2.extend(each_gt_score)

                    # the highest noise in abnormal case (HN)
                    each_hn_score = np.max(each_score_det[each_hn_idx].tolist())
                    scs1.extend([each_hn_score])

    return each_ns1, each_ss1, rts1, scs1, wts1, each_ns2, each_ss2, rts2, scs2, wts2


def set_jafroc(dets, pred_dets, pred_scores, name):
    name = name.numpy()
    batch_size = dets.shape[0]

    each_ns1, each_ss1, rts1, scs1, wts1 = [], [], [], [], []
    each_ns2, each_ss2, rts2, scs2, wts2 = [], [], [], [], []

    for i in range(batch_size):
        each_det = dets.numpy()[i, :, :, 0]
        each_pred_det = pred_dets.numpy()[i, :, :, 0]

        each_score_det = pred_scores.numpy()[i, :, :, 0]
        each_ste = np.sum(each_det).astype(int)

        if each_ste == 0:
            normal_max = np.max(each_pred_det)
            each_ss1.extend([each_ste])
            each_ns1.extend([name[i].decode()])
            rts1.extend([normal_max])
            scs1.extend([np.max(each_score_det)])
            wts1.extend([0])

            # the highest noise in normal case (HN)
            each_ss2.extend([each_ste])
            each_ns2.extend([name[i].decode()])
            rts2.extend([normal_max])
            scs2.extend([np.max(each_score_det)])
            wts2.extend([0])

        else:
            each_ss1.extend(np.repeat(1, each_ste).tolist())
            each_ns1.extend(np.repeat(name[i].decode(), each_ste).tolist())

            # ground-truth lesion
            each_ss2.extend(np.repeat(1, each_ste).tolist())
            each_ns2.extend(np.repeat(name[i].decode(), each_ste).tolist())

            each_gt_idx = np.where(each_det == 1)
            each_gt_prob = each_pred_det[each_gt_idx].tolist()
            each_gt_score = each_score_det[each_gt_idx].tolist()

            rts1.extend(each_gt_prob)
            scs1.extend(each_gt_score)
            wts1.extend(np.repeat(1 / each_ste, each_ste).tolist())

            rts2.extend(each_gt_prob)
            scs2.extend(each_gt_score)
            wts2.extend(np.repeat(1 / each_ste, each_ste).tolist())

            # the highest noise in abnormal case (HN): only JAFROC1
            each_ss1.extend([0])
            each_ns1.extend([name[i].decode()])

            each_hn_idx = np.where(each_det == 0)
            each_hn_prob = np.max(each_pred_det[each_hn_idx].tolist())
            each_hn_score = np.max(each_score_det[each_hn_idx].tolist())

            rts1.extend([each_hn_prob])
            scs1.extend([each_hn_score])
            wts1.extend([0])

    return each_ns1, each_ss1, rts1, scs1, wts1, each_ns2, each_ss2, rts2, scs2, wts2


def cal_score(abnormal_r, normal_r):
    if abnormal_r > normal_r:
        score = 1.0
    elif abnormal_r == normal_r:
        score = 0.5
    elif abnormal_r < normal_r:
        score = 0.0
    else:
        raise ValueError('Invalid values')
    return score


def calculate_jafroc(y, rating, weight):
    normal = [x == 0 for x in y]
    abnormal = [x == 1 for x in y]

    normal_rating = [*compress(rating, normal)]
    abnormal_rating = [*compress(rating, abnormal)]
    abnormal_weight = [*compress(weight, abnormal)]

    num_normal = len(normal_rating)
    num_abnormal = np.sum(abnormal_weight)

    fom = 0.0
    try:
        for wt, rt in zip(abnormal_weight, abnormal_rating):
            for nt in normal_rating:
                fom += wt * cal_score(rt, nt)
        fom = fom / (num_normal * num_abnormal)
    except:
        pass

    return fom


def built_guided_model(model, mtl_mode=False):
    model_copied = clone_model(model)
    if mtl_mode:
        model_copied.set_weights(model.get_weights()[0:-1])  # excluded mtl weights
    else:
        model_copied.set_weights(model.get_weights())

    @tf.custom_gradient
    def guided_relu(x):
        def grad(dy):
            return tf.cast(dy > 0, tf.float32) * tf.cast(x > 0,  tf.float32) * dy
        return tf.nn.relu(x), grad
    layer_dict = [layer for layer in model_copied.layers[1:] if hasattr(layer, 'activation')]
    for layer in layer_dict:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu
    return model_copied


def guided_backprop(model, img, layer_name):
    part_model = keras.Model(model.inputs, model.get_layer(layer_name).output)

    with tf.GradientTape() as tape:
        f32_img = tf.cast(img, tf.float32)
        tape.watch(f32_img)
        part_output = part_model(f32_img)

    grads = tape.gradient(part_output, f32_img)[0].numpy()

    del part_model
    return grads


def deprocess_image(x):
    x = x.copy()
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    x += 0.5
    x = np.clip(x, 0, 1)

    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')

    return x


def guided_grad_cam(guide_backprop, gradcam, target_size=(224, 224)):
    resized_gradcam = cv2.resize(gradcam, guide_backprop[:-1])
    gradcam_r = np.repeat(resized_gradcam[..., None], 3, axis=2)

    return guide_backprop * gradcam_r


def name_idx(name):
    png_name = name.decode()
    name_split = re.split('_', png_name)
    if len(name_split) <= 3:
        patient_id, start_idx = name_split[0:2]
    else:
        patient_id, start_idx = '_'.join([name_split[0], name_split[1]]), name_split[2]
    return patient_id, start_idx
