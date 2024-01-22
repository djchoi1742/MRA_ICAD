import os, glob, time, imageio
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import backend as K
from tensorflow import keras


class ShowPlotsCallback(tf.keras.callbacks.Callback):
    """
    Description
        Present image and metric per each epoch to check the sanity of training process
        - Plot specific single example of prediction mask (first batch and first time-axis)
        - Print IOU
    """
    def __init__(self, val_dataset, batch_size, t):
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.t = t

    def on_epoch_end(self, v_db, n_plt):
        print('\n> Plotting prediction mask...')
        for elem in self.val_dataset:
            if self.t == 1:
                prediction_mask = np.argmax(self.model.predict_on_batch(elem[0]), axis=-1)[0]  # w, h
                label = np.argmax(elem[1], axis=-1)[0]  # w, h
            else:
                prediction_mask = np.argmax(self.model.predict_on_batch(elem[0]), axis=-1)[0][0]  # w, h
                label = np.argmax(elem[1], axis=-1)[0][0]  # w, h

            plt.imshow(prediction_mask)
            plt.show()
            # plt.imshow(label)
            # plt.show()

            numer = 2. * np.count_nonzero(prediction_mask * label)
            denom = np.count_nonzero(prediction_mask) + np.count_nonzero(label) + 1.
            dice = numer / denom
            print('Dice Coefficient Score : %f' % dice)

            break


class SavePlotsCallback(tf.keras.callbacks.Callback):
    """
    Description
        Save predicted mask of model of validation set
    Parameters
        val_dataset (tf.data.Dataset) : same Dataset object consumed at validation time
        batch (int) : batch size
        time (int) : the length of time sequence
    """
    def __init__(self, val_dataset, batch_size, seq_len, save_path):
        # super(PlotValResultCallback, self).__init__()
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.save_path = save_path

    def on_test_begin(self, v_db):
        print('\n> Saving validation result example...')
        for elem in self.val_dataset:
            prediction_mask = self.model.predict_on_batch(elem[0])  # [batch_size, seq_len, 256, 256, 2]
            iou_list = []
            for batch_num in range(self.batch_size):
                for seq_num in range(self.seq_len):

                    if self.seq_len == 1:
                        image = elem[0][batch_num]
                        label = tf.argmax(elem[1][batch_num], axis=-1)
                        # label = elem[1][batch_num][seq_num]
                        masks = tf.argmax(prediction_mask[batch_num], axis=-1)
                    else:
                        image = elem[0][batch_num][seq_num]  # [256, 256, 1], each slice
                        label = tf.argmax(elem[1][batch_num][seq_num], axis=-1)  # [256, 256], label
                        masks = tf.argmax(prediction_mask[batch_num][seq_num], axis=-1)  # predicted mask

                    # draw 3 subplots
                    title_dict = {0: 'Original', 1: 'Label', 2: 'Prediction'}
                    plots_list = [image, label, masks]
                    whole_fig = plt.figure()
                    for j in range(len(plots_list)):
                        sub_fig = whole_fig.add_subplot(1, 3, j + 1)
                        sub_fig.imshow(np.squeeze(plots_list[j]))
                        sub_fig.set_title(title_dict[j])
                        sub_fig.axis('off')

                    # calc metrics
                    union = np.count_nonzero(np.logical_or(label, masks))
                    inter = np.count_nonzero(np.logical_and(label, masks))
                    if union is 0:
                        iou = 0
                    else:
                        iou = inter / union
                        iou_list += [iou]

                    png_index = '_'.join([str(batch_num), str(seq_num)])

                    # save plots
                    if not os.path.exists(self.save_path):
                        os.makedirs(self.save_path)
                    imageio.imwrite(os.path.join(self.save_path, png_index+'_'+str(time.time()) + '.png'), masks)
                    # plt.savefig(os.path.join(self.save_path, '%03d_%.3f.png' %
                    # ((batch_num * self.seq_len + seq_num + 1), iou)))
                    # plt.close()
            # break

        print('Figures saved at', self.save_path)
        print('IOU : %.4f' % np.mean(np.asarray(iou_list)))


class BinaryFocalLoss(keras.losses.Loss):
    def call(self, y_true, y_pred, gamma=2., alpha=0.05):

        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - \
               K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


class DiceScoreLoss(keras.losses.Loss):
    def call(self, y_true, y_pred, smooth=1.):
        numer = 2. * tf.reduce_sum(y_true * y_pred, axis=[-1, -2, -3])
        denom = tf.reduce_sum(y_true + y_pred, axis=[-1, -2, -3]) + smooth

        loss = 1. - numer / denom
        return tf.reduce_mean(loss)
        # return loss


class DiceCrossEntropyLoss(keras.losses.Loss):
    def call(self, y_true, y_pred, smooth=1.):
        numer = 2. * tf.reduce_sum(y_true * y_pred, axis=[-1, -2, -3])
        denom = tf.reduce_sum(y_true + y_pred, axis=[-1, -2, -3]) + smooth
        dice_loss = tf.reduce_mean(1. - numer / denom)
        ce_loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

        return dice_loss + ce_loss


class WeightedDiceScoreLoss(keras.losses.Loss):
    def call(self, y_true, y_pred, smooth=1., pos_weight=.90, gamma=2.):
        """
        y_true (batch, d0, ..., dn, num_classes=2) : one-hot represented binary gt mask
        y_pred (batch, d0, ..., dn, num_classes=2) : logits from neural network
        """
        y_true_bg = y_true[..., 0]  # b, t, h, w, background: white
        y_pred_bg = y_pred[..., 0]  # b, t, h, w
        y_true_fg = y_true[..., 1]  # b, t, h, w, background: black
        y_pred_fg = y_pred[..., 1]  # b, t, h, w

        numer_fg = 2. * tf.reduce_sum(y_true_fg * y_pred_fg, axis=[-1, -2])  # b, t
        denom_fg = tf.reduce_sum(y_true_fg + y_pred_fg, axis=[-1, -2]) + smooth  # b, t
        numer_bg = 2. * tf.reduce_sum(y_true_bg * y_pred_bg, axis=[-1, -2])  # b, t
        denom_bg = tf.reduce_sum(y_true_bg + y_pred_bg, axis=[-1, -2]) + smooth  # b, t

        fg_loss = pos_weight * (1 - numer_fg / denom_fg)  # b, t
        bg_loss = (1. - pos_weight) * (1 - numer_bg / denom_bg)  # b, t

        if len(y_true_fg.shape) == 4:
            return tf.reduce_mean(fg_loss + bg_loss ** gamma)
        else:
            return fg_loss + bg_loss ** gamma


class WeightedCrossEntropyLoss(keras.losses.Loss):
    def call(self, y_true, y_pred, pos_weight=128):
        y_pred = tf.clip_by_value(y_pred, keras.backend.epsilon(), 1 - keras.backend.epsilon())
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)
        return tf.reduce_mean(loss)


def dcs(y_true, y_pred):
    # import pdb; pdb.set_trace()
    y_true = tf.argmax(y_true, axis=-1)  # n, t, h, w [5, 256, 256]
    y_pred = tf.argmax(y_pred, axis=-1)  # n, t, h, w y_pred[...,0]+y_pred[...,1]=1

    numer = tf.cast(2 * tf.reduce_sum(y_true * y_pred, axis=[-1, -2]), tf.float32)
    denom = tf.cast(1 + tf.reduce_sum(y_true + y_pred, axis=[-1, -2]), tf.float32)

    if len(y_true.shape) == 4:
        return tf.reduce_mean(numer / denom)
    else:
        return numer / denom


def iou(y_true, y_pred):
    y_true = tf.argmax(y_true, axis=-1)  # n, t, h, w
    y_pred = tf.argmax(y_pred, axis=-1)  # n, t, h, w

    inter = tf.cast(tf.math.count_nonzero(y_true * y_pred), tf.float32)
    union = tf.cast(tf.math.count_nonzero(y_true + y_pred) + 1, tf.float32)

    if len(y_true.shape) == 4:
        return tf.reduce_mean(inter / union)
    else:
        return inter / union


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))