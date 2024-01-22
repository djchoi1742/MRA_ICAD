import os, random, logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tf_keras_vis as vis

from tensorflow import keras
from tensorflow.keras import models
import sys
import datetime

sys.path.append('/workspace/bitbucket/MRA')

import data.setup_ref as D
import models.model_ref as M
import models.ref_metric as metrics

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation
# tf.config.experimental.set_visible_devices(gpu[0], 'GPU')  # set visible devices
sns.set()  # apply seaborn style

"""
- Set W, H to None to use arbitrary sized image
- Set T to 1 to delete 4th dimension (time-axis) from model
"""
MODEL = 'lstm_unet'  # wide_unet, lstm_unet, att_unet, r2_unet, fcn_rnn, lstm_att_unet, unet_3d
DATASET_CATEGORY = 'clinical_mra'  # 'clinical_mra', 'synthetic_mra', 'hepatic_vessel', 'lv_vessel'
WIDTH, HEIGHT, CHANNEL, CLASSES = 256, 256, 1, 2
T, S = 5, 3
IS_TRAINING, LEARNING_RATE, BATCH, EPOCH = False, 0.001, 2, 150
PROJECT_PATH = '/workspace/MRA'
TRIAL_PATH = 'MRA_Finetune3'
CKPT_PATH = 'MRA_Finetune3'
CKPT_SERIAL = 63
CKPT_SERIAL_STR = '%03d' % CKPT_SERIAL
CKPT_NAME = 'ckpt-' + CKPT_SERIAL_STR + '.hdf5'
# CKPT_NAME = 'ckpt-080.hdf5'


# Create folders for current trial
def create_workspace(project_path=PROJECT_PATH, trial_path=TRIAL_PATH,
                     folder_list=['checkpoint', 'tensorboard', 'plots']):
    if not os.path.exists(os.path.join(project_path, 'models', trial_path)):
        os.makedirs(os.path.join(project_path, 'models', trial_path))
        print('> Created workspace for %s' % trial_path)

    for folder in folder_list:
        if not os.path.exists(os.path.join(project_path, 'models', trial_path, folder)):
            os.makedirs(os.path.join(project_path, 'models', trial_path, folder))
            print('> Created %s folder for %s' % (folder.title(), trial_path))


def create_lr_scheduler(epoch, init_lr=0.001, decay=0.96):
    if epoch <= 20:
        return init_lr
    else:
        # return init_lr * decay ** (epoch - 20)
        # return init_lr * 1 / (1 + decay * (epoch - 20))
        return init_lr


def run_mra_trainer(dataset, model=MODEL, w=WIDTH, h=HEIGHT, t=T, c=CHANNEL, class_num=CLASSES,
                    is_training=IS_TRAINING, lr=LEARNING_RATE, b=BATCH, epoch=EPOCH,
                    val_dataset=None, project_path=PROJECT_PATH,
                    trial_path=TRIAL_PATH, trial_file=CKPT_NAME, verbose=True):

    def _verbose(v):
        if v:
            print('> Training / Test Information')
            print('\tInput Shapes - Width %d, Height %d, Time Sequence %d, Channel %d' % (w, h, t, c))
            print('\tDataset Object Shapes -', dataset)
            print('\tBatch Size -', b)
            print('\tLearning Rate -', lr)
            print('\tEpoch -', epoch)
            print('\tModel -', model)

    # plot first single image sequence
    def _plot_sample(seq_len):
        for elem in dataset:
            for i in range(seq_len):
                title_dict = {0: 'Original', 1: 'Label'}
                if seq_len == 1:
                    plots_dict = {0: elem[0][i], 1: np.argmax(elem[1][i], axis=-1)}
                else:
                    plots_dict = {0: elem[0][0][i], 1: np.argmax(elem[1][0][i], axis=-1)}
                whole_fig = plt.figure()
                for j in range(2):
                    sub_fig = whole_fig.add_subplot(1, 2, j + 1)
                    sub_fig.imshow(np.squeeze(plots_dict[j]))
                    sub_fig.set_title(title_dict[j])
                    sub_fig.axis('off')
                plt.show()
            break

    saver_path = os.path.join(project_path, 'models', trial_path)
    _verbose(verbose)
    _plot_sample(t)

    if is_training:
        # callbacks
        saver_filename = os.path.join(saver_path, 'checkpoint', 'ckpt-{epoch:03d}.hdf5')
        saver_callback = keras.callbacks.ModelCheckpoint(filepath=saver_filename, monitor='val_loss',
                                                         verbose=1, save_freq='epoch',
                                                         save_best_only=False, #  save_weights_only=True
                                                         )
        plot_callback = metrics.ShowPlotsCallback(dataset, b, t)
        lr_scheduler = keras.callbacks.LearningRateScheduler(create_lr_scheduler)

        # model
        if t == 1:
            neural_network = getattr(M, model)((h, w, c), class_num, [64, 128, 196, 256, 512],
                                               is_training=is_training)
        else:
            neural_network = getattr(M, model)((t, h, w, c), class_num, [64, 128, 196, 256, 512],
                                               is_training=is_training)

        # (optional) layer freezing for fine-tuning
        # for layer in neural_network.layers[:-10]:
        #     layer.trainable = False

        # training
        neural_network.summary()
        neural_network.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=[metrics.dcs],
                               loss=metrics.WeightedDiceScoreLoss())

        # (optional) resume training
        # neural_network.load_weights(os.path.join(saver_path, 'checkpoint', trial_file))
        neural_network.fit(dataset, epochs=epoch, initial_epoch=0, shuffle=True,
                           validation_data=val_dataset, verbose=1,
                           callbacks=[plot_callback, saver_callback, lr_scheduler])

    else:
        # callbacks
        plot_path = os.path.join(saver_path, 'plots-'+CKPT_SERIAL_STR)
        if not os.path.exists(plot_path): os.makedirs(plot_path)
        whole_callback = metrics.SavePlotsCallback(dataset, b, t, plot_path)
        # whole_callback = metrics.SavePlotsCallback(dataset, b, t, os.path.join(saver_path, 'plots'))

        # model
        if t == 1:
            neural_network = getattr(M, model)((h, w, c), class_num, [64, 128, 196, 256, 512],
                                               is_training=is_training)
        else:
            neural_network = getattr(M, model)((t, h, w, c), class_num, [64, 128, 196, 256, 512],
                                               is_training=is_training)

        # testing
        neural_network.summary()
        neural_network.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                               metrics=[metrics.dcs, metrics.iou], loss=metrics.WeightedDiceScoreLoss())
        neural_network.load_weights(os.path.join(saver_path, 'checkpoint', trial_file))
        # history = neural_network.evaluate(dataset)
        history = neural_network.evaluate(dataset, callbacks=[whole_callback])
        print('loss, dcs, iou :', history)

    return


if __name__ == '__main__':
    create_workspace()
    if IS_TRAINING:
        train_db = D.get_dataset('finetune', DATASET_CATEGORY, dataset_size=1000,
                                 seq_len=T, seq_interval=S).batch(BATCH)
        val_db = D.get_dataset('finetune_val', DATASET_CATEGORY, do_augmentation=False, dataset_size=1000,
                               seq_len=T, seq_interval=T).batch(BATCH)

        run_mra_trainer(dataset=train_db, val_dataset=val_db, is_training=IS_TRAINING)
    else:
        test_db = D.get_dataset('finetune_test', DATASET_CATEGORY, do_augmentation=False, dataset_size=1000,
                                seq_len=T, seq_interval=T).batch(BATCH)

        run_mra_trainer(dataset=test_db, is_training=IS_TRAINING)
    # run_mra_fine_tune(is_training=False, is_synthetic=False)