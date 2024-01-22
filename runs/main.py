import os, random, logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import tf_keras_vis as vis

from tensorflow import keras
from tensorflow.keras import models
import sys

sys.path.append('/workspace/bitbucket/MRA')

import data.setup_ref as setup
import models.model_ref as model
import models.ref_metric as metrics
import argparse

parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, \
                                 description='', epilog=license, add_help=False)
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp001')
main_config.add_argument('--model_name', type=str, dest='model_name', default='lstm_unet')  # MODEL
main_config.add_argument('--data_name', type=str, dest='data_name', default='clinical_mra')  # DATASET_CATE
main_config.add_argument('--trial_serial', type=int, dest='trial_serial', default=1)
main_config.add_argument('--image_size', type=int, dest='image_size', default=256)  # WIDTH, HEIGHT
main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)  # CHANNEL
main_config.add_argument('--num_classes', type=int, dest='num_classes', default=2)  # CLASSES
main_config.add_argument('--seq_length', type=int, dest='seq_length', default=5)  # T (seq_len)
main_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)  # S (seq_interval)
main_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
main_config.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.001)  # LEARNING_RATE
main_config.add_argument('--batch_size', type=int, dest='batch_size', default=2)  # BATCH
main_config.add_argument('--epoch', type=int, dest='epoch', default=150)  # EPOCH
main_config.add_argument('--val_idx', type=int, dest='val_idx', default=2)

# parser.print_help()
config, unparsed = parser.parse_known_args()


logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation
sns.set()  # apply seaborn style

"""
- Set W, H to None to use arbitrary sized image
- Set T to 1 to delete 4th dimension (time-axis) from model
"""
# MODEL = 'lstm_unet'  # wide_unet, lstm_unet, att_unet, r2_unet, fcn_rnn, lstm_att_unet, unet_3d
# DATASET_CATEGORY = 'clinical_mra'  # 'clinical_mra', 'synthetic_mra', 'hepatic_vessel', 'lv_vessel'
# WIDTH, HEIGHT, CHANNEL, CLASSES = 256, 256, 1, 2
# T, S = 5, 3

weight_name = 'ckpt-'+'%03d' % config.val_idx+'.hdf5'

trial_serial_str = '%03d' % config.trial_serial

log_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'logs-%s' % trial_serial_str)
plot_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'plot-%s' % trial_serial_str)

result_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%s' % trial_serial_str)

if not os.path.exists(log_path): os.makedirs(log_path)
if not os.path.exists(plot_path): os.makedirs(plot_path)
if not os.path.exists(result_path): os.makedirs(result_path)


def create_lr_scheduler(epoch, init_lr=0.001, decay=0.96):
    if epoch <= 20:
        return init_lr
    else:
        # return init_lr * decay ** (epoch - 20)
        # return init_lr * 1 / (1 + decay * (epoch - 20))
        return init_lr


def run_mra_trainer(data_set, model_name=config.model_name,
                    w=config.image_size, h=config.image_size, t=config.seq_length,
                    c=config.channel_size, class_num=config.num_classes,
                    is_training=config.train, lr=config.learning_rate,
                    b=config.batch_size, epoch=config.epoch,
                    val_data_set=None, val_weight=weight_name, verbose=True):

    def _verbose(v):
        if v:
            print('> Training / Test Information')
            print('\tInput Shapes - Width %d, Height %d, Time Sequence %d, Channel %d' % (w, h, t, c))
            print('\tDataSet Object Shapes -', data_set)
            print('\tBatch Size -', b)
            print('\tLearning Rate -', lr)
            print('\tEpoch -', epoch)
            print('\tModel -', model)

    # plot first single image sequence
    def _plot_sample(seq_len):
        for elem in data_set:
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

    # saver_path = os.path.join(project_path, 'models', trial_path)
    # _verbose(verbose)
    # _plot_sample(t)

    if is_training:
        # callbacks
        saver_filename = os.path.join(log_path, 'ckpt-{epoch:03d}.ckpt')
        print(saver_filename)
        saver_callback = keras.callbacks.ModelCheckpoint(filepath=saver_filename, monitor='val_loss',
                                                         verbose=1, save_freq='epoch',
                                                         save_best_only=False  # save_weights_only=True
                                                         )
        plot_callback = metrics.ShowPlotsCallback(data_set, b, t)
        lr_scheduler = keras.callbacks.LearningRateScheduler(create_lr_scheduler)

        if t == 1:
            neural_network = getattr(model, model_name)((h, w, c), class_num, [64, 128, 196, 256, 512],
                                                        is_training=is_training)
        else:
            neural_network = getattr(model, model_name)((t, h, w, c), class_num, [64, 128, 196, 256, 512],
                                                        is_training=is_training)

        neural_network.summary()
        neural_network.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=[metrics.dcs],
                               loss=metrics.WeightedDiceScoreLoss())

        neural_network.fit(data_set, epochs=epoch, initial_epoch=0, shuffle=True,
                           validation_data=val_data_set, verbose=1,
                           callbacks=[plot_callback, saver_callback, lr_scheduler])

        # ex_save = os.path.join(log_path, 'ex_ckpt.hdf5')

        # neural_network.save(ex_save)
        # if not os.path.exists(ex_save):
        #     print('no saved!')
    else:
        whole_callback = metrics.SavePlotsCallback(data_set, b, t, plot_path)

        if t == 1:
            neural_network = getattr(model, model_name)((h, w, c), class_num, [64, 128, 196, 256, 512],
                                                        is_training=is_training)
        else:
            neural_network = getattr(model, model_name)((t, h, w, c), class_num, [64, 128, 196, 256, 512],
                                                        is_training=is_training)

        # testing
        neural_network.summary()
        neural_network.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                               metrics=[metrics.dcs, metrics.iou], loss=metrics.WeightedDiceScoreLoss())
        neural_network.load_weights(os.path.join(log_path, val_weight))
        # history = neural_network.evaluate(dataset)
        history = neural_network.evaluate(data_set, callbacks=[whole_callback])
        print('loss, dcs, iou :', history)

    return


if __name__ == '__main__':
    if config.train:
        train_db = setup.get_dataset('finetune', config.data_name, dataset_size=100,
                                     seq_len=config.seq_length,
                                     seq_interval=config.seq_interval).batch(config.batch_size)
        val_db = setup.get_dataset('finetune_val', config.data_name, do_augmentation=False, dataset_size=100,
                                   seq_len=config.seq_length,
                                   seq_interval=config.seq_length).batch(config.batch_size)

        run_mra_trainer(data_set=train_db, val_data_set=val_db, is_training=config.train)
    else:
        test_db = setup.get_dataset('finetune_test', config.data_name, do_augmentation=False, dataset_size=100,
                                seq_len=config.seq_length,
                                seq_interval=config.seq_length).batch(config.batch_size)

        run_mra_trainer(data_set=test_db, is_training=config.train)
    # run_mra_fine_tune(is_training=False, is_synthetic=False)