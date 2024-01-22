import os, sys, logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
import argparse, json
import re, datetime
from scipy.stats import mode

sys.path.append('/workspace/bitbucket/MRA')
from data.setup_d import DataSettingV1, INFO_PATH
import models.model_e as model_ref
import models.metric as metric
from runs.cams import *
import tf_utils.tboard as tboard


parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh09')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp009')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,4')
main_config.add_argument('--val_name', type=str, dest='val_name', default='5')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model30')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,112,160,208')
main_config.add_argument('--serial', type=int, dest='serial', default=1)
main_config.add_argument('--image_size', type=int, dest='image_size', default=64)
main_config.add_argument('--radius', type=int, dest='radius', default=80)
main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
main_config.add_argument('--max_keep', type=int, dest='max_keep', default=5)  # only use training
main_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)  # only use validation
main_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
main_config.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.00005)
main_config.add_argument('--decay_steps', type=int, dest='decay_steps', default=5000)
main_config.add_argument('--decay_rate', type=int, dest='decay_rate', default=0.94)
main_config.add_argument('--batch_size', type=int, dest='batch_size', default=1)
main_config.add_argument('--epoch', type=int, dest='epoch', default=50)
main_config.add_argument('--seq_len', type=int, dest='seq_len', default=8)
main_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)
main_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=True)
main_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=False)
main_config.add_argument('--use_se', type=lambda x: x.title() in str(True), dest='use_se', default=True)
main_config.add_argument('--cls_alpha', type=float, dest='cls_alpha', default=0.05)
main_config.add_argument('--cls_gamma', type=float, dest='cls_gamma', default=0.0)

config, unparsed = parser.parse_known_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation


serial_str = '%03d' % config.serial

log_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'logs-%s' % serial_str)
result_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%s' % serial_str)
plot_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'plot-%s' % serial_str)
plot_val_path = os.path.join(plot_path, '_'.join([config.excel_name, config.val_name]))

if not os.path.exists(log_path): os.makedirs(log_path)
if not os.path.exists(result_path): os.makedirs(result_path)
if not os.path.exists(plot_path): os.makedirs(plot_path)


if 'snubh' in config.excel_name:
    data_type = 'clinical'
elif 'cusmh' in config.excel_name:
    data_type = 'external'
else:
    raise ValueError('Invalid data type')


img_size, img_c = config.image_size, config.channel_size
seq_len, seq_interval = config.seq_len, config.seq_interval
f_num = config.f_num

df = pd.read_excel(os.path.join(INFO_PATH, config.excel_name) + '.xlsx')

d_set = DataSettingV1(df=df, train_type=config.train_name, val_type=config.val_name, data_type=data_type,
                      train=config.train, seq_len=seq_len, seq_interval=seq_interval,
                      only_ste=config.only_ste, each_ste=config.each_ste, image_size=img_size, radius=config.radius)

if config.train:
    train_db = d_set.train.batch(config.batch_size)
    train_length = d_set.train.cardinality().numpy()
    print('train length: ', train_length)

val_db = d_set.val.batch(config.batch_size)
val_length = d_set.val.cardinality().numpy()
print('val length: ', val_length)

input_size = [seq_len, img_size, img_size, img_c]

infer_name = config.model_name
infer = getattr(model_ref, infer_name)(input_size=input_size, class_num=1, f_num=f_num,
                                       is_training=config.train, use_se=config.use_se)

model = infer.model

cls_loss_fn = metric.focal_loss_sigmoid
cls_alpha, cls_gamma = config.cls_alpha, config.cls_gamma


def training():
    info_log = {
        'EXCEL_NAME': config.excel_name,
        'MODEL_NAME': config.model_name,
        'SERIAL': config.serial,
        'F_NUM': config.f_num,
        'TRAIN_NAME': config.train_name,
        'VAL_NAME': config.val_name,
        'TRAIN_LENGTH': str(train_length),
        'VAL_LENGTH': str(val_length),
        'ONLY_STE': config.only_ste,
        'EACH_STE': config.each_ste,
        'IMAGE_SIZE': config.image_size,
        'RADIUS': config.radius,
        'SEQ_LENGTH': config.seq_len,
        'SEQ_INTERVAL': config.seq_interval,
        'BATCH_SIZE': config.batch_size,
        'LEARNING_RATE': config.learning_rate,
        'DECAY_STEPS': config.decay_steps,
        'DECAY_RATE': config.decay_rate,
        'EPOCH': config.epoch,
        'USE_SE': config.use_se,
        'H_CLS_ALPHA': config.cls_alpha,
        'H_CLS_GAMMA': config.cls_gamma,
    }

    with open(os.path.join(result_path, '.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    train_summary, val_summary = tboard.tensorboard_create(log_path)
    result_name = '_'.join([config.exp_name, config.model_name, serial_str]) + '.csv'
    result_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'AUC': pd.Series()})

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    perf_per_epoch, max_perf_per_epoch, max_current_step = [], [], []
    log_string = ''
    start_time = datetime.datetime.now()

    try:
        for epoch in range(1, config.epoch + 1):
            train_cx, train_cy, train_loss = [], [], []

            for train_step, (img, _, ste, _, name) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    cls_prob = model(img)
                    train_loss_batch = cls_loss_fn(ste, cls_prob, cls_alpha, cls_gamma)

                    grads = tape.gradient(train_loss_batch, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.append(train_loss_batch)

                train_cx.extend(np.reshape(cls_prob.numpy(), (-1,)).tolist())
                train_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())

                sys.stdout.write('Step: {0:>4d}, Loss: {1:.4f} ({2})\r'.format(train_step, train_loss_batch, epoch))

            train_loss_mean = np.mean(train_loss)
            train_auc = metric.calculate_auc(train_cy, train_cx)

            train_record = {'Loss': train_loss_mean, 'AUC': train_auc}

            val_cx, val_cy, val_loss = [], [], []
            val_steps = val_length // config.batch_size + 1

            for val_step, (img, _, ste, _, name) in enumerate(val_db):

                val_cls_prob = model(img)
                val_loss_batch = cls_loss_fn(ste, val_cls_prob, cls_alpha, cls_gamma)

                val_loss.append(val_loss_batch)

                val_cx.extend(np.reshape(val_cls_prob.numpy(), (-1,)).tolist())
                val_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())

                sys.stdout.write('Evaluation [{0}/{1}], Loss: {2:.4f}\r'.
                                 format(val_step + 1, val_steps, val_loss_batch))

            val_loss_mean = np.mean(val_loss)
            val_auc = metric.calculate_auc(val_cy, val_cx)

            val_record = {'Loss': val_loss_mean, 'AUC': val_auc}

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += ' Time:{0}'.format(time_elapsed.split('.')[0])

            print('Epoch:%s Train-Loss:%.4f AUC:%.3f Val-Loss:%.4f AUC:%.3f' %
                  (epoch, train_loss_mean, train_auc, val_loss_mean, val_auc) + log_string)

            tboard.board_record_value(train_summary, train_record, epoch)
            tboard.board_record_value(val_summary, val_record, epoch)

            log_string = ''

            perf_per_epoch.append(val_auc)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_auc)

                model.save(weight_path)
                result_csv.loc[epoch] = weight_path, val_auc

            elif val_auc > min(result_csv['AUC'].tolist()):
                os.remove(result_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                result_csv = result_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_auc)

                model.save(weight_path)
                result_csv.loc[epoch] = weight_path, val_auc

            result_csv.to_csv(os.path.join(result_path, result_name))

            if epoch == config.epoch:
                break

    except KeyboardInterrupt:
        print('Result saved')
        result_csv.to_csv(os.path.join(result_path, result_name))


def validation():
    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('AUC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][(config.num_weight - 1):config.num_weight])
    print('num_ckpt: ', len(all_ckpt_paths))

    ckpt_idx = 0
    val_cx, val_cy, val_loss, names = [], [], [], []

    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        for step, (img, mask, ste, det, name) in enumerate(val_db):
            with tf.GradientTape() as tape:
                val_prob = model(img)

            loss_batch = cls_loss_fn(ste, val_prob, cls_alpha, cls_gamma)

            val_loss.append(loss_batch)
            val_cx.extend(val_prob.numpy())
            val_cy.extend(ste.numpy())

            if ckpt_idx == 0:
                name_str = [x.decode() for x in name.numpy()]
                names.extend(name_str)

            sys.stdout.write('{0} Evaluation [{1}/{2}]\r'.
                             format(os.path.basename(ckpt), step, val_length // config.batch_size))

        ckpt_idx += 1

    val_auc = metric.calculate_auc(val_cy, val_cx)
    print('\nFinal AUC: %.3f' % val_auc)

    result_csv = pd.DataFrame({'NUMBER': names, 'STE': val_cy, 'PROB': val_cx})

    result_name = '_'.join([config.model_name, config.excel_name, config.val_name, serial_str,
                            '%03d' % config.num_weight]) + '.xlsx'

    result_csv.to_excel(os.path.join(result_path, result_name), index=False)


if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()

