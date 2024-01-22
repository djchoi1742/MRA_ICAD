import os, sys, logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
import matplotlib.pyplot as plt
import seaborn as sns
import tf_keras_vis as vis
import sklearn.metrics
import pandas as pd
import argparse, time, json
import pptx, re, datetime
from pptx.util import Inches

sys.path.append('/workspace/bitbucket/MRA')
from data.setup_c import DataSettingV1, INFO_PATH
import models.model_c as model_ref
import models.ref_metric as metrics
import tf_utils.tboard as tboard
import tensorflow.keras.backend as K

parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh06')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp004')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,4')
main_config.add_argument('--val_name', type=str, dest='val_name', default='5')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model10')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,128,192,256')  # or 64,128,192,256,512
main_config.add_argument('--serial', type=int, dest='serial', default=1)
main_config.add_argument('--image_size', type=int, dest='image_size', default=256)
main_config.add_argument('--radius', type=int, dest='radius', default=80)
main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
main_config.add_argument('--num_classes', type=int, dest='num_classes', default=2)
main_config.add_argument('--max_keep', type=int, dest='max_keep', default=5)  # only use training
main_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)  # only use validation
main_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
main_config.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.00005)
main_config.add_argument('--decay_steps', type=int, dest='decay_steps', default=5000)
main_config.add_argument('--decay_rate', type=int, dest='decay_rate', default=0.94)
main_config.add_argument('--batch_size', type=int, dest='batch_size', default=2)
main_config.add_argument('--epoch', type=int, dest='epoch', default=40)
main_config.add_argument('--seq_len', type=int, dest='seq_len', default=5)
main_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)
main_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=False)
main_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=False)
main_config.add_argument('--one_hot', type=lambda x: x.title() in str(True), dest='one_hot', default=True)

config, unparsed = parser.parse_known_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation
sns.set()  # apply seaborn style

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
                      train=config.train, seq_len=config.seq_len, seq_interval=config.seq_interval,
                      only_ste=config.only_ste, each_ste=config.each_ste,
                      image_size=config.image_size, radius=config.radius, one_hot=config.one_hot)


if config.train:
    train_db = d_set.train.batch(config.batch_size)
    train_length = d_set.train.cardinality().numpy()
    print('train length: ', train_length)

val_db = d_set.val.batch(config.batch_size)
val_length = d_set.val.cardinality().numpy()
print('val length: ', val_length)


img_h, img_w, img_c = config.image_size, config.image_size, config.channel_size
class_num = 2

infer_name = config.model_name
model = getattr(model_ref, infer_name)((seq_len, img_h, img_w, img_c), class_num, f_num, is_training=config.train)


train_loss_fn = metrics.WeightedDiceScoreLoss()
val_loss_fn = metrics.WeightedDiceScoreLoss()

train_dsc_metric = metrics.dcs
val_dsc_metric = metrics.dcs

iou_metric = metrics.iou


def training():
    info_log = {
        'EXCEL_NAME': config.excel_name,
        'MODEL_NAME': config.model_name,
        'F_NUM': config.f_num,
        'TRAIN_NAME': config.train_name,
        'VAL_NAME': config.val_name,
        'TRAIN_LENGTH': str(train_length),
        'VAL_LENGTH': str(val_length),
        'ONLY_STE': config.only_ste,
        'IMAGE_SIZE': config.image_size,
        'RADIUS': config.radius,
        'SEQ_LENGTH': config.seq_len,
        'SEQ_INTERVAL': config.seq_interval,
        'BATCH_SIZE': config.batch_size,
        'LEARNING_RATE': config.learning_rate,
        'DECAY_STEPS': config.decay_steps,
        'DECAY_RATE': config.decay_rate,
        'EPOCH': config.epoch,
    }

    with open(os.path.join(result_path, '.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    # train_summary, val_summary = tboard.tensorboard_create(log_path)

    result_name = '_'.join([config.exp_name, config.model_name, serial_str]) + '.csv'
    dsc_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'DSC': pd.Series()})

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
            train_loss, train_x, train_y, train_dsc, train_iou = [], [], [], [], []
            for train_step, (img, lbl, _, _) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    logits = model(img)
                    train_loss_batch = train_loss_fn(lbl, logits)

                grads = tape.gradient(train_loss_batch, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.append(train_loss_batch)
                train_x.extend(logits.numpy())
                train_y.extend(lbl.numpy())

                train_dsc_batch = train_dsc_metric(lbl, logits)
                train_dsc.append(train_dsc_batch)

                train_iou_batch = iou_metric(lbl, logits)
                train_iou.append(train_iou_batch)

                sys.stdout.write('Step: {0:>4d}, Loss: {1:.4f} DSC: {2:.4f} IOU: {3:.4f} ({4})\r'.
                                 format(train_step, train_loss_batch, train_dsc_batch, train_iou_batch, epoch))

            train_loss_mean = np.mean(train_loss)
            train_dsc_mean = np.mean(train_dsc)

            val_loss, val_x, val_y, val_dsc, val_iou = [], [], [], [], []
            for val_step, (img, lbl, _, _) in enumerate(val_db):
                val_logits = model(img)
                val_loss_batch = val_loss_fn(lbl, val_logits)
                val_loss.append(val_loss_batch)
                val_x.extend(val_logits.numpy())
                val_y.extend(lbl.numpy())

                val_dsc_batch = val_dsc_metric(lbl, val_logits)
                val_dsc.append(val_dsc_batch)

                val_iou_batch = iou_metric(lbl, val_logits)
                val_iou.append(val_iou_batch)

                sys.stdout.write('Evaluation [{0}/{1}],  Loss: {2:.4f} DSC: {3:.4f} IOU: {4:.4f}\r'.
                                 format(val_step + 1, val_length // config.batch_size + 1,
                                        val_loss_batch, val_dsc_batch, val_iou_batch))

            val_loss_mean = np.mean(val_loss)
            val_dsc_mean = np.mean(val_dsc)

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += 'Time Elapsed: {0}'.format(time_elapsed.split('.')[0])

            print('Epoch: %s Train-Loss: %.4f Train-DSC %.4f Val-Loss %.4f Val-DSC %.4f ' %
                  (epoch, train_loss_mean, train_dsc_mean, val_loss_mean, val_dsc_mean) + log_string)

            log_string = ''
            perf_per_epoch.append(val_dsc_mean)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_dsc_mean)

                model.save(weight_path)
                dsc_csv.loc[epoch] = weight_path, val_dsc_mean

            elif val_dsc_mean > min(dsc_csv['DSC'].tolist()):
                os.remove(dsc_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                dsc_csv = dsc_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_dsc_mean)

                model.save(weight_path)
                dsc_csv.loc[epoch] = weight_path, val_dsc_mean

            dsc_csv.to_csv(os.path.join(result_path, result_name))

            if epoch == config.epoch:
                break

    except KeyboardInterrupt:
        print('Result saved')
        dsc_csv.to_csv(os.path.join(result_path, result_name))


def validation():
    if not os.path.exists(plot_val_path): os.makedirs(plot_val_path)

    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name,
                                   'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('DSC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])

    val_dsc = np.zeros([len(all_ckpt_paths), val_length * seq_len])
    val_iou = np.zeros([len(all_ckpt_paths), val_length * seq_len])
    val_pixel = np.zeros([len(all_ckpt_paths), val_length * seq_len])
    val_name = []

    print('num_ckpt: ', len(all_ckpt_paths))

    ckpt_idx = 0
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        val_loss, val_name = [], []
        for step, (img, mask, _, name) in enumerate(val_db):
            val_logits = model(img)
            val_loss_batch = val_loss_fn(mask, val_logits)
            val_loss.append(val_loss_batch)

            val_name_batch, val_dsc_batch, val_iou_batch, val_pixel_batch = \
                show_slice_cam(img.numpy(), mask.numpy(), val_logits.numpy(), name.numpy(), is_png=False,
                               save_path=plot_val_path)

            if ckpt_idx == 0:
                val_name.extend(val_name_batch)

            cnt_range = config.batch_size * seq_len

            val_dsc[ckpt_idx, step * cnt_range:step * cnt_range + len(val_dsc_batch)] = val_dsc_batch
            val_iou[ckpt_idx, step * cnt_range:step * cnt_range + len(val_iou_batch)] = val_iou_batch
            val_pixel[ckpt_idx, step * cnt_range:step * cnt_range + len(val_pixel_batch)] = val_pixel_batch

            sys.stdout.write('{0} Evaluation [{1}/{2}], DSC:{3:.4f}, IOU:{4:.4f}\r'.
                             format(os.path.basename(ckpt), step + 1, val_length // config.batch_size + 1,
                                    np.mean(val_dsc_batch), np.mean(val_iou_batch)))

        ckpt_idx += 1

    val_dsc, val_iou = np.mean(val_dsc, axis=0), np.mean(val_iou, axis=0)
    val_pixel = np.mean(val_pixel, axis=0)

    result_csv = pd.DataFrame({'NUMBER': val_name, 'PIXEL': val_pixel, 'DCS': val_dsc, 'IOU': val_iou})
    result_name = '_'.join([config.model_name, config.excel_name,
                            config.val_name, serial_str, '%03d' % config.num_weight]) + '.csv'
    result_csv.to_csv(os.path.join(result_path, result_name), index=False)

    print('\nFinal DSC: %.3f, IOU: %.3f' % (np.mean(val_dsc), np.mean(val_iou)))


def show_slice_cam(images, masks, pred_masks, name, is_png=True, num_rows=1, num_cols=3,
                   fig_size=(3 * 3, 1 * 3), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size, seq_len = pred_masks.shape[0:2]
    names, dscs, ious, ste_probs = [], [], [], []
    pixels = []
    for i in range(batch_size):
        dsc_slices = val_dsc_metric(masks[i, :, :, :], pred_masks[i, :, :, :]).numpy()
        png_name = name[i].decode()
        name_split = re.split('_', png_name)

        if len(name_split) <= 3:
            patient_id, start_idx = name_split[0:2]
        else:
            patient_id, start_idx = '_'.join([name_split[0], name_split[1]]), name_split[2]

        # patient_id, start_idx = name_split[0:2]
        mask_pixels = np.sum(masks[i, :, :, :, 1], axis=(-1, -2))

        iou_slices, ste_prob_slices = [], []
        for j in range(seq_len):
            fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            show_image = images[i, j, :, :, 0]
            show_mask = masks[i, j, :, :, 1]
            show_pred_mask = pred_masks[i, j, :, :, 1]

            # each_dsc = model_ref.each_dcs(masks[i, j, :, :, :], pred_masks[i, j, :, :, :]).numpy()
            each_dsc = dsc_slices[j]
            each_iou = iou_metric(masks[i, j, :, :, :], pred_masks[i, j, :, :, :]).numpy()

            # dsc_slices.append(each_dsc)
            iou_slices.append(each_iou)

            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])
            names.append(each_name)

            if is_png:
                ax[0].imshow(show_image, cmap='bone')
                ax[0].set_title(each_name, fontsize=7, color='black')
                ax[1].imshow(show_mask, cmap='bone')
                ax[1].set_title('Mask: ' + each_name, fontsize=7, color='navy')

                ax[2].imshow(show_pred_mask, cmap='bone')
                ax[2].set_title('DSC: %.3f, IOU: %.3f' % (each_dsc, each_iou), fontsize=7, color='blue')

                fig_name = os.path.join(save_path, each_name)
                plt.savefig(fig_name, bbox_inches='tight')

            plt.clf()
        dscs.extend(dsc_slices)
        ious.extend(iou_slices)
        pixels.extend(mask_pixels)
        ste_probs.extend(ste_prob_slices)

    return names, np.array(dscs), np.array(ious), np.array(pixels)


def show_seq_cam(images, labels, masks, name, is_png=True, num_rows=3, num_cols=5, fig_size=(5 * 2, 3 * 2)):
    batch_size, seq_len = masks.shape[0:2]
    names, dscs, ious = [], [], []

    for i in range(batch_size):
        fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
        axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
        axoff_fun(ax)

        dsc_slices = val_dsc_metric(labels[i, :, :, :], masks[i, :, :, :]).numpy()

        png_name = name[i].decode()
        name_split = re.split('_', png_name)
        patient_id, start_idx = name_split[0:2]
        iou_slices = []
        for j in range(seq_len):

            show_image = images[i, j, :, :, 0]
            show_label = labels[i, j, :, :, 1]
            show_logit = masks[i, j, :, :, 1]

            each_dsc = dsc_slices[j]
            each_iou = iou_metric(labels[i, j, :, :, :], masks[i, j, :, :, :]).numpy()
            iou_slices.append(each_iou)

            # each_name = png_name + '_%d' % j
            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])
            names.append(each_name)

            if is_png:
                ax[0, j].imshow(show_image, cmap='bone')
                ax[0, j].set_title(each_name, fontsize=7, color='black')
                ax[1, j].imshow(show_label, cmap='bone')
                ax[1, j].set_title('Mask', fontsize=7, color='green')
                ax[2, j].imshow(show_logit, cmap='bone')
                ax[2, j].set_title('DSC: %.3f,  IOU: %.3f' % (each_dsc, each_iou), fontsize=7, color='blue')

        dscs.extend(dsc_slices.tolist())
        ious.extend(iou_slices)

        if is_png:
            fig_name = os.path.join(plot_val_path, png_name)
            plt.savefig(fig_name, bbox_inches='tight')

        plt.clf()

    return names, np.array(dscs), np.array(ious)


if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()










