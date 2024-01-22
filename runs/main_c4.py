import os, sys, logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
import argparse, json
import datetime

sys.path.append('/workspace/bitbucket/MRA')
from data.setup_c import DataSettingV2, INFO_PATH, py_nii_to_sequence, nii_dataset
import models.model_c as model_ref
import models.metric as metric
from runs.cams import *
import tf_utils.tboard as tboard


parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh06')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp006')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,4')
main_config.add_argument('--val_name', type=str, dest='val_name', default='5')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model15')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,128,192,256')
main_config.add_argument('--serial', type=int, dest='serial', default=1)
main_config.add_argument('--image_size', type=int, dest='image_size', default=256)
main_config.add_argument('--radius', type=int, dest='radius', default=80)
main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
main_config.add_argument('--max_keep', type=int, dest='max_keep', default=5)  # only use training
main_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)  # only use validation
main_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
main_config.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.00005)
main_config.add_argument('--decay_steps', type=int, dest='decay_steps', default=5000)
main_config.add_argument('--decay_rate', type=int, dest='decay_rate', default=0.94)
main_config.add_argument('--batch_size', type=int, dest='batch_size', default=2)
main_config.add_argument('--epoch', type=int, dest='epoch', default=50)
main_config.add_argument('--seq_len', type=int, dest='seq_len', default=5)
main_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)
main_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=False)
main_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=False)
main_config.add_argument('--one_hot', type=lambda x: x.title() in str(True), dest='one_hot', default=False)
main_config.add_argument('--is_png', type=lambda x: x.title() in str(True), dest='is_png', default=False)
main_config.add_argument('--add_conv', type=lambda x: x.title() in str(True), dest='add_conv', default=False)
main_config.add_argument('--use_ic', type=lambda x: x.title() in str(True), dest='use_ic', default=False)
main_config.add_argument('--use_se', type=lambda x: x.title() in str(True), dest='use_se', default=False)

config, unparsed = parser.parse_known_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

import inspect
exec_file = os.path.basename(inspect.getfile(inspect.currentframe()))

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

d_set = DataSettingV2(df=df, train_type=config.train_name, val_type=config.val_name, data_type=data_type,
                      train=config.train, seq_len=seq_len, seq_interval=seq_interval,
                      only_ste=config.only_ste, each_ste=config.each_ste,
                      image_size=img_size, radius=config.radius, one_hot=config.one_hot)

input_size = [seq_len, img_size, img_size, img_c] if seq_len != 1 else [img_size, img_size, img_c]
infer_name = config.model_name

seg_loss_fn = metric.weighted_dice_score_loss
cls_loss_fn = metric.focal_loss_sigmoid

dcs_metric = metric.dcs_2d if config.seq_len == 1 else metric.dcs_3d
iou_metric = metric.iou_2d if config.seq_len == 1 else metric.iou_3d


gpu = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)
tf.config.experimental.set_memory_growth(gpu[1], True)

strategy = tf.distribute.MirroredStrategy()
print('# of Devices: {}'.format(strategy.num_replicas_in_sync))


def setup_dataset(raws, masks, indexes, stes, global_batch_size, radius=80, one_hot=False, augment=False, drop=False):
    dataset = tf.data.Dataset.from_tensor_slices((raws, masks, indexes, stes))

    if augment:
        dataset = dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True)
    dataset = dataset.map(lambda x, y, z, w: py_nii_to_sequence(x, y, z, w, seq_len, config.image_size, radius))

    dataset = dataset.map(lambda x, y, z, w: nii_dataset(x, y, z, w, one_hot, augment))
    length = dataset.cardinality()

    dataset = dataset.batch(global_batch_size, drop_remainder=drop)
    dataset = strategy.experimental_distribute_dataset(dataset)

    return dataset, length.numpy()


@tf.function
def compute_loss(mask_true, mask_prob, train=False):
    seg_loss = seg_loss_fn(mask_true, mask_prob)

    if train:
        seg_loss = tf.expand_dims(seg_loss, axis=-1)
        seg_loss = tf.nn.compute_average_loss(seg_loss, global_batch_size=config.batch_size)

    return seg_loss


@tf.function
def val_each_step(data_inputs, model):
    img, msk, st, name = data_inputs
    seg_output = model(img)
    seg_loss = compute_loss(msk, seg_output, train=False)

    return seg_loss, seg_output


@tf.function
def distributed_val_step(data_inputs, model):
    seg_loss, seg_pred = strategy.run(val_each_step, args=(data_inputs, model))
    return seg_loss, seg_pred


def training():
    @tf.function
    def train_each_step(data_inputs):
        img, msk, _, _ = data_inputs

        with tf.GradientTape() as tape:
            seg_output = model(img)
            seg_loss = compute_loss(msk, seg_output, train=True)

        grads = tape.gradient(seg_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return seg_loss, seg_output

    @tf.function
    def distributed_train_step(data_inputs):
        seg_loss, seg_pred = strategy.run(train_each_step, args=(data_inputs,))
        seg_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, seg_loss, axis=None)

        return seg_loss, seg_pred

    train_raws, train_masks, train_indexes, train_stes = d_set.train_input
    val_raws, val_masks, val_indexes, val_stes = d_set.val_input

    per_batch_size = 1
    size = per_batch_size * strategy.num_replicas_in_sync

    result_name = '_'.join([config.exp_name, config.model_name, serial_str]) + '.csv'
    dcs_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'DCS': pd.Series(), 'IOU': pd.Series()})

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=config.decay_steps,
        decay_rate=config.decay_rate,
        staircase=True)

    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

    with strategy.scope():
        train_db, train_length = setup_dataset(train_raws, train_masks, train_indexes, train_stes, size,
                                               augment=True, drop=True)
        val_db, val_length = setup_dataset(val_raws, val_masks, val_indexes, val_stes, size,
                                           augment=False, drop=False)

        info_log = {
            'EXEC_FILE': exec_file,
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
            'ADD_CONV': config.add_conv,
            'USE_IC': config.use_ic,
            'USE_SE': config.use_se
        }

        with open(os.path.join(result_path, '.info'), 'w') as f:
            f.write(json.dumps(info_log, indent=4, sort_keys=True))
            f.close()

        train_summary, val_summary = tboard.tensorboard_create(log_path)

        infer = getattr(model_ref, infer_name)(input_size=input_size, class_num=1, f_num=f_num,
                                               is_training=config.train, add_conv=config.add_conv,
                                               use_ic=config.use_ic, use_se=config.use_se)

        model = infer.model

        perf_per_epoch, max_perf_per_epoch, max_current_step = [], [], []
        log_string = ''
        start_time = datetime.datetime.now()

    try:
        for epoch in range(1, config.epoch + 1):
            train_seg_loss, train_dcs, train_iou = [], [], []

            step = 0
            for x in train_db:
                _, mask, _, _ = x
                seg_loss_batch, seg_prob = distributed_train_step(x)

                mask = tf.concat(mask.values, axis=0)
                seg_probs = tf.concat(seg_prob.values, axis=0)
                train_seg_loss.append(seg_loss_batch.numpy())

                train_dcs_batch = dcs_metric(mask, seg_probs)
                train_dcs.extend(train_dcs_batch.numpy())
                dcs_batch_mean = np.mean(train_dcs_batch)

                train_iou_batch = iou_metric(mask, seg_probs)
                train_iou.extend(train_iou_batch.numpy())
                iou_batch_mean = np.mean(train_iou_batch)

                sys.stdout.write('Step: {0:>4d}, Loss: {1:.4f} DCS: {2:.4f} IOU: {3:.4f} ({4})\r'.
                                 format(step, seg_loss_batch, dcs_batch_mean, iou_batch_mean, epoch))

                step += 1

            train_seg_loss_mean = np.mean(train_seg_loss)
            train_dcs_mean, train_iou_mean = np.mean(train_dcs), np.mean(train_iou)

            train_record = {'Loss': train_seg_loss_mean, 'DCS': train_dcs_mean, 'IOU': train_iou_mean}

            val_loss, val_dcs, val_iou = [], [], []
            val_steps = val_length // config.batch_size + 1

            val_step = 0
            for x in val_db:

                if False:
                    _, mask, ste, _ = x
                    batch_unit = mask.shape[0]
                    val_loss_batch, seg_prob = distributed_val_step(x, model)

                    val_loss_batch = tf.concat(val_loss_batch.values, axis=0)
                    val_loss_batch = val_loss_batch[tf.math.is_nan(val_loss_batch) == False]
                    mask = tf.concat(mask.values, axis=0)
                    seg_probs = tf.concat(seg_prob.values, axis=0)

                    val_loss.append(val_loss_batch)

                    val_dcs_batch = dcs_metric(mask, seg_probs)
                    val_dcs.extend(val_dcs_batch.numpy())
                    # val_dcs_batch = val_dcs_batch.numpy().reshape(batch_unit)
                    dcs_batch_mean = np.mean(val_dcs_batch)

                    val_iou_batch = iou_metric(mask, seg_probs)
                    val_iou.extend(val_iou_batch.numpy())
                    # val_iou_batch = val_iou_batch.numpy().reshape(batch_unit)
                    iou_batch_mean = np.mean(val_iou_batch)

                _, mask, ste, _ = x
                val_loss_batch, seg_prob = distributed_val_step(x, model)

                if mask.values[1].shape[0] != 0:
                    val_loss_batch = tf.reduce_mean(val_loss_batch.values).numpy()
                    mask = tf.concat(mask.values, axis=0)
                    seg_probs = tf.concat(seg_prob.values, axis=0)

                else:
                    val_loss_batch = val_loss_batch.values[0].numpy()
                    mask = mask.values[0]
                    seg_probs = seg_prob.values[0]

                val_loss.append(val_loss_batch)

                val_dcs_batch = dcs_metric(mask, seg_probs)
                val_dcs.extend(val_dcs_batch.numpy())
                dcs_batch_mean = np.mean(val_dcs_batch)

                val_iou_batch = iou_metric(mask, seg_probs)
                val_iou.extend(val_iou_batch.numpy())
                iou_batch_mean = np.mean(val_iou_batch)

                sys.stdout.write('Evaluation [{0}/{1}], Loss: {2:.4f} DCS: {3:.4f} IOU: {4:.4f}\r'.
                                 format(val_step + 1, val_steps, val_loss_batch, dcs_batch_mean, iou_batch_mean))

                val_step += 1

            val_loss_mean = np.mean(val_loss)
            val_dcs_mean, val_iou_mean = np.mean(val_dcs), np.mean(val_iou)

            val_record = {'Loss': val_loss_mean, 'DCS': val_dcs_mean, 'IOU': val_iou_mean}

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += ' Time Elapsed: {0}'.format(time_elapsed.split('.')[0])

            print('Epoch:%s Train-Loss:%.4f DCS:%.4f IOU:%.4f Val-Loss:%.4f DCS:%.4f IOU:%.4f' %
                  (epoch, train_seg_loss_mean, train_dcs_mean, train_iou_mean,
                   val_loss_mean, val_dcs_mean, val_iou_mean) + log_string)

            tboard.board_record_value(train_summary, train_record, epoch)
            tboard.board_record_value(val_summary, val_record, epoch)

            log_string = ''
            val_metric = val_dcs_mean
            perf_per_epoch.append(val_metric)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_dcs_mean, val_iou_mean

            elif val_metric > min(dcs_csv['DCS'].tolist()):
                os.remove(dcs_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                dcs_csv = dcs_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_dcs_mean, val_iou_mean

            dcs_csv.to_csv(os.path.join(result_path, result_name))

            if epoch == config.epoch:
                break

    except KeyboardInterrupt:
        print('Result saved')
        dcs_csv.to_csv(os.path.join(result_path, result_name))


def validation():
    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('DCS', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])

    per_batch_size = 1
    size = per_batch_size * strategy.num_replicas_in_sync

    with strategy.scope():
        val_raws, val_masks, val_indexes, val_stes = d_set.val_input
        val_db, val_length = setup_dataset(val_raws, val_masks, val_indexes, val_stes, size, augment=False)

        infer = getattr(model_ref, infer_name)(input_size=input_size, class_num=1, f_num=f_num,
                                               is_training=config.train, add_conv=config.add_conv,
                                               use_ic=config.use_ic, use_se=config.use_se)

        model = infer.model

    print('num_ckpt: ', len(all_ckpt_paths))

    val_loss = np.zeros([len(all_ckpt_paths), val_length])
    val_dcs = np.zeros([len(all_ckpt_paths), val_length])
    val_iou = np.zeros([len(all_ckpt_paths), val_length])
    val_name = []

    ckpt_idx = 0
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        step = 0
        for x in val_db:
            img, mask, ste, name = x

            val_loss_batch, seg_prob = distributed_val_step(x, model)

            val_loss_batch = tf.concat(val_loss_batch.values, axis=0)
            val_loss_batch = val_loss_batch[tf.math.is_nan(val_loss_batch) == False]

            mask = tf.concat(mask.values, axis=0)
            seg_probs = tf.concat(seg_prob.values, axis=0)
            batch_unit = mask.shape[0]

            val_dcs_batch = dcs_metric(mask, seg_probs)
            val_dcs_batch = val_dcs_batch.numpy().reshape(batch_unit)

            val_iou_batch = iou_metric(mask, seg_probs)
            val_iou_batch = val_iou_batch.numpy().reshape(batch_unit)

            if ckpt_idx == 0:
                name = tf.concat(name.values, axis=0)
                val_name.extend(name)

            cnt_range = config.batch_size

            val_loss[ckpt_idx, step * cnt_range:step * cnt_range + batch_unit] = val_loss_batch
            val_dcs[ckpt_idx, step * cnt_range:step * cnt_range + batch_unit] = val_dcs_batch
            val_iou[ckpt_idx, step * cnt_range:step * cnt_range + batch_unit] = val_iou_batch

            sys.stdout.write('{0} Evaluation [{1}/{2}], DCS:{3:.4f}, IOU:{4:.4f}\r'.
                             format(os.path.basename(ckpt), step, val_length // config.batch_size,
                                    np.mean(val_dcs_batch), np.mean(val_iou_batch)))

            step += 1

        ckpt_idx += 1

    val_loss = np.mean(val_loss, axis=0)
    val_dcs, val_iou = np.mean(val_dcs, axis=0), np.mean(val_iou, axis=0)
    val_name = [x.numpy().decode() for x in val_name]

    print('\nFinal Loss: %.3f DCS: %.3f, IOU: %.3f' % (np.mean(val_loss), np.mean(val_dcs), np.mean(val_iou)))

    result_csv = pd.DataFrame({'NUMBER': val_name, 'LOSS': val_loss, 'DCS': val_dcs, 'IOU': val_iou})

    result_name = '_'.join([config.model_name, config.excel_name, config.val_name, serial_str,
                            '%03d' % config.num_weight]) + '.csv'
    result_csv.to_csv(os.path.join(result_path, result_name), index=False)


def show_slice_cam_3d(images, masks, pred_masks, stes, pred_stes, cam_stes, ggc_stes, name, is_png=True,
                      num_rows=1, num_cols=5, fig_size=(5 * 2, 1 * 2), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size, seq_len = pred_masks.shape[0:2]
    names, dcss, ious, ste_probs = [], [], [], []

    for i in range(batch_size):
        patient_id, start_idx = metric.name_idx(name[i])
        dcs_slices, iou_slices, ste_prob_slices = [], [], []

        each_dcs = dcs_metric(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()
        each_iou = iou_metric(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()
        show_ste, show_pred_ste = stes[i], pred_stes[i]

        dcs_slices.append(each_dcs)
        iou_slices.append(each_iou)
        ste_prob_slices.append(show_pred_ste)

        for j in range(seq_len):
            show_image = images[i, j, :, :, 0]
            show_mask = masks[i, j, :, :, 0]
            show_pred_mask = pred_masks[i, j, :, :, 0]
            show_cam_ste = cam_stes[i, j, :, :, 0]  # grad_cam
            show_ggc_ste = ggc_stes[i, j, :, :, 0]

            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])
            names.append(each_name)

            if is_png:
                fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                ax[0].imshow(show_image, cmap='bone')
                ax[0].set_title(each_name, fontsize=7, color='black')
                ax[1].imshow(show_mask, cmap='bone')
                ax[1].set_title('Mask: ' + each_name, fontsize=7, color='navy')

                ax[2].imshow(show_pred_mask, cmap='bone')
                ax[2].set_title('DCS: %.3f, IOU: %.3f' % (each_dcs, each_iou), fontsize=7, color='blue')
                ax[3].imshow(show_image, cmap='bone')
                ax[3].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[3].set_title('Stenosis: %d Prob: %.3f' % (show_ste, show_pred_ste), fontsize=7, color='green')

                ax[4].imshow(show_ggc_ste, cmap='bone')
                ax[4].set_title('Guided Grad-CAM', fontsize=7, color='darkgoldenrod')

                fig_name = os.path.join(save_path, each_name)
                plt.savefig(fig_name, bbox_inches='tight')

                plt.clf()

        dcss.extend(dcs_slices)
        ious.extend(iou_slices)
        ste_probs.extend(ste_prob_slices)

    return names, dcss, ious, ste_probs  # np.squeeze(np.array(ste_probs))


def show_slice_cam_2d(images, masks, pred_masks, stes, pred_stes, cam_stes, ggc_stes,
                      name, is_png=True, num_rows=1, num_cols=5, fig_size=(5 * 2, 1 * 2), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = pred_masks.shape[0]

    names, dcss, ious, ste_probs = [], [], [], []
    for i in range(batch_size):
        show_image = images[i, :, :, 0]
        show_mask = masks[i, :, :, 0]
        show_pred_mask = pred_masks[i, :, :, 0]
        show_cam_ste = cam_stes[i, :, :, 0]
        show_ggc_ste = ggc_stes[i, :, :, 0]
        show_ste, show_pred_ste = stes[i], pred_stes[i]

        each_dcs = dcs_metric(masks[i, :, :, :], pred_masks[i, :, :, :]).numpy()
        each_iou = iou_metric(masks[i, :, :, :], pred_masks[i, :, :, :]).numpy()
        each_name = name[i].decode()

        dcss.append(each_dcs)
        ious.append(each_iou)
        names.append(each_name)
        ste_probs.append(show_pred_ste)

        if is_png:
            fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            ax[0].imshow(show_image, cmap='bone')
            ax[0].set_title(each_name, fontsize=7, color='black')
            ax[1].imshow(show_mask, cmap='bone')
            ax[1].set_title('Mask: ' + each_name, fontsize=7, color='navy')

            ax[2].imshow(show_pred_mask, cmap='bone')
            ax[2].set_title('DCS: %.3f, IOU: %.3f' % (each_dcs, each_iou), fontsize=7, color='blue')
            ax[3].imshow(show_image, cmap='bone')
            ax[3].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
            ax[3].set_title('Stenosis: %d Prob: %.3f' % (show_ste, show_pred_ste), fontsize=7, color='green')

            ax[4].imshow(show_ggc_ste, cmap='bone')
            ax[4].set_title('Guided Grad-CAM', fontsize=7, color='darkgoldenrod')

            fig_name = os.path.join(save_path, each_name)
            plt.savefig(fig_name, bbox_inches='tight')

    return names, dcss, ious, ste_probs





if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()

