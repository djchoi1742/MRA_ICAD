import os, sys, logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import pandas as pd
import argparse, json
import re, datetime

sys.path.append('/workspace/bitbucket/MRA')
from data.setup_d import DataSettingV1, INFO_PATH
import models.model_d as model_ref
import models.metric as metric
from runs.cams import *
import tf_utils.tboard as tboard

parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh08')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp008')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1')
main_config.add_argument('--val_name', type=str, dest='val_name', default='2')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model24')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,128,192,256')
main_config.add_argument('--serial', type=int, dest='serial', default=1)
main_config.add_argument('--image_size', type=int, dest='image_size', default=256)
main_config.add_argument('--radius', type=int, dest='radius', default=80)
main_config.add_argument('--det_size', type=int, dest='det_size', default=16)
main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
main_config.add_argument('--max_keep', type=int, dest='max_keep', default=5)  # only use training
main_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)  # only use validation
main_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
main_config.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.00005)
main_config.add_argument('--decay_steps', type=int, dest='decay_steps', default=300)
main_config.add_argument('--decay_rate', type=int, dest='decay_rate', default=0.94)
main_config.add_argument('--batch_size', type=int, dest='batch_size', default=2)
main_config.add_argument('--epoch', type=int, dest='epoch', default=40)
main_config.add_argument('--seq_len', type=int, dest='seq_len', default=5)
main_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)
main_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=False)
main_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=False)
main_config.add_argument('--one_hot', type=lambda x: x.title() in str(True), dest='one_hot', default=False)
main_config.add_argument('--cls_lambda', type=float, dest='cls_lambda', default=1.0)
main_config.add_argument('--det_lambda', type=float, dest='det_lambda', default=0.1)
main_config.add_argument('--use_ic', type=lambda x: x.title() in str(True), dest='use_ic', default=False)
main_config.add_argument('--is_png', type=lambda x: x.title() in str(True), dest='is_png', default=False)

config, unparsed = parser.parse_known_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)

import warnings

warnings.filterwarnings('ignore')

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
                      train=config.train, seq_len=seq_len, seq_interval=seq_interval,
                      only_ste=config.only_ste, each_ste=config.each_ste,
                      img_size=img_size, radius=config.radius, det_size=config.det_size, one_hot=config.one_hot)

if config.train:
    train_db = d_set.train.batch(config.batch_size)
    train_length = d_set.train.cardinality().numpy()
    print('train length: ', train_length)

val_db = d_set.val.batch(config.batch_size)
val_length = d_set.val.cardinality().numpy()
print('val length: ', val_length)

input_size = [seq_len, img_size, img_size, img_c] if seq_len != 1 else [img_size, img_size, img_c]

infer_name = config.model_name
infer = getattr(model_ref, infer_name)(input_size=input_size, f_num=f_num, is_training=config.train,
                                       det_size=config.det_size, use_ic=config.use_ic)


model, cam_model = infer.model, infer.cam_model

# seg_loss_fn = metric.weighted_dice_score_loss
cls_loss_fn = metric.focal_loss_sigmoid

# dcs_metric = metric.dcs_2d if config.seq_len == 1 else metric.dcs_3d
# iou_metric = metric.iou_2d if config.seq_len == 1 else metric.iou_3d


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
        'DET_SIZE': config.det_size,
        'SEQ_LENGTH': config.seq_len,
        'SEQ_INTERVAL': config.seq_interval,
        'BATCH_SIZE': config.batch_size,
        'LEARNING_RATE': config.learning_rate,
        'CLS_LAMBDA': config.cls_lambda,
        'DET_LAMBDA': config.det_lambda,
        'DECAY_STEPS': config.decay_steps,
        'DECAY_RATE': config.decay_rate,
        'EPOCH': config.epoch,
        'USE_IC': config.use_ic

    }

    with open(os.path.join(result_path, '.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    train_summary, val_summary = tboard.tensorboard_create(log_path)
    result_name = '_'.join([config.exp_name, config.model_name, serial_str]) + '.csv'
    dcs_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'LOSS': pd.Series(), 'FOM': pd.Series(), 'AUC': pd.Series()})

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
            train_cx, train_cy = [], []

            train_loss = []
            train_each_names, train_each_stes, train_ratings, train_weights = [], [], [], []

            for train_step, (img, mask, ste, det, name) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    det_prob = model(img)

                    train_loss_batch = config.det_lambda * metric.object_loss(det, det_prob)

                grads = tape.gradient(train_loss_batch, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.append(train_loss_batch)

                cls_prob = tf.reduce_mean(det_prob, axis=[1, 2, 3])

                train_cx.extend(cls_prob.numpy())
                train_cy.extend(ste.numpy())

                # train_cx.extend(cls_prob.numpy())
                # train_cy.extend(tf.reshape(ste, [ste.shape[0], ]).numpy())

                _, each_stes, ratings, weights = metric.set_jafroc(det, det_prob, name.numpy())

                train_each_stes.extend(each_stes)
                train_ratings.extend(ratings)
                train_weights.extend(weights)

                sys.stdout.write('Step: {0:>4d},  Loss: {1:.4f} ({2})\r'.format(train_step, train_loss_batch, epoch))

            train_loss_mean = np.mean(train_loss)
            train_jafroc = metric.calculate_jafroc(train_each_stes, train_ratings, train_weights)

            train_auc = metric.calculate_auc(train_cy, train_cx)
            train_record = {'Loss': train_loss_mean, 'AUC': train_auc, 'JAFROC': train_jafroc}

            val_cx, val_cy = [], []
            val_loss = []
            val_each_names, val_each_stes, val_ratings, val_weights = [], [], [], []

            val_steps = val_length // config.batch_size + 1
            for val_step, (img, mask, ste, det, name) in enumerate(val_db):
                val_det_prob = model(img)
                val_loss_batch = config.det_lambda * metric.object_loss(det, val_det_prob)

                val_loss.append(val_loss_batch)
                val_cls_prob = tf.reduce_mean(val_det_prob, axis=[1, 2, 3])

                val_cx.extend(val_cls_prob.numpy())
                val_cy.extend(ste.numpy())
                # val_cx.extend(val_cls_prob.numpy())
                # val_cy.extend(tf.reshape(ste, [ste.shape[0], ]).numpy())

                _, each_stes, ratings, weights = metric.set_jafroc(det, det_prob, name.numpy())

                val_each_stes.extend(each_stes)
                val_ratings.extend(ratings)
                val_weights.extend(weights)

                sys.stdout.write('Evaluation [{0}/{1}],  Loss: {2:.4f}\r'.
                                 format(val_step+1, val_steps, val_loss_batch))

            val_loss_mean = np.mean(val_loss)

            val_auc = metric.calculate_auc(val_cy, val_cx)
            val_jafroc = metric.calculate_jafroc(val_each_stes, val_ratings, val_weights)

            val_record = {'Loss': val_loss_mean, 'AUC': val_auc, 'JAFROC': val_jafroc}

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += ' Elapsed: {0}'.format(time_elapsed.split('.')[0])

            print('Epoch: %s Train-Loss: %.4f Train-FOM %.3f Train-AUC %.3f Val-Loss %.4f Val-FOM %.3f Val-AUC %.3f' %
                  (epoch, train_loss_mean, train_jafroc, train_auc, val_loss_mean, val_jafroc, val_auc) + log_string)

            tboard.board_record_value(train_summary, train_record, epoch)
            tboard.board_record_value(val_summary, val_record, epoch)

            log_string = ''
            val_metric = val_jafroc
            perf_per_epoch.append(val_metric)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_loss_mean, val_jafroc, val_auc

            elif val_metric < max(dcs_csv['LOSS'].tolist()):
                os.remove(dcs_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                dcs_csv = dcs_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_loss_mean, val_jafroc, val_auc

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
    weight_auc_csv = weight_auc_csv.sort_values('AUC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])

    cnt_range = config.batch_size
    show_slice_cam = show_slice_cam_3d if config.seq_len != 1 else show_slice_cam_2d
    if config.seq_len == 1:
        gen_grad_cam = gen_grad_cam_2d
    # elif config.each_ste:
    elif cam_model.outputs[0].shape[1] != 1:
        gen_grad_cam = gen_grad_cam_lstm
    else:
        gen_grad_cam = gen_grad_cam_3d

    probs = np.zeros([len(all_ckpt_paths), val_length])

    names = []

    print('num_ckpt: ', len(all_ckpt_paths))

    gt_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                          'PROB': pd.Series(), 'DETECTED': pd.Series(dtype=int)})

    pred_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                            'PROB': pd.Series(), 'GT': pd.Series(dtype=int)})

    ckpt_idx = 0
    val_cx, val_cy = [], []

    val_each_names, val_each_stes, val_ratings, val_weights = [], [], [], []
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        loss = []
        for step, (img, mask, ste, det, name) in enumerate(val_db):

            with tf.GradientTape() as tape:
                det_prob = model(img)

            loss_batch = config.det_lambda * metric.object_loss(det, det_prob)

            cls_prob = tf.reduce_mean(det_prob, axis=[1, 2, 3])
            val_cx.extend(cls_prob.numpy())

            loss.append(loss_batch)

            each_names, each_stes, ratings, weights = metric.set_jafroc(det, det_prob, name.numpy())

            val_each_names.extend(each_names)
            val_each_stes.extend(each_stes)
            val_ratings.extend(ratings)
            val_weights.extend(weights)

            name_batch, prob_batch, gt_batch, pred_batch = \
                show_slice_cam(images=img.numpy(), masks=mask.numpy(),
                               stes=ste.numpy(), pred_stes=cls_prob.numpy(), dets=det, pred_dets=det_prob,
                               name=name.numpy(), is_png=config.is_png, save_path=plot_val_path)

            if ckpt_idx == 0:
                name_str = [x.decode() for x in name.numpy()]
                names.extend(name_batch) if config.seq_len == 1 else names.extend(name_str)
                # val_cy.extend(tf.reshape(ste, [ste.shape[0], ]).numpy())
                val_cy.extend(ste.numpy())

            probs[ckpt_idx, step * cnt_range:step * cnt_range + len(prob_batch)] = prob_batch

            gt_df = gt_df.append(gt_batch)
            pred_df = pred_df.append(pred_batch)

            sys.stdout.write('{0} Evaluation [{1}/{2}]\r'.
                             format(os.path.basename(ckpt), step, val_length // config.batch_size))

        ckpt_idx += 1

    val_ste_prob = np.mean(probs, axis=0)
    val_auc = metric.calculate_auc(val_cy, val_cx)
    val_jafroc = metric.calculate_jafroc(val_each_stes, val_ratings, val_weights)

    print('\nFinal FOM: %.3f AUC: %.3f' % (val_jafroc, val_auc))

    jafroc_df = pd.DataFrame({'NUMBER': val_each_names, 'LABEL': val_each_stes,
                              'PROB': val_ratings, 'WEIGHT': val_weights})

    result_name = '_'.join([config.model_name, config.excel_name, config.val_name, serial_str,
                            '%03d' % config.num_weight]) + '.xlsx'

    writer = pd.ExcelWriter(os.path.join(result_path, result_name), engine='xlsxwriter')

    result_csv = pd.DataFrame({'NUMBER': names, 'STE': val_cy, 'STE_PROB': val_ste_prob})

    result_csv.to_excel(writer, sheet_name='TOTAL', index=False)
    jafroc_df.to_excel(writer, sheet_name='JAFROC', index=False)
    gt_df.to_excel(writer, sheet_name='GT', index=False)
    pred_df.to_excel(writer, sheet_name='PRED', index=False)

    writer.save()


def show_slice_cam_3d(images, masks, stes, pred_stes, dets, pred_dets, name, is_png=True,
                      num_rows=1, num_cols=4, fig_size=(4 * 2, 1 * 2), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = images.shape[0]
    names, lbls, probs = [], [], []

    gt_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                          'PROB': pd.Series(), 'DETECTED': pd.Series(dtype=int)})

    pred_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                            'PROB': pd.Series(), 'GT': pd.Series(dtype=int)})

    for i in range(batch_size):
        patient_id, start_idx = metric.name_idx(name[i])
        show_ste, show_pred_ste = stes[i], pred_stes[i]

        lbls.append(show_ste)
        probs.append(show_pred_ste)
        names.append(name[i])

        for j in range(seq_len):
            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])

            show_image = images[i, j, :, :, 0]
            show_mask = masks[i, j, :, :, 0]
            # show_pred_mask = pred_masks[i, j, :, :, 0]

            show_det = dets.numpy()[i, j, :, :, 0]
            show_pred_det = pred_dets.numpy()[i, j, :, :, 0]
            show_pred_bin_det = np.where(show_pred_det < 0.8, 0.0, 1.0)  # threshold: 0.9

            show_gt_idx = np.where(show_det == 1)
            gt_idx_h, gt_idx_w = show_gt_idx
            show_gt_prob = show_pred_det[show_gt_idx]
            show_gt_loc = show_pred_bin_det[show_gt_idx]
            num_gt = len(show_gt_prob)

            if num_gt > 0:
                gt_each = pd.DataFrame({'PATIENT_ID': np.repeat(each_name, num_gt),
                                        'HEIGHT': gt_idx_h, 'WIDTH': gt_idx_w,
                                        'PROB': show_gt_prob, 'DETECTED': show_gt_loc.astype(int)})
                gt_df = gt_df.append(gt_each)

            show_pred_idx = np.where(show_pred_bin_det == 1)
            pred_idx_h, pred_idx_w = show_pred_idx
            show_pred_prob = show_pred_det[show_pred_idx]
            show_pred_loc = show_det[show_pred_idx]
            num_pred = len(show_pred_prob)

            if num_pred > 0:
                pred_each = pd.DataFrame({'PATIENT_ID': np.repeat(each_name, num_pred),
                                          'HEIGHT': pred_idx_h, 'WIDTH': pred_idx_w,
                                          'PROB': show_pred_prob, 'GT': show_pred_loc.astype(int)})
                pred_df = pred_df.append(pred_each)

            rsz_show_det = skimage.transform.resize(show_det, show_image.shape)
            rsz_show_pred_det = skimage.transform.resize(show_pred_bin_det, show_image.shape)

            if is_png:
                fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                ax[0].imshow(show_image, cmap='bone')
                ax[0].set_title(each_name, fontsize=6, color='black')

                ax[1].imshow(show_mask, cmap='bone')
                ax[1].set_title('Mask: ' + each_name, fontsize=6, color='navy')

                ax[2].imshow(show_image, cmap='bone')
                ax[2].imshow(rsz_show_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[2].set_title('Stenosis: %d' % show_ste, fontsize=6, color='darkgreen')

                ax[3].imshow(rsz_show_det, cmap='bone')
                ax[3].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[3].set_title('Stenosis Prob.: %.3f' % show_pred_ste, fontsize=6, color='darkgreen')

                fig_name = os.path.join(save_path, each_name)
                plt.savefig(fig_name, bbox_inches='tight')

            plt.clf()

    return names, probs, gt_df, pred_df


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

