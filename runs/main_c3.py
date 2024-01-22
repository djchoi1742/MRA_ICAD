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
# from data.setup_c import DataSettingV1, INFO_PATH  # w/o detection map
from data.setup_d import DataSettingV1, INFO_PATH  # with detection map
import models.model_c as model_ref
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
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model15')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,112,160,208')
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
main_config.add_argument('--mtl_mode', type=lambda x: x.title() in str(True), dest='mtl_mode', default=False)
main_config.add_argument('--cls_lambda', type=float, dest='cls_lambda', default=1.0)
main_config.add_argument('--cls_alpha', type=float, dest='cls_alpha', default=0.05)
main_config.add_argument('--cls_gamma', type=float, dest='cls_gamma', default=0.0)

config, unparsed = parser.parse_known_args()

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')

import inspect
exec_file = os.path.basename(inspect.getfile(inspect.currentframe()))

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
                      image_size=img_size, radius=config.radius, det_size=16, one_hot=config.one_hot)

if config.train:
    train_db = d_set.train.batch(config.batch_size)
    train_length = d_set.train.cardinality().numpy()
    print('train length: ', train_length)

val_db = d_set.val.batch(config.batch_size)
val_length = d_set.val.cardinality().numpy()
print('val length: ', val_length)

input_size = [seq_len, img_size, img_size, img_c] if seq_len != 1 else [img_size, img_size, img_c]

infer_name = config.model_name

infer = getattr(model_ref, infer_name)(input_size=input_size, class_num=1, f_num=f_num, is_training=config.train,
                                       add_conv=config.add_conv, use_ic=config.use_ic, use_se=config.use_se,
                                       mtl_mode=config.mtl_mode)

model, cam_model = infer.model, infer.cam_model

seg_loss_fn = metric.weighted_dice_score_loss
cls_loss_fn = metric.focal_loss_sigmoid

dcs_metric = metric.dcs_2d if config.seq_len == 1 else metric.dcs_3d
iou_metric = metric.iou_2d if config.seq_len == 1 else metric.iou_3d

cls_alpha, cls_gamma = config.cls_alpha, config.cls_gamma


def training():
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
        'USE_SE': config.use_se,
        'H_CLS_ALPHA': config.cls_alpha,
        'H_CLS_GAMMA': config.cls_gamma
    }

    with open(os.path.join(result_path, '.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    train_summary, val_summary = tboard.tensorboard_create(log_path)
    result_name = '_'.join([config.exp_name, config.model_name, serial_str]) + '.csv'
    dcs_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'METRIC': pd.Series(),
                            'DCS': pd.Series(), 'IOU': pd.Series(), 'AUC': pd.Series()})

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
            # train_sx, train_sy = [], []
            train_cx, train_cy = [], []

            train_loss, train_seg_loss, train_cls_loss, train_dcs, train_iou = [], [], [], [], []
            for train_step, (img, mask, ste, _, _) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    seg_prob, cls_prob = model(img)

                    seg_loss_batch = seg_loss_fn(mask, seg_prob)
                    cls_loss_batch = cls_loss_fn(ste, cls_prob, cls_alpha, cls_gamma)
                    loss_vars = [seg_loss_batch, cls_loss_batch]

                    if config.mtl_mode:
                        train_loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
                    else:
                        train_loss_batch = seg_loss_batch + config.cls_lambda * cls_loss_batch

                grads = tape.gradient(train_loss_batch, model.params)
                optimizer.apply_gradients(zip(grads, model.params))

                # grads = tape.gradient(train_loss_batch, model.trainable_variables)
                # optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.append(train_loss_batch)
                train_seg_loss.append(seg_loss_batch)
                train_cls_loss.append(cls_loss_batch)

                # train_sx.extend(seg_prob.numpy())
                # train_sy.extend(mask.numpy())

                if config.each_ste:
                    train_cx.extend(np.reshape(cls_prob.numpy(), (-1,)).tolist())
                    train_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())
                else:
                    train_cx.extend(cls_prob.numpy())
                    train_cy.extend(ste.numpy())

                train_dcs_batch = dcs_metric(mask, seg_prob)
                train_dcs.extend(train_dcs_batch)
                dcs_batch_mean = np.mean(train_dcs_batch)

                train_iou_batch = iou_metric(mask, seg_prob)
                train_iou.extend(train_iou_batch)
                iou_batch_mean = np.mean(train_iou_batch)

                sys.stdout.write('Step: {0:>4d}, Loss: {1:.4f} Seg Loss: {2:.4f} Cls Loss: {3:.4f}'
                                 ' DCS: {4:.4f} IOU: {5:.4f} ({6})\r'.
                                 format(train_step, train_loss_batch, seg_loss_batch, cls_loss_batch,
                                        dcs_batch_mean, iou_batch_mean, epoch))

            train_loss_mean = np.mean(train_loss)
            train_seg_loss_mean, train_cls_loss_mean = np.mean(train_seg_loss), np.mean(train_cls_loss)
            train_dcs_mean, train_iou_mean = np.mean(train_dcs), np.mean(train_iou)
            train_auc = metric.calculate_auc(train_cy, train_cx)

            train_record = {'Loss': train_loss_mean, 'Seg_Loss': train_seg_loss_mean, 'Cls_Loss': train_cls_loss_mean,
                            'DCS': train_dcs_mean, 'IOU': train_iou_mean, 'AUC': train_auc}

            val_cx, val_cy = [], []
            val_loss, val_seg_loss, val_cls_loss, val_dcs, val_iou = [], [], [], [], []
            val_steps = val_length // config.batch_size + 1

            for val_step, (img, mask, ste, _, _) in enumerate(val_db):
                val_seg_prob, val_cls_prob = model(img)

                val_seg_loss_batch = seg_loss_fn(mask, val_seg_prob)
                val_cls_loss_batch = cls_loss_fn(ste, val_cls_prob, cls_alpha, cls_gamma)
                loss_vars = [val_seg_loss_batch, val_cls_loss_batch]

                if config.mtl_mode:
                    val_loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
                else:
                    val_loss_batch = val_seg_loss_batch + config.cls_lambda * val_cls_loss_batch

                val_loss.append(val_loss_batch)
                val_seg_loss.append(val_seg_loss_batch)
                val_cls_loss.append(val_cls_loss_batch)

                if config.each_ste:
                    val_cx.extend(np.reshape(val_cls_prob.numpy(), (-1,)).tolist())
                    val_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())
                else:
                    val_cx.extend(val_cls_prob.numpy())
                    val_cy.extend(ste.numpy())

                val_dcs_batch = dcs_metric(mask, val_seg_prob)
                val_dcs.extend(val_dcs_batch)
                dcs_batch_mean = np.mean(val_dcs_batch)

                val_iou_batch = iou_metric(mask, val_seg_prob)
                val_iou.extend(val_iou_batch)
                iou_batch_mean = np.mean(val_iou_batch)

                sys.stdout.write('Evaluation [{0}/{1}],  Loss: {2:.4f} Seg Loss: {3:.4f} Cls Loss: {4:.4f}'
                                 ' DCS: {5:.4f} IOU: {6:.4f}\r'.
                                 format(val_step + 1, val_steps, val_loss_batch,
                                        val_seg_loss_batch, val_cls_loss_batch, dcs_batch_mean, iou_batch_mean))

            val_loss_mean = np.mean(val_loss)
            val_seg_loss_mean, val_cls_loss_mean = np.mean(val_seg_loss), np.mean(val_cls_loss)
            val_dcs_mean, val_iou_mean = np.mean(val_dcs), np.mean(val_iou)
            val_auc = metric.calculate_auc(val_cy, val_cx)

            val_record = {'Loss': val_loss_mean, 'Seg_Loss': val_seg_loss_mean, 'Cls_Loss': val_cls_loss_mean,
                          'DCS': val_dcs_mean, 'IOU': val_iou_mean, 'AUC': val_auc}

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += ' Time Elapsed: {0}'.format(time_elapsed.split('.')[0])

            print('Epoch:%s Train-Loss:%.4f Seg:%.4f Cls:%.4f DCS:%.4f IOU:%.4f AUC:%.4f '
                  'Val-Loss:%.4f Seg:%.4f Cls:%.4f DCS:%.4f IOU:%.4f AUC:%.4f' %
                  (epoch, train_loss_mean, train_seg_loss_mean, train_cls_loss_mean,
                   train_dcs_mean, train_iou_mean, train_auc,
                   val_loss_mean, val_seg_loss_mean, val_cls_loss_mean,
                   val_dcs_mean, val_iou_mean, val_auc) + log_string)

            tboard.board_record_value(train_summary, train_record, epoch)
            tboard.board_record_value(val_summary, val_record, epoch)

            log_string = ''
            val_metric = np.mean([val_dcs_mean, val_auc])
            perf_per_epoch.append(val_metric)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_metric, val_dcs_mean, val_iou_mean, val_auc

            elif val_metric > min(dcs_csv['METRIC'].tolist()):
                os.remove(dcs_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                dcs_csv = dcs_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_metric, val_dcs_mean, val_iou_mean, val_auc

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
    weight_auc_csv = weight_auc_csv.sort_values('METRIC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][(config.num_weight-1):config.num_weight])

    # show_slice_cam = show_slice_cam_3d if config.seq_len != 1 else show_slice_cam_2d
    show_slice_cam = show_slice_cam_seq if config.each_ste else show_slice_cam_3d

    if config.seq_len == 1:
        gen_grad_cam = gen_grad_cam_2d
    elif cam_model.outputs[0].shape[1] != 1:
        gen_grad_cam = gen_grad_cam_lstm
    else:
        gen_grad_cam = gen_grad_cam_3d

    # val_dcs = np.zeros([len(all_ckpt_paths), val_length])
    # val_iou = np.zeros([len(all_ckpt_paths), val_length])
    # val_prob = np.zeros([len(all_ckpt_paths), val_length])
    # val_name = []

    dcss, ious, probs = [], [], []
    names = []

    print('num_ckpt: ', len(all_ckpt_paths))

    guided_model = metric.built_guided_model(model, mtl_mode=config.mtl_mode)

    ckpt_idx = 0
    val_cx, val_cy = [], []
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        val_loss, val_name = [], []
        for step, (img, mask, ste, det, name) in enumerate(val_db):

            with tf.GradientTape() as tape:
                cam_layers, seg_prob, cls_prob = cam_model(img)

            grad_cams = gen_grad_cam(cam_layers, cls_prob, tape, infer)
            gb = metric.guided_backprop(guided_model, img, infer.cam_layer_name)
            guided_grad_cams = gb * grad_cams
            ggc = metric.deprocess_image(guided_grad_cams)

            seg_loss_batch = seg_loss_fn(mask, seg_prob)
            cls_loss_batch = cls_loss_fn(ste, cls_prob, cls_alpha, cls_gamma)
            loss_vars = [seg_loss_batch, cls_loss_batch]

            if config.mtl_mode:
                loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
            else:
                loss_batch = seg_loss_batch + config.cls_lambda * cls_loss_batch
            val_loss.append(loss_batch)

            # val_cls_prob_each = tf.reshape(val_cls_prob, [ste.shape[0], ]).numpy()
            # val_cx.extend(val_cls_prob_each)

            # val_cx.extend(val_cls_prob.numpy())
            # val_loss_batch = val_seg_loss_batch + val_cls_loss_batch

            name_batch, dcs_batch, iou_batch, prob_batch = \
                show_slice_cam(images=img.numpy(), masks=mask.numpy(), pred_masks=seg_prob.numpy(),
                               stes=ste.numpy(), pred_stes=cls_prob.numpy(), cam_stes=grad_cams, ggc_stes=ggc,
                               dets=det.numpy(), name=name.numpy(), is_png=config.is_png,
                               save_path=plot_val_path)

            if ckpt_idx == 0:
                name_str = [x.decode() for x in name.numpy()]
                # val_name.extend(val_name_batch) if config.seq_len == 1 else val_name.extend(name_str)
                names.extend(name_batch) if config.each_ste else names.extend(name_str)
                # val_cy.extend(ste.numpy())
                val_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())

            # val_dcs[ckpt_idx, step * cnt_range:step * cnt_range + len(val_dcs_batch)] = val_dcs_batch
            # val_iou[ckpt_idx, step * cnt_range:step * cnt_range + len(val_iou_batch)] = val_iou_batch
            # val_prob[ckpt_idx, step * cnt_range:step * cnt_range + len(val_iou_batch)] = val_prob_batch

            dcss.extend(dcs_batch)
            ious.extend(iou_batch)
            probs.extend(np.reshape(prob_batch, (-1,)).tolist())

            sys.stdout.write('{0} Evaluation [{1}/{2}], DCS:{3:.4f}, IOU:{4:.4f}\r'.
                             format(os.path.basename(ckpt), step, val_length // config.batch_size,
                                    np.mean(dcs_batch), np.mean(iou_batch)))

        ckpt_idx += 1

    val_dcs, val_iou = np.mean(dcss), np.mean(ious)
    val_auc = metric.calculate_auc(val_cy, probs)
    # val_dcs, val_iou = np.mean(val_dcs, axis=0), np.mean(val_iou, axis=0)
    # val_prob = np.mean(val_prob, axis=0)

    print('\nFinal DCS: %.3f, IOU: %.3f, AUC: %.3f' % (np.mean(val_dcs), np.mean(val_iou), val_auc))

    result_csv = pd.DataFrame({'NUMBER': names, 'DCS': dcss, 'IOU': ious, 'STE': val_cy, 'PROB': probs})

    result_name = '_'.join([config.model_name, config.excel_name, config.val_name, serial_str,
                            '%03d' % config.num_weight]) + '.csv'
    result_csv.to_csv(os.path.join(result_path, result_name), index=False)


def show_slice_cam_seq(images, masks, pred_masks, stes, pred_stes, cam_stes, ggc_stes, dets, name, is_png=True,
                       num_rows=2, num_cols=3, fig_size=(3 * 1.3, 2 * 1.3), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = pred_masks.shape[0]
    names, dcss, ious, lbls, probs = [], [], [], [], []

    for i in range(batch_size):
        patient_id, start_idx = metric.name_idx(name[i])

        each_dcs = dcs_metric(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()
        each_iou = iou_metric(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()

        dcss.extend(np.repeat(each_dcs, config.seq_len))
        ious.extend(np.repeat(each_iou, config.seq_len).tolist())

        for j in range(config.seq_len):
            show_ste, show_pred_ste = stes[i, j], pred_stes[i, j]
            lbls.append(show_ste)
            probs.append(show_pred_ste)

            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])
            names.append(each_name)

            if is_png:
                show_image = images[i, j, :, :, 0]
                show_mask = masks[i, j, :, :, 0]
                show_pred_mask = pred_masks[i, j, :, :, 0]
                show_cam_ste = cam_stes[i, j, :, :, 0]  # grad_cam
                show_ggc_ste = ggc_stes[i, j, :, :, 0]

                show_det = dets[i, j, :, :, 0]
                rsz_show_det = skimage.transform.resize(show_det, images.shape[2:4])

                fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                ax[0, 0].imshow(show_image, cmap='bone')
                ax[0, 0].set_title(each_name, fontsize=6, color='black')

                ax[0, 1].imshow(show_mask, cmap='bone')
                ax[0, 1].set_title('Mask: ' + each_name, fontsize=6, color='navy')

                ax[0, 2].imshow(show_pred_mask, cmap='bone')
                ax[0, 2].set_title('DCS: %.3f, IOU: %.3f' % (each_dcs, each_iou), fontsize=6, color='blue')

                ax[1, 0].imshow(show_image, cmap='bone')
                ax[1, 0].imshow(rsz_show_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 0].set_title('Stenosis: %d' % show_ste, fontsize=6, color='darkgreen')

                ax[1, 1].imshow(show_image, cmap='bone')
                ax[1, 1].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 1].set_title('Prob: %.3f' % show_pred_ste, fontsize=6, color='green')

                ax[1, 2].imshow(show_ggc_ste, cmap='bone')
                ax[1, 2].set_title('Guided Grad-CAM', fontsize=6, color='darkgoldenrod')

                fig_name = os.path.join(save_path, each_name)
                plt.savefig(fig_name, bbox_inches='tight')

                plt.clf()

        # dcss.extend(dcs_slices)
        # ious.extend(iou_slices)
        # probs.extend(prob_slices)

    return names, dcss, ious, probs


def show_slice_cam_2d(images, masks, pred_masks, stes, pred_stes, cam_stes, ggc_stes, dets,
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


def show_slice_cam_3d(images, masks, pred_masks, stes, pred_stes, cam_stes, ggc_stes, dets, name, is_png=True,
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


if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()

