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
from data.fig_23 import DataSettingV2, INFO_PATH
import models.model_d as model_ref
import models.metric as metric
from runs.cams import *
import tf_utils.tboard as tboard


parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh09')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp009')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,4')
main_config.add_argument('--val_name', type=str, dest='val_name', default='1,2')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model28')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,112,160,208')
main_config.add_argument('--serial', type=int, dest='serial', default=24)
main_config.add_argument('--image_size', type=int, dest='image_size', default=256)
main_config.add_argument('--radius', type=int, dest='radius', default=80)
main_config.add_argument('--det_size', type=int, dest='det_size', default=16)
main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
main_config.add_argument('--max_keep', type=int, dest='max_keep', default=5)  # only use training
main_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)  # only use validation
main_config.add_argument('--train', type=lambda x: x.title() in str(True), dest='train', default=False)
main_config.add_argument('--learning_rate', type=float, dest='learning_rate', default=0.00005)
main_config.add_argument('--decay_steps', type=int, dest='decay_steps', default=5000)
main_config.add_argument('--decay_rate', type=int, dest='decay_rate', default=0.94)
main_config.add_argument('--batch_size', type=int, dest='batch_size', default=2)
main_config.add_argument('--epoch', type=int, dest='epoch', default=50)
main_config.add_argument('--seq_len', type=int, dest='seq_len', default=8)
main_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)
main_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=False)
main_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=True)
main_config.add_argument('--one_hot', type=lambda x: x.title() in str(True), dest='one_hot', default=False)
main_config.add_argument('--use_ic', type=lambda x: x.title() in str(True), dest='use_ic', default=False)
main_config.add_argument('--use_se', type=lambda x: x.title() in str(True), dest='use_se', default=False)
main_config.add_argument('--is_png', type=lambda x: x.title() in str(True), dest='is_png', default=False)
main_config.add_argument('--mtl_mode', type=lambda x: x.title() in str(True), dest='mtl_mode', default=True)
main_config.add_argument('--cls_lambda', type=float, dest='cls_lambda', default=1.0)  # naive weight sum
main_config.add_argument('--det_lambda', type=float, dest='det_lambda', default=0.01)  # naive weight sum
main_config.add_argument('--cls_alpha', type=float, dest='cls_alpha', default=0.05)
main_config.add_argument('--cls_gamma', type=float, dest='cls_gamma', default=0.0)
main_config.add_argument('--det_alpha', type=float, dest='det_alpha', default=0.01)
main_config.add_argument('--det_gamma', type=float, dest='det_gamma', default=2.0)
main_config.add_argument('--init_w', type=float, dest='init_w', default=0.1)  # for detect loss

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

d_set = DataSettingV2(df=df, train_type=config.train_name, val_type=config.val_name,
                      patient_id='13352132', data_type=data_type,
                      train=config.train, seq_len=seq_len, seq_interval=seq_interval,
                      only_ste=config.only_ste, each_ste=config.each_ste,
                      image_size=img_size, radius=config.radius, det_size=config.det_size, one_hot=config.one_hot)

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
                                       det_size=config.det_size, seq_len=config.seq_len,
                                       use_ic=config.use_ic, use_se=config.use_se, mtl_mode=config.mtl_mode)


model, cam_model = infer.model, infer.cam_model

seg_loss_fn = metric.weighted_dice_score_loss
cls_loss_fn = metric.focal_loss_sigmoid
det_loss_fn = metric.object_loss

dcs_metric = metric.dcs_2d if config.seq_len == 1 else metric.dcs_3d
iou_metric = metric.iou_2d if config.seq_len == 1 else metric.iou_3d

if config.each_ste:
    jafroc_fn = metric.set_jafroc_2d if config.seq_len == 1 else metric.set_jafroc_seq
else:
    jafroc_fn = metric.set_jafroc

cls_alpha, cls_gamma = config.cls_alpha, config.cls_gamma
det_alpha, det_gamma, init_w = config.det_alpha, config.det_gamma, config.init_w


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
        'USE_IC': config.use_ic,
        'USE_SE': config.use_se,
        'MTL_MODE': config.mtl_mode,
        'H_CLS_ALPHA': config.cls_alpha,
        'H_CLS_GAMMA': config.cls_gamma,
        'H_DET_ALPHA': config.det_alpha,
        'H_DET_GAMMA': config.det_gamma,
        'INIT_W': config.init_w
    }

    with open(os.path.join(result_path, '.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    train_summary, val_summary = tboard.tensorboard_create(log_path)
    result_name = '_'.join([config.exp_name, config.model_name, serial_str]) + '.csv'
    dcs_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'METRIC': pd.Series(), 'DCS': pd.Series(), 'IOU': pd.Series(),
                            'FOM1': pd.Series(), 'FOM2': pd.Series(), 'AUC': pd.Series()})

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
            train_loss, train_seg_loss, train_cls_loss, train_det_loss, train_dcs, train_iou = [], [], [], [], [], []
            train_ens1, train_ess1, train_rts1, train_wts1 = [], [], [], []
            train_ens2, train_ess2, train_rts2, train_wts2 = [], [], [], []

            for train_step, (img, mask, ste, det, name) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    seg_prob, cls_prob, det_prob = model(img)

                    seg_loss_batch = seg_loss_fn(mask, seg_prob)
                    cls_loss_batch = cls_loss_fn(ste, cls_prob, cls_alpha, cls_gamma)
                    det_loss_batch = det_loss_fn(det, det_prob, det_alpha, det_gamma, init_w)
                    # det_loss_batch = 0.1 * metric.object_loss(det, det_prob)  # exp009,Model28,serial 0,4,8
                    # det_loss_batch = metric.weighted_object_loss(det, det_prob)  # exp009,Model28,serial 6
                    # det_loss_batch = 0.1 * metric.weighted_object_loss(det, det_prob)  # exp009,Model28,serial 7
                    loss_vars = [seg_loss_batch, cls_loss_batch, det_loss_batch]

                    if config.mtl_mode:
                        train_loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
                    else:
                        cls_loss_batch = config.cls_lambda * cls_loss_batch
                        det_loss_batch = config.det_lambda * det_loss_batch
                        train_loss_batch = seg_loss_batch + cls_loss_batch + det_loss_batch

                grads = tape.gradient(train_loss_batch, model.params)
                optimizer.apply_gradients(zip(grads, model.params))

                train_loss.append(train_loss_batch)
                train_seg_loss.append(seg_loss_batch)
                train_cls_loss.append(cls_loss_batch)
                train_det_loss.append(det_loss_batch)

                train_dcs_batch = dcs_metric(mask, seg_prob)
                train_dcs.extend(train_dcs_batch)
                dcs_batch_mean = np.mean(train_dcs_batch)

                train_iou_batch = iou_metric(mask, seg_prob)
                train_iou.extend(train_iou_batch)
                iou_batch_mean = np.mean(train_iou_batch)

                if config.each_ste:
                    train_cx.extend(np.reshape(cls_prob.numpy(), (-1,)).tolist())
                    train_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())
                    # ens1, ess1, rts1, wts1, ens2, ess2, rts2, wts2 = jafroc_fn(det, det_prob, name)
                else:
                    train_cx.extend(cls_prob.numpy())
                    train_cy.extend(ste.numpy())

                ens1, ess1, rts1, _, wts1, ens2, ess2, rts2, _, wts2 = jafroc_fn(det, det_prob, None, name)

                train_ens1.extend(ens1)
                train_ess1.extend(ess1)
                train_rts1.extend(rts1)
                train_wts1.extend(wts1)

                train_ens2.extend(ens2)
                train_ess2.extend(ess2)
                train_rts2.extend(rts2)
                train_wts2.extend(wts2)

                sys.stdout.write('Step: {0:>4d}, Loss: {1:.4f} Seg: {2:.4f} Cls: {3:.4f} Det: {4:.4f} '
                                 'DCS: {5:.4f} IOU: {6:.4f} ({7})\r'.
                                 format(train_step, train_loss_batch, seg_loss_batch, cls_loss_batch, det_loss_batch,
                                        dcs_batch_mean, iou_batch_mean, epoch))

            train_loss_mean = np.mean(train_loss)
            train_seg_loss_mean = np.mean(train_seg_loss)
            train_cls_loss_mean = np.mean(train_cls_loss)
            train_det_loss_mean = np.mean(train_det_loss)
            train_dcs_mean = np.mean(train_dcs)
            train_iou_mean = np.mean(train_iou)

            train_auc = metric.calculate_auc(train_cy, train_cx)
            train_fom1 = metric.calculate_jafroc(train_ess1, train_rts1, train_wts1)
            train_fom2 = metric.calculate_jafroc(train_ess2, train_rts2, train_wts2)

            train_record = {'Loss': train_loss_mean, 'Seg_Loss': train_seg_loss_mean,
                            'Cls_Loss': train_cls_loss_mean, 'Det_Loss': train_det_loss_mean,
                            'DCS': train_dcs_mean, 'IOU': train_iou_mean, 'AUC': train_auc,
                            'JAFROC1': train_fom1, 'JAFROC2': train_fom2}

            val_cx, val_cy = [], []
            val_loss, val_seg_loss, val_det_loss, val_cls_loss, val_dcs, val_iou = [], [], [], [], [], []
            val_ens1, val_ess1, val_rts1, val_wts1 = [], [], [], []
            val_ens2, val_ess2, val_rts2, val_wts2 = [], [], [], []
            val_steps = val_length // config.batch_size + 1

            for val_step, (img, mask, ste, det, name) in enumerate(val_db):
                val_seg_prob, val_cls_prob, val_det_prob = model(img)

                val_seg_loss_batch = seg_loss_fn(mask, val_seg_prob)
                val_cls_loss_batch = cls_loss_fn(ste, val_cls_prob, cls_alpha, cls_gamma)
                val_det_loss_batch = det_loss_fn(det, val_det_prob, det_alpha, det_gamma, init_w)

                # val_det_loss_batch = 0.1 * metric.object_loss(det, val_det_prob)  # exp009,Model28,serial 0,4,8
                # val_det_loss_batch = metric.weighted_object_loss(det, val_det_prob)  # exp009,Model28,serial 6
                # val_det_loss_batch = 0.1 * metric.weighted_object_loss(det, val_det_prob)  # exp009,Model28,serial 7
                loss_vars = [val_seg_loss_batch, val_cls_loss_batch, val_det_loss_batch]

                if config.mtl_mode:
                    val_loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
                else:
                    val_cls_loss_batch = config.cls_lambda * val_cls_loss_batch
                    val_det_loss_batch = config.det_lambda * val_det_loss_batch
                    val_loss_batch = val_seg_loss_batch + val_cls_loss_batch + val_det_loss_batch

                val_loss.append(val_loss_batch)
                val_seg_loss.append(val_seg_loss_batch)
                val_cls_loss.append(val_cls_loss_batch)
                val_det_loss.append(val_det_loss_batch)

                if config.each_ste:
                    val_cx.extend(np.reshape(val_cls_prob.numpy(), (-1,)).tolist())
                    val_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())
                    # ens1, ess1, rts1, wts1, ens2, ess2, rts2, wts2 = jafroc_fn(det, val_det_prob, name)
                else:
                    val_cx.extend(val_cls_prob.numpy())
                    val_cy.extend(ste.numpy())

                ens1, ess1, rts1, _, wts1, ens2, ess2, rts2, _, wts2 = jafroc_fn(det, val_det_prob, None, name)
                # ens1, ess1, rts1, wts1, ens2, ess2, rts2, wts2 = jafroc_fn(det, val_det_prob, name)

                val_dcs_batch = dcs_metric(mask, val_seg_prob)
                val_dcs.extend(val_dcs_batch)
                dcs_batch_mean = np.mean(val_dcs_batch)

                val_iou_batch = iou_metric(mask, val_seg_prob)
                val_iou.extend(val_iou_batch)
                iou_batch_mean = np.mean(val_iou_batch)

                val_ens1.extend(ens1)
                val_ess1.extend(ess1)
                val_rts1.extend(rts1)
                val_wts1.extend(wts1)

                val_ens2.extend(ens2)
                val_ess2.extend(ess2)
                val_rts2.extend(rts2)
                val_wts2.extend(wts2)

                sys.stdout.write('Evaluation [{0}/{1}], Loss: {2:.4f} Seg: {3:.4f}'
                                 ' Cls: {4:.4f} Det: {5:.4f} DCS: {6:.4f} IOU: {7:.4f}\r'.
                                 format(val_step + 1, val_steps, val_loss_batch, val_seg_loss_batch,
                                        val_cls_loss_batch, val_det_loss_batch, dcs_batch_mean, iou_batch_mean))

            val_loss_mean = np.mean(val_loss)
            val_seg_loss_mean = np.mean(val_seg_loss)
            val_cls_loss_mean = np.mean(val_cls_loss)
            val_det_loss_mean = np.mean(val_det_loss)
            val_dcs_mean = np.mean(val_dcs)
            val_iou_mean = np.mean(val_iou)

            val_auc = metric.calculate_auc(val_cy, val_cx)
            val_fom1 = metric.calculate_jafroc(val_ess1, val_rts1, val_wts1)
            val_fom2 = metric.calculate_jafroc(val_ess2, val_rts2, val_wts2)

            val_record = {'Loss': val_loss_mean, 'Seg_Loss': val_seg_loss_mean,
                          'Cls_Loss': val_cls_loss_mean, 'Det_Loss': val_det_loss_mean,
                          'DCS': val_dcs_mean, 'IOU': val_iou_mean, 'AUC': val_auc,
                          'JAFROC1': val_fom1, 'JAFROC2': val_fom2}

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += ' Time:{0}'.format(time_elapsed.split('.')[0])

            print('Epoch:%s '
                  'Train-Seg:%.4f Cls:%.4f Det:%.4f DCS:%.3f IOU:%.3f FOM1:%.3f FOM2:%.3f AUC:%.3f '
                  'Val-Seg:%.4f Cls:%.4f Det:%.4f DCS:%.3f IOU:%.3f FOM1:%.3f FOM2:%.3f AUC:%.3f' %
                  (epoch,
                   train_seg_loss_mean, train_cls_loss_mean, train_det_loss_mean,
                   train_dcs_mean, train_iou_mean, train_fom1, train_fom2, train_auc,
                   val_seg_loss_mean, val_cls_loss_mean, val_det_loss_mean,
                   val_dcs_mean, val_iou_mean, val_fom1, val_fom2, val_auc) + log_string)

            tboard.board_record_value(train_summary, train_record, epoch)
            tboard.board_record_value(val_summary, val_record, epoch)

            log_string = ''
            val_metric = np.mean([val_dcs_mean, val_fom1, val_fom2, val_auc])
            perf_per_epoch.append(val_metric)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_metric, val_dcs_mean, val_iou_mean, val_fom1, val_fom2, val_auc

            elif val_metric > min(dcs_csv['METRIC'].tolist()):
                os.remove(dcs_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                dcs_csv = dcs_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_metric, val_dcs_mean, val_iou_mean, val_fom1, val_fom2, val_auc

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
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][(config.num_weight - 1):config.num_weight])

    dcss, ious, probs, names = [], [], [], []
    print('num_ckpt: ', len(all_ckpt_paths))

    guided_model = metric.built_guided_model(model, mtl_mode=config.mtl_mode)

    ckpt_idx = 0
    val_cy = []
    val_ens1, val_ess1, val_rts1, val_scs1, val_wts1 = [], [], [], [], []
    val_ens2, val_ess2, val_rts2, val_scs2, val_wts2 = [], [], [], [], []
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        loss = []
        for step, (img, mask, ste, det, loc, name) in enumerate(val_db):

            with tf.GradientTape() as tape:
                cam_layers, seg_prob, cls_prob, det_prob, det_gap, det_score = cam_model(img)

            # grad_cams = gen_grad_cam(cam_layers, cls_prob, tape, infer)
            # gb = metric.guided_backprop(guided_model, img, infer.cam_layer_name)
            # guided_grad_cams = gb * grad_cams
            # ggc = metric.deprocess_image(guided_grad_cams)

            seg_loss_batch = seg_loss_fn(mask, seg_prob)
            cls_loss_batch = cls_loss_fn(ste, cls_prob, cls_alpha, cls_gamma)
            det_loss_batch = det_loss_fn(det, det_prob, det_alpha, det_gamma, init_w)  # serial 1

            loss_vars = [seg_loss_batch, cls_loss_batch, det_loss_batch]

            if config.mtl_mode:
                loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
            else:
                cls_loss_batch = config.cls_lambda * cls_loss_batch
                det_loss_batch = config.det_lambda * det_loss_batch
                loss_batch = seg_loss_batch + cls_loss_batch + det_loss_batch

            loss.append(loss_batch)

            if config.each_ste:
                val_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())

                if config.seq_len != 1:
                    ens1, ess1, rts1, scs1, wts1, ens2, ess2, rts2, scs2, wts2 = \
                        metric.set_jafroc_seq(det, det_prob, det_score, name)
                else:
                    ens1, ess1, rts1, scs1, wts1, ens2, ess2, rts2, scs2, wts2 = \
                        metric.set_jafroc_2d(det, det_prob, det_score, name)

            else:
                val_cy.extend(ste.numpy())
                ens1, ess1, rts1, scs1, wts1, ens2, ess2, rts2, scs2, wts2 = \
                    metric.set_jafroc(det, det_prob, det_score, name)

            val_ens1.extend(ens1)
            val_ess1.extend(ess1)
            val_rts1.extend(rts1)
            val_scs1.extend(scs1)
            val_wts1.extend(wts1)

            val_ens2.extend(ens2)
            val_ess2.extend(ess2)
            val_rts2.extend(rts2)
            val_scs2.extend(scs2)
            val_wts2.extend(wts2)

            show_slice_cam_seq_fig1(images=img.numpy(), masks=mask.numpy(), pred_masks=seg_prob.numpy(),
                                   stes=ste.numpy(), pred_stes=cls_prob.numpy(),
                                   dets=det, pred_dets=det_prob, pred_scores=det_score, name=name.numpy(),
                                   is_png=config.is_png, save_path=plot_val_path)  # Fig. 1

        ckpt_idx += 1


def show_slice_cam_seq_fig1(images, masks, pred_masks, stes, pred_stes, dets, pred_dets, pred_scores,
                           name, is_png=True, num_rows=8, num_cols=1, save_path=plot_val_path):  # Fig. 1
    raw_img_path = os.path.join(save_path, 'raw_img')
    if not os.path.exists(raw_img_path):
        os.makedirs(raw_img_path)
    pred_mask_det_path = os.path.join(save_path, 'pred_mask_det')
    if not os.path.exists(pred_mask_det_path):
        os.makedirs(pred_mask_det_path)

    fig_size = (1.2, 1.2)

    for i in range(config.batch_size):
        patient_id, start_idx = metric.name_idx(name[i])

        for j in range(config.seq_len):
            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + j)])

            fig, ax = plt.subplots(1, 1, figsize=fig_size)
            fig.subplots_adjust(top=1.00, bottom=0.00, left=0.00, right=1.00, hspace=0.00, wspace=0.00)
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            show_image = images[i, j, :, :, 0]
            ax.imshow(show_image, cmap='gray')

            fig.tight_layout(pad=0.00, h_pad=0.00, w_pad=0.00)

            fig_img_name = os.path.join(raw_img_path, each_name+'_img')
            plt.savefig(fig_img_name, dpi=300)
            plt.close()

            fig, ax = plt.subplots(1, 1, figsize=fig_size)
            fig.subplots_adjust(top=1.00, bottom=0.00, left=0.00, right=1.00, hspace=0.00, wspace=0.00)
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            show_mask = masks[i, j, :, :, 0]
            show_pred_mask = pred_masks[i, j, :, :, 0]
            show_pred_det = pred_dets.numpy()[i, j, :, :, 0]
            show_pred_det_bin = np.where(show_pred_det <= 0.8, 0.0, 1.0)
            rsz_show_pred_det = skimage.transform.resize(show_pred_det_bin, images.shape[2:4])

            # ax.imshow(show_mask, cmap='gray')
            ax.imshow(show_pred_mask, cmap='gray')
            ax.imshow(rsz_show_pred_det, cmap=plt.cm.hot, alpha=0.5, interpolation='nearest')

            fig.tight_layout(pad=0.00, h_pad=0.00, w_pad=0.00)

            fig_mask_name = os.path.join(pred_mask_det_path, each_name+'_pred')
            plt.savefig(fig_mask_name, dpi=300)
            print(fig_img_name)
            plt.close()





if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()

