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
from data.setup_f import DataSettingV1, INFO_PATH
import models.model_d as model_ref
import models.metric as metric
from runs.cams import *
import tf_utils.tboard as tboard
# For Comparison 2

parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh09_2')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp009')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,4')
main_config.add_argument('--val_name', type=str, dest='val_name', default='5')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model33')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,112,160,208')
main_config.add_argument('--serial', type=int, dest='serial', default=1)
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

d_set = DataSettingV1(df=df, train_type=config.train_name, val_type=config.val_name, data_type=data_type,
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
                                       det_size=config.det_size, use_ic=config.use_ic, use_se=config.use_se,
                                       mtl_mode=config.mtl_mode)

model, cam_model = infer.model, infer.cam_model

cls_loss_fn = metric.focal_loss_sigmoid
det_loss_fn = metric.object_loss

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
    dcs_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'METRIC': pd.Series(),
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
            train_loss, train_cls_loss, train_det_loss = [], [], []
            train_ens1, train_ess1, train_rts1, train_wts1 = [], [], [], []
            train_ens2, train_ess2, train_rts2, train_wts2 = [], [], [], []

            for train_step, (img, mask, ste, det, name) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    cls_prob, det_prob = model([img, mask])

                    cls_loss_batch = cls_loss_fn(ste, cls_prob, cls_alpha, cls_gamma)
                    det_loss_batch = det_loss_fn(det, det_prob, det_alpha, det_gamma, init_w)

                    loss_vars = [cls_loss_batch, det_loss_batch]

                    if config.mtl_mode:
                        train_loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
                    else:
                        cls_loss_batch = config.cls_lambda * cls_loss_batch
                        det_loss_batch = config.det_lambda * det_loss_batch
                        train_loss_batch = cls_loss_batch + det_loss_batch

                grads = tape.gradient(train_loss_batch, model.params)
                optimizer.apply_gradients(zip(grads, model.params))

                train_loss.append(train_loss_batch)
                train_cls_loss.append(cls_loss_batch)
                train_det_loss.append(det_loss_batch)

                if config.each_ste:
                    train_cx.extend(np.reshape(cls_prob.numpy(), (-1,)).tolist())
                    train_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())
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

                sys.stdout.write('Step: {0:>4d}, Loss: {1:.4f} Cls: {2:.4f} Det: {3:.4f} ({4})\r'.
                                 format(train_step, train_loss_batch, cls_loss_batch, det_loss_batch, epoch))

            train_loss_mean = np.mean(train_loss)
            train_cls_loss_mean = np.mean(train_cls_loss)
            train_det_loss_mean = np.mean(train_det_loss)

            train_auc = metric.calculate_auc(train_cy, train_cx)
            train_fom1 = metric.calculate_jafroc(train_ess1, train_rts1, train_wts1)
            train_fom2 = metric.calculate_jafroc(train_ess2, train_rts2, train_wts2)

            train_record = {'Loss': train_loss_mean, 'Cls_Loss': train_cls_loss_mean, 'Det_Loss': train_det_loss_mean,
                            'AUC': train_auc, 'JAFROC1': train_fom1, 'JAFROC2': train_fom2}

            val_cx, val_cy = [], []
            val_loss, val_det_loss, val_cls_loss = [], [], []
            val_ens1, val_ess1, val_rts1, val_wts1 = [], [], [], []
            val_ens2, val_ess2, val_rts2, val_wts2 = [], [], [], []
            val_steps = val_length // config.batch_size + 1

            for val_step, (img, mask, ste, det, name) in enumerate(val_db):
                val_cls_prob, val_det_prob = model([img, mask])

                val_cls_loss_batch = cls_loss_fn(ste, val_cls_prob, cls_alpha, cls_gamma)
                val_det_loss_batch = det_loss_fn(det, val_det_prob, det_alpha, det_gamma, init_w)
                loss_vars = [val_cls_loss_batch, val_det_loss_batch]

                if config.mtl_mode:
                    val_loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
                else:
                    val_cls_loss_batch = config.cls_lambda * val_cls_loss_batch
                    val_det_loss_batch = config.det_lambda * val_det_loss_batch
                    val_loss_batch = val_cls_loss_batch + val_det_loss_batch

                val_loss.append(val_loss_batch)
                val_cls_loss.append(val_cls_loss_batch)
                val_det_loss.append(val_det_loss_batch)

                if config.each_ste:
                    val_cx.extend(np.reshape(val_cls_prob.numpy(), (-1,)).tolist())
                    val_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())
                else:
                    val_cx.extend(val_cls_prob.numpy())
                    val_cy.extend(ste.numpy())

                ens1, ess1, rts1, _, wts1, ens2, ess2, rts2, _, wts2 = jafroc_fn(det, val_det_prob, None, name)

                val_ens1.extend(ens1)
                val_ess1.extend(ess1)
                val_rts1.extend(rts1)
                val_wts1.extend(wts1)

                val_ens2.extend(ens2)
                val_ess2.extend(ess2)
                val_rts2.extend(rts2)
                val_wts2.extend(wts2)

                sys.stdout.write('Evaluation [{0}/{1}], Loss: {2:.4f} Cls: {3:.4f} Det: {4:.4f}\r'.
                                 format(val_step + 1, val_steps, val_loss_batch,
                                        val_cls_loss_batch, val_det_loss_batch))

            val_loss_mean = np.mean(val_loss)
            val_cls_loss_mean = np.mean(val_cls_loss)
            val_det_loss_mean = np.mean(val_det_loss)
            val_auc = metric.calculate_auc(val_cy, val_cx)
            val_fom1 = metric.calculate_jafroc(val_ess1, val_rts1, val_wts1)
            val_fom2 = metric.calculate_jafroc(val_ess2, val_rts2, val_wts2)

            val_record = {'Loss': val_loss_mean, 'Cls_Loss': val_cls_loss_mean, 'Det_Loss': val_det_loss_mean,
                          'AUC': val_auc, 'JAFROC1': val_fom1, 'JAFROC2': val_fom2}

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += ' Time:{0}'.format(time_elapsed.split('.')[0])

            print('Epoch:%s '
                  'Train-Cls:%.4f Det:%.4f FOM1:%.3f FOM2:%.3f AUC:%.3f '
                  'Val-Cls:%.4f Det:%.4f FOM1:%.3f FOM2:%.3f AUC:%.3f' %
                  (epoch,
                   train_cls_loss_mean, train_det_loss_mean, train_fom1, train_fom2, train_auc,
                   val_cls_loss_mean, val_det_loss_mean, val_fom1, val_fom2, val_auc) + log_string)

            tboard.board_record_value(train_summary, train_record, epoch)
            tboard.board_record_value(val_summary, val_record, epoch)

            log_string = ''
            val_metric = np.mean([val_fom1, val_fom2, val_auc])
            perf_per_epoch.append(val_metric)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_metric, val_fom1, val_fom2, val_auc

            elif val_metric > min(dcs_csv['METRIC'].tolist()):
                os.remove(dcs_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                dcs_csv = dcs_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_metric)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_metric, val_fom1, val_fom2, val_auc

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

    show_slice_cam = show_slice_cam_seq if config.each_ste else show_slice_cam_3d

    if config.seq_len == 1:
        show_slice_cam = show_slice_cam_2d

    gen_grad_cam = gen_grad_cam_2d if config.seq_len == 1 else gen_grad_cam_lstm

    dcss, ious, probs, names = [], [], [], []
    print('num_ckpt: ', len(all_ckpt_paths))

    # guided_model = metric.built_guided_model(model, mtl_mode=config.mtl_mode)
    gt_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                          'PROB': pd.Series(), 'DETECTED': pd.Series(dtype=int)})

    pred_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                            'PROB': pd.Series(), 'GT': pd.Series(dtype=int)})

    ckpt_idx = 0
    val_cy = []
    val_ens1, val_ess1, val_rts1, val_scs1, val_wts1 = [], [], [], [], []
    val_ens2, val_ess2, val_rts2, val_scs2, val_wts2 = [], [], [], [], []
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        loss = []
        for step, (img, mask, ste, det, name) in enumerate(val_db):

            with tf.GradientTape() as tape:
                # val_cls_prob, val_det_prob = model([img, mask])
                cam_layers, cls_prob, det_prob, det_gap, det_score = cam_model([img, mask])

            # grad_cams = gen_grad_cam(cam_layers, cls_prob, tape, infer)
            # gb = metric.guided_backprop(guided_model, img, infer.cam_layer_name)
            # guided_grad_cams = gb * grad_cams
            # ggc = metric.deprocess_image(guided_grad_cams)

            cls_loss_batch = cls_loss_fn(ste, cls_prob, cls_alpha, cls_gamma)
            det_loss_batch = det_loss_fn(det, det_prob, det_alpha, det_gamma, init_w)  # serial 1

            loss_vars = [cls_loss_batch, det_loss_batch]

            if config.mtl_mode:
                loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
            else:
                cls_loss_batch = config.cls_lambda * cls_loss_batch
                det_loss_batch = config.det_lambda * det_loss_batch
                loss_batch = cls_loss_batch + det_loss_batch

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

            name_batch, dcs_batch, iou_batch, prob_batch, gt_batch, pred_batch = \
                show_slice_cam(images=img.numpy(), masks=mask.numpy(), pred_masks=mask.numpy(),
                               stes=ste.numpy(), pred_stes=cls_prob.numpy(),
                               dets=det, pred_dets=det_prob, pred_scores=det_score, name=name.numpy(),
                               is_png=config.is_png, save_path=plot_val_path)

            if ckpt_idx == 0:
                name_str = [x.decode() for x in name.numpy()]
                names.extend(name_batch) if config.each_ste else names.extend(name_str)

            probs.extend(prob_batch)

            gt_df = gt_df.append(gt_batch)
            pred_df = pred_df.append(pred_batch)

            sys.stdout.write('{0} Evaluation [{1}/{2}]\r'.
                             format(os.path.basename(ckpt), step, val_length // config.batch_size))

        ckpt_idx += 1

    val_auc = metric.calculate_auc(val_cy, probs)

    val_jafroc1 = metric.calculate_jafroc(val_ess1, val_rts1, val_wts1)
    val_jafroc2 = metric.calculate_jafroc(val_ess2, val_rts2, val_wts2)

    print('\nFinal FOM1: %.3f, FOM2: %.3f AUC: %.3f' % (val_jafroc1, val_jafroc2, val_auc))

    result_name = '_'.join([config.model_name, config.excel_name, config.val_name, serial_str,
                            '%03d' % config.num_weight]) + '.xlsx'

    writer = pd.ExcelWriter(os.path.join(result_path, result_name), engine='xlsxwriter')

    result_csv = pd.DataFrame({'NUMBER': names, 'STE': val_cy, 'PROB': probs})
    jafroc1_df = pd.DataFrame({'NUMBER': val_ens1, 'LABEL': val_ess1, 'PROB': val_rts1,
                               'SCORE': val_scs1, 'WEIGHT': val_wts1})
    jafroc2_df = pd.DataFrame({'NUMBER': val_ens2, 'LABEL': val_ess2, 'PROB': val_rts2,
                               'SCORE': val_scs2, 'WEIGHT': val_wts2})

    result_csv.to_excel(writer, sheet_name='TOTAL', index=False)
    jafroc1_df.to_excel(writer, sheet_name='JAFROC1', index=False)
    jafroc2_df.to_excel(writer, sheet_name='JAFROC2', index=False)
    gt_df.to_excel(writer, sheet_name='GT', index=False)
    pred_df.to_excel(writer, sheet_name='PRED', index=False)

    writer.save()


def show_slice_cam_seq_fig(images, masks, pred_masks, stes, pred_stes, dets, pred_dets, pred_scores,
                           name, is_png=True, num_rows=8, num_cols=1, save_path=plot_val_path):
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

            show_pred_mask = pred_masks[i, j, :, :, 0]
            show_pred_det = pred_dets.numpy()[i, j, :, :, 0]
            mode_pred_det = mode(show_pred_det, axis=None)[0][0]
            show_pred_det_bin = np.where(show_pred_det <= 0.8, 0.0, 1.0)

            # show_pred_det_bin = np.where(show_pred_det <= np.percentile(show_pred_det, 0.9), 0.0, 1.0)
            # show_pred_det_bin = np.where(show_pred_det <= mode_pred_det, 0, show_pred_det)
            rsz_show_pred_det = skimage.transform.resize(show_pred_det_bin, images.shape[2:4])

            ax.imshow(show_pred_mask, cmap='bone')
            ax.imshow(rsz_show_pred_det, cmap=plt.cm.hot, alpha=0.5, interpolation='nearest')

            fig.tight_layout(pad=0.00, h_pad=0.00, w_pad=0.00)

            fig_mask_name = os.path.join(pred_mask_det_path, each_name)
            plt.savefig(fig_mask_name, dpi=300)
            plt.close()


def show_slice_cam_seq(images, masks, pred_masks, stes, pred_stes,
                       dets, pred_dets, pred_scores, name, is_png=True, num_rows=3, num_cols=3,
                       fig_size=(3 * 1.3, 3 * 1.3), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = pred_masks.shape[0]
    names, dcss, ious, lbls, probs = [], [], [], [], []

    gt_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                          'PROB': pd.Series(), 'DETECTED': pd.Series(dtype=int)})

    pred_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                            'PROB': pd.Series(), 'GT': pd.Series(dtype=int)})

    for i in range(batch_size):
        patient_id, start_idx = metric.name_idx(name[i])

        each_dcs = metric.dcs_3d(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()
        each_iou = metric.iou_3d(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()

        dcss.extend(np.repeat(each_dcs, config.seq_len))
        ious.extend(np.repeat(each_iou, config.seq_len).tolist())

        for j in range(config.seq_len):
            show_ste, show_pred_ste = stes[i, j], pred_stes[i, j]

            lbls.append(show_ste)
            probs.append(show_pred_ste)

            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + j + 1)])
            names.append(each_name)

            show_det = dets.numpy()[i, j, :, :, 0]
            show_pred_det = pred_dets.numpy()[i, j, :, :, 0]
            mode_pred_det = mode(show_pred_det, axis=None)[0][0]

            show_pred_det_show = np.where(show_pred_det <= mode_pred_det, 0.0, show_pred_det)
            show_pred_bin_det = np.where(show_pred_det < 0.5, 0.0, 1.0)  # threshold: 0.9
            show_pred_score = pred_scores.numpy()[i, j, :, :, 0]

            show_gt_idx = np.where(show_det == 1)
            gt_idx_h, gt_idx_w = show_gt_idx
            show_gt_prob = show_pred_det[show_gt_idx]
            show_gt_score = show_pred_score[show_gt_idx]
            show_gt_loc = show_pred_bin_det[show_gt_idx]
            num_gt = len(show_gt_prob)

            show_lesion_prob = np.mean(show_gt_prob) if show_ste == 1 else np.max(show_pred_det)
            show_lesion_score = np.mean(show_gt_score) if show_ste == 1 else np.max(show_pred_score)

            show_pred_idx = np.where(show_pred_bin_det == 1)
            pred_idx_h, pred_idx_w = show_pred_idx
            show_pred_prob = show_pred_det[show_pred_idx]
            show_pred_loc = show_det[show_pred_idx]
            num_pred = len(show_pred_prob)

            rsz_show_det = skimage.transform.resize(show_det, images.shape[2:4])
            rsz_show_pred_det = skimage.transform.resize(show_pred_det_show, images.shape[2:4])

            if num_gt > 0:
                gt_each = pd.DataFrame({'PATIENT_ID': np.repeat(each_name, num_gt),
                                        'HEIGHT': gt_idx_h, 'WIDTH': gt_idx_w,
                                        'PROB': show_gt_prob, 'DETECTED': show_gt_loc.astype(int)})
                gt_df = gt_df.append(gt_each)

            if num_pred > 0:
                pred_each = pd.DataFrame({'PATIENT_ID': np.repeat(each_name, num_pred),
                                          'HEIGHT': pred_idx_h, 'WIDTH': pred_idx_w,
                                          'PROB': show_pred_prob, 'GT': show_pred_loc.astype(int)})
                pred_df = pred_df.append(pred_each)

            if is_png:
                show_image = images[i, j, :, :, 0]
                show_mask = masks[i, j, :, :, 0]
                show_pred_mask = pred_masks[i, j, :, :, 0]
                # show_cam_ste = cam_stes[i, j, :, :, 0]  # grad_cam
                # show_ggc_ste = ggc_stes[i, j, :, :, 0]

                fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                sub_title = 'Lesion' if show_ste == 1 else 'Noise'
                sub_color = 'red' if show_ste == 1 else 'blue'

                ax[0, 0].imshow(show_image, cmap='bone')
                ax[0, 0].set_title(each_name, fontsize=6, color='black')

                ax[0, 1].imshow(show_mask, cmap='bone')
                ax[0, 1].set_title('Mask: ' + each_name, fontsize=6, color='navy')

                ax[0, 2].imshow(show_pred_mask, cmap=plt.cm.gray, alpha=0.5, interpolation='nearest')
                ax[0, 2].set_title('DCS: %.3f, IOU: %.3f' % (each_dcs, each_iou), fontsize=6, color='blue')

                ax[1, 0].imshow(show_image, cmap='bone')
                ax[1, 0].imshow(rsz_show_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 0].set_title('Stenosis: %d' % show_ste, fontsize=6, color='darkgreen')

                ax[1, 1].imshow(rsz_show_det, cmap='bone')
                ax[1, 1].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 1].set_title(sub_title + ' Prob.: %.3f' % show_lesion_prob, fontsize=6, color=sub_color)

                ax[1, 2].imshow(show_pred_mask, cmap='bone')
                ax[1, 2].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 2].set_title(sub_title + ' Score: %.3f' % show_lesion_score, fontsize=6, color=sub_color)

                ax[2, 0].imshow(show_pred_det, cmap=plt.cm.gray, alpha=0.5, interpolation='nearest')
                ax[2, 0].set_title('Detection output', fontsize=6, color='darkorange')

                ax[2, 1].imshow(show_image, cmap='bone')
                ax[2, 1].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[2, 1].set_title('Overall Prob.: %.3f' % show_pred_ste, fontsize=6, color=sub_color)

                ax[2, 2].imshow(show_ggc_ste, cmap='bone')
                ax[2, 2].set_title('Guided Grad-CAM', fontsize=6, color='darkgoldenrod')

                fig_name = os.path.join(save_path, each_name)
                plt.savefig(fig_name, bbox_inches='tight')

            plt.clf()

    return names, dcss, ious, probs, gt_df, pred_df


def show_slice_cam_3d(images, masks, pred_masks, stes, pred_stes, pred_scores, cam_stes, ggc_stes,
                      dets, pred_dets, name, is_png=True, num_rows=3, num_cols=3,
                      fig_size=(3 * 1.3, 3 * 1.3), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = pred_masks.shape[0]
    names, dcss, ious, lbls, probs = [], [], [], [], []

    gt_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                          'PROB': pd.Series(), 'DETECTED': pd.Series(dtype=int)})

    pred_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                            'PROB': pd.Series(), 'GT': pd.Series(dtype=int)})

    for i in range(batch_size):
        patient_id, start_idx = metric.name_idx(name[i])

        each_dcs = dcs_metric(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()
        each_iou = iou_metric(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()
        show_ste, show_pred_ste = stes[i], pred_stes[i]

        dcss.extend(each_dcs)
        ious.extend(each_iou)

        lbls.extend([show_ste])
        probs.extend(show_pred_ste)

        names.extend(name[i])

        show_det = dets.numpy()[i, :, :, 0]
        show_pred_det = pred_dets.numpy()[i, :, :, 0]
        mode_pred_det = mode(show_pred_det, axis=None)[0][0]

        show_pred_det_show = np.where(show_pred_det <= mode_pred_det, 0.0, show_pred_det)
        show_pred_bin_det = np.where(show_pred_det < 0.5, 0.0, 1.0)  # threshold: 0.9
        show_pred_score = pred_scores.numpy()[i, :, :, 0]

        show_gt_idx = np.where(show_det == 1)
        gt_idx_h, gt_idx_w = show_gt_idx
        show_gt_prob = show_pred_det[show_gt_idx]
        show_gt_score = show_pred_score[show_gt_idx]

        show_gt_loc = show_pred_bin_det[show_gt_idx]
        num_gt = len(show_gt_prob)

        show_lesion_prob = np.mean(show_gt_prob) if show_ste == 1 else np.max(show_pred_det)
        show_lesion_score = np.mean(show_gt_score) if show_ste == 1 else np.max(show_pred_score)

        show_pred_idx = np.where(show_pred_bin_det == 1)
        pred_idx_h, pred_idx_w = show_pred_idx
        show_pred_prob = show_pred_det[show_pred_idx]
        show_pred_loc = show_det[show_pred_idx]
        num_pred = len(show_pred_prob)

        rsz_show_det = skimage.transform.resize(show_det, images.shape[2:4])
        rsz_show_pred_det = skimage.transform.resize(show_pred_det_show, images.shape[2:4])

        if num_gt > 0:
            gt_each = pd.DataFrame({'PATIENT_ID': np.repeat(name[i].decode(), num_gt),
                                    'HEIGHT': gt_idx_h, 'WIDTH': gt_idx_w,
                                    'PROB': show_gt_prob, 'DETECTED': show_gt_loc.astype(int)})
            gt_df = gt_df.append(gt_each)

        if num_pred > 0:
            pred_each = pd.DataFrame({'PATIENT_ID': np.repeat(name[i].decode(), num_pred),
                                      'HEIGHT': pred_idx_h, 'WIDTH': pred_idx_w,
                                      'PROB': show_pred_prob, 'GT': show_pred_loc.astype(int)})
            pred_df = pred_df.append(pred_each)

        for j in range(seq_len):
            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])

            show_image = images[i, j, :, :, 0]
            show_mask = masks[i, j, :, :, 0]
            show_pred_mask = pred_masks[i, j, :, :, 0]
            show_cam_ste = cam_stes[i, j, :, :, 0]  # grad_cam
            show_ggc_ste = ggc_stes[i, j, :, :, 0]

            if is_png:
                fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                sub_title = 'Lesion' if show_ste == 1 else 'Noise'
                sub_color = 'red' if show_ste == 1 else 'blue'

                ax[0, 0].imshow(show_image, cmap='bone')
                ax[0, 0].set_title(each_name, fontsize=6, color='black')

                ax[0, 1].imshow(show_mask, cmap='bone')
                ax[0, 1].set_title('Mask: ' + each_name, fontsize=6, color='navy')

                ax[0, 2].imshow(show_pred_mask, cmap=plt.cm.gray, alpha=0.5, interpolation='nearest')
                ax[0, 2].set_title('DCS: %.3f, IOU: %.3f' % (each_dcs, each_iou), fontsize=6, color='blue')

                ax[1, 0].imshow(show_image, cmap='bone')
                ax[1, 0].imshow(rsz_show_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 0].set_title('Stenosis: %d' % show_ste, fontsize=6, color='darkgreen')

                ax[1, 1].imshow(rsz_show_det, cmap='bone')
                ax[1, 1].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 1].set_title(sub_title + ' Prob.: %.3f' % show_lesion_prob, fontsize=6, color=sub_color)

                ax[1, 2].imshow(show_pred_mask, cmap='bone')
                ax[1, 2].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 2].set_title(sub_title + ' Score: %.3f' % show_lesion_score, fontsize=6, color=sub_color)

                ax[2, 0].imshow(show_pred_det, cmap=plt.cm.gray, alpha=0.5, interpolation='nearest')
                ax[2, 0].set_title('Detection output', fontsize=6, color='darkorange')

                ax[2, 1].imshow(show_image, cmap='bone')
                ax[2, 1].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[2, 1].set_title('Overall Prob.: %.3f' % show_pred_ste, fontsize=6, color=sub_color)

                ax[2, 2].imshow(show_ggc_ste, cmap='bone')
                ax[2, 2].set_title('Guided Grad-CAM', fontsize=6, color='darkgoldenrod')

                fig_name = os.path.join(save_path, each_name)
                plt.savefig(fig_name, bbox_inches='tight')

            plt.clf()

    return names, dcss, ious, probs, gt_df, pred_df


def show_slice_cam_2d(images, masks, pred_masks, stes, pred_stes, pred_scores, cam_stes, ggc_stes,
                      dets, pred_dets, name, is_png=True, num_rows=3, num_cols=3,
                      fig_size=(3 * 1.3, 3 * 1.3), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = pred_masks.shape[0]
    names, dcss, ious, lbls, probs = [], [], [], [], []

    gt_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                          'PROB': pd.Series(), 'DETECTED': pd.Series(dtype=int)})

    pred_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                            'PROB': pd.Series(), 'GT': pd.Series(dtype=int)})

    if True:
        case_path = os.path.join(config.data_path, config.exp_name, 'case', '2D U-Net', 'plot2d')
        if not os.path.exists(case_path):
            os.makedirs(case_path)

    for i in range(batch_size):
        each_dcs = dcs_metric(masks[i, :, :, :], pred_masks[i, :, :, :]).numpy()[0]
        each_iou = iou_metric(masks[i, :, :, :], pred_masks[i, :, :, :]).numpy()[0]

        name_split = re.split('_', name[i].decode())
        each_name = '_'.join([name_split[0], name_split[-1]])

        case_patient_path = os.path.join(case_path, name_split[0])
        if not os.path.exists(case_patient_path):
            os.makedirs(case_patient_path)

        show_ste, show_pred_ste = stes[i], pred_stes[i]

        dcss.append(each_dcs)
        ious.append(each_iou)
        names.append(each_name)

        lbls.append(show_ste)
        probs.append(show_pred_ste)

        show_det = dets.numpy()[i, :, :, 0]
        show_pred_det = pred_dets.numpy()[i, :, :, 0]
        mode_pred_det = mode(show_pred_det, axis=None)[0][0]

        show_pred_det_show = np.where(show_pred_det <= mode_pred_det, 0.0, show_pred_det)
        show_pred_score = pred_scores.numpy()[i, :, :, 0]
        show_pred_bin_det = np.where(show_pred_det < 0.5, 0.0, 1.0)  # threshold: 0.9

        show_gt_idx = np.where(show_det == 1)
        gt_idx_h, gt_idx_w = show_gt_idx
        show_gt_prob = show_pred_det[show_gt_idx]
        show_gt_score = show_pred_score[show_gt_idx]
        show_gt_loc = show_pred_bin_det[show_gt_idx]
        num_gt = len(show_gt_prob)

        show_lesion_prob = np.mean(show_gt_prob) if show_ste == 1 else np.max(show_pred_det)
        show_lesion_score = np.mean(show_gt_score) if show_ste == 1 else np.max(show_pred_score)

        show_pred_idx = np.where(show_pred_bin_det == 1)
        pred_idx_h, pred_idx_w = show_pred_idx
        show_pred_prob = show_pred_det[show_pred_idx]
        show_pred_loc = show_det[show_pred_idx]
        num_pred = len(show_pred_prob)

        if num_gt > 0:
            gt_each = pd.DataFrame({'PATIENT_ID': np.repeat(each_name, num_gt),
                                    'HEIGHT': gt_idx_h, 'WIDTH': gt_idx_w,
                                    'PROB': show_gt_prob, 'DETECTED': show_gt_loc.astype(int)})
            gt_df = gt_df.append(gt_each)

        if num_pred > 0:
            pred_each = pd.DataFrame({'PATIENT_ID': np.repeat(each_name, num_pred),
                                      'HEIGHT': pred_idx_h, 'WIDTH': pred_idx_w,
                                      'PROB': show_pred_prob, 'GT': show_pred_loc.astype(int)})
            pred_df = pred_df.append(pred_each)

        if is_png:
            show_image = images[i, :, :, 0]
            show_mask = masks[i, :, :, 0]
            show_pred_mask = pred_masks[i, :, :, 0]
            show_cam_ste = cam_stes[i, :, :, 0]
            show_ggc_ste = ggc_stes[i, :, :, 0]

            rsz_show_det = skimage.transform.resize(show_det, images.shape[1:3])
            rsz_show_pred_det = skimage.transform.resize(show_pred_det_show, images.shape[1:3])

            if True:

                fig, ax = plt.subplots(1, 3, figsize=(3, 1))
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                ax[0].imshow(show_image, cmap='bone')
                ax[0].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                # ax[0].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')

                ax[1].imshow(show_pred_mask, cmap=plt.cm.gray, alpha=0.5, interpolation='nearest')
                ax[1].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')

                ax[2].imshow(rsz_show_det, cmap='bone')
                ax[2].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')

                fig.tight_layout()
                pred_name = os.path.join(case_patient_path, each_name)

                plt.savefig(pred_name, bbox_inches='tight', dpi=300)

            if False:
                fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                sub_title = 'Lesion' if show_ste == 1 else 'Noise'
                sub_color = 'red' if show_ste == 1 else 'blue'

                ax[0, 0].imshow(show_image, cmap='bone')
                ax[0, 0].set_title(each_name, fontsize=6, color='black')

                ax[0, 1].imshow(show_mask, cmap='bone')
                ax[0, 1].set_title('Mask: ' + each_name, fontsize=6, color='navy')

                ax[0, 2].imshow(show_pred_mask, cmap=plt.cm.gray, alpha=0.5, interpolation='nearest')
                ax[0, 2].set_title('DCS: %.3f, IOU: %.3f' % (each_dcs, each_iou), fontsize=6, color='blue')

                ax[1, 0].imshow(show_image, cmap='bone')
                ax[1, 0].imshow(rsz_show_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 0].set_title('Stenosis: %d' % show_ste, fontsize=6, color='darkgreen')

                ax[1, 1].imshow(rsz_show_det, cmap='bone')
                ax[1, 1].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 1].set_title(sub_title + ' Prob.: %.3f' % show_lesion_prob, fontsize=6, color=sub_color)

                ax[1, 2].imshow(show_pred_mask, cmap='bone')
                ax[1, 2].imshow(rsz_show_pred_det, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[1, 2].set_title(sub_title + ' Score: %.3f' % show_lesion_score, fontsize=6, color=sub_color)

                ax[2, 0].imshow(show_pred_det, cmap=plt.cm.gray, alpha=0.5, interpolation='nearest')
                ax[2, 0].set_title('Detection output', fontsize=6, color='darkorange')

                ax[2, 1].imshow(show_image, cmap='bone')
                ax[2, 1].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[2, 1].set_title('Overall Prob.: %.3f' % show_pred_ste, fontsize=6, color=sub_color)

                ax[2, 2].imshow(show_ggc_ste, cmap='bone')
                ax[2, 2].set_title('Guided Grad-CAM', fontsize=6, color='darkgoldenrod')

                fig_name = os.path.join(save_path, each_name)
                plt.savefig(fig_name, bbox_inches='tight')

    return names, dcss, ious, probs, gt_df, pred_df


if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()

