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
import models.model_d as model_ref
import models.metric as metric
from runs.cams import *
import tf_utils.tboard as tboard


parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh10')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp010')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,4')
main_config.add_argument('--val_name', type=str, dest='val_name', default='5')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model28')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,112,160,208')
main_config.add_argument('--serial', type=int, dest='serial', default=7)
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
main_config.add_argument('--det_cutoff', type=float, dest='det_cutoff', default=0.25)

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
                                       det_size=config.det_size, seq_len=config.seq_len,
                                       use_ic=config.use_ic, use_se=config.use_se, mtl_mode=config.mtl_mode)


model, cam_model = infer.model, infer.cam_model

seg_loss_fn = metric.weighted_dice_score_loss
cls_loss_fn = metric.focal_loss_sigmoid
det_loss_fn = metric.object_loss

dcs_metric = metric.dcs_3d
iou_metric = metric.iou_3d
jafroc_fn = metric.set_jafroc_seq

cls_alpha, cls_gamma = config.cls_alpha, config.cls_gamma
det_alpha, det_gamma, init_w = config.det_alpha, config.det_gamma, config.init_w


def validation():
    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('METRIC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][(config.num_weight - 1):config.num_weight])

    dcss, ious, probs, names = [], [], [], []
    print('num_ckpt: ', len(all_ckpt_paths))

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
                cam_layers, seg_prob, cls_prob, det_prob, det_gap, det_score = cam_model(img)

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

            val_cy.extend(np.reshape(ste.numpy(), (-1,)).tolist())

            ens1, ess1, rts1, scs1, wts1, ens2, ess2, rts2, scs2, wts2 = \
                metric.set_jafroc_seq(det, det_prob, det_score, name)

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
                slice_lesion_prob(name=name.numpy(), masks=mask.numpy(), pred_masks=seg_prob.numpy(),
                                  stes=ste.numpy(), pred_stes=cls_prob.numpy(),
                                  dets=det, pred_dets=det_prob, det_cutoff=config.det_cutoff)

            if ckpt_idx == 0:
                name_str = [x.decode() for x in name.numpy()]
                names.extend(name_batch) if config.each_ste else names.extend(name_str)

            dcss.extend(dcs_batch)
            ious.extend(iou_batch)
            probs.extend(prob_batch)

            gt_df = gt_df.append(gt_batch)
            pred_df = pred_df.append(pred_batch)

            sys.stdout.write('{0} Evaluation [{1}/{2}], DCS:{3:.4f}, IOU:{4:.4f}\r'.
                             format(os.path.basename(ckpt), step, val_length // config.batch_size,
                                    np.mean(dcs_batch), np.mean(iou_batch)))

        ckpt_idx += 1

    val_dcs, val_iou = np.mean(dcss), np.mean(ious)
    val_auc = metric.calculate_auc(val_cy, probs)

    val_jafroc1 = metric.calculate_jafroc(val_ess1, val_rts1, val_wts1)
    val_jafroc2 = metric.calculate_jafroc(val_ess2, val_rts2, val_wts2)

    print('\nFinal DCS: %.3f, IOU: %.3f, FOM1: %.3f, FOM2: %.3f AUC: %.3f' %
          (np.mean(val_dcs), np.mean(val_iou), val_jafroc1, val_jafroc2, val_auc))

    result_name = '_'.join([config.model_name, config.excel_name, config.val_name, serial_str,
                            '%03d' % config.num_weight, '%.2f' % config.det_cutoff]) + '.xlsx'

    writer = pd.ExcelWriter(os.path.join(result_path, result_name), engine='xlsxwriter')

    result_csv = pd.DataFrame({'NUMBER': names, 'DCS': dcss, 'IOU': ious, 'STE': val_cy, 'PROB': probs})

    jafroc1_df = pd.DataFrame({'NUMBER': val_ens1, 'LABEL': val_ess1, 'PROB': val_rts1, 'WEIGHT': val_wts1})
    jafroc2_df = pd.DataFrame({'NUMBER': val_ens2, 'LABEL': val_ess2, 'PROB': val_rts2, 'WEIGHT': val_wts2})

    result_csv.to_excel(writer, sheet_name='TOTAL', index=False)
    jafroc1_df.to_excel(writer, sheet_name='JAFROC1', index=False)
    jafroc2_df.to_excel(writer, sheet_name='JAFROC2', index=False)
    gt_df.to_excel(writer, sheet_name='GT', index=False)
    pred_df.to_excel(writer, sheet_name='PRED', index=False)

    writer.save()


def slice_lesion_prob(name, masks, pred_masks, stes, pred_stes, dets, pred_dets, det_cutoff):
    batch_size = stes.shape[0]
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

            show_pred_bin_det = np.where(show_pred_det < det_cutoff, 0.0, 1.0)  # threshold: 0.25 or 0.40

            show_gt_idx = np.where(show_det == 1)
            gt_idx_h, gt_idx_w = show_gt_idx
            show_gt_prob = show_pred_det[show_gt_idx]
            show_gt_loc = show_pred_bin_det[show_gt_idx]
            num_gt = len(show_gt_prob)

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

            plt.close()

    return names, dcss, ious, probs, gt_df, pred_df


if __name__ == '__main__':
    print('Clinical Validation')
    validation()

