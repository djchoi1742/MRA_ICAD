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
main_config.add_argument('--serial', type=str, dest='serial', default='1,2,3,4')
main_config.add_argument('--esb_serial', type=int, dest='esb_serial', default=1)
main_config.add_argument('--image_size', type=int, dest='image_size', default=256)
main_config.add_argument('--radius', type=int, dest='radius', default=80)
main_config.add_argument('--det_size', type=int, dest='det_size', default=16)
main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
main_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)  # only use validation
main_config.add_argument('--batch_size', type=int, dest='batch_size', default=2)
main_config.add_argument('--seq_len', type=int, dest='seq_len', default=8)
main_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)
main_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=False)
main_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=True)
main_config.add_argument('--one_hot', type=lambda x: x.title() in str(True), dest='one_hot', default=False)
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


# try:
#     serial_list = []
    # is_ensemble = False
# except:
    # is_ensemble = True

serial_list = re.split(',', config.serial)
serial_str = 'e' + '%03d' % int(config.esb_serial)
is_ensemble = True if len(serial_list) > 1 else False


serial_str = 'e%03d' % int(config.esb_serial)
result_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%s' % serial_str)
plot_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'plot-%s' % serial_str)
plot_val_path = os.path.join(plot_path, '_'.join([config.excel_name, config.val_name]))


if not os.path.exists(result_path): os.makedirs(result_path)
if not os.path.exists(plot_path): os.makedirs(plot_path)
if not os.path.exists(plot_val_path): os.makedirs(plot_val_path)


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
                      train=False, seq_len=seq_len, seq_interval=seq_interval,
                      only_ste=config.only_ste, each_ste=config.each_ste,
                      image_size=img_size, radius=config.radius, det_size=config.det_size, one_hot=config.one_hot)

val_db = d_set.val.batch(config.batch_size)
val_length = d_set.val.cardinality().numpy()
print('val length: ', val_length)

input_size = [seq_len, img_size, img_size, img_c] if seq_len != 1 else [img_size, img_size, img_c]

infer_name = config.model_name
infer = getattr(model_ref, infer_name)(input_size=input_size, f_num=f_num, is_training=False,
                                       det_size=config.det_size, use_ic=False, use_se=False, mtl_mode=config.mtl_mode)


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


def restore_weight(data_path, exp_name, model_name, trial_serial, num_weight):
    weight_auc_path = os.path.join(data_path, exp_name, model_name, 'result-%03d' % trial_serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([exp_name, model_name,
                                                                         '%03d' % trial_serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('METRIC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:num_weight])
    return all_ckpt_paths


def validation():
    if is_ensemble:
        all_ckpt_paths = []
        for idx in serial_list:
            each_ckpt_paths = restore_weight(config.data_path, config.exp_name, config.model_name,
                                             int(idx), config.num_weight)
            all_ckpt_paths = all_ckpt_paths + each_ckpt_paths
    else:
        all_ckpt_paths = restore_weight(config.data_path, config.exp_name, config.model_name,
                                        int(config.serial), config.num_weight)

    show_slice_cam = show_slice_cam_seq
    gen_grad_cam = gen_grad_cam_lstm

    num_ckpt = len(all_ckpt_paths)
    print('num_ckpt: ', num_ckpt)

    info_log = {
        'esb_ckpt': all_ckpt_paths
    }

    with open(os.path.join(result_path, 'esb.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    guided_model = metric.built_guided_model(model, mtl_mode=config.mtl_mode)
    gt_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                          'PROB': pd.Series(), 'DETECTED': pd.Series(dtype=int)})

    pred_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'HEIGHT': pd.Series(), 'WIDTH': pd.Series(),
                            'PROB': pd.Series(), 'GT': pd.Series(dtype=int)})

    dcss, ious, probs, names = [], [], [], []

    imgs = np.zeros([val_length, config.seq_len, img_size, img_size, img_c])
    masks = np.zeros([val_length, config.seq_len, img_size, img_size, img_c])
    cls_labels = np.zeros([val_length, config.seq_len])
    det_labels = np.zeros([val_length, config.seq_len, config.det_size, config.det_size, img_c])
    img_names = np.zeros([val_length, ], dtype=object)

    seg_probs = np.zeros([num_ckpt, val_length, config.seq_len, img_size, img_size, img_c])
    cls_probs = np.zeros([num_ckpt, val_length, config.seq_len])
    det_probs = np.zeros([num_ckpt, val_length, config.seq_len, config.det_size, config.det_size, img_c])
    det_scores = np.zeros([num_ckpt, val_length, config.seq_len, config.det_size, config.det_size, img_c])

    # step_range = config.batch_size * config.seq_len
    for ckpt_idx, ckpt_path in enumerate(all_ckpt_paths):
        model.load_weights(ckpt_path)

        loss = []
        for step, (img, mask, ste, det, name) in enumerate(val_db):
            with tf.GradientTape() as tape:
                cam_layers, seg_prob, cls_prob, det_prob, det_gap, det_score = cam_model(img)

            grad_cams = gen_grad_cam(cam_layers, cls_prob, tape, infer)
            gb = metric.guided_backprop(guided_model, img, infer.cam_layer_name)
            # guided_grad_cams = gb * grad_cams
            # ggc = metric.deprocess_image(guided_grad_cams)

            seg_loss_batch = seg_loss_fn(mask, seg_prob)
            cls_loss_batch = cls_loss_fn(ste, cls_prob, cls_alpha, cls_gamma)
            det_loss_batch = det_loss_fn(det, det_prob, det_alpha, det_gamma, init_w)  # serial 1
            loss_vars = [seg_loss_batch, cls_loss_batch, det_loss_batch]

            seg_probs[ckpt_idx, step * config.batch_size:step * config.batch_size + len(ste)] = seg_prob
            cls_probs[ckpt_idx, step * config.batch_size:step * config.batch_size + len(ste)] = cls_prob
            det_probs[ckpt_idx, step * config.batch_size:step * config.batch_size + len(ste)] = det_prob
            det_scores[ckpt_idx, step * config.batch_size:step * config.batch_size + len(ste)] = det_score

            if config.mtl_mode:
                loss_batch = metric.multi_task_loss(loss_vars, infer.log_vars)
            else:
                cls_loss_batch = config.cls_lambda * cls_loss_batch
                det_loss_batch = config.det_lambda * det_loss_batch
                loss_batch = seg_loss_batch + cls_loss_batch + det_loss_batch

            loss.append(loss_batch)

            if False:
                name_batch, dcs_batch, iou_batch, prob_batch, gt_batch, pred_batch = \
                    show_slice_cam(images=img.numpy(), masks=mask.numpy(), pred_masks=seg_prob.numpy(),
                                   stes=ste.numpy(), pred_stes=cls_prob.numpy(), cam_stes=grad_cams, ggc_stes=ggc,
                                   dets=det, pred_dets=det_prob, pred_scores=det_score, name=name.numpy(),
                                   is_png=config.is_png, save_path=plot_val_path)

            if ckpt_idx == 0:
                # name_str = [x.decode() for x in name.numpy()]
                imgs[step * config.batch_size:step * config.batch_size + len(ste)] = img
                masks[step * config.batch_size:step * config.batch_size + len(ste)] = mask
                cls_labels[step * config.batch_size:step * config.batch_size + len(ste)] = ste
                det_labels[step * config.batch_size:step * config.batch_size + len(ste)] = det
                img_names[step * config.batch_size:step * config.batch_size + len(ste)] = name

                name_batch = show_slice_name(seg_prob.numpy(), name.numpy())
                names.extend(name_batch)

            sys.stdout.write('{0} Evaluation [{1}/{2}] \r'.
                             format(ckpt_path, step, val_length // config.batch_size))

    seg_probs_mean, cls_probs_mean = np.mean(seg_probs, axis=0), np.mean(cls_probs, axis=0)
    det_probs_mean, det_scores_mean = np.mean(det_probs, axis=0), np.mean(det_scores, axis=0)

    val_ens1, val_ess1, val_rts1, val_scs1, val_wts1, val_ens2, val_ess2, val_rts2, val_scs2, val_wts2 = \
        metric.set_jafroc_seq_np(det_labels, det_probs_mean, det_scores_mean, img_names)

    cls_labels_reshape, cls_probs_reshape = np.reshape(cls_labels, [-1, ]), np.reshape(cls_probs_mean, [-1, ])

    val_auc = metric.calculate_auc(cls_labels_reshape, cls_probs_reshape)
    val_jafroc1 = metric.calculate_jafroc(val_ess1, val_rts1, val_wts1)
    val_jafroc2 = metric.calculate_jafroc(val_ess2, val_rts2, val_wts2)

    val_dcs, val_iou = np.mean(dcss), np.mean(ious)

    print('\nFinal DCS: %.3f, IOU: %.3f, FOM1: %.3f, FOM2: %.3f AUC: %.3f' %
          (np.mean(val_dcs), np.mean(val_iou), val_jafroc1, val_jafroc2, val_auc))

    result_name = '_'.join([config.model_name, config.excel_name, config.val_name, serial_str,
                            '%03d' % config.num_weight]) + '.xlsx'

    writer = pd.ExcelWriter(os.path.join(result_path, result_name), engine='xlsxwriter')

    result_csv = pd.DataFrame({'NUMBER': names, 'STE': cls_labels_reshape, 'PROB': cls_probs_reshape})
    jafroc1_df = pd.DataFrame({'NUMBER': val_ens1, 'LABEL': val_ess1, 'PROB': val_rts1,
                               'SCORE': val_scs1, 'WEIGHT': val_wts1})

    jafroc2_df = pd.DataFrame({'NUMBER': val_ens2, 'LABEL': val_ess2, 'PROB': val_rts2,
                               'SCORE': val_scs2, 'WEIGHT': val_wts2})

    result_csv.to_excel(writer, sheet_name='TOTAL', index=False)
    jafroc1_df.to_excel(writer, sheet_name='JAFROC1', index=False)
    jafroc2_df.to_excel(writer, sheet_name='JAFROC2', index=False)
    writer.save()


def show_slice_name(pred_masks, name):
    batch_size = pred_masks.shape[0]
    names = []
    for i in range(batch_size):
        patient_id, start_idx = metric.name_idx(name[i])

        for j in range(config.seq_len):
            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + j + 1)])
            names.append(each_name)

    return names


def show_slice_cam_seq(images, masks, pred_masks, stes, pred_stes, cam_stes, ggc_stes,
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
            # show_pred_det2 = show_pred_det[1:16, :]
            # rsz_show_pred_det = skimage.transform.resize(show_pred_det2,  show_pred_det2.shape*np.array(16))

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
                show_cam_ste = cam_stes[i, j, :, :, 0]  # grad_cam
                show_ggc_ste = ggc_stes[i, j, :, :, 0]

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


if __name__ == '__main__':
    validation()
