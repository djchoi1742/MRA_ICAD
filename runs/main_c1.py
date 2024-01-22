import os, sys, logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse, json
import re, datetime
import skimage.transform
import nibabel as nib

sys.path.append('/workspace/bitbucket/MRA')
from data.setup_d import DataSettingV1, INFO_PATH
import models.model_c as model_ref
import models.metric as metric
import tf_utils.tboard as tboard
from plots.fig import select_train_groups, EachDataSetting


parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh09')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp009')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,4')
main_config.add_argument('--val_name', type=str, dest='val_name', default='5')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model03')
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
main_config.add_argument('--seq_len', type=int, dest='seq_len', default=8)
main_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)
main_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=False)
main_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=True)
main_config.add_argument('--one_hot', type=lambda x: x.title() in str(True), dest='one_hot', default=False)
main_config.add_argument('--is_png', type=lambda x: x.title() in str(True), dest='is_png', default=False)

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
infer = getattr(model_ref, infer_name)(input_size=input_size, f_num=f_num, is_training=config.train)


model = infer.model
seg_loss_fn = metric.weighted_dice_score_loss

dcs_metric = metric.dcs_2d if config.seq_len == 1 else metric.dcs_3d
iou_metric = metric.iou_2d if config.seq_len == 1 else metric.iou_3d


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
        'EPOCH': config.epoch
    }

    with open(os.path.join(result_path, '.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    train_summary, val_summary = tboard.tensorboard_create(log_path)
    result_name = '_'.join([config.exp_name, config.model_name, serial_str]) + '.csv'
    dcs_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'DCS': pd.Series(), 'IOU': pd.Series()})

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
            train_sx, train_sy = [], []
            train_loss, train_dcs, train_iou = [], [], []
            for train_step, (img, mask, _, _, name) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    seg_prob = model(img)

                    train_loss_batch = seg_loss_fn(mask, seg_prob)

                grads = tape.gradient(train_loss_batch, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.append(train_loss_batch)

                train_sx.extend(seg_prob.numpy())
                train_sy.extend(mask.numpy())

                train_dcs_batch = dcs_metric(mask, seg_prob)
                train_dcs.extend(train_dcs_batch)
                dcs_batch_mean = np.mean(train_dcs_batch)

                train_iou_batch = iou_metric(mask, seg_prob)
                train_iou.extend(train_iou_batch)
                iou_batch_mean = np.mean(train_iou_batch)

                sys.stdout.write('Step: {0:>4d}, Loss: {1:.4f} DCS: {2:.4f} IOU: {3:.4f} ({4})\r'.
                                 format(train_step, train_loss_batch, dcs_batch_mean, iou_batch_mean, epoch))

            train_loss_mean = np.mean(train_loss)
            train_dcs_mean, train_iou_mean = np.mean(train_dcs), np.mean(train_iou)

            train_record = {'Loss': train_loss_mean, 'DCS': train_dcs_mean, 'IOU': train_iou_mean}

            val_sx, val_sy = [], []
            val_loss, val_dcs, val_iou = [], [], []
            val_steps = val_length // config.batch_size + 1

            for val_step, (img, mask, _, _, name) in enumerate(val_db):
                val_seg_prob = model(img)

                val_loss_batch = seg_loss_fn(mask, val_seg_prob)
                val_loss.append(val_loss_batch)

                val_sx.extend(val_seg_prob.numpy())
                val_sy.extend(mask.numpy())

                val_dcs_batch = dcs_metric(mask, val_seg_prob)
                val_dcs.extend(val_dcs_batch)
                dcs_batch_mean = np.mean(val_dcs_batch)

                val_iou_batch = iou_metric(mask, val_seg_prob)
                val_iou.extend(val_iou_batch)
                iou_batch_mean = np.mean(val_iou_batch)

                sys.stdout.write('Evaluation [{0}/{1}], Loss: {2:.4f} DCS: {3:.4f} IOU: {4:.4f}\r'.
                                 format(val_step + 1, val_steps, val_loss_batch, dcs_batch_mean, iou_batch_mean))

            val_loss_mean = np.mean(val_loss)
            val_dcs_mean, val_iou_mean = np.mean(val_dcs), np.mean(val_iou)

            val_record = {'Loss': val_loss_mean, 'DCS': val_dcs_mean, 'IOU': val_iou_mean}

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += ' Time Elapsed: {0}'.format(time_elapsed.split('.')[0])

            print('Epoch:%s Train-Loss:%.4f DCS:%.4f IOU:%.4f Val-Loss:%.4f DCS:%.4f IOU:%.4f' %
                  (epoch, train_loss_mean, train_dcs_mean, train_iou_mean,
                   val_loss_mean, val_dcs_mean, val_iou_mean) + log_string)

            tboard.board_record_value(train_summary, train_record, epoch)
            tboard.board_record_value(val_summary, val_record, epoch)

            log_string = ''

            perf_per_epoch.append(val_dcs_mean)
            weight_path = os.path.join(log_path, 'ckpt-' + '%03d' % epoch + '.hdf5')

            if epoch < config.max_keep + 1:
                max_current_step.append(epoch)
                max_perf_per_epoch.append(val_dcs_mean)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_dcs_mean, val_iou_mean

            elif val_dcs_mean > min(dcs_csv['DCS'].tolist()):
                os.remove(dcs_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                dcs_csv = dcs_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_dcs_mean)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_dcs_mean, val_iou_mean

            dcs_csv.to_csv(os.path.join(result_path, result_name))

            if epoch == config.epoch:
                break

    except KeyboardInterrupt:
        print('Result saved')
        dcs_csv.to_csv(os.path.join(result_path, result_name))


def validation_save():

    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name,
                                   'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('DCS', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])
    ckpt = all_ckpt_paths[0]

    seq_len, img_size, img_c= config.seq_len, config.image_size, 1

    df = pd.read_excel(os.path.join(INFO_PATH, config.excel_name) + '.xlsx')
    if config.only_ste:
        dfs = df[df['LABEL'] == 1]
    else:
        dfs = df

    group_values = dfs['GROUP'].tolist()
    type_groups = [*map(int, re.split(',', config.val_name))]
    dfs = dfs[[*map(lambda x: select_train_groups(x, type_groups), group_values)]]
    dfs['PATIENT_ID'] = dfs['PATIENT_ID'].astype(str)
    dfs_id = dfs['PATIENT_ID'].tolist()

    if 'snubh' in config.excel_name:
        d_type = 'clinical'
    elif 'cusmh' in config.excel_name:
        d_type = 'external'
    else:
        raise ValueError('Invalid data type')

    pred_mask_path = os.path.join('/workspace/MRA/RAW/', d_type, 'stenosis_pred_mask')

    model.load_weights(ckpt)
    for patient_idx in dfs_id:
        case = EachDataSetting(config.excel_name, patient_id=patient_idx, data_type=d_type, seq_len=config.seq_len)

        num_slices = seq_len * case.val.cardinality().numpy()

        img_3d = np.zeros((num_slices, img_size, img_size, 1))
        pred_mask_3d = np.zeros((num_slices, img_size, img_size, 1))

        val_db = case.val.batch(config.batch_size)

        for step, (img, mask, _, _, name) in enumerate(val_db):
            val_seg_prob = model(img)
            img_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = img.numpy()[0, :, :, :, :]
            pred_mask_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = val_seg_prob.numpy()[0, :, :, :, :]

        # img_3d = np.transpose(np.squeeze(img_3d), (1, 2, 0))
        # img_nii = nib.Nifti1Image(img_3d, None, nib.Nifti1Header())
        # img_name = '_'.join([str(patient_idx), 'img']) + '.nii.gz'
        # nib.save(img_nii, os.path.join(pred_mask_path, img_name))

        pred_mask_3d = np.transpose(np.squeeze(pred_mask_3d), (1, 2, 0))
        pred_mask_nii = nib.Nifti1Image(pred_mask_3d, None, nib.Nifti1Header())
        pred_mask_name = '_'.join([str(patient_idx), 'pred', 'mask'])+'.nii.gz'

        nib.save(pred_mask_nii, os.path.join(pred_mask_path, pred_mask_name))
        print(os.path.join(pred_mask_path, pred_mask_name))


def validation():
    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name,
                                   'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('DCS', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])

    slice_cam = show_slice_cam_3d if config.seq_len != 1 else show_slice_cam_2d

    val_loss = np.zeros([len(all_ckpt_paths), val_length])
    val_dcs = np.zeros([len(all_ckpt_paths), val_length])
    val_iou = np.zeros([len(all_ckpt_paths), val_length])
    val_cy, val_name = [], []

    print('num_ckpt: ', len(all_ckpt_paths))

    ckpt_idx = 0
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        val_name = []
        for step, (img, mask, _, _, name) in enumerate(val_db):
            val_seg_prob = model(img)
            val_loss_batch = seg_loss_fn(mask, val_seg_prob, each=True)
            import pdb; pdb.set_trace()
            val_name_batch, val_dcs_batch, val_iou_batch = \
                slice_cam(img.numpy(), mask.numpy(), val_seg_prob.numpy(), name.numpy(),
                          is_png=config.is_png, save_path=plot_val_path)

            # if config.model_name == 'Model09':
            #     ste = tf.where(tf.reduce_sum(ste, axis=1) >= 1, 1, 0)

            if ckpt_idx == 0:
                name_str = [x.decode() for x in name.numpy()]
                val_name.extend(val_name_batch) if config.seq_len == 1 else val_name.extend(name_str)

            cnt_range = config.batch_size

            val_loss[ckpt_idx, step * cnt_range:step * cnt_range + len(val_dcs_batch)] = val_loss_batch
            val_dcs[ckpt_idx, step * cnt_range:step * cnt_range + len(val_dcs_batch)] = val_dcs_batch
            val_iou[ckpt_idx, step * cnt_range:step * cnt_range + len(val_iou_batch)] = val_iou_batch

            sys.stdout.write('{0} Evaluation [{1}/{2}], DCS:{3:.4f}, IOU:{4:.4f}\r'.
                             format(os.path.basename(ckpt), step, val_length // config.batch_size,
                                    np.mean(val_dcs_batch), np.mean(val_iou_batch)))

        ckpt_idx += 1

    val_loss, val_dcs, val_iou = np.mean(val_loss, axis=0), np.mean(val_dcs, axis=0), np.mean(val_iou, axis=0)

    result_csv = pd.DataFrame({'NUMBER': val_name, 'LOSS': val_loss, 'DCS': val_dcs, 'IOU': val_iou})

    result_name = '_'.join([config.model_name, config.excel_name, config.val_name,
                            serial_str, '%03d' % config.num_weight]) + '.csv'
    result_csv.to_csv(os.path.join(result_path, result_name), index=False)

    print('\nFinal Loss: %.3f DCS: %.3f, IOU: %.3f' % (np.mean(val_loss), np.mean(val_dcs), np.mean(val_iou)))


def gen_grad_cam(cam_layer, seq_len, loss, tape):
    grads = tape.gradient(loss, cam_layer)

    weights = tf.reduce_mean(grads, axis=(1, 2, 3))
    cam = np.zeros(cam_layer.shape[0:4], dtype=np.float32)
    img_size = config.image_size

    batch_size = cam_layer.shape[0]
    heatmaps = np.zeros((batch_size, seq_len, img_size, img_size), dtype=np.float32)

    for batch in range(batch_size):  # batch size
        for index, w in enumerate(weights[batch]):  # each weights of batch
            cam[batch, :, :, :] += w * cam_layer[batch, :, :, :, index]
        cam_resize = skimage.transform.resize(cam[batch, :, :, :], (seq_len, img_size, img_size))
        # cam_resize = np.maximum(cam_resize, 0)  # ReLU
        # heatmaps[batch, :, :, :] = (cam_resize - cam_resize.min()) / (cam_resize.max() - cam_resize.min())
        heatmaps[batch, :, :, :] = cam_resize

    heatmaps = np.expand_dims(heatmaps, axis=-1)

    return heatmaps


def show_slice_cam_2d(images, masks, pred_masks, name, is_png=True, num_rows=1, num_cols=3,
                      fig_size=(3 * 3, 1 * 3), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = pred_masks.shape[0]

    names, dcss, ious, ste_probs = [], [], [], []
    for i in range(batch_size):
        fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
        axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
        axoff_fun(ax)

        show_image = images[i, :, :, 0]
        show_mask = masks[i, :, :, 0]
        show_pred_mask = pred_masks[i, :, :, 0]
        each_name = name[i].decode()

        each_dcs = dcs_metric(masks[i, :, :, :], pred_masks[i, :, :, :]).numpy()
        each_iou = iou_metric(masks[i, :, :, :], pred_masks[i, :, :, :]).numpy()

        dcss.append(each_dcs)
        ious.append(each_iou)
        names.append(each_name)

        if is_png:
            ax[0].imshow(show_image, cmap='bone')
            ax[0].set_title(each_name, fontsize=7, color='black')
            ax[1].imshow(show_mask, cmap='bone')
            ax[1].set_title('Mask: ' + each_name, fontsize=7, color='navy')

            ax[2].imshow(show_pred_mask, cmap='bone')
            ax[2].set_title('DCS: %.3f, IOU: %.3f' % (each_dcs, each_iou), fontsize=7, color='blue')

            fig_name = os.path.join(save_path, each_name)
            plt.savefig(fig_name, bbox_inches='tight')

    return names, dcss, ious


def show_slice_cam_3d(images, masks, pred_masks, name, is_png=True, num_rows=1, num_cols=3,
                      fig_size=(3 * 3, 1 * 3), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size, seq_len = pred_masks.shape[0:2]
    names, dcss, ious, ste_probs = [], [], [], []

    for i in range(batch_size):
        patient_id, start_idx = metric.name_idx(name[i])
        dcs_slices, iou_slices = [], []

        each_dcs = dcs_metric(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()
        each_iou = iou_metric(masks[i, :, :, :, :], pred_masks[i, :, :, :, :]).numpy()

        dcs_slices.append(each_dcs)
        iou_slices.append(each_iou)

        if is_png:
            for j in range(seq_len):
                fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                show_image = images[i, j, :, :, 0]
                show_mask = masks[i, j, :, :, 0]
                show_pred_mask = pred_masks[i, j, :, :, 0]

                each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])
                names.append(each_name)

                if is_png:
                    ax[0].imshow(show_image, cmap='bone')
                    ax[0].set_title(each_name, fontsize=7, color='black')
                    ax[1].imshow(show_mask, cmap='bone')
                    ax[1].set_title('Mask: ' + each_name, fontsize=7, color='navy')

                    ax[2].imshow(show_pred_mask, cmap='bone')
                    ax[2].set_title('DCS: %.3f, IOU: %.3f' % (each_dcs, each_iou), fontsize=7, color='blue')

                    fig_name = os.path.join(save_path, each_name)
                    plt.savefig(fig_name, bbox_inches='tight')

                plt.close()

        dcss.extend(dcs_slices)
        ious.extend(iou_slices)

    return names, dcss, ious


if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation_save()
        # validation()

