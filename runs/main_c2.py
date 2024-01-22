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
import skimage.transform

sys.path.append('/workspace/bitbucket/MRA')
from data.setup_c import DataSettingV1, INFO_PATH
import models.model_c as model_ref
import models.metric as metric
from runs.cams import *
import tf_utils.tboard as tboard

parser = argparse.ArgumentParser()
main_config = parser.add_argument_group('network setting (must be provided)')

main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh07')
main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp007')
main_config.add_argument('--train_name', type=str, dest='train_name', default='1,2,3,4')
main_config.add_argument('--val_name', type=str, dest='val_name', default='5')
main_config.add_argument('--model_name', type=str, dest='model_name', default='Model11')
main_config.add_argument('--f_num', type=str, dest='f_num', default='64,128,192,256')  # or 64,128,192,256,512
main_config.add_argument('--serial', type=int, dest='serial', default=1)
main_config.add_argument('--image_size', type=int, dest='image_size', default=256)
main_config.add_argument('--radius', type=int, dest='radius', default=80)
main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
# main_config.add_argument('--num_classes', type=int, dest='num_classes', default=2)
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
main_config.add_argument('--one_hot', type=lambda x: x.title() in str(True), dest='one_hot', default=False)
main_config.add_argument('--add_conv', type=lambda x: x.title() in str(True), dest='add_conv', default=False)


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
# if not os.path.exists(plot_val_path): os.makedirs(plot_val_path)


if 'snubh' in config.excel_name:
    data_type = 'clinical'
elif 'cusmh' in config.excel_name:
    data_type = 'external'
else:
    raise ValueError('Invalid data type')


img_size, img_c = config.image_size, config.channel_size
seq_len, seq_interval = config.seq_len, config.seq_interval
# ste_unit = seq_len if config.each_ste else 1
f_num = config.f_num

df = pd.read_excel(os.path.join(INFO_PATH, config.excel_name) + '.xlsx')

d_set = DataSettingV1(df=df, train_type=config.train_name, val_type=config.val_name, data_type=data_type,
                      train=config.train, seq_len=seq_len, seq_interval=seq_interval,
                      only_ste=config.only_ste, each_ste=config.each_ste,
                      image_size=img_size, radius=config.radius, one_hot=config.one_hot)

if config.train:
    train_db = d_set.train.batch(config.batch_size)
    train_length = d_set.train.cardinality().numpy()
    print('train length: ', train_length)

val_db = d_set.val.batch(config.batch_size)
val_length = d_set.val.cardinality().numpy()
print('val length: ', val_length)

input_size = [seq_len, img_size, img_size, img_c] if seq_len != 1 else [img_size, img_size, img_c]
time_axis = True if seq_len != 1 else False

infer_name = config.model_name
infer = getattr(model_ref, infer_name)(input_size=input_size, class_num=1, f_num=f_num,
                                       t=time_axis, add_conv=config.add_conv, is_training=config.train)


model, cam_model = infer.model, infer.cam_model

cls_loss_fn = metric.focal_loss_sigmoid


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
        'ADD_CONV': config.add_conv
    }

    with open(os.path.join(result_path, '.info'), 'w') as f:
        f.write(json.dumps(info_log, indent=4, sort_keys=True))
        f.close()

    train_summary, val_summary = tboard.tensorboard_create(log_path)
    result_name = '_'.join([config.exp_name, config.model_name, serial_str]) + '.csv'
    dcs_csv = pd.DataFrame({'WEIGHT_PATH': pd.Series(), 'AUC': pd.Series()})

    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=config.learning_rate,
        decay_steps=int(train_length),
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
            for train_step, (img, _, ste, _) in enumerate(train_db):
                with tf.GradientTape() as tape:
                    cls_prob = model(img)
                    cls_loss_batch = 1.0 * cls_loss_fn(ste, cls_prob)

                grads = tape.gradient(cls_loss_batch, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))

                train_loss.append(cls_loss_batch.numpy())
                train_cx.extend(cls_prob.numpy())
                train_cy.extend(ste.numpy())

                sys.stdout.write('Step: {0:>4d},  Loss: {1:.4f} ({2})\r'.format(train_step, cls_loss_batch, epoch))

            train_loss_mean = np.mean(train_loss)
            train_auc = metric.calculate_auc(train_cy, train_cx)
            train_record = {'Loss': train_loss_mean, 'AUC': train_auc}

            val_cx, val_cy = [], []
            val_loss = []
            val_steps = val_length // config.batch_size + 1

            for val_step, (img, _, ste, _) in enumerate(val_db):
                val_cls_prob = model(img)
                val_cls_loss_batch = cls_loss_fn(ste, val_cls_prob)

                val_loss.append(val_cls_loss_batch.numpy())
                val_cx.extend(val_cls_prob.numpy())
                val_cy.extend(ste.numpy())

                sys.stdout.write('Evaluation [{0}/{1}],  Loss: {2:.4f}\r'.
                                 format(val_step+1, val_steps, val_cls_loss_batch))

            val_loss_mean = np.mean(val_loss)
            val_auc = metric.calculate_auc(val_cy, val_cx)
            val_record = {'Loss': val_loss_mean, 'AUC': val_auc}

            time_elapsed = str(datetime.datetime.now() - start_time)
            log_string += ' Time Elapsed: {0}'.format(time_elapsed.split('.')[0])

            print('Epoch: %s Train-Loss: %.4f Train-AUC %.4f Val-Loss %.4f Val-AUC %.4f' %
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
                dcs_csv.loc[epoch] = weight_path, val_auc

            elif val_auc > min(dcs_csv['AUC'].tolist()):
                os.remove(dcs_csv.loc[max_current_step[0], 'WEIGHT_PATH'])
                dcs_csv = dcs_csv.drop(max_current_step[0])
                max_current_step.pop(0)
                max_current_step.append(epoch)
                max_perf_per_epoch.pop(0)
                max_perf_per_epoch.append(val_auc)

                model.save(weight_path)
                dcs_csv.loc[epoch] = weight_path, val_auc

            dcs_csv.to_csv(os.path.join(result_path, result_name))

            if epoch == config.epoch:
                break

    except KeyboardInterrupt:
        print('Result saved')
        dcs_csv.to_csv(os.path.join(result_path, result_name))


def _gen_grad_cam(cam_layer, seq_len, loss, tape):
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

        heatmaps[batch, :, :, :] = cam_resize

    heatmaps = np.expand_dims(heatmaps, axis=-1)

    return heatmaps


def validation():
    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('AUC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:int(config.num_weight)])

    # batch_unit = 1
    show_slice_cam = show_slice_cam_3d if config.seq_len != 1 else show_slice_cam_2d

    if config.seq_len == 1:
        gen_grad_cam = gen_grad_cam_2d
    elif cam_model.outputs[0].shape[1] != 1:
        gen_grad_cam = gen_grad_cam_lstm
    else:
        gen_grad_cam = gen_grad_cam_3d

    val_ste_prob = np.zeros([len(all_ckpt_paths), val_length])
    val_name = []

    print('num_ckpt: ', len(all_ckpt_paths))
    guided_model = metric.built_guided_model(model)

    ckpt_idx = 0
    val_cx, val_cy = [], []
    for ckpt in all_ckpt_paths:
        model.load_weights(ckpt)

        val_loss, val_name = [], []
        for step, (img, mask, ste, name) in enumerate(val_db):

            with tf.GradientTape() as tape:
                cam_layers, val_cls_prob = cam_model(img)

            val_cls_loss_batch = cls_loss_fn(ste, val_cls_prob)
            val_cx.extend(val_cls_prob.numpy())
            val_loss.append(val_cls_loss_batch)

            grad_cams = gen_grad_cam(cam_layers, val_cls_prob, tape, infer)
            gb = metric.guided_backprop(guided_model, img, infer.cam_layer_name)
            guided_grad_cams = gb * grad_cams
            ggc = metric.deprocess_image(guided_grad_cams)

            val_name_batch, val_ste_prob_batch = \
                show_slice_cam(images=img.numpy(), masks=mask.numpy(), stes=ste.numpy(),
                               pred_stes=val_cls_prob.numpy(), cam_stes=grad_cams, ggc_stes=ggc, name=name.numpy(),
                               is_png=False, save_path=plot_val_path)

            if ckpt_idx == 0:
                name_str = [x.decode() for x in name.numpy()]
                val_name.extend(val_name_batch) if config.seq_len == 1 else val_name.extend(name_str)
                val_cy.extend(ste.numpy())

            cnt_range = config.batch_size
            val_ste_prob[ckpt_idx, step * cnt_range:step * cnt_range + len(val_ste_prob_batch)] = val_ste_prob_batch

            sys.stdout.write('{0} Evaluation [{1}/{2}]\r'.
                             format(os.path.basename(ckpt), step, val_length // config.batch_size))

        ckpt_idx += 1

    val_ste_prob = np.mean(val_ste_prob, axis=0)

    fpr, tpr, _ = sklearn.metrics.roc_curve(val_cy, val_ste_prob, drop_intermediate=False)
    val_auc = sklearn.metrics.auc(fpr, tpr)
    print('\nFinal AUC: %.3f' % val_auc)

    result_csv = pd.DataFrame({'NUMBER': val_name, 'STE': val_cy, 'STE_PROB': val_ste_prob})
    result_name = '_'.join([config.model_name, config.excel_name, config.val_name,
                            serial_str, '%03d' % config.num_weight]) + '.csv'

    result_csv.to_csv(os.path.join(result_path, result_name), index=False)


def show_slice_cam_3d(images, masks, stes, pred_stes, cam_stes, ggc_stes, name, is_png=True,
                      num_rows=1, num_cols=4, fig_size=(4 * 2, 1 * 2), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = pred_stes.shape[0]
    names, ste_probs = [], []

    for i in range(batch_size):
        patient_id, start_idx = metric.name_idx(name[i])

        ste_prob_slices = []
        show_ste, show_pred_ste = stes[i], pred_stes[i]
        ste_prob_slices.append(show_pred_ste)

        for j in range(seq_len):
            fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            show_image = images[i, j, :, :, 0]
            show_mask = masks[i, j, :, :, 0]

            show_cam_ste = cam_stes[i, j, :, :, 0]  # grad_cam
            show_ggc_ste = ggc_stes[i, j, :, :, 0]

            each_name = '_'.join([patient_id, '%03d' % (int(start_idx) + 1 + j)])
            names.append(each_name)

            if is_png:
                ax[0].imshow(show_image, cmap='bone')
                ax[0].set_title(each_name, fontsize=7, color='black')
                ax[1].imshow(show_mask, cmap='bone')
                ax[1].set_title('Mask', fontsize=7, color='green')
                ax[2].imshow(show_image, cmap='bone')
                ax[2].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
                ax[2].set_title('Stenosis: %d Prob: %.3f' % (show_ste, show_pred_ste), fontsize=7, color='green')
                ax[3].imshow(show_ggc_ste, cmap='bone')
                ax[3].set_title('Guided Grad-CAM', fontsize=7, color='darkgoldenrod')

                fig_name = os.path.join(save_path, each_name)
                plt.savefig(fig_name, bbox_inches='tight')

            plt.clf()

        ste_probs.extend(ste_prob_slices)

    return names, ste_probs


def show_slice_cam_2d(images, masks, stes, pred_stes, cam_stes, ggc_stes,
                      name, is_png=True, num_rows=1, num_cols=4, fig_size=(4 * 2, 1 * 2), save_path=plot_val_path):
    if not os.path.exists(save_path): os.makedirs(save_path)
    batch_size = masks.shape[0]

    names, ste_probs = [], []
    for i in range(batch_size):
        fig, ax = plt.subplots(num_rows, num_cols, figsize=fig_size)
        axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
        axoff_fun(ax)

        show_image = images[i, :, :, 0]
        show_mask = masks[i, :, :, 0]
        show_cam_ste = cam_stes[i, :, :, 0]
        show_ggc_ste = ggc_stes[i, :, :, 0]

        show_ste, show_pred_ste = stes[i], pred_stes[i]
        each_name = name[i].decode()

        names.append(each_name)
        ste_probs.append(show_pred_ste)

        if is_png:
            ax[0].imshow(show_image, cmap='bone')
            ax[0].set_title(each_name, fontsize=7, color='black')
            ax[1].imshow(show_mask, cmap='bone')
            ax[1].set_title('Mask: ' + each_name, fontsize=7, color='navy')

            ax[2].imshow(show_image, cmap='bone')
            ax[2].imshow(show_cam_ste, cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
            ax[2].set_title('Stenosis: %d Prob: %.3f' % (show_ste, show_pred_ste), fontsize=7, color='green')

            ax[3].imshow(show_ggc_ste, cmap='bone')
            ax[3].set_title('Guided Grad-CAM', fontsize=7, color='darkgoldenrod')

            fig_name = os.path.join(save_path, each_name)
            plt.savefig(fig_name, bbox_inches='tight')

    return names, ste_probs


if __name__ == '__main__':
    if config.train:
        print('Training')
        training()
    else:
        print('Validation')
        validation()










