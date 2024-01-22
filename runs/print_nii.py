import os
import sys
import logging
import numpy as np
import tensorflow as tf
import SimpleITK as sitk
import matplotlib.pyplot as plt
import sklearn.metrics
import pandas as pd
import nibabel as nib
import argparse
import re
import ast
import skimage.transform
from scipy.stats import mode

sys.path.append('/workspace/bitbucket/MRA')
# from data.setup_d import DataSettingV1, INFO_PATH
import models.model_d as model_ref
import models.metric as metric
import runs.cams
from runs.cams import gen_grad_cam_2d, gen_grad_cam_lstm


ROOT_PATH = '/workspace/MRA/'
INFO_PATH = os.path.join(ROOT_PATH, 'info')


def nii_each_extract(excel_name, raw_path, patient_id):
    df = pd.read_excel(os.path.join(INFO_PATH, excel_name+'.xlsx'))
    df['PATIENT_ID'] = df['PATIENT_ID'].astype(str)
    df_each = df[df['PATIENT_ID'] == patient_id]

    df_dict = df_each.to_dict('records')[0]

    total_z = df_dict['PIXEL_Z']
    slice_s, slice_e = df_dict['S_VALID'], df_dict['E_VALID']
    rev = df_dict['REVERSE']

    nii_path = os.path.join(raw_path, df_dict['NII_FOLDER'], df_dict['NII_RAW'])
    mask_path = os.path.join(raw_path, df_dict['NII_FOLDER'], df_dict['NII_MASK'])
    ste_s, ste_e = df_dict['S_SLICE'], df_dict['E_SLICE']

    l_coord = df_dict['LOCAL_COORD']
    l_coord = ast.literal_eval(l_coord)

    return nii_path, mask_path, slice_s, slice_e, ste_s, ste_e, l_coord, rev, total_z


def _extract_loc(seq_idx, loc_zxy):
    loc_slice = []

    which_idx = 0
    num_idx = 0
    for idx in seq_idx:
        if idx in [*loc_zxy]:
            for xy in loc_zxy[idx]:
                loc_slice.append([which_idx, xy[0], xy[1]])
                num_idx += 1

        which_idx += 1

    return str(loc_slice)


def nii_each(nii_raw, nii_mask, slice_start, slice_end, ste_start, ste_end,
             coord, rev, n_end, seq_len=8, seq_interval=8, each_ste=True):
    def index_start(start_slice, end_slice):
        start_indexes = np.arange(start_slice, end_slice, seq_interval)
        end_indexes = start_indexes + seq_len

        valid_indexes = end_indexes <= end_slice
        start_indexes = start_indexes[valid_indexes]
        return start_indexes

    def path_start(paths, total_slices):
        paths_list = np.repeat(paths, len(total_slices))
        return paths_list.tolist()

    def reverse_check(rev, total_slices):
        reverses = np.repeat(True, len(total_slices)) if rev == 'y' else np.repeat(False, len(total_slices))
        return reverses.tolist()

    def stenosis_extract(x, slices):
        return np.isin(x, slices).astype(np.int32)

    def include_stenosis(start_index, l_start, l_end, l_coord, reverse, n_e):
        slice_range = [*map(lambda x: np.arange(x, x + seq_len), start_index)]
        start_idx, end_idx = [*map(int, re.split('-', l_start))], [*map(int, re.split('-', l_end))]

        if len(start_idx) == 1 and start_idx[0] == 0:
            if each_ste:
                labels = np.reshape(np.repeat(np.zeros([seq_len], dtype=np.int32), len(start_index)),
                                    [len(start_index), seq_len]).tolist()
            else:
                labels = np.repeat(0, len(start_index)).tolist()
            locs = np.repeat('[]', len(labels)).tolist()

        else:
            stenosis_slices = list()

            for s_idx, e_idx in zip(start_idx, end_idx):
                if reverse == 'y':
                    s_idx, e_idx = n_e + 1 - e_idx, n_e + 1 - s_idx
                stenosis_slices.extend(np.arange(int(s_idx - 1), int(e_idx - 1) + 1))

            stenosis_slices = np.array(stenosis_slices)

            if each_ste:
                labels = [*map(lambda x: stenosis_extract(x, stenosis_slices), slice_range)]
            else:
                labels = [*map(lambda x: 1 if np.sum(np.isin(x, stenosis_slices)) >= 1 else 0, slice_range)]

            locs_zxy = dict()
            z_idx = []
            for zxy in l_coord:
                if zxy[0] - 1 not in z_idx:
                    locs_zxy[zxy[0] - 1] = [zxy[1:3]]
                else:
                    locs_zxy[zxy[0] - 1].append(zxy[1:3])
                z_idx.append(zxy[0] - 1)

            locs = [*map(lambda x: _extract_loc(x, locs_zxy), slice_range)]

        return labels, locs

    slice_index = index_start(slice_start, slice_end).tolist()
    nii_raw = path_start(nii_raw, slice_index)
    nii_mask = path_start(nii_mask, slice_index)
    nii_ste, nii_loc = include_stenosis(slice_index, ste_start, ste_end, coord, rev, n_end)
    reverse_list = reverse_check(rev, slice_index)
    print('# of MRA sequences: ', len(slice_index))

    return nii_raw, nii_mask, slice_index, nii_ste, nii_loc, reverse_list


def nii_to_sequence(nii, nii_mask, idx, ste, loc, seq_len=5, image_size=256, radius=80, det_size=16, each_ste=False):
    nii = nii.numpy().decode()
    nii_mask = nii_mask.numpy().decode()

    img = nib.load(nii).dataobj[:, :, idx:idx + seq_len]
    ori_w, ori_h = img.shape[0:2]
    mask = nib.load(nii_mask).dataobj[:, :, idx:idx + seq_len]
    sx, sy, sz = nib.load(nii).header.get_zooms()
    msx, msy, msz = nib.load(nii_mask).header.get_zooms()
    radius = radius.numpy()

    def loc_convert(zxy):
        lz, lx, ly = zxy
        cx, cy = int(ori_w / 2), int(ori_h / 2)
        x1, y1 = int(cx - radius / sx), int(cy - radius / sy)
        x2, y2 = int(cx + radius / sx), int(cy + radius / sy)
        c_w, c_h = x2 - x1, y2 - y1
        c_lx, c_ly = lx - x1, ly - y1
        r_lx, r_ly = ((image_size / c_h) * c_lx).numpy(), ((image_size / c_w) * c_ly).numpy()
        return [lz, int(r_lx), int(r_ly)]

    def img_preprocess(each_seq, sp_x=sx, sp_y=sy):
        each_seq = np.squeeze(each_seq)
        each_seq = np.transpose(each_seq, [1, 0])

        y, x = each_seq.shape[0:2]
        cx, cy = int(x / 2), int(y / 2)

        x1, y1 = int(cx - radius / sp_x), int(cy - radius / sp_y)  # 80: radius
        x2, y2 = int(cx + radius / sp_x), int(cy + radius / sp_y)  # 80: radius

        each_seq = each_seq[max(0, y1):min(y, y2), max(0, x1):min(x, x2)]
        each_seq = skimage.transform.resize(each_seq, [image_size, image_size], preserve_range=True)

        each_seq = np.expand_dims(each_seq, axis=-1)
        return each_seq

    img_split = np.split(img, img.shape[-1], axis=-1)  # 3d array to list of 2d arrays
    mask_split = np.split(mask, mask.shape[-1], axis=-1)

    img = np.array([*map(lambda x: img_preprocess(x, sx, sy), img_split)])  # (d, h, w, c)
    mask = np.array([*map(lambda x: img_preprocess(x, msx, msy), mask_split)])  # (d, h, w, c)

    name = re.sub('.nii.gz', '', os.path.basename(nii))
    name = '_'.join([name, '%03d' % idx, '%03d' % (idx + seq_len)])

    loc_list = ast.literal_eval(loc.numpy().decode())
    # down_size = 32  # Model24, serial2: 16
    det_mask = np.zeros([seq_len, det_size, det_size])

    ratio = image_size.numpy() / det_size
    if len(loc_list) > 0:
        loc_gt = [*map(loc_convert, loc_list)]
        for zxy in loc_gt:
            z, x, y = zxy
            y_center, x_center = int(y / ratio), int(x / ratio)
            # det_mask[z, y_center - 1:y_center + 2, x_center - 1:x_center + 2] = 1
            det_mask[z, y_center, x_center] = 1

    if seq_len == 1:
        img = np.reshape(img, (img.shape[1:4]))  # (h, w, c)
        mask = np.reshape(mask, (mask.shape[1:4]))  # (h, w, c)
        det_mask = np.reshape(det_mask, [det_size, det_size])

    if seq_len == 1:
        ste = np.where(np.sum(det_mask) > 0, 1, 0)
    else:
        ste = np.where(np.sum(det_mask, axis=(1, 2)) > 0, 1, 0)

    det_mask = np.expand_dims(det_mask, axis=-1)

    return img, mask, ste, det_mask, name


def py_nii_to_sequence(img, mask, index, ste, loc, seq_len, img_size, radius, det_size, each_ste):
    imgs, masks, stes, dets, names = tf.py_function(nii_to_sequence,
                                                    [img, mask, index, ste, loc, seq_len, img_size,
                                                     radius, det_size, each_ste],
                                                    [tf.float32, tf.float32, tf.int32, tf.float32, tf.string])

    if seq_len != 1:
        imgs.set_shape([seq_len, None, None, 1])
        masks.set_shape([seq_len, None, None, 1])
    else:
        imgs.set_shape([None, None, 1])
        masks.set_shape([None, None, 1])

    return imgs, masks, stes, dets, names


def nii_dataset(image, mask, ste, det, name):
    def img_preprocess(each_seq):
        each_seq = tf.image.per_image_standardization(each_seq)
        return each_seq

    def mask_preprocess(each_seq):
        each_seq = tf.where(each_seq < 0.5, 0, 1)
        return each_seq

    if len(image.shape) == 4:
        stack_image = tf.unstack(image, axis=0)
        stack_mask = tf.unstack(mask, axis=0)
        image = tf.stack([*map(img_preprocess, stack_image)], axis=0)
        mask = tf.stack([*map(mask_preprocess, stack_mask)], axis=0)
    else:
        image = img_preprocess(image)
        mask = mask_preprocess(mask)

    return image, mask, ste, det, name


def nii_data_setting(imgs, masks, indexes, stes, locs, seq_len=8, img_size=256, radius=80, det_size=16, each_ste=True):
    dataset = tf.data.Dataset.from_tensor_slices((imgs, masks, indexes, stes, locs))

    dataset = dataset.map(lambda x, y, z, w, v: py_nii_to_sequence(x, y, z, w, v, seq_len, img_size, radius, det_size,
                                                                   each_ste))
    dataset = dataset.map(lambda x, y, z, w, v: nii_dataset(x, y, z, w, v))

    return dataset


class EachDataSetting:
    def __init__(self, excel_name, patient_id, data_type='clinical', seq_len=8,
                 each_ste=False, img_size=256, radius=80, det_size=16, **kwargs):
        raw_path = os.path.join(ROOT_PATH, 'RAW', data_type)

        val_raw, val_mask, val_ss, val_se, val_ste_s, val_ste_e, val_coord, val_rev, val_n = \
            nii_each_extract(excel_name=excel_name, raw_path=raw_path, patient_id=patient_id)

        val_raw, val_mask, val_index, val_ste, val_loc, val_rev = \
            nii_each(nii_raw=val_raw, nii_mask=val_mask, slice_start=val_ss, slice_end=val_se,
                     ste_start=val_ste_s, ste_end=val_ste_e, coord=val_coord,
                     rev=val_rev, n_end=val_n, seq_len=seq_len, seq_interval=seq_len,
                     each_ste=each_ste)

        self.val = nii_data_setting(imgs=val_raw, masks=val_mask, indexes=val_index, stes=val_ste, locs=val_loc,
                                    seq_len=seq_len, img_size=img_size, radius=radius, det_size=det_size)


def select_train_groups(x, group_list):
    out = True if x in group_list else False
    return out


def restore_weight(data_path, exp_name, model_name, trial_serial, num_weight):
    weight_auc_path = os.path.join(data_path, exp_name, model_name, 'result-%03d' % trial_serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([exp_name, model_name,
                                                                         '%03d' % trial_serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('METRIC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:num_weight])
    return all_ckpt_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    main_config = parser.add_argument_group('network setting (must be provided)')

    main_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
    main_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh09')
    main_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp009')
    main_config.add_argument('--val_name', type=str, dest='val_name', default='6')
    main_config.add_argument('--patient_id', type=str, dest='patient_id', default='30476110')
    main_config.add_argument('--model_name', type=str, dest='model_name', default='Model28')
    main_config.add_argument('--f_num', type=str, dest='f_num', default='64,112,160,208')
    main_config.add_argument('--serial', type=int, dest='serial', default=24)
    main_config.add_argument('--image_size', type=int, dest='image_size', default=256)
    main_config.add_argument('--radius', type=int, dest='radius', default=80)
    main_config.add_argument('--num_weight', type=int, dest='num_weight', default=1)
    main_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=True)
    main_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=True)
    main_config.add_argument('--det_size', type=int, dest='det_size', default=16)
    main_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
    main_config.add_argument('--seq_len', type=int, dest='seq_len', default=8)
    main_config.add_argument('--batch_size', type=int, dest='batch_size', default=1)

    config, unparsed = parser.parse_known_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)

    import warnings
    warnings.filterwarnings('ignore')

    if 'snubh' in config.excel_name:
        d_type = 'clinical'
        patient_id = str(config.patient_id)
    elif 'cusmh' in config.excel_name:
        d_type = 'external'
        patient_id = config.patient_id
    else:
        raise ValueError('Invalid data type')

    show_name = {'Model28': 'Full model', 'Model232': 'Full model_wo_seg', 'Model203': 'Without LM',
                 'Model242': 'Without CM', 'Model282': 'Without MO',
                 'Model22': '2D U-Net', 'Model221': '2D U-Net_wo_seg',
                 'Model262': '3D U-Net', 'Model264': '3D U-Net_wo_seg',
                 'Model03': 'Only seg'}

    # case_path = os.path.join(config.data_path, config.exp_name, 'case', show_name[config.model_name],
    #                          '_'.join([config.excel_name, config.val_name]))
    nii_case_path = os.path.join(config.data_path, config.exp_name, 'nii', show_name[config.model_name],
                                 '_'.join([config.excel_name, config.val_name]))

    if not os.path.exists(nii_case_path):
        os.makedirs(nii_case_path)

    gpu = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation

    serial_str = '%03d' % int(config.serial)
    log_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'logs-%s' % serial_str)
    result_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%s' % serial_str)
    if not os.path.exists(log_path): os.makedirs(log_path)
    if not os.path.exists(result_path): os.makedirs(result_path)

    weight_auc_path = os.path.join(config.data_path, config.exp_name,
                                   config.model_name, 'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values('METRIC', ascending=False)
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][(config.num_weight - 1):config.num_weight])
    ckpt = all_ckpt_paths[0]

    num_ckpt = len(all_ckpt_paths)
    print('num_ckpt: ', num_ckpt)

    seq_len, img_size, img_c, det_size = config.seq_len, config.image_size, 1, config.det_size

    df = pd.read_excel(os.path.join(INFO_PATH, config.excel_name) + '.xlsx')
    if config.only_ste:
        df = df[df['LABEL'] == 1]

    group_values = df['GROUP'].tolist()
    type_groups = [*map(int, re.split(',', config.val_name))]
    df = df[[*map(lambda x: select_train_groups(x, type_groups), group_values)]]
    df['PATIENT_ID'] = df['PATIENT_ID'].astype(str)

    df = df[df['PATIENT_ID'] == patient_id]

    dfs_id = df['PATIENT_ID'].tolist()

    input_size = [seq_len, img_size, img_size, img_c] if seq_len != 1 else [img_size, img_size, img_c]

    infer_name = config.model_name
    infer = getattr(model_ref, infer_name)(input_size=input_size, class_num=1, f_num='64,112,160,208',
                                           is_training=False, mtl_mode=True)
    model, cam_model = infer.model, infer.cam_model
    gen_grad_cam = gen_grad_cam_lstm if config.seq_len != 1 else gen_grad_cam_2d

    roc_det_df = pd.DataFrame({'PATIENT_ID': pd.Series(), 'THRESHOLD': pd.Series(), 'AUC': pd.Series()})

    for patient_idx in dfs_id:
        print(patient_idx)

        case_patient_path = os.path.join(nii_case_path, patient_idx)
        if not os.path.exists(case_patient_path): os.makedirs(case_patient_path)

        case = EachDataSetting(config.excel_name, patient_id=patient_idx, data_type=d_type, seq_len=config.seq_len)

        num_slices = seq_len * case.val.cardinality().numpy()

        img_3d = np.zeros([num_slices, img_size, img_size, img_c])
        mask_3d = np.zeros([num_slices, img_size, img_size, img_c])

        cls = np.zeros([num_slices, ])
        det_3d = np.zeros([num_slices, config.det_size, config.det_size, img_c])
        img_names = np.zeros([num_slices, ], dtype=object)

        pcls = np.zeros([num_slices, ])
        pmask_3d = np.zeros([num_slices, img_size, img_size, img_c])
        pred_cls_probs = np.zeros([num_slices])
        pdet_3d = np.zeros([num_slices, config.det_size, config.det_size, img_c])
        pscore_3d = np.zeros([num_slices, config.det_size, config.det_size, img_c])

        val_db = case.val.batch(config.batch_size)

        print('ckpt: ', ckpt)
        model.load_weights(ckpt)

        for step, (img, mask, ste, det, name) in enumerate(val_db):
            with tf.GradientTape() as tape:
                if show_name[config.model_name] in \
                        ['Full model', 'Spider U-Net', '2D U-Net', '3D U-Net', 'Without MO']:
                    cam_layers, seg_prob, cls_prob, det_prob, det_gap, det_score = cam_model(img)
                    cams = None

                elif 'wo_seg' in show_name[config.model_name]:
                    cam_layers, cls_prob, det_prob, det_gap, det_score = cam_model(img)
                    seg_prob = mask
                    cams = None

                elif show_name[config.model_name] == 'Without LM':
                    cam_layers, seg_prob, cls_prob = cam_model(img)
                    det_prob = det
                    cams = gen_grad_cam(cam_layers, cls_prob, tape, infer)

                elif show_name[config.model_name] == 'Without CM':
                    cam_layers, seg_prob, det_prob, det_gap = cam_model(img)
                    cls_prob = tf.reduce_max(det_prob, axis=[2, 3, 4])

                    cams = None

            if config.seq_len != 1:
                img_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = img[0, :, :, :, :]
                mask_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = mask[0, :, :, :, :]
                det_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = det[0, :, :, :, :]

                cls[step * seq_len:(step + 1) * seq_len, ] = ste.numpy()[0, :]
                pcls[step * seq_len:(step + 1) * seq_len, ] = cls_prob.numpy()[0, :]
                pmask_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = seg_prob.numpy()[0, :, :, :, :]
                pdet_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = det_prob.numpy()[0, :, :, :, :]
                print(np.max(det_prob.numpy()[0, :, :, :, :]))
                # pscore_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = det_score.numpy()[0, :, :, :, :]

                det_split = np.split(det.numpy()[0, :, :, :, :], config.seq_len)


            else:  # 2D U-Net
                img_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = img
                mask_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = mask
                det_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = det

                cls[step * seq_len:(step + 1) * seq_len, ] = ste.numpy()
                pcls[step * seq_len:(step + 1) * seq_len, ] = cls_prob.numpy()
                pmask_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = seg_prob.numpy()
                pdet_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = det_prob.numpy()
                # pscore_3d[step * seq_len:(step + 1) * seq_len, :, :, :] = det_score.numpy()

            img_names[step * config.batch_size:step * config.batch_size + len(ste)] = name

        img_name = '_'.join(['img', show_name[config.model_name], patient_idx])
        mask_name = '_'.join(['mask', show_name[config.model_name], patient_idx])
        cls_name = '_'.join(['cls', show_name[config.model_name], patient_idx])
        det_name = '_'.join(['det', show_name[config.model_name], patient_idx])

        pmask_name = '_'.join(['pmask', show_name[config.model_name], patient_idx])
        pcls_name = '_'.join(['pcls', show_name[config.model_name], patient_idx])
        pdet_name = '_'.join(['pdet', show_name[config.model_name], patient_idx])

        img_3d = np.transpose(np.squeeze(img_3d), (1, 2, 0))
        mask_3d = np.transpose(np.squeeze(mask_3d), (1, 2, 0))

        # mask_3d = mask_3d.copy()  # for 24807461
        # mask_3d[:, 200:, :] = 0  # for 24807461

        det_3d = np.transpose(np.squeeze(det_3d), (1, 2, 0))

        pmask_3d = np.transpose(np.squeeze(pmask_3d), (1, 2, 0))
        pdet_3d = np.transpose(np.squeeze(pdet_3d), (1, 2, 0))

        print(patient_idx, np.max(pdet_3d), np.min(pdet_3d), np.median(pdet_3d))

        img_nii = nib.Nifti1Image(img_3d, None, nib.Nifti1Header())
        mask_nii = nib.Nifti1Image(mask_3d, None, nib.Nifti1Header())
        det_nii = nib.Nifti1Image(det_3d, None, nib.Nifti1Header())

        pred_mask_nii = nib.Nifti1Image(pmask_3d, None, nib.Nifti1Header())
        pred_det_nii = nib.Nifti1Image(pdet_3d, None, nib.Nifti1Header())

        nib.save(img_nii, os.path.join(case_patient_path, img_name))
        nib.save(mask_nii, os.path.join(case_patient_path, mask_name))
        pd.DataFrame(cls).to_excel(os.path.join(case_patient_path, cls_name + '.xlsx'), index=False)
        nib.save(det_nii, os.path.join(case_patient_path, det_name))

        pd.DataFrame(pcls).to_excel(os.path.join(case_patient_path, pcls_name+'.xlsx'), index=False)
        nib.save(pred_mask_nii, os.path.join(case_patient_path, pmask_name))
        nib.save(pred_det_nii, os.path.join(case_patient_path, pdet_name))

        print(os.path.join(case_patient_path, img_name))
        print(os.path.join(case_patient_path, mask_name))
        print(os.path.join(case_patient_path, cls_name))
        print(os.path.join(case_patient_path, det_name))
        print(os.path.join(case_patient_path, pmask_name))
        print(os.path.join(case_patient_path, pcls_name))
        print(os.path.join(case_patient_path, pdet_name))




