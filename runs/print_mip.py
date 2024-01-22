import os
import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import re
import random
import nibabel as nib
import matplotlib.pyplot as plt
import skimage.transform
import pydicom as dcm
import scipy.ndimage
from scipy.stats import mode
import argparse
import sys
import json

import warnings
warnings.filterwarnings('ignore')

sys.path.append('/workspace/bitbucket/MRA')


def rotate_image(mra_img, angle):  # (z, y, x)
    # size = mra_img.shape
    batch = np.squeeze(mra_img)
    batch_rot = np.zeros(batch.shape)
    for i in range(batch.shape[0]):
        image1 = np.squeeze(batch[i, :, :])
        rot_image_slice = scipy.ndimage.interpolation.rotate(image1, angle, mode='nearest', reshape=False)
        batch_rot[i, :, :] = rot_image_slice
    return batch_rot


def img_diff(a_img, b_img):
    diff = (a_img - b_img) ** 2
    return np.sum(diff)


def select_train_groups(x, group_list):
    out = True if x in group_list else False
    return out


if __name__ == '__main__':
    import logging

    parser = argparse.ArgumentParser()
    setup_config = parser.add_argument_group('dataset setting')
    setup_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
    setup_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp009')
    setup_config.add_argument('--model_name', type=str, dest='model_name', default='Model28')
    setup_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh09')
    setup_config.add_argument('--val_name', type=str, dest='val_name', default='6')
    setup_config.add_argument('--patient_id', type=str, dest='patient_id', default='30476110')
    setup_config.add_argument('--add_raw', type=lambda x: x.title() in str(True), dest='add_raw', default=False)
    setup_config.add_argument('--det_cutoff', type=float, dest='det_cutoff', default=0.8)
    config, unparsed = parser.parse_known_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)

    import warnings
    warnings.filterwarnings('ignore')

    INFO_PATH = os.path.join(config.data_path, 'info')

    if 'snubh' in config.excel_name:
        d_type = 'clinical'
    elif 'cusmh' in config.excel_name:
        d_type = 'external'
    else:
        raise ValueError('Invalid data type')

    patient_id = config.patient_id

    show_name = {'Model28': 'Full model', 'Model232': 'Full model_wo_seg', 'Model203': 'Without LM',
                 'Model242': 'Without CM', 'Model282': 'Without MO',
                 'Model22': '2D U-Net', 'Model221': '2D U-Net_wo_seg',
                 'Model262': '3D U-Net', 'Model264': '3D U-Net_wo_seg',
                 'Model03': 'Only seg'}

    # gpu = tf.config.list_physical_devices('GPU')
    # tf.config.experimental.set_memory_growth(gpu[0], True)  # dynamic memory allocation
    model_name = show_name[config.model_name]

    # model_name = 'Full model'
    case_nii_path = os.path.join(config.data_path, config.exp_name, 'nii', show_name[config.model_name],
                                 '_'.join([config.excel_name, config.val_name]))

    case_list = sorted(os.listdir(case_nii_path))
    print('case_nii_path: ', case_nii_path)
    print(len(case_list))

    for folder in case_list:
        if folder == config.patient_id:

            img_nii_name = '_'.join(['img', model_name, folder]) + '.nii'

            mask_nii_name = '_'.join(['mask', model_name, folder]) + '.nii'
            pmask_nii_name = '_'.join(['pmask', model_name, folder]) + '.nii'

            pcls_xlsx_name = '_'.join(['pcls', model_name, folder]) + '.xlsx'
            det_nii_name = '_'.join(['det', model_name, folder]) + '.nii'
            pdet_nii_name = '_'.join(['pdet', model_name, folder]) + '.nii'

            img_nii = nib.load(os.path.join(case_nii_path, folder, img_nii_name))
            img = np.squeeze(img_nii.dataobj)
            img = np.transpose(img, (2, 0, 1))

            mask_nii = nib.load(os.path.join(case_nii_path, folder, mask_nii_name))
            mask = np.squeeze(mask_nii.dataobj)
            mask = np.transpose(mask, (2, 0, 1))

            if folder == '24807461':
                mask = mask.copy()  # for 24807461
                mask[:, 200:, :] = 0  # for 24807461
                mask[57, 165:, :] = 0
                mask[62, 180:, :] = 0
                mask[63, 180:, :] = 0

            # import pdb; pdb.set_trace()
            pcls_xlsx = pd.read_excel(os.path.join(case_nii_path, folder, pcls_xlsx_name))

            pmask_nii = nib.load(os.path.join(case_nii_path, folder, pmask_nii_name))
            pmask = np.squeeze(pmask_nii.dataobj)
            pmask = np.transpose(pmask, (2, 0, 1))

            det_nii = nib.load(os.path.join(case_nii_path, folder, det_nii_name))
            det = np.squeeze(det_nii.dataobj)
            det = np.transpose(det, (2, 0, 1))
            rsz_det = skimage.transform.resize(det, (det.shape[0], 256, 256))

            pdet_nii = nib.load(os.path.join(case_nii_path, folder, pdet_nii_name))
            pdet = np.squeeze(pdet_nii.dataobj)
            pdet = np.transpose(pdet, (2, 0, 1))
            pdet = np.squeeze(pdet)

            info_log = {'PATIENT_ID': folder,
                        'DET_CUTOFF': config.det_cutoff,
                        'DET_MAX': float(np.max(pdet)),
                        'DET_MIN': float(np.min(pdet)),
                        'DET_MEDIAN': float(np.median(pdet))}

            with open(os.path.join(case_nii_path, folder, 'det_cutoff.info'), 'w') as f:
                f.write(json.dumps(info_log, indent=4, sort_keys=True))
                f.close()

            print(folder, np.max(pdet), np.min(pdet), np.median(pdet))

            # pdet_bin = np.where(pdet > 0.7, 1.0, 0.0)

            pdet_bin = np.where(pdet > config.det_cutoff, pdet, 0.0)

            array_pcls = np.squeeze(np.array(pcls_xlsx))
            rsz_pdet = skimage.transform.resize(pdet_bin, (pdet.shape[0], 256, 256))

            # resize: twice transverse direction
            len_z, len_y, len_x = img.shape

            img_cube = np.zeros((len_x, len_x, len_x))
            img_resize = skimage.transform.resize(img, (len_z * 2, len_y, len_x))

            mask_cube = np.zeros((len_x, len_x, len_x))
            mask_resize = skimage.transform.resize(mask, (len_z * 2, len_y, len_x))

            pcls_cube = np.zeros((len_x, len_x, len_x))
            pcls_vector = np.concatenate([*map(lambda x: np.repeat(x, 2).tolist(), array_pcls)])

            pcls_resize = np.multiply(np.ones((len_z * 2, len_y, len_x)), np.expand_dims(pcls_vector, axis=[-2, -1]))

            pmask_cube = np.zeros((len_x, len_x, len_x))
            pmask_resize = skimage.transform.resize(pmask, (len_z * 2, len_y, len_x))

            rsz_det_cube = np.zeros((len_x, len_x, len_x))
            rsz_det_resize = skimage.transform.resize(rsz_det, (len_z * 2, len_y, len_x))

            rsz_pdet_cube = np.zeros((len_x, len_x, len_x))
            rsz_pdet_resize = skimage.transform.resize(rsz_pdet, (len_z * 2, len_y, len_x))

            # import pdb; pdb.set_trace()

            x_r, z_r = len_x / 2, len_z * 2 / 2
            if len_z * 2 < len_x:
                img_cube[int(x_r - z_r):int(x_r + z_r), :, :] = img_resize
                mask_cube[int(x_r - z_r):int(x_r + z_r), :, :] = mask_resize
                pcls_cube[int(x_r - z_r):int(x_r + z_r), :, :] = pcls_resize
                pmask_cube[int(x_r - z_r):int(x_r + z_r), :, :] = pmask_resize
                rsz_det_cube[int(x_r - z_r):int(x_r + z_r), :, :] = rsz_det_resize
                rsz_pdet_cube[int(x_r - z_r):int(x_r + z_r), :, :] = rsz_pdet_resize

            else:
                img_cube[:, :, :] = img_resize[int(z_r - x_r):int(z_r + x_r), :, :]
                mask_cube[:, :, :] = mask_resize[int(z_r - x_r):int(z_r + x_r), :, :]
                pcls_cube[:, :, :] = pcls_resize[int(z_r - x_r):int(z_r + x_r), :, :]
                pmask_cube[:, :, :] = pmask_resize[int(z_r - x_r):int(z_r + x_r), :, :]
                rsz_det_cube[:, :, :] = rsz_det_resize[int(z_r - x_r):int(z_r + x_r), :, :]
                rsz_pdet_cube[:, :, :] = rsz_pdet_resize[int(z_r - x_r):int(z_r + x_r), :, :]

            # Save MIP Image: Coronal View (with confidence score, predicted lesion)
            coronal_all_path = os.path.join(case_nii_path, folder, 'coronal_all')
            if not os.path.exists(coronal_all_path): os.makedirs(coronal_all_path)
            axis_order, angle = (2, 0, 1), 90
            img_trans = np.transpose(img_cube, axis_order)
            img_trans_degree = rotate_image(img_trans, 180-angle)

            pmask_trans = np.transpose(pmask_cube, axis_order)
            pmask_trans_degree = rotate_image(pmask_trans, 180 - angle)

            pcls_trans = np.transpose(pcls_cube, axis_order)
            pcls_trans_degree = rotate_image(pcls_trans, 180 - angle)

            rsz_pdet_trans = np.transpose(rsz_pdet_cube, axis_order)
            rsz_pdet_trans_degree = rotate_image(rsz_pdet_trans, 180-angle)

            if False:
                fig, ax = plt.subplots(1, 1, figsize=(2, 2))
                axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
                axoff_fun(ax)

                # plt.rcParams["figure.figsize"] = (2, 2)
                # ax.axis('off')
                ax.imshow(np.flipud(np.transpose(np.max(img_trans_degree, axis=1))), cmap='gray')
                ax.imshow(np.flipud(np.transpose(np.max(pmask_trans_degree, axis=1))), cmap=plt.cm.pink,
                           alpha=0.3, interpolation='nearest')

                ax.imshow(np.flipud(np.mean(pcls_cube, axis=1)), cmap=plt.cm.gray,  alpha=0.5, interpolation='nearest')

                ax.imshow(np.flipud(np.transpose(np.max(rsz_pdet_trans_degree, axis=1))),
                           cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')

                leg = ax.get_legend()
                leg.legendHandles[0].set_color('red')
                leg.legendHandles[1].set_color('yellow')

                ax.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
                plt_name = '_'.join([model_name, folder, 'coronal', 'all', '%03d' % angle])+'.png'
                plt.savefig(os.path.join(coronal_all_path, plt_name), dpi=300, pad_inches=0, bbox_inches='tight')
                plt.close()

            # Save MIP Image: Coronal View (only image)
            if config.add_raw:
                coronal_img_path = os.path.join(case_nii_path, folder, 'coronal_img')
                if not os.path.exists(coronal_img_path): os.makedirs(coronal_img_path)
                for degree in range(0, 195, 15):
                    axis_order, angle = (2, 0, 1), degree
                    img_trans = np.transpose(img_cube, axis_order)
                    img_trans_degree = rotate_image(img_trans, 180-angle)

                    mask_trans = np.transpose(mask_cube, axis_order)
                    mask_trans_degree = rotate_image(mask_trans, 180 - angle)

                    # rsz_det_trans = np.transpose(rsz_det_cube, axis_order)
                    # rsz_det_trans_degree = rotate_image(rsz_det_trans, 180-angle)

                    plt.rcParams["figure.figsize"] = (2, 2)
                    plt.axis('off')
                    plt.imshow(np.flipud(np.transpose(np.max(img_trans_degree, axis=1))), cmap='gray')
                    plt.imshow(np.flipud(np.transpose(np.max(mask_trans_degree, axis=1))), cmap=plt.cm.hot,
                               alpha=0.3, interpolation='nearest')

                    plt.subplots_adjust(top=1.00, bottom=0.00, left=0.00, right=1.00, hspace=0, wspace=0)
                    plt_name = '_'.join([model_name, folder, 'coronal', 'img', '%03d' % angle])+'.png'
                    plt.savefig(os.path.join(coronal_img_path, plt_name), dpi=300, pad_inches=0, bbox_inches='tight')

                    plt.close()

            if config.add_raw:
                coronal_img_path = os.path.join(case_nii_path, folder, 'coronal_mask')
                if not os.path.exists(coronal_img_path): os.makedirs(coronal_img_path)
                for degree in range(0, 195, 15):
                    axis_order, angle = (2, 0, 1), degree
                    img_trans = np.transpose(img_cube, axis_order)
                    img_trans_degree = rotate_image(img_trans, 180-angle)

                    mask_trans = np.transpose(mask_cube, axis_order)
                    mask_trans_degree = rotate_image(mask_trans, 180-angle)

                    plt.rcParams["figure.figsize"] = (2, 2)
                    plt.axis('off')
                    plt.imshow(np.flipud(np.transpose(np.max(img_trans_degree, axis=1))), cmap='gray')

                    plt.imshow(np.flipud(np.transpose(np.max(mask_trans_degree, axis=1))), cmap=plt.cm.hot,
                               alpha=0.3, interpolation='nearest')

                    plt.subplots_adjust(top=1.00, bottom=0.00, left=0.00, right=1.00, hspace=0, wspace=0)
                    plt_name = '_'.join([model_name, folder, 'coronal', 'mask', '%03d' % angle])+'.png'
                    plt.savefig(os.path.join(coronal_img_path, plt_name), dpi=300, pad_inches=0, bbox_inches='tight')
                    plt.close()

            if config.add_raw:
                coronal_raw_path = os.path.join(case_nii_path, folder, 'coronal_raw')
                if not os.path.exists(coronal_raw_path): os.makedirs(coronal_raw_path)
                for degree in range(0, 195, 15):
                    axis_order, angle = (2, 0, 1), degree
                    img_trans = np.transpose(img_cube, axis_order)
                    img_trans_degree = rotate_image(img_trans, 180-angle)

                    mask_trans = np.transpose(mask_cube, axis_order)
                    mask_trans_degree = rotate_image(mask_trans, 180-angle)

                    rsz_det_trans = np.transpose(rsz_det_cube, axis_order)
                    rsz_det_trans_degree = rotate_image(rsz_det_trans, 180-angle)

                    plt.rcParams["figure.figsize"] = (2, 2)
                    plt.axis('off')
                    plt.imshow(np.flipud(np.transpose(np.max(img_trans_degree, axis=1))), cmap='gray')

                    plt.imshow(np.flipud(np.transpose(np.max(mask_trans_degree, axis=1))), cmap=plt.cm.hot,
                               alpha=0.3, interpolation='nearest')

                    plt.imshow(np.flipud(np.transpose(np.max(rsz_det_trans_degree, axis=1))),
                               cmap=plt.cm.hot, alpha=0.5, interpolation='nearest')

                    plt.subplots_adjust(top=1.00, bottom=0.00, left=0.00, right=1.00, hspace=0, wspace=0)
                    plt_name = '_'.join([model_name, folder, 'coronal', 'raw', '%03d' % angle])+'.png'
                    plt.savefig(os.path.join(coronal_raw_path, plt_name), dpi=300, pad_inches=0, bbox_inches='tight')
                    plt.close()

            # Save MIP Image: Coronal View (with predicted lesion)
            coronal_pred_path = os.path.join(case_nii_path, folder, 'coronal_pred')
            if not os.path.exists(coronal_pred_path): os.makedirs(coronal_pred_path)
            for degree in range(0, 195, 15):
                axis_order, angle = (2, 0, 1), degree
                img_trans = np.transpose(img_cube, axis_order)
                img_trans_degree = rotate_image(img_trans, 180-angle)

                pmask_trans = np.transpose(pmask_cube, axis_order)
                pmask_trans_degree = rotate_image(pmask_trans, 180-angle)

                rsz_pdet_trans = np.transpose(rsz_pdet_cube, axis_order)
                rsz_pdet_trans_degree = rotate_image(rsz_pdet_trans, 180-angle)

                plt.rcParams["figure.figsize"] = (2, 2)
                plt.axis('off')
                plt.imshow(np.flipud(np.transpose(np.max(img_trans_degree, axis=1))), cmap='gray')
                plt.imshow(np.flipud(np.transpose(np.max(pmask_trans_degree, axis=1))), cmap=plt.cm.pink,
                           alpha=0.3, interpolation='nearest')

                plt.imshow(np.flipud(np.mean(np.where(pcls_resize > 0.5, 1, 0), axis=1)),
                           cmap=plt.cm.gray, alpha=0.5, interpolation='nearest',
                           vmin=0, vmax=1)  # add

                plt.imshow(np.flipud(np.transpose(np.max(rsz_pdet_trans_degree, axis=1))),
                           cmap=plt.cm.jet, alpha=0.5, interpolation='nearest',
                           vmin=0, vmax=1)  # 2022.04.15

                plt.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0, wspace=0)
                plt_name = '_'.join([model_name, folder, 'coronal', 'pred', '%03d' % angle])+'.png'
                plt.savefig(os.path.join(coronal_pred_path, plt_name), dpi=300, pad_inches=0, bbox_inches='tight')
                plt.close()

            # Save MIP Image: Sagittal View (w/o predicted lesion)
            if config.add_raw:
                sagittal_raw_path = os.path.join(case_nii_path, folder, 'sagittal_raw')
                if not os.path.exists(sagittal_raw_path): os.makedirs(sagittal_raw_path)
                for degree in reversed(range(180, 375, 15)):
                    axis_order, angle = (0, 1, 2), degree
                    img_trans = np.transpose(img_cube, axis_order)
                    img_trans_degree = rotate_image(img_trans, angle)

                    mask_trans = np.transpose(mask_cube, axis_order)
                    mask_trans_degree = rotate_image(mask_trans, angle)

                    rsz_det_trans = np.transpose(rsz_det_cube, axis_order)
                    rsz_det_trans_degree = rotate_image(rsz_det_trans, 180-angle)

                    plt.rcParams["figure.figsize"] = (2, 2)
                    plt.axis('off')
                    plt.imshow(np.flipud(np.max(img_trans_degree, axis=1)), cmap='gray')
                    plt.imshow(np.flipud(np.transpose(np.max(mask_trans_degree, axis=1))), cmap=plt.cm.pink,
                               alpha=0.3, interpolation='nearest')
                    plt.imshow(np.flipud(np.transpose(np.max(rsz_det_trans_degree, axis=1))),
                               cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')

                    plt.subplots_adjust(top=1.00, bottom=0.00, left=0.00, right=1.00, hspace=0, wspace=0)
                    plt_name = '_'.join([model_name, folder, 'sagittal', 'raw', '%03d' % (360 - angle)])+'.png'
                    plt.savefig(os.path.join(sagittal_raw_path, plt_name), dpi=300, pad_inches=0, bbox_inches='tight')
                    plt.axis('off')

            # Save MIP Image: Sagittal View (with predicted lesion)
            sagittal_pred_path = os.path.join(case_nii_path, folder, 'sagittal_pred')
            if not os.path.exists(sagittal_pred_path): os.makedirs(sagittal_pred_path)
            for degree in reversed(range(180, 375, 15)):
                axis_order, angle = (0, 1, 2), degree
                img_trans = np.transpose(img_cube, axis_order)
                img_trans_degree = rotate_image(img_trans, angle)

                pmask_trans = np.transpose(pmask_cube, axis_order)
                pmask_trans_degree = rotate_image(pmask_trans, angle)

                rsz_pred_det_trans = np.transpose(rsz_pdet_cube, axis_order)
                rsz_pred_det_trans_degree = rotate_image(rsz_pred_det_trans, angle)

                plt.rcParams["figure.figsize"] = (2, 2)
                plt.axis('off')
                plt.imshow(np.flipud(np.max(img_trans_degree, axis=1)), cmap='gray')
                plt.imshow(np.flipud(np.max(pmask_trans_degree, axis=1)), cmap=plt.cm.pink,
                           alpha=0.3, interpolation='nearest')

                plt.imshow(np.flipud(np.mean(np.where(pcls_resize > 0.5, 1, 0), axis=1)),
                           cmap=plt.cm.gray, alpha=0.5, interpolation='nearest',
                           vmin=0, vmax=1)  # add

                plt.imshow(np.flipud(np.max(rsz_pred_det_trans_degree, axis=1)), cmap=plt.cm.jet,
                           alpha=0.5, interpolation='nearest',
                           vmin=0, vmax=1)  # 2022.04.15

                plt.subplots_adjust(top=1.00, bottom=0.00, left=0.00, right=1.00, hspace=0, wspace=0)
                plt_name = '_'.join([model_name, folder, 'sagittal', 'pred', '%03d' % (360 - angle)])+'.png'
                plt.savefig(os.path.join(sagittal_pred_path, plt_name), dpi=300, pad_inches=0, bbox_inches='tight')

            print(folder)