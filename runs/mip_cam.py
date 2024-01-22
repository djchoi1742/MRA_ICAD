import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import pydicom as dcm
import sys
sys.path.append('/workspace/bitbucket/MRA')
from data.setup_c import *
import models.model_c as model_ref
import models.metric as metric
from runs.cams import *

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure


def nii_extract_id(df, raw_path, patient_id):
    dfs = df[df['PATIENT_ID'] == int(patient_id)]

    total_slice = dfs['PIXEL_Z'].tolist()
    slice_start, slice_end = dfs['S_VALID'].tolist(), dfs['E_VALID'].tolist()
    reverse = dfs['REVERSE'].tolist()

    nii_raw_path = dfs.apply(lambda row: os.path.join(raw_path, row['NII_FOLDER'], row['NII_RAW']), axis=1).tolist()
    nii_mask_path = dfs.apply(lambda row: os.path.join(raw_path, row['NII_FOLDER'], row['NII_MASK']), axis=1).tolist()

    ste_start, ste_end = dfs['S_SLICE'].tolist(), dfs['E_SLICE'].tolist()
    ste_start, ste_end = [*map(str, ste_start)], [*map(str, ste_end)]

    return nii_raw_path, nii_mask_path, slice_start, slice_end, ste_start, ste_end, reverse, total_slice


def plot_3d(image, threshold=-0.75):
    projection = image.transpose(2, 1, 0)
    verts, faces, normals, values = measure.marching_cubes_lewiner(projection, threshold)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    mesh = Poly3DCollection(verts[faces], alpha=0.1)
    face_color = [0.0, 0.0, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    ax.set_xlim(0, projection.shape[0])
    ax.set_ylim(0, projection.shape[1])
    ax.set_zlim(0, projection.shape[2])

    # plt.show()


def create_mip(np_img, slice_num=15):
    ''' create the mip image from original image, slice_num is the number of
    slices for maximum intensity projection'''
    np_img = (np_img - np.mean(np_img)) / np.std(np_img)
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i - img_shape[0])
        np_mip[i, :, :] = np.amax(np_img[start:i + 1], 0)
    return np_mip


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feature_map(feature):
    feature = - (feature - np.mean(feature)) / np.std(feature)

    return np.mean(feature, axis=-1)


def feature_sigmoid(feature):
    feature = np.mean(feature, axis=-1)
    feature = - (feature - np.mean(feature)) / np.std(feature)
    feature_prob = sigmoid(feature)

    feature_prob[feature_prob < 0.75] = 0.0
    return feature_prob


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    setup_config = parser.add_argument_group('dataset setting')
    setup_config.add_argument('--data_path', type=str, dest='data_path', default='/workspace/MRA')
    setup_config.add_argument('--excel_name', type=str, dest='excel_name', default='snubh09')
    setup_config.add_argument('--exp_name', type=str, dest='exp_name', default='exp009')
    setup_config.add_argument('--seq_len', type=int, dest='seq_len', default=8)
    setup_config.add_argument('--seq_interval', type=int, dest='seq_interval', default=3)
    setup_config.add_argument('--only_ste', type=lambda x: x.title() in str(True), dest='only_ste', default=True)
    setup_config.add_argument('--each_ste', type=lambda x: x.title() in str(True), dest='each_ste', default=True)
    setup_config.add_argument('--one_hot', type=lambda x: x.title() in str(True), dest='one_hot', default=False)
    setup_config.add_argument('--model_name', type=str, dest='model_name', default='Model28')
    setup_config.add_argument('--f_num', type=str, dest='f_num', default='64,112,160,208')
    setup_config.add_argument('--serial', type=int, dest='serial', default=24)
    setup_config.add_argument('--image_size', type=int, dest='image_size', default=256)
    setup_config.add_argument('--channel_size', type=int, dest='channel_size', default=1)
    setup_config.add_argument('--patient_id', type=str, dest='patient_id', default='24807461')
    setup_config.add_argument('--type', type=int, dest='type', default=2)
    config, unparsed = parser.parse_known_args()
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

    serial_str = '%03d' % config.serial
    plot_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'plot-%s' % serial_str)
    cube_path = os.path.join(plot_path, 'cube')
    if not os.path.exists(cube_path): os.makedirs(cube_path)

    df_path = os.path.join(INFO_PATH, config.excel_name) + '.xlsx'
    df = pd.read_excel(df_path)

    if 'snubh' in config.excel_name:
        d_type = 'clinical'
    elif 'cusmh' in config.excel_name:
        d_type = 'external'
    else:
        raise ValueError('Invalid data type')

    img_size, img_c = config.image_size, config.channel_size
    seq_len = config.seq_len
    f_num = config.f_num

    RAW_PATH = os.path.join(ROOT_PATH, 'RAW', d_type)
    VIEW_PATH = os.path.join(ROOT_PATH, config.exp_name, 'view')
    if not os.path.exists(VIEW_PATH): os.makedirs(VIEW_PATH)

    raw_path = os.path.join(ROOT_PATH, 'RAW', d_type)
    val_raws, val_masks, val_s_sts, val_s_ens, val_ste_sts, val_ste_ens, val_revs, val_ns = \
        nii_extract_id(df=df, raw_path=raw_path, patient_id=config.patient_id)

    val_raws, val_masks, val_indexes, val_stes, val_revs = \
        nii_each_read(nii_raw=val_raws, nii_mask=val_masks,
                      slice_start=val_s_sts, slice_end=val_s_ens,
                      ste_start=val_ste_sts, ste_end=val_ste_ens, reverse=val_revs, n_end=val_ns,
                      seq_len=seq_len, seq_interval=seq_len, each_ste=config.each_ste)

    val = nii_data_setting(imgs=val_raws, masks=val_masks, indexes=val_indexes, stes=val_stes,
                           seq_len=config.seq_len, image_size=img_size, radius=80, one_hot=False,
                           augment=False, shuffle=False)

    input_size = [seq_len, img_size, img_size, img_c] if seq_len != 1 else [img_size, img_size, img_c]

    infer_name = config.model_name
    infer = getattr(model_ref, infer_name)(input_size=input_size, class_num=1, f_num=f_num, is_training=False,
                                           add_conv=config.add_conv)

    model = infer.model
    if config.type > 0:
        cam_model = infer.cam_model
    guided_model = metric.built_guided_model(model)

    if config.seq_len == 1:
        gen_grad_cam = gen_grad_cam_2d
    else:
    # elif config.each_ste:
        gen_grad_cam = gen_grad_cam_lstm
    # else:
    #     gen_grad_cam = gen_grad_cam_3d

    z_total = config.seq_len * val.cardinality().numpy()

    cam_cube = np.zeros([z_total, img_size, img_size])
    seg_cube = np.zeros([z_total, img_size, img_size])
    img_cube = np.zeros([z_total, img_size, img_size])
    cls_value = np.zeros([z_total])
    ste_value = np.zeros([z_total])

    seg0 = np.zeros([z_total, 32, 32])
    seg1 = np.zeros([z_total, 64, 64])
    seg2 = np.zeros([z_total, 128, 128])
    seg3 = np.zeros([z_total, 256, 256])

    metrics = ['DSC', 'AUC', 'METRIC']
    val_metric = metrics[config.type]

    weight_auc_path = os.path.join(config.data_path, config.exp_name, config.model_name, 'result-%03d' % config.serial)
    weight_auc_csv = pd.read_csv(os.path.join(weight_auc_path, '_'.join([config.exp_name, config.model_name,
                                                                         '%03d' % config.serial]) + '.csv'))
    weight_auc_csv = weight_auc_csv.sort_values(val_metric, ascending=False)  # DSC or AUC or METRIC
    all_ckpt_paths = list(weight_auc_csv['WEIGHT_PATH'][0:5])
    ckpt_path = all_ckpt_paths[0]
    print(ckpt_path)
    idx = 0
    for batch in val.batch(1):
        model.load_weights(ckpt_path)

        img, mask, ste, name = batch

        with tf.GradientTape() as tape:
            if config.type == 0:
                seg_prob = model(img)
            else:
                if config.type == 1:
                    cam_layers, cls_prob = cam_model(img)
                else:
                    cam_layers, seg_prob, cls_prob = cam_model(img)
                    seg_output = infer.seg_model(img)

        seg_list = [x for x in seg_output.values()]
        # seg_mean = [*map(feature_map, seg_list)]
        seg_mean = [*map(feature_sigmoid, seg_list)]

        img_cube[idx:idx + seq_len, :, :] = img[0, :, :, 0] if seq_len == 1 else img[0, :, :, :, 0]

        if config.type > 0:
            grad_cams = gen_grad_cam(cam_layers, cls_prob, tape, infer)
            gb = metric.guided_backprop(guided_model, img, infer.cam_layer_name)
            guided_grad_cams = gb * grad_cams
            ggc = metric.deprocess_image(guided_grad_cams)

            cam_cube[idx:idx + seq_len, :, :] = grad_cams[0, :, :, 0] if seq_len == 1 else grad_cams[0, :, :, :, 0]

            ste_value[idx:idx + seq_len] = ste.numpy()
            cls_value[idx:idx + seq_len] = cls_prob

        if config.type != 1:
            seg_cube[idx:idx + seq_len, :, :] = seg_prob[0, :, :, 0] if seq_len == 1 else seg_prob[0, :, :, :, 0]

            # seg0[idx:idx + seq_len, :, :] = seg_mean[0][0, :, :] if seq_len == 1 else seg_mean[0][0, :, :]
            # seg1[idx:idx + seq_len, :, :] = seg_mean[1][0, :, :] if seq_len == 1 else seg_mean[1][0, :, :]
            # seg2[idx:idx + seq_len, :, :] = seg_mean[2][0, :, :] if seq_len == 1 else seg_mean[2][0, :, :]
            # seg3[idx:idx + seq_len, :, :] = seg_mean[3][0, :, :] if seq_len == 1 else seg_mean[3][0, :, :]

        idx += seq_len

        print(img.shape, mask.shape, ste.shape, name)

    img_mip = create_mip(img_cube)

    if config.type != 1:
        seg_mip = create_mip(seg_cube)

    if config.type > 0:
        cam_mip = create_mip(cam_cube)

    png_name = config.patient_id + '.png'

    if config.type == 0:  # only segmentation
        for i in range(img_mip.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(2 * 3, 1 * 3))
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            img_mip_tr = np.transpose(img_mip, [1, 2, 0])
            seg_mip_tr = np.transpose(seg_mip, [1, 2, 0])

            ax[0].imshow(img_mip_tr[:, :, i], cmap='bone')
            ax[0].set_title(config.patient_id + '_%03d' % (i+1) + ': IMG MIP', fontsize=7, color='black')

            ax[1].imshow(seg_mip_tr[:, :, i], cmap='bone')
            ax[1].set_title('SEG MIP', fontsize=7, color='navy')

            save_path = os.path.join(cube_path, config.patient_id + '_img')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, config.patient_id + '_%03d' % (i+1) + '.png'), bbox_inches='tight')

            plt.clf()

    elif config.type == 1:  # only classification
        for i in range(img_mip.shape[0]):
            fig, ax = plt.subplots(1, 3, figsize=(3 * 3, 1 * 3))
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            img_mip_tr = np.transpose(img_mip, [1, 2, 0])
            cam_mip_tr = np.transpose(cam_mip, [1, 2, 0])

            ax[0].imshow(img_mip_tr[:, :, i], cmap='bone')
            ax[0].set_title(config.patient_id + '_%03d' % (i+1) + ': IMG MIP', fontsize=7, color='black')

            ax[1].imshow(cam_mip_tr[:, :, i], cmap='bone')
            ax[1].set_title('CAM MIP', fontsize=7, color='navy')

            ax[2].imshow(img_mip_tr[:, :, i], cmap='bone')
            ax[2].imshow(cam_mip_tr[:, :, i], cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
            ax[2].set_title('IMG & CAM MIP', fontsize=7, color='green')

            save_path = os.path.join(cube_path, config.patient_id + '_img')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, config.patient_id + '_%03d' % (i+1) + '.png'), bbox_inches='tight')

            plt.clf()

    elif config.type == 2:  # joint segmentation and classification
        for i in range(img_mip.shape[0]):
            fig, ax = plt.subplots(2, 4, figsize=(4 * 3, 2 * 3))
            axoff_fun = np.vectorize(lambda ax: ax.axis('off'))
            axoff_fun(ax)

            img_mip_tr = np.transpose(img_mip, [1, 2, 0])
            seg_mip_tr = np.transpose(seg_mip, [1, 2, 0])
            cam_mip_tr = np.transpose(cam_mip, [1, 2, 0])
            ste_v, cls_v = ste_value[i], cls_value[i]

            ax[0, 0].imshow(img_cube[i, :, :], cmap='bone')
            ax[0, 0].set_title(config.patient_id + '_%03d' % (i+1) + ': IMG', fontsize=7, color='black')

            ax[0, 1].imshow(seg_cube[i, :, :], cmap='bone')
            ax[0, 1].set_title(config.patient_id + '_%03d' % (i+1) + ': Mask', fontsize=7, color='black')

            ax[0, 2].imshow(img_cube[i, :, :], cmap='bone')
            ax[0, 2].imshow(cam_cube[i, :, :], cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
            ax[0, 2].set_title(config.patient_id + '_%03d' % (i+1) + ': CAM', fontsize=7, color='black')

            ax[1, 0].imshow(img_mip_tr[:, :, i], cmap='bone')
            ax[1, 0].set_title(config.patient_id + '_%03d' % (i+1) + ': IMG MIP', fontsize=7, color='black')

            ax[1, 1].imshow(seg_mip_tr[:, :, i], cmap='bone')
            ax[1, 1].set_title('SEG MIP' + ' Stenosis: %d' % ste_v, fontsize=7, color='black')

            ax[1, 2].imshow(cam_mip_tr[:, :, i], cmap='bone')
            ax[1, 2].set_title('CAM MIP' + ' Prob: %.3f' % cls_v, fontsize=7, color='navy')

            ax[1, 3].imshow(img_mip_tr[:, :, i], cmap='bone')
            ax[1, 3].imshow(cam_mip_tr[:, :, i], cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
            ax[1, 3].set_title('IMG & CAM MIP', fontsize=7, color='green')

            ax[0, 3].imshow(seg_mip_tr[:, :, i], cmap='bone')
            ax[0, 3].imshow(cam_mip_tr[:, :, i], cmap=plt.cm.seismic, alpha=0.5, interpolation='nearest')
            ax[0, 3].set_title('SEG MIP & CAM MIP', fontsize=7, color='green')

            save_path = os.path.join(cube_path, config.patient_id + '_img')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            plt.savefig(os.path.join(save_path, config.patient_id + '_%03d' % (i+1) + '.png'), bbox_inches='tight')

            plt.clf()

