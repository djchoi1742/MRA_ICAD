import numpy as np
import SimpleITK as sitk
import os
from os import listdir
from os.path import isfile, join
import glob
import matplotlib.pyplot as plt


def create_mip(np_img, slices_num=15):
    ''' create the mip image from original image, slice_num is the number of
    slices for maximum intensity projection'''
    np_img = (np_img - np.mean(np_img)) / np.std(np_img)
    img_shape = np_img.shape
    np_mip = np.zeros(img_shape)
    for i in range(img_shape[0]):
        start = max(0, i - img_shape[0])
        np_mip[i, :, :] = np.amax(np_img[start:i + 1], 0)
    return np_mip


def main():
    output_dir = '/data1/SNUBH/MRA/RAW/'
    files = '/data1/SNUBH/MRA/RAW/clinical/stenosis_nii/10051579.nii.gz'
    if files:
        sitk_img = sitk.ReadImage(files)
        np_img = sitk.GetArrayFromImage(sitk_img)
        np_mip = create_mip(np_img)
        sitk_mip = sitk.GetImageFromArray(np_mip)
        sitk_mip.SetOrigin(sitk_img.GetOrigin())
        sitk_mip.SetSpacing(sitk_img.GetSpacing())
        sitk_mip.SetDirection(sitk_img.GetDirection())
        writer = sitk.ImageFileWriter()
        writer.SetFileName(join(output_dir, 'mip.nii.gz'))
        writer.Execute(sitk_mip)


def nii_read():
    files = '/data1/SNUBH/MRA/RAW/clinical/stenosis_nii/10051579.nii.gz'
    sitk_img = sitk.ReadImage(files)
    np_img = sitk.GetArrayFromImage(sitk_img)
    np_mip = create_mip(np_img)
    return np_mip


if __name__ == '__main__':
    # main()
    from mpl_toolkits.mplot3d import Axes3D
    from skimage.transform import resize
    from matplotlib import cm

    # IMG_DIM = 50

    def make_ax(grid=False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(grid)
        return ax


    def explode(data):
        shape_arr = np.array(data.shape)
        size = shape_arr[:3] * 2 - 1
        exploded = np.zeros(np.concatenate([size, shape_arr[3:]]), dtype=data.dtype)
        exploded[::2, ::2, ::2] = data
        return exploded


    def expand_coordinates(indices):
        x, y, z = indices
        x[1::2, :, :] += 1
        y[:, 1::2, :] += 1
        z[:, :, 1::2] += 1
        return x, y, z


    def normalize(arr):
        arr_min = np.min(arr)
        return (arr - arr_min) / (np.max(arr) - arr_min)


    def scale_by(arr, fac):
        mean = np.mean(arr)
        return (arr - mean) * fac + mean


    def plot_cube(cube, angle=320):
        cube = normalize(cube)

        facecolors = cm.bone(cube)
        facecolors[:, :, :, -1] = cube
        facecolors = explode(facecolors)

        # import pdb; pdb.set_trace()

        filled = facecolors[:, :, :, -1] != 0

        y_lim, x_lim, z_lim = filled.shape
        x, y, z = expand_coordinates(np.indices(np.array(filled.shape) + 1))

        fig = plt.figure(figsize=(30 / 2.54, 30 / 2.54))
        ax = fig.gca(projection='3d')
        ax.view_init(30, angle)
        ax.set_xlim(right=x_lim)
        ax.set_ylim(top=y_lim)
        ax.set_zlim(top=z_lim)

        ax.voxels(x, y, z, filled, facecolors=facecolors, shade=False)
        plt.show()

    arr = nii_read()

    transformed = np.clip(scale_by(np.clip(normalize(arr) - 0.1, 0, 1) ** 0.4, 2) - 0.1, 0, 1)
    # tr_max = np.max(transformed)
    # transformed[transformed < tr_max] = 0

    resized = resize(transformed, (32, 32, 8), mode='constant')
    print(transformed.shape)
    plot_cube(resized)
