import numpy as np
import os
import pydicom as dcm
import nibabel as nib
import glob
import matplotlib.pyplot as plt
import pandas as pd
import skimage.transform
import cv2
import re
import itertools
import logging
import ast

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.disable(logging.WARNING)

import warnings
warnings.filterwarnings('ignore')


BASE_PATH = '/workspace/MRA'
INFO_PATH = os.path.join(BASE_PATH, 'info')
analysis_path = os.path.join(BASE_PATH, 'Validation', 'Analysis')

gt_path = os.path.join(analysis_path, 'gt')


ii = 1

if ii == 0:
    gt_df_name = 'validation_220708.xlsx'
    gt_df_path = os.path.join(gt_path, gt_df_name)

    gt_df = pd.read_excel(gt_df_path)
    gt_df = gt_df.drop('LOCAL_COORD', axis=1)

    coord_df_name = 'coord_group_220708.xlsx'
    coord_df_path = os.path.join(gt_path, coord_df_name)

    coord_df = pd.read_excel(coord_df_path)
    coord_df = coord_df[coord_df['INVALID'] == 0]
    coord_df = coord_df[['RANDOM_ID', 'LOCAL_COORD']]

    coord_total = pd.DataFrame({'RANDOM_ID': pd.Series(dtype='object'),
                                'LOCAL_COORD': pd.Series(dtype='object'),
                                'CHECK_SLICE': pd.Series(dtype='object')})

    num = 0
    random_id_list = []

    for idx, row in gt_df.iterrows():
        random_id = gt_df.loc[idx, 'RANDOM_ID']

        if random_id not in random_id_list:
            random_id_list.append(random_id)
            num += 1
            coord_df_id = coord_df[coord_df['RANDOM_ID'] == random_id]

            count = 0
            locs = []
            slide_indexes = []
            for loc_row, loc_idx in coord_df_id.iterrows():

                local_coord = coord_df_id.loc[loc_row, 'LOCAL_COORD']
                locs.append(ast.literal_eval(local_coord))
                slide_indexes.append(ast.literal_eval(local_coord)[0])

                count += 1

            coord_total.loc[idx, 'RANDOM_ID'] = str(random_id)
            coord_total.loc[idx, 'LOCAL_COORD'] = str(locs)
            coord_total.loc[idx, 'CHECK_SLICE'] = str([*set(sorted(slide_indexes))])

    merge_df = pd.merge(gt_df, coord_total, on='RANDOM_ID')
    merge_df['GROUP'] = 6

    merge_df.to_excel(os.path.join(gt_path, 'validation_220713.xlsx'), index=False)


if ii == 1:
    merge_df = pd.read_excel(os.path.join(gt_path, 'validation_220713-02.xlsx'))

    merge_df_snubh = merge_df[merge_df['INSTITUTION'] == 'snubh']
    merge_df_cusmh = merge_df[merge_df['INSTITUTION'] == 'cusmh']

    merge_df_snubh.to_excel(os.path.join(INFO_PATH, 'snubh11.xlsx'), index=False)
    merge_df_cusmh.to_excel(os.path.join(INFO_PATH, 'cusmh11.xlsx'), index=False)


