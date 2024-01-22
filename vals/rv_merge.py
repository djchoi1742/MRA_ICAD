import os
import sys
import re
import pandas as pd
import glob
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
import ast
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

sys.path.append('/workspace/bitbucket/MRA')


if __name__ == '__main__':
    import argparse
    import logging

    parser = argparse.ArgumentParser()
    setup_config = parser.add_argument_group('setting')
    setup_config.add_argument('--base_path', type=str, dest='base_path', default='/workspace/MRA')
    setup_config.add_argument('--rv_idx', type=int, dest='rv_idx', default=1)
    setup_config.add_argument('--ss_idx', type=int, dest='ss_idx', default=2)
    setup_config.add_argument('--gt_df', type=str, dest='gt_df', default='validation_220725')
    config, unparsed = parser.parse_known_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)

    import warnings
    warnings.filterwarnings('ignore')

    # Set paths
    analysis_path = os.path.join(config.base_path, 'Validation', 'Analysis')
    gt_path = os.path.join(analysis_path, 'gt')
    review_path = os.path.join(analysis_path, 'review')
    result_path = os.path.join(analysis_path, 'sess_result')

    # reviewer_name = re.split('_', config.file_name)[-2]
    reviewer_dict = dict({1: 'JJH', 2: 'CSJ', 3: 'CHE', 4: 'BSH', 5: 'CHS'})

    # Load gt_df: validation_yymmdd.xlsx
    gt_df_name = config.gt_df + '.xlsx'
    gt_df_path = os.path.join(gt_path, gt_df_name)
    gt_df = pd.read_excel(gt_df_path)

    # Load df: reviewer result
    file_name = '_'.join(['MRA review', 'R'+str(config.rv_idx), reviewer_dict[config.rv_idx],
                          'session'+str(config.ss_idx)]) + '.xlsx'

    df_path = os.path.join(review_path, file_name)
    df = pd.read_excel(df_path)

    df['PATIENT_NO'] = df['PATIENT_NO'].astype('object')
    df['REPORT_COORD'] = df.apply(lambda row: '[' + ', '.join([str(row['SLICE_Z']), str(row['COORD_X']),
                                                               str(row['COORD_Y'])]) + ']', axis=1)

    loc_df = df
    loc_total = pd.DataFrame({'RANDOM_ID': pd.Series(dtype='object'), 'PATIENT_NO': pd.Series(dtype='object'),
                              'LESION_NUM': pd.Series(dtype='int'), 'REVIEWER_COORD': pd.Series(dtype='object'),
                              'REVIEWER_SCORE': pd.Series(dtype='object'), 'REVIEWER_TIME': pd.Series(dtype='int')})

    num = 0
    patient_id_list = []
    for row, idx in loc_df.iterrows():
        patient_id = loc_df.loc[row, 'PATIENT_NO']
        reading_time = int(loc_df.loc[row, 'READING_TIME'])
        if patient_id not in patient_id_list:
            patient_id_list.append(patient_id)
            num += 1
            loc_df_id = loc_df[loc_df['PATIENT_NO'] == patient_id]

            locs = []
            scores = []
            for loc_row, loc_idx in loc_df_id.iterrows():
                scores.append(ast.literal_eval(str(loc_df_id.loc[loc_row, 'CONFIDENCE_SCORE'])))

                if loc_df_id.loc[loc_row, 'REPORT_COORD'] == '[0, 0, 0]':
                    pass
                else:
                    locs.append(ast.literal_eval(loc_df_id.loc[loc_row, 'REPORT_COORD']))

            loc_total.loc[row, 'RANDOM_ID'] = 'TEST' + '%03d' % patient_id
            loc_total.loc[row, 'LESION_NUM'] = len(locs)
            loc_total.loc[row, 'PATIENT_NO'] = str(patient_id)
            loc_total.loc[row, 'REVIEWER_COORD'] = str(locs)
            loc_total.loc[row, 'REVIEWER_TIME'] = reading_time
            loc_total.loc[row, 'REVIEWER_SCORE'] = str(scores)

    loc_total.index = range(0, len(loc_total))

    # Set result name
    name_part = 'reviewer%02d' % config.rv_idx

    save_df_name = name_part + '_s%d' % config.ss_idx + '.xlsx'
    save_merge_name = name_part + '_merge_s%d' % config.ss_idx + '.xlsx'

    save_df_path = os.path.join(result_path, save_df_name)
    save_merge_path = os.path.join(result_path, save_merge_name)

    print(save_df_path)
    print(save_merge_path)

    loc_total.to_excel(save_df_path, index=False)  # output: reviewer00.xlsx

    loc_df_merge = pd.merge(loc_total, gt_df, on='RANDOM_ID')
    loc_df_merge.to_excel(save_merge_path, index=False)  # output: reviewer00_merge.xlsx
