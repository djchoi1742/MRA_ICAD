import os
import pandas as pd
import glob
import numpy as np
import nibabel as nib
import ast
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import datetime
import logging
import argparse
import re


def _distance_3d(p1, p2):
    squared_dist = np.sum((np.array(p1) - np.array(p2)) ** 2, axis=0)
    dist = np.sqrt(squared_dist)
    return dist


def match_lesion(each_rv_coord, k_group_coords):
    dist_list = []
    for g in k_group_coords:
        btn_distance = _distance_3d(each_rv_coord, g)
        if btn_distance < 20:
            dist_list.append(btn_distance)

    if len(dist_list) > 0:
        return True, np.min(dist_list)
    else:
        return False, None
        

rv_df = pd.DataFrame({'RANDOM_ID': pd.Series(dtype='object'), 'LABEL': pd.Series(dtype='int'),
                      'REVIEWER_COORD': pd.Series(dtype='object'), 'LesionID': pd.Series(dtype='int'),
                      'DISTANCE': pd.Series(dtype='float'), 'Rating': pd.Series(dtype='int'),
                      'IS_LESION': pd.Series(dtype='bool')})

# truth_df = pd.DataFrame({'RANDOM_ID': pd.Series(dtype='object'), 'LABEL': pd.Series(dtype='int'),
#                          'LesionID': pd.Series(dtype='int'), 'Weight': pd.Series(dtype='float')})


group_colors = {1: 'firebrick', 2: 'royalblue', 3: 'gold', 4: 'darkturquoise', 5: 'navy', 6: 'darkorange'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    setup_config = parser.add_argument_group('setting')
    setup_config.add_argument('--base_path', type=str, dest='base_path', default='/workspace/MRA')
    setup_config.add_argument('--rv_idx', type=int, dest='rv_idx', default=1)
    setup_config.add_argument('--ss_idx', type=int, dest='ss_idx', default=2)
    setup_config.add_argument('--coord_df', type=str, dest='coord_df', default='coord_group_220818')

    config, unparsed = parser.parse_known_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)

    import warnings
    warnings.filterwarnings('ignore')

    raw_path = os.path.join(config.base_path, 'RAW')
    analysis_path = os.path.join(config.base_path, 'Validation', 'Analysis')
    
    gt_path = os.path.join(analysis_path, 'gt')
    result_path = os.path.join(analysis_path, 'sess_result')

    name_part = 'reviewer%02d' % config.rv_idx
    sess_part = 'sess%d' % config.ss_idx
    reviewer_xlsx = name_part + '_merge_s%d' % config.ss_idx + '.xlsx'
    rv_save_name = name_part + '_result_s%d' % config.ss_idx + '.xlsx'
    print('**', reviewer_xlsx, rv_save_name)

    plot_path = os.path.join(analysis_path, '_'.join([name_part, sess_part]))
    if not os.path.exists(plot_path):
        os.makedirs(plot_path)

    reviewer_path = os.path.join(os.path.join(result_path, reviewer_xlsx))
    df = pd.read_excel(reviewer_path)
    print('**Load: ', reviewer_path)

    coord_xlsx = config.coord_df + '.xlsx'
    coord_df = pd.read_excel(os.path.join(gt_path, coord_xlsx))
    coord_df = coord_df[coord_df['INVALID'] == 0]

    rv_df_idx = 0

    for idx, row in df.iterrows():
        random_id = df.loc[idx, 'RANDOM_ID']
        random_id_idx = int(re.search('\d{3}', random_id).group())

        select = df[df['RANDOM_ID'] == random_id]

        institution, label, vessel_num = df.loc[idx, ['INSTITUTION', 'LABEL', 'VESSEL_NUM']]

        if institution == 'snubh':
            folder_name = 'clinical'
        elif institution == 'cusmh':
            folder_name = 'external'
        else:
            raise ValueError

        if label == 0:
            sub_folder_name, case_title, case_color = 'normal_mask', 'Normal group', 'navy'
        elif label == 1:
            sub_folder_name, case_title, case_color = 'stenosis_mask', 'Disease group', 'firebrick'
        else:
            raise ValueError

        mask_name = select['NII_MASK'].tolist()[0]
        mask_path = os.path.join(raw_path, folder_name, sub_folder_name, mask_name)

        local_coord = select['LOCAL_COORD'].item()
        reviewer_coord, reviewer_score = select['REVIEWER_COORD'].item(), select['REVIEWER_SCORE'].item()
        lc_coord_list = ast.literal_eval(local_coord)
        rv_coord_list, rv_score_list = ast.literal_eval(reviewer_coord), ast.literal_eval(reviewer_score)
        s_valid, e_valid = select['S_VALID'].item(), select['E_VALID'].item()

        coord_mask = nib.load(mask_path).dataobj[:, :, s_valid:e_valid]
        coord_mask = np.transpose(coord_mask, (2, 0, 1))
        coord_pixel = np.where(coord_mask == 1)

        fig = plt.figure()
        fig.set_size_inches((10, 10))
        ax = fig.add_subplot(projection='3d')
        ax.patch.set_facecolor('snow')
        ax.view_init(270, 270)

        vessels_x, vessels_y, vessels_z = coord_pixel[1], coord_pixel[2], coord_pixel[0]
        ax.scatter(vessels_x, vessels_y, vessels_z, s=0.3, alpha=0.1, color='lightgray')
        ax.set_xlim(np.min(vessels_x) - 50, np.max(vessels_x) + 50)
        ax.set_ylim(np.min(vessels_y) - 50, np.max(vessels_y) + 50)
        ax.set_zlim(np.min(vessels_z), np.max(vessels_z))

        check_color = 'blue' if vessel_num == len(rv_coord_list) else 'red'

        if label == 1:
            coord_df_id = coord_df[coord_df['RANDOM_ID'] == random_id]
            gt_coords = coord_df_id['LOCAL_COORD'].tolist()
            group_coords = coord_df_id['KMEANS_GROUP'].tolist()

            for c_idx, c_row in coord_df_id.iterrows():
                lz, lx, ly =  ast.literal_eval(coord_df_id.loc[c_idx, 'LOCAL_COORD'])
                k_group = coord_df_id.loc[c_idx, 'KMEANS_GROUP']

                ax.text(lx, ly, lz, s=4, fontsize=8, va='center', ha='center',
                        text=str(k_group), color=group_colors[k_group], zorder=3)

            # for each_group in sorted(list(set(group_coords))):
            #     truth_df.loc[truth_idx] = random_id, 1, each_group, 1 / vessel_num
            #     truth_idx += 1

        else:
            # truth_df.loc[truth_idx] = random_id, 0, 0, 0
            # truth_idx += 1
            predict_coord = []

        rv_df_id = pd.DataFrame({'RANDOM_ID': pd.Series(dtype='object'), 'LABEL': pd.Series(dtype=np.int),
                                 'REVIEWER_COORD': pd.Series(dtype='object'),
                                 'LesionID': pd.Series(dtype=np.int), 'DISTANCE': pd.Series(dtype=np.float),
                                 'Rating': pd.Series(dtype='int'), 'IS_LESION': pd.Series(dtype=np.bool)})

        if len(rv_coord_list) == 0:
            rv_df_id.loc[rv_df_idx] = random_id, label, np.NaN, np.NaN, np.NaN, rv_score_list[0], np.NaN
            rv_df_idx += 1

            print(random_id, label, np.NaN, np.NaN, np.NaN, rv_score_list[0], np.NaN)

        else:
            for rv_coord, rv_score in zip(rv_coord_list, rv_score_list):
                matched_dict, distance_dict = {}, {}

                if label == 1:  # Disease group
                    for each_group in sorted(list(set(group_coords))):
                        g_coords = coord_df_id[coord_df_id['KMEANS_GROUP'] == each_group]['LOCAL_COORD'].tolist()
                        g_coords = [*map(ast.literal_eval, g_coords)]
                        matched, distance = match_lesion(rv_coord, g_coords)

                        if matched:
                            matched_dict[each_group] = matched
                            distance_dict[each_group] = distance

                    matched_list = [*matched_dict.values()]
                    distance_list = [*distance_dict.values()]

                    if len(matched_list) >= 1:
                        match_group = [k for k, v in distance_dict.items() if v == np.min(distance_list)][0]
                        distance, is_lesion = distance_dict[match_group], True

                    else:  # False positive in Disease group (no match vessel)
                        match_group, distance, is_lesion = np.NaN, np.NaN, False

                else:  # False positive in Normal group
                    match_group, distance, is_lesion = np.NaN, np.NaN, False

                rv_df_id.loc[rv_df_idx] = random_id, label, rv_coord, match_group, distance, rv_score, is_lesion

                print(random_id, label, rv_coord, match_group, distance, rv_score, is_lesion)

                rv_df_idx += 1

                rv_color = 'black' if is_lesion else 'darkgray'
                z, x, y = rv_coord
                ax.text(x=x, y=y, z=z, s=8, fontsize=25, va='center', ha='center', text='*', color=rv_color)

        rv_df = rv_df.append(rv_df_id)

        rv_df_id_is_lesion = rv_df_id['IS_LESION'].tolist()
        tp_lesions = np.sum([x == True for x in rv_df_id_is_lesion])
        fp_lesions = np.sum([x == False for x in rv_df_id_is_lesion])

        result_png = '_'.join([sess_part, name_part, random_id]) + '.png'
        plt.figtext(0.5, 0.79, ', '.join(['Session %d' % config.ss_idx, name_part.capitalize(), random_id,
                                          case_title]), ha='center', color=case_color)

        plt.figtext(0.5, 0.77, '# of lesions: %d, Recorded lesions: %d' % (vessel_num, len(rv_coord_list)),
                    ha='center', color=check_color)
        plt.figtext(0.5, 0.75, 'True positive lesions: %d, False positive lesions: %d' %
                    (tp_lesions, fp_lesions), ha='center')
        plt.savefig(os.path.join(plot_path, result_png), dpi=300, pad_inches=0, bbox_inches='tight')
        plt.close()

    # if config.truth_df:
    #     truth_df.to_excel(os.path.join(gt_path, 'truth_220622.xlsx'), index=False)

    rv_df.to_excel(os.path.join(result_path, rv_save_name), index=False)  # output: reviewer00_result_s1.xlsx
