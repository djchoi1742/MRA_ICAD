import os
import pandas as pd
import logging
import argparse


truth_df = pd.DataFrame({'RANDOM_ID': pd.Series(dtype='object'), 'LABEL': pd.Series(dtype='int'),
                         'LesionID': pd.Series(dtype='int'), 'Weight': pd.Series(dtype='float')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    setup_config = parser.add_argument_group('setting')
    setup_config.add_argument('--base_path', type=str, dest='base_path', default='/workspace/MRA')
    setup_config.add_argument('--gt_df', type=str, dest='gt_df', default='validation_220708')
    setup_config.add_argument('--truth_df', type=str, dest='truth_df', default='truth_220708')
    config, unparsed = parser.parse_known_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    logging.disable(logging.WARNING)

    import warnings
    warnings.filterwarnings('ignore')

    analysis_path = os.path.join(config.base_path, 'Validation', 'Analysis')

    gt_path = os.path.join(analysis_path, 'gt')
    df_path = os.path.join(gt_path, config.gt_df + '.xlsx')
    df = pd.read_excel(df_path)

    truth_idx = 0
    for idx, row in df.iterrows():
        random_id, label, vessel_num = df.loc[idx, ['RANDOM_ID', 'LABEL', 'VESSEL_NUM']]

        if label == 1:
            df_id = df[df['RANDOM_ID'] == random_id]

            for num in range(1, vessel_num + 1):
                truth_df.loc[truth_idx] = random_id, 1, num, 1 / vessel_num
                truth_idx += 1

        else:
            truth_df.loc[truth_idx] = random_id, 0, 0, 0
            truth_idx += 1

    if config.truth_df:
        truth_df.to_excel(os.path.join(gt_path, config.truth_df + '.xlsx'), index=False)
