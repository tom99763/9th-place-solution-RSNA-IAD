from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import time

dataset_name = 'rsna-intracranial-aneurysm-detection'

def download_by_location(api, df_loc):
    for i in range(df_loc.shape[0]):
        sample = df_loc.iloc[i]
        SeriesInstanceUID = sample['SeriesInstanceUID']
        SOPInstanceUID = sample['SOPInstanceUID']
        path = f'series/{SeriesInstanceUID}/{SOPInstanceUID}.dcm'
        api.competition_download_file(dataset_name, path, path='./rsna_data/loc_slices')
        time.sleep(5)


def download_data():
    api = KaggleApi()
    api.authenticate()
    api.competition_download_file(
        dataset_name, 'train.csv', path='./rsna_data')
    api.competition_download_file(dataset_name, 'train_localizers.csv',
                                  path='./rsna_data')
    df_meta = pd.read_csv('./rsna_data/train.csv')
    df_loc = pd.read_csv('./rsna_data/train_localizers.csv')
    download_by_location(api, df_loc)


if __name__ == '__main__':
    download_data()