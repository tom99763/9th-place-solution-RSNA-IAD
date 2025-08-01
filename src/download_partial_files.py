from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import numpy as np
import time
import os
from tqdm import tqdm

dataset_name = 'rsna-intracranial-aneurysm-detection'
mf_dicom_uids = pd.read_csv("./multiframe_dicoms.csv")
# We don't want to include multiframe dicoms as we can't get there z axis
# Discussion: https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/discussion/591546
ignore_uids = [
    "1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068",
    "1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444",
    "1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242"
] + list(mf_dicom_uids["SeriesInstanceUID"])



def download_by_location(api, df_loc):
    for i in tqdm(range(df_loc.shape[0])):
        sample = df_loc.iloc[i]
        SeriesInstanceUID = sample['SeriesInstanceUID']
        SOPInstanceUID = sample['SOPInstanceUID']
        if SeriesInstanceUID not in ignore_uids:
            path = f'series/{SeriesInstanceUID}/{SOPInstanceUID}.dcm'
            api.competition_download_file(dataset_name, path, path='./rsna_data/loc_slices')

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


