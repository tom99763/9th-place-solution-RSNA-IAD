import os
import numpy as np
from rsna_datasets.cnn_25D_v2 import *
from tqdm import tqdm


def main():
    proc = DICOMPreprocessorKaggle()
    data_path = './data/series'

    if not os.path.exists('./data/processed'):
        os.makedirs('./data/processed')

    for uid in tqdm(os.listdir(data_path)):
        try:
            series_path = f'{data_path}/{uid}'
            vol = proc.process_series(series_path)
            d, h, w = vol.shape
            np.savez(f"./data/processed/{uid}.npz"
                     , vol=vol
                     , original_shape=np.array([d, h, w])
                     )
        except Exception as e:
            print(e)
            pass
if __name__ == '__main__':
    main()