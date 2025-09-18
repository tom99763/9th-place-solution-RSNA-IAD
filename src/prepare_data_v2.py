import os
import numpy as np
from rsna_datasets.cnn_25D_v2 import *
from tqdm import tqdm


def main():
    proc = DICOMPreprocessorKaggle()
    data_path = './data/series'

    if not os.path.exists('./data/processed'):
        os.makedirs('./data/processed')

    for name in tqdm(os.listdir(data_path)):
        series_path = f'{data_path}/{name}'
        vol = proc.process_series(series_path)
        d, h, w = vol.shape
        np.savez("./data/processed/{uid}.npz"
                 , vol=vol
                 , original_shape=np.array([d, h, w])
                 )

if __name__ == '__main__':
    main()