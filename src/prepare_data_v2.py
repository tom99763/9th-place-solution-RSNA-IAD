import os
import numpy as np
from rsna_datasets.cnn_25D_v2 import *
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed


def process_and_save(uid, data_path):
    try:
        series_path = f"{data_path}/{uid}"
        vol, mask = proc.process_series(series_path)
        np.savez(f"./data/processed/{uid}.npz", vol=vol, mask=mask)
    except Exception as e:
        return f"{uid}: {e}"
    return None  # success

def main():

    if not os.path.exists('./data/processed'):
        os.makedirs('./data/processed')

    uids = os.listdir(data_path)

    with ProcessPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(process_and_save, uid, data_path): uid for uid in uids}
        for f in tqdm(as_completed(futures), total=len(futures)):
            err = f.result()
            if err:
                print(err)
               

if __name__ == '__main__':

    proc = DICOMPreprocessorKaggle()
    data_path = './data/series'
    main()
