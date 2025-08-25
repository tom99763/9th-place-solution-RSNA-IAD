import pandas as pd
import numpy as np
import SimpleITK as sitk
import pydicom
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import cupy as cp
from cupyx.scipy.ndimage import zoom
import ast

import hydra
from omegaconf import OmegaConf
from pathlib import Path

import pytorch_lightning as pl

from tqdm import tqdm





def load_series2vol(series_path, series_id=None, spacing_tolerance=1e-3, resample=False, default_thickness=1.0, max_workers=24):
    reader = sitk.ImageSeriesReader()

    # Get all series IDs
    series_ids = reader.GetGDCMSeriesIDs(series_path)
    if not series_ids:
        raise RuntimeError(f"No DICOM series found in {series_path}")

    # Pick first if not specified
    series_id = str(series_ids[0] if series_id is None else series_id)

    # Get file names for the series
    all_files = reader.GetGDCMSeriesFileNames(series_path, series_id)

    # --- Parallel metadata read (fast size check) ---
    def get_size(f):
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        return (int(ds.Rows), int(ds.Columns)), f, ds

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        sizes = list(ex.map(get_size, all_files))


    # Pick the most common size
    most_common_size = Counter(s[0] for s in sizes).most_common(1)[0][0]
    files = [f for (sz, f, _) in sizes if sz == most_common_size]
    sortedfiles = sorted(sizes, key=lambda s: s[2].ImagePositionPatient[-1])
    sortedfiles = [s[1] for s in sortedfiles]

    # --- Now read the actual image series ---
    reader.SetFileNames(files)
    image = reader.Execute()

    # --- Fix zero thickness ---
    spacing = list(image.GetSpacing())
    if spacing[2] == 0:
        spacing[2] = default_thickness
        image.SetSpacing(spacing)

    # --- Optional resample ---
    if resample and abs(spacing[2] - spacing[0]) > spacing_tolerance:
        new_spacing = [spacing[0], spacing[1], spacing[0]]
        new_size = [
            int(round(image.GetSize()[0] * spacing[0] / new_spacing[0])),
            int(round(image.GetSize()[1] * spacing[1] / new_spacing[1])),
            int(round(image.GetSize()[2] * spacing[2] / new_spacing[2]))
        ]
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        image = resampler.Execute(image)

    # Convert to numpy array
    volume = sitk.GetArrayFromImage(image)
    return volume, sortedfiles

def apply_cta_window(image, window_level=300, window_width=600):
    """
    Applies a window to the HU values to highlight blood vessels.
    """
    min_val = window_level - window_width // 2
    max_val = window_level + window_width // 2
    windowed_image = np.clip(image, min_val, max_val)
    return windowed_image


def normalize(image):
    """
    Normalizes the image intensity to the range [0, 1].
    """
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val > min_val:
        return (image - min_val) / (max_val - min_val)
    return image


def process_single_series(uid, series_path, traindf, labeldf, cfg):
    vol,files = load_series2vol(series_path, uid)
    row = traindf[traindf["SeriesInstanceUID"] == uid]
    modality = row["Modality"].iloc[0]
    sopinstances = [f.split("/")[-1][:-4] for f in files]
    
    tmp = pydicom.dcmread(f"{files[0]}").pixel_array

    assert tmp.shape[0] == vol.shape[1] and tmp.shape[1] == vol.shape[2]

    depth,height,width = vol.shape

    target_depth  = cfg.preprocess.volume.depth
    target_height = cfg.preprocess.volume.height
    target_width  = cfg.preprocess.volume.width

    vol = cp.asarray(vol)
    vol = zoom(vol, (target_depth/depth, target_height/height, target_width/width), order=3)
    vol = cp.asnumpy(vol)


    coords = [] #z,y,x
    location_labels = []
    for _, label in labeldf[labeldf["SeriesInstanceUID"] == uid].iterrows():
        val = ast.literal_eval(label["coordinates"])
        sopuid = label["SOPInstanceUID"]
        z = sopinstances.index(sopuid)
        loc = label["location"]
        location_labels.append(LABELS_TO_IDX[loc])
        # assert z
        coords.append((int(z*target_depth/depth)
                       , int(val["y"]*target_height/height)
                       , int(val["x"]*target_width/width)))

    if modality == "CTA":
        vol = apply_cta_window(vol
                               ,window_level=cfg.preprocess.cta.window_level
                               ,window_width=cfg.preprocess.cta.window_width
                               )
    vol = (normalize(vol) * 255).astype(np.uint8)

    ## TODO: Add output for faulty labels

    np.savez(f"{cfg.preprocess.data_path}/processed/{uid}.npz"
             , vol=vol
             , coords=np.array(coords)
             , location_labels=np.array(location_labels)
             )


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    root_data_path = Path(cfg.preprocess.data_path)
    traindf = pd.read_csv(root_data_path / "train.csv")
    labeldf = pd.read_csv(root_data_path / "train_localizers.csv")
    traindf = traindf[traindf["SeriesInstanceUID"].isin(labeldf["SeriesInstanceUID"])].reset_index(drop=True)

    for uid in tqdm(traindf["SeriesInstanceUID"].unique().tolist()):
        try:
            process_single_series(uid, root_data_path / f"series/{uid}", traindf, labeldf, cfg)
        except Exception as e:
            print(f"unable to process {uid}")
            print(e)


if __name__ == "__main__":

    LABELS_TO_IDX = {
        'Anterior Communicating Artery': 1,
        'Basilar Tip': 2,
        'Left Anterior Cerebral Artery': 3,
        'Left Infraclinoid Internal Carotid Artery': 4,
        'Left Middle Cerebral Artery': 5,
        'Left Posterior Communicating Artery': 6,
        'Left Supraclinoid Internal Carotid Artery': 7,
        'Other Posterior Circulation': 8,
        'Right Anterior Cerebral Artery': 9,
        'Right Infraclinoid Internal Carotid Artery': 10,
        'Right Middle Cerebral Artery': 11,
        'Right Posterior Communicating Artery': 12,
        'Right Supraclinoid Internal Carotid Artery': 13
    }
    main()

