import pandas as pd
import numpy as np
import pydicom

from concurrent.futures import ThreadPoolExecutor
import cupy as cp
from cupyx.scipy.ndimage import zoom
import ast

import hydra
from omegaconf import OmegaConf
from pathlib import Path

import pytorch_lightning as pl

import os
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold

from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm

def normalize(volume):
    vmin, vmax = volume.min(), volume.max()
    if vmax > vmin:  # avoid div by zero
        volume = (volume - vmin) / (vmax - vmin)
    return volume

def load_dicom_folder(uid, folder_path: str, labeldf):
    """
    Load all DICOM series from a folder and stack them into 3D volumes.

    Parameters
    ----------
    folder_path : str
        Path to the folder containing DICOM files.

    Returns
    -------
    volumes : dict[str, np.ndarray]
        Dictionary mapping SeriesInstanceUID -> 3D numpy array (num_slices, H, W).
    metadata : dict[str, dict]
        Dictionary mapping SeriesInstanceUID -> metadata dictionary.
    """
    dicom_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith(".dcm")
    ]

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {folder_path}")

    # Group files by SeriesInstanceUID
    series_dict = defaultdict(list)
    for filepath in dicom_files:
        try:
            sopInstance = filepath.split("/")[-1][:-4]
            ds = pydicom.dcmread(filepath, force=True)
            series_dict[ds.SeriesInstanceUID].append((ds, sopInstance))
        except Exception as e:
            print(f"Skipping {filepath}, error: {e}")

    volumes = {}
    metadata = {}

    if 1 < len(series_dict.keys()):
        print(folder_path)

    for series_uid, slices in series_dict.items():
        # Sort slices (by z-pos if possible, otherwise by InstanceNumber)
        try:
            slices.sort(key=lambda s: float(s[0].ImagePositionPatient[2]))
        except Exception:
            slices.sort(key=lambda s: int(s[0].InstanceNumber))

        # Convert to numpy array
        volume = np.stack([s[0].pixel_array for s in slices], axis=0)
        sop_instance_uids_sorted = [s[1] for s in slices]
        slices = [s[0] for s in slices]

        # Apply RescaleSlope and RescaleIntercept if present (e.g., CT scans)
        slope = getattr(slices[0], "RescaleSlope", 1)
        intercept = getattr(slices[0], "RescaleIntercept", 0)
        volume = volume.astype(np.float32) * slope + intercept
        
        volumes[series_uid] = volume.squeeze()

        # Metadata
        metadata[series_uid] = {
            "Modality": getattr(slices[0], "Modality", None),
            "SeriesDescription": getattr(slices[0], "SeriesDescription", None),
            "PixelSpacing": getattr(slices[0], "PixelSpacing", None),
            "SliceThickness": getattr(slices[0], "SliceThickness", None),
            "Shape": volume.squeeze().shape,
            "Coords": [],
            "Label": []
        }
        
        labels = labeldf[labeldf["SeriesInstanceUID"] == uid]

        for _, label in labels.iterrows():
            sopInstance = label["SOPInstanceUID"]
            coord = label["coordinates"]
            coord = ast.literal_eval(coord)
            location = label["location"]

            if "f" in coord:
                for series_uid in metadata.keys():
                    metadata[series_uid]["Coords"].append([coord["f"], coord["y"], coord["x"]])
                    metadata[series_uid]["Label"].append(LABELS_TO_IDX[location])
            else:
                z = sop_instance_uids_sorted.index(sopInstance)
                for series_uid in metadata.keys():
                    metadata[series_uid]["Coords"].append([z, coord["y"], coord["x"]])
                    metadata[series_uid]["Label"].append(LABELS_TO_IDX[location])

    return volumes, metadata

def apply_ct_window(image: np.ndarray, window_level=250, window_width=700) -> np.ndarray:
    lower = window_level - (window_width / 2)
    upper = window_level + (window_width / 2)
    image = np.clip(image, lower, upper)
    image = ((image - lower) / (window_width + 1e-7)) * 255.0

    return image

def generate_sphere_mask(shape, center, radius):
    """
    Generate a binary segmentation mask of a sphere inside a 3D volume.

    Parameters
    ----------
    shape : tuple of int
        The shape of the output 3D array (e.g., (depth, height, width)).
    center : tuple of float
        The (z, y, x) coordinates of the sphere center.
    radius : float
        The radius of the sphere.

    Returns
    -------
    mask : np.ndarray
        A binary mask with ones inside the sphere and zeros outside.
    """
    z = np.arange(shape[0])[:, None, None]
    y = np.arange(shape[1])[None, :, None]
    x = np.arange(shape[2])[None, None, :]

    dist_sq = ((z - center[0])**2 +
               (y - center[1])**2 +
               (x - center[2])**2)

    mask = dist_sq <= radius**2
    return mask.astype(np.uint8)


def process_series(uid, root_path, labeldf, cfg):

    vol, metadata = load_dicom_folder(uid, root_path / f"series/{uid}", labeldf)
    metadata = metadata[uid]
    vol = vol[uid].squeeze()

    orig_z, orig_h, orig_w = metadata["Shape"]
    target_z, target_h, target_w = cfg.preprocess.target_shape

    vol = cp.asarray(vol)
    vol = zoom(vol, (target_z/orig_z, target_h/orig_h, target_w/orig_w), order=3)
    vol = cp.asnumpy(vol)

    mask = np.zeros_like(vol)

    if "CT" in metadata["Modality"]:
        vol = apply_ct_window(vol)
    vol = (normalize(vol) * 255.0).astype(np.uint8)

    for (cz,cy,cx) in metadata["Coords"]:

        mask += generate_sphere_mask(mask.shape
                , (cz * target_z / orig_z, cy * target_h / orig_h, cx * target_w / orig_w)
                , cfg.preprocess.radius)

    np.savez_compressed(root_path / f"processed/{uid}.npz", vol=vol, mask=mask, metadata=metadata)
        # Free CuPy memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

def run_parallel(uids, root_path, labeldf, cfg, num_workers=3):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_series, uid, root_path, labeldf, cfg): uid
            for uid in uids
        }

        results = []
        for future in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                uid = futures[future]
                print(f"Error processing {uid}: {e}")
        return results


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg):
    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)
    root_path = Path(cfg.preprocess.data_path)

    traindf = pd.read_csv(root_path / "train.csv")
    labeldf = pd.read_csv(root_path / "train_localizers.csv")

    skf = StratifiedKFold( n_splits=cfg.preprocess.n_folds
                         , random_state=cfg.seed
                         , shuffle=True)

    traindf["fold_id"] = 0
    for i, (_, test_index) in enumerate(skf.split(traindf["SeriesInstanceUID"], traindf["Aneurysm Present"])):
        traindf.loc[test_index, "fold_id"] = i

    uids = traindf["SeriesInstanceUID"].unique().tolist()

    run_parallel(uids, root_path, labeldf, cfg, num_workers=cfg.preprocess.cores)

    traindf.to_csv(root_path / "processed/train.csv", index=False)

if __name__ == "__main__":

    LABELS_TO_IDX = {
                'Anterior Communicating Artery': 0,
                'Basilar Tip': 1,
                'Left Anterior Cerebral Artery': 2,
                'Left Infraclinoid Internal Carotid Artery': 3,
                'Left Middle Cerebral Artery': 4,
                'Left Posterior Communicating Artery': 5,
                'Left Supraclinoid Internal Carotid Artery': 6,
                'Other Posterior Circulation': 7,
                'Right Anterior Cerebral Artery': 8,
                'Right Infraclinoid Internal Carotid Artery': 9,
                'Right Middle Cerebral Artery': 10,
                'Right Posterior Communicating Artery': 11,
                'Right Supraclinoid Internal Carotid Artery': 12
    }
    main()

