""" Script to perform inference with vesselFM."""

import logging
import warnings
from pathlib import Path

import torch
import torch.nn.functional as F
import hydra
import numpy as np
from tqdm import tqdm
from huggingface_hub import hf_hub_download
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist
from utils.data import generate_transforms
from utils.io import determine_reader_writer
from preprocess import *
import os
import SimpleITK as sitk
import pydicom
from concurrent.futures import ThreadPoolExecutor
from collections import Counter


warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def load_model(cfg, device):
    try:
        logger.info(f"Loading model from {cfg.ckpt_path}.")
        ckpt = torch.load(Path(cfg.ckpt_path), map_location=device, weights_only=False)['state_dict']
        ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
        print('load local ckpt')
    except Exception as e:
        print(f'Error:{e}')
        logger.info(f"Loading model from Hugging Face.")
        hf_hub_download(repo_id='bwittmann/vesselFM', filename='meta.yaml')  # required to track downloads
        ckpt = torch.load(
            hf_hub_download(repo_id='bwittmann/vesselFM', filename='vesselFM_base.pt'),
            map_location=device, weights_only=True
        )
        print('load hugging face ckpt')

    model = hydra.utils.instantiate(cfg.model)
    model.load_state_dict(ckpt)
    return model


def load_series2vol(series_path, series_id=None, spacing_tolerance=1e-3, resample=False, default_thickness=1.0, max_workers=20):
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
        return (int(ds.Rows), int(ds.Columns)), f

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        sizes = list(ex.map(get_size, all_files))

    # Pick the most common size
    most_common_size = Counter(s[0] for s in sizes).most_common(1)[0][0]
    files = [f for (sz, f) in sizes if sz == most_common_size]

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
    return volume

def resample(image, factor=None, target_shape=None):
    if factor == 1:
        return image

    if target_shape:
        _, _, new_d, new_h, new_w = target_shape
    else:
        _, _, d, h, w = image.shape
        new_d, new_h, new_w = int(round(d / factor)), int(round(h / factor)), int(round(w / factor))
    return F.interpolate(image, size=(new_d, new_h, new_w), mode="trilinear", align_corners=False)


def read_dicom_info(path):
    """Read DICOM metadata needed for sorting and decoding."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    try:
        z = float(ds.ImagePositionPatient[2])  # Preferred
    except AttributeError:
        z = float(ds.InstanceNumber)           # Fallback
    return path, z, ds

def read_pixel_data(path, rescale_slope, rescale_intercept):
    """Read pixel data and apply rescale."""
    ds = pydicom.dcmread(path)  # full read with pixels
    arr = ds.pixel_array.astype(np.float32)
    if rescale_slope is not None and rescale_intercept is not None:
        arr = arr * rescale_slope + rescale_intercept
    return arr

def load_volume_fast(directory, max_workers=20):
    dcm_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.dcm')]
    if not dcm_files:
        raise RuntimeError(f"No DICOM files found in {directory}")

    # Step 1: Read metadata in parallel (stop_before_pixels=True)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        meta_info = list(executor.map(read_dicom_info, dcm_files))

    # Step 2: Pick most common size
    sizes = [(int(ds.Rows), int(ds.Columns)) for _, _, ds in meta_info]
    most_common_size = Counter(sizes).most_common(1)[0][0]

    # Step 3: Keep only files with the most common size
    filtered_info = [(p, z, ds) for (p, z, ds) in meta_info if (int(ds.Rows), int(ds.Columns)) == most_common_size]

    # Step 4: Sort by slice position
    filtered_info.sort(key=lambda x: x[1])
    sorted_paths = [m[0] for m in filtered_info]
    first_ds = filtered_info[0][2]
    rescale_slope = getattr(first_ds, "RescaleSlope", None)
    rescale_intercept = getattr(first_ds, "RescaleIntercept", None)

    # Step 5: Get slice shape
    height, width = most_common_size
    depth = len(sorted_paths)

    volume = np.zeros((depth, height, width), dtype=np.float32)

    def load_into_array(idx_path):
        idx, path = idx_path
        volume[idx] = read_pixel_data(path, rescale_slope, rescale_intercept)

    # Step 6: Load pixels in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(load_into_array, enumerate(sorted_paths)))

    return volume

@hydra.main(config_path="configs", config_name="extraction", version_base="1.3.2")
def main(cfg):
    # seed libraries
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # set device
    logger.info(f"Using device {cfg.device}.")
    device = cfg.device

    # load model and ckpt
    print('load_model')
    model = load_model(cfg, device)
    model.to(device)
    model.eval()

    # init pre-processing transforms
    print('process')
    transforms = generate_transforms(cfg.transforms_config)

    # i/o
    output_folder = Path(cfg.output_folder)
    output_folder.mkdir(exist_ok=True)

    series_paths = cfg.series_path
    logger.info(f"Found {len(os.listdir(series_paths))} series in {cfg.series_path}.")

    # init sliding window inferer
    logger.debug(f"Sliding window patch size: {cfg.patch_size}")
    logger.debug(f"Sliding window batch size: {cfg.batch_size}.")
    logger.debug(f"Sliding window overlap: {cfg.overlap}.")
    inferer = SlidingWindowInfererAdapt(
        roi_size=cfg.patch_size, sw_batch_size=cfg.batch_size, overlap=cfg.overlap,
        mode=cfg.mode, sigma_scale=cfg.sigma_scale, padding_mode=cfg.padding_mode
    )

    with torch.no_grad():
        for idx, uid in tqdm(enumerate(sorted(os.listdir(cfg.series_path))), total=len(os.listdir(series_paths)), desc="Processing series."):
            if os.path.exists(f'{cfg.output_folder}/{uid}'):
                continue
            image_path = os.path.join(cfg.series_path, uid)
            preds = []
            for scale in cfg.tta.scales:
                image = load_volume_fast(image_path)
                image = transforms(image.astype(np.float32))[None].to(device)

                # apply test time augmentation
                if cfg.tta.invert:
                    image = 1 - image if image.mean() > cfg.tta.invert_mean_thresh else image

                if cfg.tta.equalize_hist:
                    image_np = image.cpu().squeeze().numpy()
                    image_equal_hist_np = equalize_hist(image_np, nbins=cfg.tta.hist_bins)
                    image = torch.from_numpy(image_equal_hist_np).to(image.device)[None][None]

                original_shape = image.shape
                image = resample(image, factor=scale)
                logits = inferer(image, model)
                logits = resample(logits, target_shape=original_shape)
                preds.append(logits.cpu().squeeze())

            # merging
            if cfg.merging.max:
                pred = torch.stack(preds).max(dim=0)[0].sigmoid()
            else:
                pred = torch.stack(preds).mean(dim=0).sigmoid()

            extract_and_save(uid, cfg.output_folder, image, pred, model,
                         target_layer = cfg.target_layer,
                         N = cfg.num_sampling_points, threshold = cfg.threshold)
    logger.info("Done.")


if __name__ == "__main__":
    main()