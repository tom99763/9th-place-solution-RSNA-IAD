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

def load_series2vol(series_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(series_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
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
            image_path = os.path.join(cfg.series_path, uid)
            preds = []
            for scale in cfg.tta.scales:
                image = load_series2vol(image_path)
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