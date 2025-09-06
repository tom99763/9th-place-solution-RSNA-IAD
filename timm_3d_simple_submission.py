import os
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import polars as pl
import pydicom
import torch
import timm_3d
from scipy.ndimage import zoom

# -----------------------------
# Config
# -----------------------------
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
TARGET_SHAPE = (32, 384, 384)  # (D, H, W)
MODEL_NAME = os.getenv("TIMM3D_MODEL_NAME", "tf_efficientnetv2_s.in21k_ft_in1k")
CKPT_PATH = os.getenv("TIMM3D_CKPT")  # e.g., /kaggle/input/my-weights/best.ckpt

# Label order matches training script
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# -----------------------------
# I/O helpers
# -----------------------------
def _ordered_dcm_paths(series_dir: Path) -> List[Path]:
    dcm_files = [p for p in series_dir.rglob("*.dcm")]  # recurse just in case
    if not dcm_files:
        return []
    items: List[Tuple[Tuple[float, float], Path]] = []
    for p in dcm_files:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
            if hasattr(ds, "SliceLocation"):
                z = float(ds.SliceLocation)
            elif hasattr(ds, "ImagePositionPatient") and len(ds.ImagePositionPatient) >= 3:
                z = float(ds.ImagePositionPatient[-1])
            else:
                z = float(getattr(ds, "InstanceNumber", 0))
            items.append(((z, float(getattr(ds, "InstanceNumber", 0))), p))
        except Exception:
            items.append(((float('inf'), float('inf')), p))
    items.sort(key=lambda x: (x[0][0], x[0][1]))
    return [p for _, p in items]


def _read_dicom_frames_hu(path: Path) -> List[np.ndarray]:
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))
    frames: List[np.ndarray] = []
    if pix.ndim == 2:
        frames.append(pix.astype(np.float32) * slope + intercept)
    elif pix.ndim == 3:
        # If RGB (H,W,3)
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            gray = pix[..., 0].astype(np.float32)
            frames.append(gray * slope + intercept)
        else:
            for i in range(pix.shape[0]):
                frames.append(pix[i].astype(np.float32) * slope + intercept)
    return frames


def _load_volume(series_path: Path) -> Optional[np.ndarray]:
    paths = _ordered_dcm_paths(series_path)
    if not paths:
        return None
    slices: List[np.ndarray] = []
    for p in paths:
        try:
            frames = _read_dicom_frames_hu(p)
            if frames:
                # take first frame from each file for consistency and normalize per-slice
                frm = frames[0].astype(np.float32)
                mn, mx = float(frm.min()), float(frm.max())
                if mx - mn < 1e-6:
                    norm = np.zeros_like(frm, dtype=np.float32)
                else:
                    norm = (frm - mn) / (mx - mn)
                slices.append(norm)
        except Exception:
            continue
    if not slices:
        return None
    # Stack to (D, H, W)
    vol = np.stack(slices, axis=0).astype(np.float32)
    return vol


def _preprocess_volume(vol: np.ndarray) -> torch.Tensor:
    # Expect per-slice normalized volume in [0,1]; just resize.
    d, h, w = vol.shape
    zoom_factors = (
        TARGET_SHAPE[0] / max(d, 1),
        TARGET_SHAPE[1] / max(h, 1),
        TARGET_SHAPE[2] / max(w, 1),
    )
    vol_resized = zoom(vol, zoom_factors, order=1)

    # To tensor (1,1,D,H,W)
    t = torch.from_numpy(vol_resized).unsqueeze(0).unsqueeze(0).float()
    return t


# -----------------------------
# Model loading
# -----------------------------
_model = None



def load_model() -> torch.nn.Module:
    global _model
    if _model is not None:
        return _model

    model = timm_3d.create_model(
        MODEL_NAME,
        pretrained=False,
        num_classes=len(LABEL_COLS),
        in_chans=1,
        global_pool='avg',
        drop_path_rate=0.0,
        drop_rate=0.0,
    )

    # Resolve checkpoint path
    ckpt_path: Optional[Path] = None
    if CKPT_PATH and Path(CKPT_PATH).exists():
        ckpt_path = Path(CKPT_PATH)

    if ckpt_path is None:
        print("Warning: No checkpoint found. Using randomly initialized weights.")
    else:
        try:
            ckpt = torch.load(str(ckpt_path), map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)
            # Strip 'model.' prefix from a Lightning checkpoint
            cleaned = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    cleaned[k.replace('model.', '', 1)] = v
                elif not any(k.startswith(p) for p in ['loss_fn', 'pos_weight', 'hparams']):
                    cleaned[k] = v
            missing, unexpected = model.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
            print(f"Loaded checkpoint: {ckpt_path}")
        except Exception as e:
            print(f"Failed to load checkpoint {ckpt_path}: {e}. Proceeding with random weights.")

    model.eval().to(DEVICE)
    _model = model
    return _model


# -----------------------------
# Inference API expected by RSNA server
# -----------------------------
@torch.no_grad()
def _predict_inner(series_path: str) -> pl.DataFrame:
    series_dir = Path(series_path)
    vol = _load_volume(series_dir)
    if vol is None:
        return pl.DataFrame(data=[[0.1] * len(LABEL_COLS)], schema=LABEL_COLS, orient='row')

    x = _preprocess_volume(vol).to(DEVICE)  # (1,1,32,384,384)
    model = load_model()

    logits = model(x)  # (1, 14)
    probs = torch.sigmoid(logits).squeeze(0).float().cpu().numpy().tolist()

    return pl.DataFrame(data=[probs], schema=LABEL_COLS, orient='row')


def predict(series_path: str) -> pl.DataFrame:
    try:
        return _predict_inner(series_path)
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        return pl.DataFrame(data=[[0.1] * len(LABEL_COLS)], schema=LABEL_COLS, orient='row')
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# -----------------------------
# Entrypoint (works both locally and on Kaggle)
# -----------------------------
if __name__ == "__main__":
    st = time.time()
    try:
        import kaggle_evaluation.rsna_inference_server as rsna
        server = rsna.RSNAInferenceServer(predict)
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            server.serve()
        else:
            server.run_local_gateway()
            sub = pl.read_parquet('/kaggle/working/submission.parquet') if os.path.exists('/kaggle/working/submission.parquet') else None
            if sub is not None:
                print(sub.head())
    except Exception as e:
        # Simple local CLI usage: python timm_3d_simple_submission.py /path/to/series_dir
        import sys
        series = sys.argv[1] if len(sys.argv) > 1 else None
        if not series:
            print(f"Startup error: {e}. Provide a series path for local run.")
            raise SystemExit(1)
        df = predict(series)
        print(df)
    finally:
        print(f"Total time: {time.time() - st:.2f}s")
