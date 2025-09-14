"""Visualize a random (or specified) nnU-Net case (image + label) from Dataset ID.

Usage examples:
  python3 visualize_nnunet_sample.py --dataset-id 001 --case-id bea9a55937b6f34672075975163520c8
  python3 visualize_nnunet_sample.py --dataset-id 902 --case-id fe3f842e426f615b9ebbf72f729cdbc7
  python3 visualize_nnunet_sample.py --dataset-id 902 --num-slices 12 --save out.png

Assumes environment variables or defaults:
  nnUNet_raw = ./data/nnUNet_raw

Shows axial slices with mask overlay. Random selection weighted to cases having positives unless --include-neg is set.
"""
import argparse
import os
import random
from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

RAW_BASE = Path(os.environ.get("nnUNet_raw", "./data/nnUNet_raw"))


def list_cases(dataset_id: int, split: str = "imagesTr"):
    ds_glob = f"Dataset{dataset_id:03d}_*"
    # Assume single match
    matches = list(RAW_BASE.glob(ds_glob))
    if not matches:
        raise FileNotFoundError(f"No dataset folder matching {ds_glob} under {RAW_BASE}")
    ds_root = matches[0]
    img_dir = ds_root / split
    cases = []
    for p in img_dir.glob("*_0000.nii.gz"):
        cid = p.name.replace('_0000.nii.gz', '')
        cases.append((cid, p))
    return ds_root, cases


def load_case(ds_root: Path, case_id: str, split: str = "imagesTr"):
    img_path = ds_root / split / f"{case_id}_0000.nii.gz"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = nib.load(str(img_path))
    vol = img.get_fdata().astype(np.float32)  # shape (X,Y,Z)
    vol = np.transpose(vol, (2, 1, 0))  # back to (Z,Y,X)

    mask = None
    if split == "imagesTr":
        lab_path = ds_root / "labelsTr" / f"{case_id}.nii.gz"
        if lab_path.exists():
            lab = nib.load(str(lab_path))
            mask = lab.get_fdata().astype(np.uint8)
            mask = np.transpose(mask, (2, 1, 0))
    return vol, mask


def pick_random_case(cases, ds_root: Path, prefer_positive: bool = True):
    if not prefer_positive:
        return random.choice(cases)[0]
    positives = []
    for cid, _ in cases:
        lab = ds_root / "labelsTr" / f"{cid}.nii.gz"
        if lab.exists():
            data = nib.load(str(lab)).get_fdata()
            if np.any(data > 0):
                positives.append(cid)
    if positives:
        return random.choice(positives)
    return random.choice(cases)[0]


def choose_slices(mask, depth: int, num_slices: int):
    if mask is not None and np.any(mask > 0):
        pos = np.where(mask > 0)[0]
        z_min, z_max = int(pos.min()), int(pos.max())
        span = max(1, z_max - z_min + 1)
        idxs = np.linspace(z_min, z_max, min(num_slices, span), dtype=int).tolist()
        # If fewer than requested, pad with random unique slices
        while len(idxs) < num_slices:
            cand = random.randint(0, depth - 1)
            if cand not in idxs:
                idxs.append(cand)
        return sorted(idxs)
    # fallback: evenly spaced
    return np.linspace(0, depth - 1, num_slices, dtype=int).tolist()


def plot_case(vol, mask, case_id: str, num_slices: int, save: str = None):
    depth = vol.shape[0]
    slices = choose_slices(mask, depth, num_slices)
    n = len(slices)
    fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n))
    if n == 1:
        axes = np.array([[axes[0], axes[1]]]) if isinstance(axes, np.ndarray) else np.array([[axes, axes]])
    for i, z in enumerate(slices):
        img = vol[z]
        axes[i, 0].imshow(img, cmap='gray')
        axes[i, 0].set_title(f"{case_id} z={z}")
        axes[i, 0].axis('off')
        overlay_ax = axes[i, 1]
        overlay_ax.imshow(img, cmap='gray')
        if mask is not None:
            overlay_ax.imshow(mask[z], cmap='autumn', alpha=0.5, vmin=0, vmax=1)
        overlay_ax.set_title("overlay" if mask is not None else "no mask")
        overlay_ax.axis('off')
    plt.tight_layout()

    plt.savefig(save, dpi=150, bbox_inches='tight')
    print(f"Saved figure to {save}")



def main():
    ap = argparse.ArgumentParser(description="Visualize a random or specified nnU-Net case")
    ap.add_argument('--dataset-id', default=902, type=int, required=False)
    ap.add_argument('--case-id', type=str, help='Specific case id (hashed) to visualize')
    ap.add_argument('--split', type=str, default='imagesTr', choices=['imagesTr', 'imagesTs'])
    ap.add_argument('--num-slices', type=int, default=48)
    ap.add_argument('--save', default="vizualize.png", type=str, help='Save path (PNG). If omitted, displays GUI window')
    ap.add_argument('--include-neg', action='store_true', help='Allow random selection of negative cases')
    args = ap.parse_args()

    ds_root, cases = list_cases(args.dataset_id, args.split)
    if not cases:
        raise SystemExit("No cases found")
    case_id = args.case_id or pick_random_case(cases, ds_root, prefer_positive=not args.include_neg)
    print(f"Selected case: {case_id}")
    vol, mask = load_case(ds_root, case_id, args.split)
    print(f"Volume shape: {vol.shape} | Mask present: {mask is not None} | Pos voxels: {int(mask.sum()) if mask is not None else 'N/A'}")
    plot_case(vol, mask, case_id, args.num_slices, args.save)

if __name__ == '__main__':
    main()
