#!/usr/bin/env python3
"""Diagnostic script to identify problematic nnU-Net cases (empty or degenerate label volumes).

Usage:
  python check_nnunet_dataset.py \
      --dataset-path data/nnUNet_raw/Dataset901_RSNAAneurysm \
      [--limit 50] [--verbose]

It scans labelsTr/*.nii.gz and corresponding imagesTr/*_0000.nii.gz and reports:
  - Missing image or label file pairs
  - Labels with zero voxels
  - Labels whose shape has a zero dimension
  - Labels with a single voxel (suspiciously tiny)
  - Intensity/image shape mismatch

It can optionally attempt to regenerate a minimal single-voxel label at the original centroid
stored in the NIfTI header (if prepared by this codebase) or skip fix (default). For now we only
report; fixing should be done at data generation level.

Exit code is 0 even if problems are found (so you can still see output in CI); review the
printed summary.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import List

try:
    import nibabel as nib  # type: ignore
    import numpy as np
except Exception as e:  # pragma: no cover
    print(f"Failed to import nibabel/numpy: {e}", file=sys.stderr)
    sys.exit(1)


def analyze_case(label_fp: Path, images_dir: Path, verbose: bool=False):
    problems: List[str] = []
    try:
        lab_img = nib.load(str(label_fp))
        lab = lab_img.get_fdata(dtype=np.float32)
    except Exception as e:
        return [f"cannot_load_label:{e}"], None
    if lab.size == 0:
        problems.append("zero_size_array")
    if 0 in lab.shape:
        problems.append(f"zero_dim_shape:{lab.shape}")
    voxels = int((lab > 0).sum())
    if voxels == 0:
        problems.append("empty_mask")
    elif voxels == 1:
        problems.append("single_voxel_mask")
    # Locate image
    # label_fp.stem yields '<uid>' without '.nii.gz'; we expect image file '<uid>_0000.nii.gz'
    case_id = label_fp.name.replace('.nii.gz','')
    img_fp = images_dir / f"{case_id}_0000.nii.gz"
    if not img_fp.exists():
        problems.append("missing_image_file")
        return problems, voxels
    try:
        img = nib.load(str(img_fp)).get_fdata(dtype=np.float32)
    except Exception as e:
        problems.append(f"cannot_load_image:{e}")
        return problems, voxels
    if img.shape != lab.shape:
        problems.append(f"shape_mismatch:image={img.shape},label={lab.shape}")
    return problems, voxels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset-path', type=Path, required=True, help='Path to DatasetXXX_NAME directory (raw)')
    ap.add_argument('--limit', type=int, default=0, help='Process only first N labels (debug)')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    labels_dir = args.dataset_path / 'labelsTr'
    images_dir = args.dataset_path / 'imagesTr'
    if not labels_dir.exists():
        print(f"labelsTr not found: {labels_dir}")
        sys.exit(1)
    label_files = sorted(labels_dir.glob('*.nii.gz'))
    if args.limit > 0:
        label_files = label_files[:args.limit]
    bad_cases = []
    empty_cases = []
    single_voxel_cases = []

    for fp in label_files:
        problems, voxels = analyze_case(fp, images_dir, args.verbose)
        # Only print if there is at least one real problem besides missing_image_file that is incorrect
        if problems:
            print(f"CASE {fp.stem}: {';'.join(problems)} (voxels={voxels})")
        if problems:
            if any(p.startswith('empty_mask') or p.startswith('zero_size') for p in problems):
                empty_cases.append(fp.stem)
            if any(p.startswith('single_voxel_mask') for p in problems):
                single_voxel_cases.append(fp.stem)
            bad_cases.append(fp.stem)
    print('\nSummary:')
    print(f" Total labels scanned: {len(label_files)}")
    print(f" Problematic cases: {len(bad_cases)}")
    print(f"   Empty masks: {len(empty_cases)}")
    print(f"   Single-voxel masks: {len(single_voxel_cases)}")
    if bad_cases:
        (args.dataset_path / 'problem_cases.txt').write_text('\n'.join(bad_cases))
        print(f"Wrote list to {args.dataset_path / 'problem_cases.txt'}")
    if empty_cases:
        (args.dataset_path / 'empty_masks.txt').write_text('\n'.join(empty_cases))
        print(f"Wrote empty mask list to {args.dataset_path / 'empty_masks.txt'}")
    if single_voxel_cases:
        (args.dataset_path / 'single_voxel_masks.txt').write_text('\n'.join(single_voxel_cases))
        print(f"Wrote single voxel mask list to {args.dataset_path / 'single_voxel_masks.txt'}")
    if bad_cases:
        print('\nNext steps:')
        print(' - Remove or regenerate these cases before rerunning nnUNet preprocessing.')
        print(' - Or modify prepare script to skip them earlier.')

if __name__ == '__main__':
    main()
