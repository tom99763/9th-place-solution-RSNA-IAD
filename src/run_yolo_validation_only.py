"""
Run only the validation phase for pre-trained YOLO fold weights.

Behavior:
- For each fold f in --folds, runs the existing series-level validation script
  using the weight file from --weights-template (with {fold}).
- Saves per-fold outputs (metrics.json, per_series_predictions.csv, per_location_auc.csv)
  under the parent of the weights folder, in a 'series_validation' directory per fold.

Usage example:
    python -m src.run_yolo_validation_only \
        --weights-template \
        "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_new_stratification_fold{fold}/weights/best.pt" \
        --folds 0,1,2,3,4 --val-batch 16 --slice-step 1 --mip-window 0
"""
import argparse
from pathlib import Path
import sys
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    ap = argparse.ArgumentParser(description='Validate YOLO aneurysm models for specified folds')
    # Weights/template and folds
    ap.add_argument(
        '--weights-template',
        type=str,
        default='/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_new_stratification_fold{fold}/weights/best.pt',
        help='Path template to best.pt with {fold} placeholder'
    )
    ap.add_argument('--folds', type=str, default='0,1,2,3,4', help='Comma-separated fold IDs to validate')

    # Validation script options (mirrors yolo_multiclass_validation.py)
    ap.add_argument('--val-batch', type=int, default=16, help='Batch size for inference')
    ap.add_argument('--slice-step', type=int, default=1, help='Process every Nth slice')
    ap.add_argument('--mip-window', type=int, default=0, help='Half-window for MIP; 0 = per-slice mode')
    ap.add_argument('--mip-img-size', type=int, default=0, help='Optional resize of MIP/slice to this square size')
    ap.add_argument('--mip-no-overlap', action='store_true', help='Use non-overlapping MIP windows')
    ap.add_argument('--max-slices', type=int, default=0, help='Cap slices/windows per series (debug)')
    ap.add_argument('--series-limit', type=int, default=0, help='Limit number of validation series (debug)')
    ap.add_argument('--bgr-mode', action='store_true', help='Use BGR 3-slice stacking mode')
    ap.add_argument('--single-cls', action='store_true', help='Single class mode (binary)')

    # W&B override (optional)
    ap.add_argument('--wandb', action='store_true', help='Enable W&B logging for validation')
    ap.add_argument('--wandb-project', type=str, default='', help='W&B project (override)')
    ap.add_argument('--wandb-entity', type=str, default='', help='W&B entity (override)')
    ap.add_argument('--wandb-group', type=str, default='', help='W&B group (override)')
    ap.add_argument('--wandb-tags', type=str, default='', help='Comma-separated W&B tags')
    ap.add_argument('--wandb-resume-id', type=str, default='', help='Resume W&B run id (optional)')
    return ap.parse_args()


def _get_wandb_resume_info(save_dir: Path):
    """Try to read W&B metadata from a training output folder.
    Returns dict with keys: id, project, entity, group, name; or None.
    """
    import json as _json

    wandb_dir = save_dir / 'wandb'
    if not wandb_dir.exists():
        return None
    run_dir = None
    latest = wandb_dir / 'latest-run'
    if latest.exists():
        try:
            rel = latest.read_text().strip()
            if rel:
                cand = wandb_dir / rel
                if cand.exists():
                    run_dir = cand
        except Exception:
            pass
    if run_dir is None:
        candidates = [p for p in wandb_dir.glob('run-*') if p.is_dir()]
        if not candidates:
            return None
        run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
    md_paths = [run_dir / 'wandb-metadata.json', run_dir / 'files' / 'wandb-metadata.json']
    meta = {}
    for mp in md_paths:
        if mp.exists():
            try:
                meta = _json.loads(mp.read_text())
                break
            except Exception:
                pass
    if not meta:
        return None
    return {
        'id': meta.get('id') or meta.get('run_id') or '',
        'project': meta.get('project') or '',
        'entity': meta.get('entity') or '',
        'group': meta.get('group') or '',
        'name': meta.get('name') or '',
    }


def run():
    args = parse_args()

    # Determine which validation script to use
    val_script = ROOT / 'yolo_multiclass_validation.py'
    if not val_script.exists():
        raise SystemExit(f"Validation script not found at {val_script}")

    folds: List[int] = [int(x) for x in args.folds.split(',') if x.strip() != '']

    import subprocess

    for f in folds:
        weights_path = Path(args.weights_template.format(fold=f))
        if not weights_path.exists():
            raise SystemExit(f"best.pt not found for fold {f}: {weights_path}")

        # Save dir = parent of 'weights' directory
        save_dir = weights_path.parent.parent
        out_dir = save_dir / 'series_validation'
        out_dir.mkdir(parents=True, exist_ok=True)

        # Build validation command
        cmd = [
            sys.executable,
            str(val_script),
            '--weights', str(weights_path),
            '--val-fold', str(f),
            '--out-dir', str(out_dir),
            '--batch-size', str(args.val_batch),
            '--slice-step', str(args.slice_step),
            '--mip-window', str(args.mip_window),
            '--mip-img-size', str(args.mip_img_size),
        ]
        if args.mip_no_overlap:
            cmd.append('--mip-no-overlap')
        if args.max_slices:
            cmd += ['--max-slices', str(args.max_slices)]
        if args.series_limit:
            cmd += ['--series-limit', str(args.series_limit)]
        if args.bgr_mode:
            cmd.append('--bgr-mode')
        if args.single_cls:
            cmd.append('--single-cls')

        # Try to attach validation logging to the same W&B run as training
        wandb_info = _get_wandb_resume_info(save_dir)
        if wandb_info or args.wandb or args.wandb_project:
            cmd.append('--wandb')
            if args.wandb_project:
                cmd += ['--wandb-project', args.wandb_project]
            elif wandb_info and wandb_info.get('project'):
                cmd += ['--wandb-project', wandb_info['project']]
            if args.wandb_entity:
                cmd += ['--wandb-entity', args.wandb_entity]
            elif wandb_info and wandb_info.get('entity'):
                cmd += ['--wandb-entity', wandb_info['entity']]
            if args.wandb_group:
                cmd += ['--wandb-group', args.wandb_group]
            elif wandb_info and wandb_info.get('group'):
                cmd += ['--wandb-group', wandb_info['group']]
            if args.wandb_tags:
                cmd += ['--wandb-tags', args.wandb_tags]
            if args.wandb_resume_id:
                cmd += ['--wandb-resume-id', args.wandb_resume_id]

        print('Running validation:', ' '.join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    run()

"""
Example direct invocation:
python3 -m src.run_yolo_validation_only \
  --weights-template "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_new_stratification_fold{fold}/weights/best.pt" \
  --folds 0,1,2,3,4 --val-batch 16 --slice-step 1 --mip-window 0
"""


