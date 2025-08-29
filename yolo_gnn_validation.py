"""Enhanced YOLO validation with GNN post-processing for aneurysm detection.

This extends the standard YOLO validation by applying a Graph Neural Network
to refine predictions based on spatial relationships between detections.
"""
import argparse
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from tqdm import tqdm

# Import base YOLO validation functionality
sys.path.insert(0, str(Path(__file__).parent))
from yolo_multiclass_validation import (
    parse_args as base_parse_args,
    read_dicom_frames_hu, min_max_normalize, collect_series_slices,
    _run_validation_for_fold as base_validation_fold
)

# Import GNN components
from src.gnn_yolo import create_gnn_model, YOLOGNNProcessor

sys.path.insert(0, "ultralytics-timm")
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from configs.data_config import data_path, LABELS_TO_IDX


def parse_args():
    """Extended argument parser with GNN options."""
    ap = base_parse_args()
    
    # GNN-specific arguments
    ap.add_argument('--use-gnn', action='store_true', 
                   help='Apply GNN post-processing to YOLO predictions')
    ap.add_argument('--gnn-weights', type=str, default='', 
                   help='Path to trained GNN weights (optional)')
    ap.add_argument('--gnn-hidden-dim', type=int, default=64,
                   help='GNN hidden dimension')
    ap.add_argument('--gnn-distance-threshold', type=float, default=0.3,
                   help='Distance threshold for GNN edge creation')
    
    return ap.parse_args()


def run_validation_with_gnn(args: argparse.Namespace, weights_path: str, 
                           fold_id: int) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Run validation with GNN post-processing."""
    
    # Setup GNN if requested
    gnn_processor = None
    if args.use_gnn:
        gnn_model = create_gnn_model(
            num_classes=len(LABELS_TO_IDX), 
            hidden_dim=args.gnn_hidden_dim
        )
        
        # Load GNN weights if provided
        if args.gnn_weights:
            gnn_model.load_state_dict(torch.load(args.gnn_weights))
            print(f"Loaded GNN weights from {args.gnn_weights}")
        else:
            print("Using untrained GNN model")
            
        gnn_model.eval()
        gnn_processor = YOLOGNNProcessor(
            gnn_model, 
            distance_threshold=args.gnn_distance_threshold
        )
    
    # Load data similar to base validation
    data_root = Path(data_path)
    series_root = data_root / 'series'
    train_df = pd.read_csv(data_root / 'train_df.csv') if (data_root / 'train_df.csv').exists() else pd.read_csv(data_root / 'train.csv')
    
    val_series = train_df[train_df['fold_id'] == fold_id]['SeriesInstanceUID'].unique().tolist()
    if args.series_limit:
        val_series = val_series[:args.series_limit]
        
    print(f"Validation fold {fold_id}: {len(val_series)} series")
    if args.use_gnn:
        print(f"Using GNN with distance threshold: {args.gnn_distance_threshold}")
    
    model = YOLO(weights_path)
    
    series_probs: Dict[str, float] = {}
    cls_labels: List[int] = []
    loc_labels: List[np.ndarray] = []
    series_pred_loc_probs: List[np.ndarray] = []
    series_pred_counts: Dict[str, int] = {}
    
    LOCATION_LABELS = sorted(list(LABELS_TO_IDX.keys()))
    N_LOC = len(LOCATION_LABELS)
    
    for sid in tqdm(val_series, desc="Validating series", unit="series"):
        series_dir = series_root / sid
        if not series_dir.exists():
            continue
            
        dicoms = collect_series_slices(series_dir)
        if not dicoms:
            continue
            
        # Process slices and collect YOLO results
        yolo_results = []
        total_dets = 0
        
        for dcm_path in dicoms[::args.slice_step]:
            if args.max_slices and len(yolo_results) >= args.max_slices:
                break
                
            try:
                frames = read_dicom_frames_hu(dcm_path)
            except Exception:
                continue
                
            for frame in frames:
                img_uint8 = min_max_normalize(frame)
                if img_uint8.ndim == 2:
                    import cv2
                    img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
                
                # Run YOLO inference
                results = model.predict([img_uint8], verbose=False, conf=0.01)
                if results and results[0].boxes is not None:
                    yolo_results.extend(results)
                    total_dets += len(results[0].boxes)
        
        # Apply GNN post-processing if enabled
        if args.use_gnn and gnn_processor and yolo_results:
            gnn_output = gnn_processor.process_series(yolo_results)
            series_conf = float(gnn_output['series_conf'])
            per_class_probs = gnn_output['location_probs'].numpy()
        else:
            # Standard YOLO aggregation (max confidence)
            max_conf_all = 0.0
            per_class_max = np.zeros(N_LOC, dtype=np.float32)
            
            for result in yolo_results:
                if result.boxes is None or len(result.boxes) == 0:
                    continue
                    
                confs = result.boxes.conf
                clses = result.boxes.cls
                
                for i in range(len(confs)):
                    c = float(confs[i])
                    k = int(clses[i])
                    
                    if c > max_conf_all:
                        max_conf_all = c
                    if 0 <= k < N_LOC and c > per_class_max[k]:
                        per_class_max[k] = c
            
            series_conf = max_conf_all
            per_class_probs = per_class_max
        
        series_probs[sid] = series_conf
        series_pred_loc_probs.append(per_class_probs.copy())
        series_pred_counts[sid] = total_dets
        
        # Get labels
        row = train_df[train_df['SeriesInstanceUID'] == sid].iloc[0]
        cls_labels.append(int(row['Aneurysm Present']))
        
        loc_vec = np.zeros(N_LOC, dtype=np.float32)
        for idx, name in enumerate(LOCATION_LABELS):
            if name in row:
                try:
                    loc_vec[idx] = float(row[name])
                except Exception:
                    loc_vec[idx] = 0.0
        loc_labels.append(loc_vec)
    
    # Compute metrics (same as base validation)
    from sklearn.metrics import roc_auc_score
    import math
    
    if not series_probs:
        return {
            'cls_auc': float('nan'),
            'loc_macro_auc': float('nan'),
            'combined_mean': float('nan'),
            'fold_id': fold_id,
        }, pd.DataFrame()
    
    y_true = np.array(cls_labels)
    y_scores = np.array([series_probs[sid] for sid in series_probs.keys()])
    cls_auc = roc_auc_score(y_true, y_scores)
    
    loc_labels_arr = np.stack(loc_labels)
    loc_pred_arr = np.stack(series_pred_loc_probs)
    per_loc_aucs = []
    
    for i in range(N_LOC):
        try:
            auc_i = roc_auc_score(loc_labels_arr[:, i], loc_pred_arr[:, i])
        except ValueError:
            auc_i = float('nan')
        per_loc_aucs.append(auc_i)
    
    loc_macro_auc = np.nanmean(per_loc_aucs)
    combined_mean = (cls_auc + loc_macro_auc) / 2
    
    print(f"Classification AUC: {cls_auc:.4f}")
    print(f"Location macro AUC: {loc_macro_auc:.4f}")
    print(f"Combined metric: {combined_mean:.4f}")
    if args.use_gnn:
        print(f"[GNN-enhanced predictions]")
    
    # Build results dataframe
    out_rows = []
    keys = list(series_probs.keys())
    for idx, sid in enumerate(keys):
        row_df = train_df[train_df['SeriesInstanceUID'] == sid]
        label = int(row_df['Aneurysm Present'].iloc[0]) if not row_df.empty else 0
        row = {
            'SeriesInstanceUID': sid,
            'aneurysm_prob': float(series_probs[sid]),
            'label_aneurysm': label,
            'num_detections': int(series_pred_counts.get(sid, 0)),
        }
        probs = series_pred_loc_probs[idx]
        for i, name in enumerate(LOCATION_LABELS):
            row[f'loc_prob_{i}'] = float(probs[i])
        out_rows.append(row)
    
    per_series_df = pd.DataFrame(out_rows)
    
    metrics = {
        'fold_id': fold_id,
        'cls_auc': float(cls_auc),
        'loc_macro_auc': float(loc_macro_auc) if not math.isnan(loc_macro_auc) else float('nan'),
        'combined_mean': float(combined_mean) if not math.isnan(combined_mean) else float('nan'),
        'per_loc_aucs': {LOCATION_LABELS[i]: float(per_loc_aucs[i]) if not math.isnan(per_loc_aucs[i]) else float('nan') for i in range(N_LOC)},
        'num_series': len(keys),
        'gnn_enabled': args.use_gnn
    }
    
    return metrics, per_series_df


def main():
    args = parse_args()
    
    if not args.weights:
        raise SystemExit('Provide --weights path to YOLO model')
    
    # Import torch here to avoid unnecessary imports
    if args.use_gnn:
        global torch
        import torch
    
    metrics, df = run_validation_with_gnn(args, args.weights, args.val_fold)
    
    # Save results if output directory specified
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(out_dir / 'per_series_predictions_gnn.csv', index=False)
        
        import json
        with open(out_dir / 'metrics_gnn.json', 'w') as f:
            json.dump(metrics, f, indent=2)
            
        print(f"Results saved to {out_dir}")


if __name__ == '__main__':
    main()