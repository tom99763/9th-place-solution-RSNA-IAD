#!/usr/bin/env python3
"""
Evaluate weighted AUC per plane view for OOF predictions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pathlib import Path

# Configuration
N_LOC = 13  # Number of location classes
ANEURYSM_WEIGHT = 13
LOCATION_WEIGHT = 1


def compute_weighted_auc(df_subset):
    """
    Compute weighted AUC for a subset of predictions.
    
    Args:
        df_subset: DataFrame with columns aneurysm_prob, label_aneurysm, 
                   loc_prob_0..12, loc_label_0..12
    
    Returns:
        dict with cls_auc, loc_macro_auc, and competition_score
    """
    if len(df_subset) < 2:
        return {
            'cls_auc': np.nan,
            'loc_macro_auc': np.nan,
            'competition_score': np.nan,
            'n_samples': len(df_subset)
        }
    
    # Classification AUC (aneurysm presence)
    y_true = df_subset['label_aneurysm'].values
    y_scores = df_subset['aneurysm_prob'].values
    
    if len(np.unique(y_true)) < 2:
        cls_auc = np.nan
    else:
        try:
            cls_auc = roc_auc_score(y_true, y_scores)
        except Exception:
            cls_auc = np.nan
    
    # Location AUCs
    per_loc_aucs = []
    for i in range(N_LOC):
        loc_label_col = f'loc_label_{i}'
        loc_prob_col = f'loc_prob_{i}'
        
        if loc_label_col not in df_subset.columns or loc_prob_col not in df_subset.columns:
            per_loc_aucs.append(np.nan)
            continue
        
        loc_labels = df_subset[loc_label_col].values
        loc_probs = df_subset[loc_prob_col].values
        
        if len(np.unique(loc_labels)) < 2:
            per_loc_aucs.append(np.nan)
        else:
            try:
                auc_i = roc_auc_score(loc_labels, loc_probs)
                per_loc_aucs.append(auc_i)
            except Exception:
                per_loc_aucs.append(np.nan)
    
    loc_macro_auc = np.nanmean(per_loc_aucs)
    
    # Compute competition score (weighted AUC)
    valid_loc_aucs = [auc for auc in per_loc_aucs if not np.isnan(auc)]
    
    if valid_loc_aucs and not np.isnan(cls_auc):
        total_weights = ANEURYSM_WEIGHT + LOCATION_WEIGHT * N_LOC
        weighted_sum = ANEURYSM_WEIGHT * cls_auc + LOCATION_WEIGHT * sum(valid_loc_aucs)
        competition_score = weighted_sum / total_weights
    else:
        competition_score = cls_auc if not np.isnan(cls_auc) else np.nan
    
    return {
        'cls_auc': cls_auc,
        'loc_macro_auc': loc_macro_auc,
        'competition_score': competition_score,
        'n_samples': len(df_subset),
        'n_positive': int(y_true.sum()),
        'n_valid_locations': len(valid_loc_aucs)
    }


def main():
    # Paths
    series_views_path = Path('data/series_views.csv')
    
    # Prediction paths for each fold
    pred_paths = [
        Path('/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_more_negatives_fold02/series_validation/fold_0/per_series_predictions.csv'),
        Path('/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_more_negatives_fold13/series_validation/fold_1/per_series_predictions.csv'),
        Path('/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_more_negatives_fold2/series_validation/fold_2/per_series_predictions.csv'),
        Path('/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_more_negatives_fold3/series_validation/fold_3/per_series_predictions.csv'),
        Path('/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_locations/cv_y11m_more_negatives_fold4/series_validation/fold_4/per_series_predictions.csv'),
    ]
    
    # Check if series views exists
    if not series_views_path.exists():
        print(f"Error: {series_views_path} not found")
        return
    
    print(f"Loading series views from: {series_views_path}")
    df_views = pd.read_csv(series_views_path)
    
    # Process each fold
    all_fold_results = []
    all_preds = []
    
    for fold_idx, pred_path in enumerate(pred_paths, 1):
        if not pred_path.exists():
            print(f"Warning: {pred_path} not found, skipping fold {fold_idx}")
            continue
        
        print(f"\n{'='*80}")
        print(f"FOLD {fold_idx}")
        print(f"{'='*80}")
        print(f"Loading predictions from: {pred_path}")
        
        df_preds = pd.read_csv(pred_path)
        df_preds['fold'] = fold_idx
        all_preds.append(df_preds)
        
        # Merge on SeriesInstanceUID
        df_merged = df_preds.merge(df_views[['SeriesInstanceUID', 'view']], 
                                    on='SeriesInstanceUID', 
                                    how='left')
        
        print(f"Total series in predictions: {len(df_preds)}")
        print(f"Series with view labels: {df_merged['view'].notna().sum()}")
        print(f"Series missing view labels: {df_merged['view'].isna().sum()}")
        
        # Compute overall metrics for this fold
        print(f"\n{'-'*60}")
        print("OVERALL METRICS (All Views)")
        print(f"{'-'*60}")
        overall_metrics = compute_weighted_auc(df_merged)
        overall_metrics['fold'] = fold_idx
        overall_metrics['view'] = 'all'
        all_fold_results.append(overall_metrics)
        
        for k, v in overall_metrics.items():
            if k not in ['fold', 'view']:
                print(f"{k:25s}: {v:.6f}" if not np.isnan(v) and isinstance(v, float) else f"{k:25s}: {v}")
        
        # Compute metrics per view
        print(f"\n{'-'*60}")
        print("METRICS PER PLANE VIEW")
        print(f"{'-'*60}")
        
        df_with_view = df_merged[df_merged['view'].notna()].copy()
        views = sorted(df_with_view['view'].unique())
        
        for view in views:
            df_view = df_with_view[df_with_view['view'] == view]
            metrics = compute_weighted_auc(df_view)
            metrics['fold'] = fold_idx
            metrics['view'] = view
            all_fold_results.append(metrics)
            
            print(f"\nView: {view}")
            for k, v in metrics.items():
                if k not in ['fold', 'view']:
                    print(f"  {k:25s}: {v:.6f}" if not np.isnan(v) and isinstance(v, float) else f"  {k:25s}: {v}")
        
        # Save per-fold results
        fold_results = [r for r in all_fold_results if r['fold'] == fold_idx]
        df_fold_results = pd.DataFrame(fold_results)
        output_path = pred_path.parent / 'metrics_by_view.csv'
        df_fold_results.to_csv(output_path, index=False)
        print(f"\nFold {fold_idx} results saved to: {output_path}")
    
    # Aggregate all folds
    if all_preds:
        print(f"\n\n{'='*80}")
        print("AGGREGATED RESULTS ACROSS ALL FOLDS")
        print(f"{'='*80}")
        
        df_all_preds = pd.concat(all_preds, ignore_index=True)
        df_all_merged = df_all_preds.merge(df_views[['SeriesInstanceUID', 'view']], 
                                           on='SeriesInstanceUID', 
                                           how='left')
        
        # Overall metrics (all folds combined)
        print(f"\n{'-'*60}")
        print("OVERALL METRICS (All Folds, All Views)")
        print(f"{'-'*60}")
        overall_all = compute_weighted_auc(df_all_merged)
        for k, v in overall_all.items():
            print(f"{k:25s}: {v:.6f}" if not np.isnan(v) and isinstance(v, float) else f"{k:25s}: {v}")
        
        # Per view metrics (all folds combined)
        print(f"\n{'-'*60}")
        print("METRICS PER PLANE VIEW (All Folds Combined)")
        print(f"{'-'*60}")
        
        df_with_view_all = df_all_merged[df_all_merged['view'].notna()].copy()
        views_all = sorted(df_with_view_all['view'].unique())
        
        combined_view_results = []
        for view in views_all:
            df_view = df_with_view_all[df_with_view_all['view'] == view]
            metrics = compute_weighted_auc(df_view)
            metrics['view'] = view
            combined_view_results.append(metrics)
            
            print(f"\nView: {view}")
            for k, v in metrics.items():
                if k != 'view':
                    print(f"  {k:25s}: {v:.6f}" if not np.isnan(v) and isinstance(v, float) else f"  {k:25s}: {v}")
        
        # Save all results
        df_all_results = pd.DataFrame(all_fold_results)
        output_all_path = Path('metrics_by_view_all_folds.csv')
        df_all_results.to_csv(output_all_path, index=False)
        print(f"\n\nAll fold results saved to: {output_all_path}")
        
        # Summary table
        print(f"\n{'='*80}")
        print("SUMMARY TABLE BY FOLD AND VIEW")
        print(f"{'='*80}")
        summary_cols = ['fold', 'view', 'n_samples', 'n_positive', 'cls_auc', 'loc_macro_auc', 'competition_score']
        print(df_all_results[summary_cols].to_string(index=False))
        
        # Average metrics across folds
        print(f"\n{'='*80}")
        print("AVERAGE METRICS ACROSS FOLDS")
        print(f"{'='*80}")
        avg_by_view = df_all_results.groupby('view').agg({
            'cls_auc': 'mean',
            'loc_macro_auc': 'mean', 
            'competition_score': 'mean',
            'n_samples': 'sum',
            'n_positive': 'sum'
        }).reset_index()
        print(avg_by_view.to_string(index=False))


if __name__ == '__main__':
    main()

