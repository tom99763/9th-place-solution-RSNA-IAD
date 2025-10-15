import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from sklearn.model_selection import train_test_split

model_dicts = {
    'YOLO-11M_2.5D': {
        "fold0": {"y11_yolo11m": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold0/series_validation/fold_0/per_series_predictions.csv"},
        "fold1": {"y11_yolo11m": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold1/series_validation/fold_1/per_series_predictions.csv"},
        "fold2": {"y11_yolo11m": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold22/series_validation/fold_2/per_series_predictions.csv"},
        "fold3": {"y11_yolo11m": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold3/series_validation/fold_3/per_series_predictions.csv"},
        "fold4": {"y11_yolo11m": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/yolo-11m-2.5D_fold4/series_validation/fold_4/per_series_predictions.csv"},
    },
    #'EfficientNetV2-S': {
    #    "fold0": {"effnetv2s": "yolo_aneurysm_location_all_negatives/cv_effnetv2s_v2_drop_path_fold0/series_validation/fold_0/per_series_predictions.csv"},
    #    "fold1": {"effnetv2s": "yolo_aneurysm_location_all_negatives/cv_effnetv2s_v2_drop_path_fold1/series_validation/fold_1/per_series_predictions.csv"},
    #    "fold2": {"effnetv2s": "yolo_aneurysm_location_all_negatives/cv_effnetv2s_v2_drop_path_fold2/series_validation/fold_2/per_series_predictions.csv"},
    #    "fold3": {"effnetv2s": "yolo_aneurysm_location_all_negatives/cv_effnetv2s_v2_drop_path_fold3/series_validation/fold_3/per_series_predictions.csv"},
    #    "fold4": {"effnetv2s": "yolo_aneurysm_location_all_negatives/cv_effnetv2s_v2_drop_path_fold4/series_validation/fold_4/per_series_predictions.csv"}
    #},
    '2.5D effnetv2_s': {
        "fold0": {"effnetv2s": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/cv_effnetv2_s_drop_path_25d_fold0/series_validation/fold_0/per_series_predictions.csv"},
        "fold1": {"effnetv2s": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/cv_effnetv2_s_drop_path_25d_fold1/series_validation/fold_1/per_series_predictions.csv"},
        "fold2": {"effnetv2s": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/cv_effnetv2_s_drop_path_25d_fold2/series_validation/fold_2/per_series_predictions.csv"},
        "fold3": {"effnetv2s": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/cv_effnetv2_s_drop_path_25d_fold3/series_validation/fold_3/per_series_predictions.csv"},
        "fold4": {"effnetv2s": "/home/sersasj/RSNA-IAD-Codebase/yolo_aneurysm_location_all_negatives/cv_effnetv2_s_drop_path_25d_fold4/series_validation/fold_4/per_series_predictions.csv"},

    }
}
# merge 2.5d into a single df

def calculate_metrics(df):
    """Calculate AUC metrics for a dataframe."""
    # Classification AUC
    #aneurysm prob = max(loc)
    cls_auc = roc_auc_score(df["label_aneurysm"], df["aneurysm_prob"]) if len(df["label_aneurysm"].unique()) > 1 else 0.0

    # Location-specific AUCs
    loc_aucs = [roc_auc_score(df[f"loc_label_{i}"], df[f"loc_prob_{i}"])
                for i in range(13)
                if f"loc_prob_{i}" in df.columns and f"loc_label_{i}" in df.columns and len(df[f"loc_label_{i}"].unique()) > 1]

    return cls_auc, np.mean(loc_aucs) if loc_aucs else 0.0

def print_table(results, model_name):
    """Print results in a concise table format."""
    print(f"\n{model_name} Results:")
    print("| Fold      | loc_macro_auc | cls_auc | combined_mean |")
    print("|-----------|---------------|---------|---------------|")

    # Sort fold names numerically (fold0, fold1, fold2, fold3)
    sorted_folds = sorted(results.keys(), key=lambda x: int(x.replace('fold', '')))

    fold_results = [(fold_name, results[fold_name][1], results[fold_name][0]) for fold_name in sorted_folds]

    for fold_name, loc_auc, cls_auc in fold_results:
        combined = (loc_auc + cls_auc) / 2
        print(f"| val_{fold_name} | {loc_auc:.4f} | {cls_auc:.4f} | {combined:.4f} |")

    if fold_results:
        avg_loc, avg_cls = np.mean(list(zip(*fold_results))[1:3], axis=1)
        avg_combined = (avg_loc + avg_cls) / 2
        print(f"| Avg       | {avg_loc:.4f} | {avg_cls:.4f} | {avg_combined:.4f} |")


def evaluate_model(model_dict):
    """Evaluate single model performance by fold."""
    return {fold_name: calculate_metrics(pd.read_csv(path))
            for fold_name, models in model_dict.items()
            for path in models.values()}

def evaluate_models_ensemble(models_dict):
    """Evaluate ensemble performance by averaging predictions across model architectures.
    
    Uses max at ensemble level: average location probabilities first, then take max.
    """
    # Get all folds and collect dataframes for each fold
    fold_data = {}
    for fold_name in {fold for model_dict in models_dict.values() for fold in model_dict.keys()}:
        fold_dfs = [pd.read_csv(path) for model_dict in models_dict.values()
                   if fold_name in model_dict for path in model_dict[fold_name].values()]

        if len(fold_dfs) >= 2:
            merged = fold_dfs[0][['SeriesInstanceUID']].copy()

            # Average location probabilities, then take max
            loc_prob_cols = [col for col in fold_dfs[0].columns if col.startswith('loc_prob_')]

            # Average location probabilities across models
            for col in loc_prob_cols:
                merged[col] = np.mean([df[col] for df in fold_dfs], axis=0)

            # Take max of averaged location probabilities for classification
            merged['aneurysm_prob'] = np.max(merged[loc_prob_cols].values, axis=1)

            # Copy non-probability columns from first model
            other_cols = [col for col in fold_dfs[0].columns
                         if col not in merged.columns or col == 'SeriesInstanceUID']
            for col in other_cols:
                if col != 'SeriesInstanceUID':
                    merged[col] = fold_dfs[0][col].values

            fold_data[fold_name] = merged

    return {fold_name: calculate_metrics(df) for fold_name, df in fold_data.items()}



if __name__ == "__main__":
    # Evaluate and print individual models
    for model_name in model_dicts:
        print_table(evaluate_model(model_dicts[model_name]), model_name)

    # Evaluate ensemble with max at ensemble level
    print(f"\n{'='*60}")
    print("ENSEMBLE RESULTS")
    print(f"{'='*60}")

    ensemble_results = evaluate_models_ensemble(model_dicts)
    print_table(ensemble_results, "Ensemble: Max at Ensemble Level")


