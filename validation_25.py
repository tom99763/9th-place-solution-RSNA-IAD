import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pathlib import Path
import sys
import pickle
import json
sys.path.append('./src')
from configs.data_config import *
from hydra.utils import instantiate
from tqdm import tqdm
import cv2

torch.set_float32_matmul_precision('medium')

def preprocess_slice_2d(slice_img):
    """
    Apply 2D preprocessing - single slice replicated 3 times.
    """
    # Create 3-channel image (required for most models)
    img = np.stack([slice_img] * 3, axis=-1)
    # Convert to tensor format (HWC to CHW)
    img = torch.from_numpy(img.transpose(2, 0, 1))
    return img

def preprocess_slice_2_5d(slice_files, data_path, slice_idx, num_adjacent_slices=1):
    """
    Apply 2.5D preprocessing - load adjacent slices like training.
    """
    # Build lookup for this series
    slice_lookup = {}
    for filename in slice_files:
        # Extract slice index from filename (format: series_uid_XXX.npz)
        slice_num = int(filename.split('_')[-1].replace('.npz', ''))
        slice_lookup[slice_num] = filename
    
    # Get adjacent slice indices (same as training logic)
    slice_indices = list(range(slice_idx - num_adjacent_slices, 
                             slice_idx + num_adjacent_slices + 1))
    
    # Load current slice first as fallback
    current_filename = None
    for filename in slice_files:
        if filename.endswith(f"_{slice_idx:03d}.npz"):
            current_filename = filename
            break
    
    if not current_filename:
        # Fallback to 2D if we can't find the slice
        slice_path = data_path / "individual_slices" / slice_files[0]
        with np.load(slice_path) as data:
            slice_img = data['slice'].astype(np.float32)
        return preprocess_slice_2d(slice_img)
    
    # Load current slice
    slice_path = data_path / "individual_slices" / current_filename
    with np.load(slice_path) as data:
        current_slice = data['slice'].astype(np.float32)
    
    # Load adjacent slices
    all_slices = []
    for adj_idx in slice_indices:
        if adj_idx in slice_lookup:
            adj_filename = slice_lookup[adj_idx]
            adj_path = data_path / "individual_slices" / adj_filename
            with np.load(adj_path) as data:
                adj_slice = data['slice'].astype(np.float32)
        else:
            # Use current slice as fallback
            adj_slice = current_slice
        all_slices.append(adj_slice)
    
    # Stack as channels (same as training)
    img = np.stack(all_slices, axis=-1)

    # Convert to tensor
    img = torch.from_numpy(img.transpose(2, 0, 1))  # HWC to CHW
    return img

# META-CLASSIFIER AGGREGATION METHODS
def aggregate_max(pred_cls_probs, pred_locs_probs):
    """
    Máximo: Usar a probabilidade máxima entre todas as fatias
    """
    series_cls_prob = pred_cls_probs.max()
    series_loc_probs = pred_locs_probs.max(axis=0)
    return series_cls_prob, series_loc_probs

def aggregate_weighted_mean(pred_cls_probs, pred_locs_probs, weight_power=2.0):
    """
    Média ponderada: Dar mais peso às fatias com maior probabilidade
    """
    # Usar probabilidades de classificação como pesos
    weights = np.power(pred_cls_probs, weight_power)
    weights = weights / (weights.sum() + 1e-8)  # Normalizar pesos
    
    # Agregação ponderada para classificação
    series_cls_prob = (pred_cls_probs * weights).sum()
    
    # Agregação ponderada para localização
    series_loc_probs = (pred_locs_probs * weights[:, np.newaxis]).sum(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_top_k_mean(pred_cls_probs, pred_locs_probs, k=5):
    """
    Top-k média: Média das k fatias com maiores probabilidades
    """
    # Encontrar top-k índices baseado na probabilidade de classificação
    k = min(k, len(pred_cls_probs))
    top_k_indices = np.argpartition(pred_cls_probs, -k)[-k:]
    
    # Agregar apenas as top-k fatias
    series_cls_prob = pred_cls_probs[top_k_indices].mean()
    series_loc_probs = pred_locs_probs[top_k_indices].mean(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_attention_weighted(pred_cls_probs, pred_locs_probs, temperature=2.0):
    """
    Attention-based: Usar softmax das probabilidades como pesos de atenção
    """
    # Calcular pesos de atenção usando softmax
    attention_logits = pred_cls_probs / temperature
    attention_weights = np.exp(attention_logits) / np.exp(attention_logits).sum()
    
    # Agregação com atenção para classificação
    series_cls_prob = (pred_cls_probs * attention_weights).sum()
    
    # Agregação com atenção para localização
    series_loc_probs = (pred_locs_probs * attention_weights[:, np.newaxis]).sum(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_ensemble(pred_cls_probs, pred_locs_probs, methods=['max', 'weighted_mean', 'top_k'], weights=None):
    """
    Ensemble: Combinar múltiplas estratégias de agregação
    """
    if weights is None:
        weights = [1.0] * len(methods)
    
    assert len(weights) == len(methods), "Número de pesos deve ser igual ao número de métodos"
    
    results_cls = []
    results_loc = []
    
    # Calcular cada método
    for method in methods:
        if method == 'max':
            cls_prob, loc_probs = aggregate_max(pred_cls_probs, pred_locs_probs)
        elif method == 'weighted_mean':
            cls_prob, loc_probs = aggregate_weighted_mean(pred_cls_probs, pred_locs_probs)
        elif method == 'top_k':
            cls_prob, loc_probs = aggregate_top_k_mean(pred_cls_probs, pred_locs_probs)
        elif method == 'attention':
            cls_prob, loc_probs = aggregate_attention_weighted(pred_cls_probs, pred_locs_probs)
        
        results_cls.append(cls_prob)
        results_loc.append(loc_probs)
    
    # Média ponderada dos resultados
    weights = np.array(weights) / np.sum(weights)
    
    series_cls_prob = np.sum([w * cls for w, cls in zip(weights, results_cls)])
    series_loc_probs = np.sum([w * loc for w, loc in zip(weights, results_loc)], axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_quantile_based(pred_cls_probs, pred_locs_probs, quantile=0.8):
    """
    Quantile-based: Use quantiles instead of max or mean
    """
    series_cls_prob = np.quantile(pred_cls_probs, quantile)
    series_loc_probs = np.quantile(pred_locs_probs, quantile, axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_robust_mean(pred_cls_probs, pred_locs_probs, trim_ratio=0.2):
    """
    Robust mean: Remove outliers before averaging
    """
    n_trim = int(len(pred_cls_probs) * trim_ratio)
    
    if n_trim > 0 and len(pred_cls_probs) > n_trim * 2:
        # Sort and trim extreme values
        sorted_indices = np.argsort(pred_cls_probs)
        keep_indices = sorted_indices[n_trim:-n_trim]
        
        series_cls_prob = pred_cls_probs[keep_indices].mean()
        series_loc_probs = pred_locs_probs[keep_indices].mean(axis=0)
    else:
        # Fallback to regular mean if too few samples
        series_cls_prob = pred_cls_probs.mean()
        series_loc_probs = pred_locs_probs.mean(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_confidence_weighted_max(pred_cls_probs, pred_locs_probs, confidence_threshold=0.7):
    """
    Confidence-weighted max: Only consider high-confidence predictions
    """
    # Only use slices above confidence threshold
    high_conf_mask = pred_cls_probs > confidence_threshold
    
    if high_conf_mask.sum() > 0:
        series_cls_prob = pred_cls_probs[high_conf_mask].max()
        series_loc_probs = pred_locs_probs[high_conf_mask].max(axis=0)
    else:
        # Fallback to regular max if no high-confidence predictions
        series_cls_prob = pred_cls_probs.max()
        series_loc_probs = pred_locs_probs.max(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_adaptive_top_k(pred_cls_probs, pred_locs_probs, base_k=5, confidence_threshold=0.5):
    """
    Adaptive top-k: Vary k based on number of confident predictions
    """
    # Count confident predictions
    confident_predictions = (pred_cls_probs > confidence_threshold).sum()
    
    # Adapt k based on confident predictions
    if confident_predictions >= base_k:
        k = min(confident_predictions, len(pred_cls_probs))
    else:
        k = max(3, min(base_k, len(pred_cls_probs)))  # At least 3, at most base_k
    
    top_k_indices = np.argpartition(pred_cls_probs, -k)[-k:]
    
    series_cls_prob = pred_cls_probs[top_k_indices].mean()
    series_loc_probs = pred_locs_probs[top_k_indices].mean(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_max_with_support(pred_cls_probs, pred_locs_probs, support_ratio=0.3):
    """
    Max with support: Combine max with proportion of supporting evidence
    """
    max_prob = pred_cls_probs.max()
    
    # Calculate support: proportion of slices above certain threshold
    support_threshold = max_prob * 0.5  # Half of max probability
    support = (pred_cls_probs > support_threshold).mean()
    
    # Weighted combination of max and support
    series_cls_prob = max_prob * (1 - support_ratio) + support * support_ratio
    
    # For locations, use max but weighted by support
    max_loc_indices = np.argmax(pred_cls_probs)
    series_loc_probs = pred_locs_probs[max_loc_indices] * (1 - support_ratio) + \
                      pred_locs_probs.mean(axis=0) * support_ratio
    
    return series_cls_prob, series_loc_probs

def aggregate_exponential_decay(pred_cls_probs, pred_locs_probs, decay_rate=0.5):
    """
    Exponential decay: Weight by rank with exponential decay
    """
    # Sort by probability (descending)
    sorted_indices = np.argsort(pred_cls_probs)[::-1]
    sorted_cls_probs = pred_cls_probs[sorted_indices]
    sorted_loc_probs = pred_locs_probs[sorted_indices]
    
    # Create exponential decay weights
    ranks = np.arange(len(sorted_cls_probs))
    weights = np.exp(-decay_rate * ranks)
    weights = weights / weights.sum()
    
    series_cls_prob = (sorted_cls_probs * weights).sum()
    series_loc_probs = (sorted_loc_probs * weights[:, np.newaxis]).sum(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_peak_detection(pred_cls_probs, pred_locs_probs, prominence_threshold=0.1):
    """
    Peak detection: Find peaks in probability distribution
    """
    from scipy.signal import find_peaks
    
    # Find peaks in the probability sequence
    peaks, properties = find_peaks(pred_cls_probs, prominence=prominence_threshold)
    
    if len(peaks) > 0:
        # Use the highest peak
        peak_values = pred_cls_probs[peaks]
        best_peak_idx = peaks[np.argmax(peak_values)]
        
        # Weight around the peak (neighboring slices)
        window_size = 3
        start_idx = max(0, best_peak_idx - window_size)
        end_idx = min(len(pred_cls_probs), best_peak_idx + window_size + 1)
        
        peak_window_cls = pred_cls_probs[start_idx:end_idx]
        peak_window_loc = pred_locs_probs[start_idx:end_idx]
        
        series_cls_prob = peak_window_cls.max()
        series_loc_probs = peak_window_loc.max(axis=0)
    else:
        # Fallback to max if no peaks found
        series_cls_prob = pred_cls_probs.max()
        series_loc_probs = pred_locs_probs.max(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_consensus_voting(pred_cls_probs, pred_locs_probs, vote_threshold=0.6, min_voters=3):
    """
    Consensus voting: Only count slices that 'vote' for aneurysm presence
    """
    # Slices that "vote" for aneurysm (above threshold)
    voters = pred_cls_probs > vote_threshold
    
    if voters.sum() >= min_voters:
        # Use voters' average
        series_cls_prob = pred_cls_probs[voters].mean()
        series_loc_probs = pred_locs_probs[voters].mean(axis=0)
    else:
        # Not enough consensus, use top-k
        k = min(3, len(pred_cls_probs))
        top_k_indices = np.argpartition(pred_cls_probs, -k)[-k:]
        series_cls_prob = pred_cls_probs[top_k_indices].mean()
        series_loc_probs = pred_locs_probs[top_k_indices].mean(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_bayesian_fusion(pred_cls_probs, pred_locs_probs, prior_positive=0.1):
    """
    Bayesian fusion: Update belief iteratively
    """
    # Start with prior
    posterior_prob = prior_positive
    
    # Update with each slice using Bayes rule
    for prob in pred_cls_probs:
        # Likelihood ratio
        likelihood_ratio = prob / (1 - prob + 1e-8)
        
        # Update posterior
        posterior_prob = (likelihood_ratio * posterior_prob) / \
                        (likelihood_ratio * posterior_prob + (1 - posterior_prob) + 1e-8)
    
    # For locations, use weighted average based on classification confidence
    weights = pred_cls_probs / (pred_cls_probs.sum() + 1e-8)
    series_loc_probs = (pred_locs_probs * weights[:, np.newaxis]).sum(axis=0)
    
    return posterior_prob, series_loc_probs

def aggregate_outlier_robust_max(pred_cls_probs, pred_locs_probs, outlier_threshold=2.0):
    """
    Outlier-robust max: Remove statistical outliers before taking max
    """
    if len(pred_cls_probs) < 3:
        return pred_cls_probs.max(), pred_locs_probs.max(axis=0)
    
    # Calculate z-scores
    mean_prob = pred_cls_probs.mean()
    std_prob = pred_cls_probs.std()
    
    if std_prob > 0:
        z_scores = np.abs((pred_cls_probs - mean_prob) / std_prob)
        # Keep non-outliers
        non_outlier_mask = z_scores < outlier_threshold
        
        if non_outlier_mask.sum() > 0:
            filtered_cls = pred_cls_probs[non_outlier_mask]
            filtered_loc = pred_locs_probs[non_outlier_mask]
        else:
            filtered_cls = pred_cls_probs
            filtered_loc = pred_locs_probs
    else:
        filtered_cls = pred_cls_probs
        filtered_loc = pred_locs_probs
    
    return filtered_cls.max(), filtered_loc.max(axis=0)

def aggregate_sigmoid_weighted(pred_cls_probs, pred_locs_probs, temperature=2.0, shift=0.5):
    """
    Sigmoid-weighted: Use sigmoid function to create smooth weights
    """
    # Apply sigmoid weighting with temperature and shift
    shifted_probs = pred_cls_probs - shift
    sigmoid_weights = 1 / (1 + np.exp(-temperature * shifted_probs))
    sigmoid_weights = sigmoid_weights / (sigmoid_weights.sum() + 1e-8)
    
    series_cls_prob = (pred_cls_probs * sigmoid_weights).sum()
    series_loc_probs = (pred_locs_probs * sigmoid_weights[:, np.newaxis]).sum(axis=0)
    
    return series_cls_prob, series_loc_probs

def aggregate_momentum_based(pred_cls_probs, pred_locs_probs, momentum=0.9):
    """
    Momentum-based: Give more weight to consecutive high predictions
    """
    if len(pred_cls_probs) < 2:
        return pred_cls_probs.max(), pred_locs_probs.max(axis=0)
    
    # Calculate momentum-weighted scores
    momentum_scores = np.zeros_like(pred_cls_probs)
    momentum_scores[0] = pred_cls_probs[0]
    
    for i in range(1, len(pred_cls_probs)):
        momentum_scores[i] = momentum * momentum_scores[i-1] + (1 - momentum) * pred_cls_probs[i]
    
    # Find the slice with highest momentum score
    best_momentum_idx = np.argmax(momentum_scores)
    
    # Use a window around the best momentum slice
    window_size = 2
    start_idx = max(0, best_momentum_idx - window_size)
    end_idx = min(len(pred_cls_probs), best_momentum_idx + window_size + 1)
    
    window_cls = pred_cls_probs[start_idx:end_idx]
    window_loc = pred_locs_probs[start_idx:end_idx]
    
    series_cls_prob = window_cls.max()
    series_loc_probs = window_loc.max(axis=0)
    
    return series_cls_prob, series_loc_probs

@torch.no_grad()
def generate_slice_predictions(slice_files, model, data_path, image_mode="2D", num_adjacent_slices=1):
    """
    Generate predictions for all slices in a series and return raw probabilities.
    This is the expensive part that we want to do only once.
    """
    # Extract slice indices from filenames for 2.5D mode
    slice_indices = []
    if image_mode == "2.5D":
        for filename in slice_files:
            slice_idx = int(filename.split('_')[-1].replace('.npz', ''))
            slice_indices.append(slice_idx)
    
    slices = []
    
    if image_mode == "2D":
        # 2D mode: process each slice individually
        for slice_filename in slice_files:
            slice_path = data_path / "individual_slices" / slice_filename
            
            with np.load(slice_path) as data:
                slice_img = data['slice'].astype(np.float32)
            
            processed_slice = preprocess_slice_2d(slice_img)
            slices.append(processed_slice)
    
    elif image_mode == "2.5D":
        # 2.5D mode: process with adjacent slices like training
        for slice_idx in slice_indices:
            processed_slice = preprocess_slice_2_5d(
                slice_files, data_path, slice_idx, num_adjacent_slices
            )
            slices.append(processed_slice)
    
    if not slices:
        # Return default predictions if no slices
        return np.array([0.1]), np.array([[0.1] * 13])
    
    # Stack slices into batch
    volume = torch.stack(slices).cuda()

    pred_cls = []
    pred_locs = []

    # Process slices in batches (same as training)
    batch_size = 12
    for batch_idx in range(0, volume.shape[0], batch_size):
        batch_slices = volume[batch_idx:batch_idx+batch_size]
        pc, pl = model(batch_slices)
        pred_cls.append(pc.squeeze(-1))  # Remove last dimension like training
        pred_locs.append(pl)

    pred_cls = torch.cat(pred_cls)
    pred_locs = torch.cat(pred_locs)

    # Apply sigmoid to get probabilities
    pred_cls_probs = torch.sigmoid(pred_cls).cpu().numpy()
    pred_locs_probs = torch.sigmoid(pred_locs).cpu().numpy()
    
    return pred_cls_probs, pred_locs_probs

def apply_aggregation_method(pred_cls_probs, pred_locs_probs, method_name, method_kwargs):
    """
    Apply a specific aggregation method to the slice-level predictions.
    """
    if method_name == "max":
        return aggregate_max(pred_cls_probs, pred_locs_probs)
    
    elif method_name == "weighted_mean":
        weight_power = method_kwargs.get('weight_power', 2.0)
        return aggregate_weighted_mean(pred_cls_probs, pred_locs_probs, weight_power)
    
    elif method_name == "top_k":
        k = method_kwargs.get('k', 5)
        return aggregate_top_k_mean(pred_cls_probs, pred_locs_probs, k)
    
    elif method_name == "attention":
        temperature = method_kwargs.get('temperature', 2.0)
        return aggregate_attention_weighted(pred_cls_probs, pred_locs_probs, temperature)
    
    elif method_name == "ensemble":
        methods = method_kwargs.get('methods', ['max', 'weighted_mean', 'top_k'])
        weights = method_kwargs.get('weights', None)
        return aggregate_ensemble(pred_cls_probs, pred_locs_probs, methods, weights)
    
    elif method_name == "quantile":
        quantile = method_kwargs.get('quantile', 0.8)
        return aggregate_quantile_based(pred_cls_probs, pred_locs_probs, quantile)
    
    elif method_name == "robust_mean":
        trim_ratio = method_kwargs.get('trim_ratio', 0.2)
        return aggregate_robust_mean(pred_cls_probs, pred_locs_probs, trim_ratio)
    elif method_name == "max_with_support":
        support_ratio = method_kwargs.get('support_ratio', 0.3)
        return aggregate_max_with_support(pred_cls_probs, pred_locs_probs, support_ratio)
    
    elif method_name == "exponential_decay":
        decay_rate = method_kwargs.get('decay_rate', 0.5)
        return aggregate_exponential_decay(pred_cls_probs, pred_locs_probs, decay_rate)
    
    elif method_name == "peak_detection":
        prominence_threshold = method_kwargs.get('prominence_threshold', 0.1)
        return aggregate_peak_detection(pred_cls_probs, pred_locs_probs, prominence_threshold)
    
    elif method_name == "consensus_voting":
        vote_threshold = method_kwargs.get('vote_threshold', 0.6)
        min_voters = method_kwargs.get('min_voters', 3)
        return aggregate_consensus_voting(pred_cls_probs, pred_locs_probs, vote_threshold, min_voters)
    
    elif method_name == "bayesian_fusion":
        prior_positive = method_kwargs.get('prior_positive', 0.1)
        return aggregate_bayesian_fusion(pred_cls_probs, pred_locs_probs, prior_positive)
    
    elif method_name == "outlier_robust_max":
        outlier_threshold = method_kwargs.get('outlier_threshold', 2.0)
        return aggregate_outlier_robust_max(pred_cls_probs, pred_locs_probs, outlier_threshold)
    
    elif method_name == "sigmoid_weighted":
        temperature = method_kwargs.get('temperature', 2.0)
        shift = method_kwargs.get('shift', 0.5)
        return aggregate_sigmoid_weighted(pred_cls_probs, pred_locs_probs, temperature, shift)
    
    elif method_name == "momentum_based":
        momentum = method_kwargs.get('momentum', 0.9)
        return aggregate_momentum_based(pred_cls_probs, pred_locs_probs, momentum)
    
    else:
        # Fallback to max if unknown method
        print(f"Warning: Unknown aggregation method '{method_name}', using 'max'")
        return aggregate_max(pred_cls_probs, pred_locs_probs)

class LitTimmClassifier(pl.LightningModule):
    def __init__(self, model=None, cfg=None):
        super().__init__()
        
        # If loading from checkpoint, model and cfg will be None initially
        if model is not None:
            self.model = model
        if cfg is not None:
            self.cfg = cfg
            self.save_hyperparameters()  # Saves args to checkpoint

    def forward(self, x):
        return self.model(x)

def save_predictions_cache(predictions, cache_path):
    """Save predictions to cache file"""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(predictions, f)
    print(f"Predictions cached to: {cache_path}")

def load_predictions_cache(cache_path):
    """Load predictions from cache file"""
    try:
        with open(cache_path, 'rb') as f:
            predictions = pickle.load(f)
        print(f"Loaded predictions from cache: {cache_path}")
        return predictions
    except FileNotFoundError:
        print(f"Cache file not found: {cache_path}")
        return None

def generate_or_load_predictions(cfg, pl_model, data_path, val_series_uids, val_slice_df, 
                                train_df_slices, label_df_slices, force_regenerate=False):
    """
    Generate predictions or load from cache. This is the expensive operation we want to do once.
    """
    # Create cache filename based on model and fold
    checkpoint_name = "models/slice_based_efficientnet_v2_baseline_25d-epoch=119-val_kaggle_score=0.8410fold_id=0.ckpt"
    cache_path = Path('prediction_cache') / f"{checkpoint_name}_fold_{cfg.fold_id}_predictions.pkl"
    
    # Try to load from cache first
    if not force_regenerate:
        cached_predictions = load_predictions_cache(cache_path)
        if cached_predictions is not None:
            return cached_predictions
    
    print("🔄 Generating slice-level predictions (this may take a while)...")
    
    all_predictions = {}
    cls_labels = []
    loc_labels = []
    
    for uid in tqdm(val_series_uids, desc="Generating predictions"):
        # Get series-level binary label
        series_row = train_df_slices[train_df_slices["SeriesInstanceUID"] == uid]
        if series_row.empty:
            continue
        
        series_row = series_row.iloc[0]
        has_aneurysm = int(series_row["Aneurysm Present"])
        cls_labels.append(has_aneurysm)
        
        # Get location labels
        loc_label = np.zeros(13, dtype=np.float32)
        if has_aneurysm:
            series_locations = label_df_slices[label_df_slices["SeriesInstanceUID"] == uid]
            for _, loc_row in series_locations.iterrows():
                location = loc_row["location"]
                if location in LABELS_TO_IDX:
                    loc_label[LABELS_TO_IDX[location]] = 1
        
        loc_labels.append(loc_label)

        # Get all slice files for this series
        series_slices = val_slice_df[val_slice_df["series_uid"] == uid]
        slice_files = series_slices["slice_filename"].tolist()
        
        if not slice_files:
            # Default predictions for series without slices
            pred_cls_probs = np.array([0.1])
            pred_locs_probs = np.array([[0.1] * 13])
        else:
            # Generate predictions for this series
            pred_cls_probs, pred_locs_probs = generate_slice_predictions(
                slice_files, pl_model, data_path, 
                image_mode=cfg.image_mode,
                num_adjacent_slices=getattr(cfg, 'num_adjacent_slices', 1)
            )
        
        all_predictions[uid] = {
            'cls_probs': pred_cls_probs,
            'loc_probs': pred_locs_probs,
            'cls_label': has_aneurysm,
            'loc_label': loc_label
        }
    
    # Save to cache
    save_predictions_cache(all_predictions, cache_path)
    
    return all_predictions

def test_aggregation_methods(predictions_dict):
    """
    Test all aggregation methods on cached predictions.
    This is fast since we already have the slice-level predictions.
    """
    methods_to_test = {
        'max': {},
        'weighted_mean': {'weight_power': 2.0},
        'weighted_mean_strong': {'weight_power': 4.0},
        'top_k_3': {'k': 3},
        'top_k_5': {'k': 5},
        'top_k_10': {'k': 10},
        'attention': {'temperature': 1.0},
        'attention_sharp': {'temperature': 0.5},
        'quantile_75': {'quantile': 0.75},
        'quantile_90': {'quantile': 0.90},
        'robust_mean': {'trim_ratio': 0.2},
        'ensemble': {
            'methods': ['max', 'weighted_mean', 'top_k'],
            'weights': [0.4, 0.4, 0.2]
        },
        'ensemble_conservative': {
            'methods': ['weighted_mean', 'top_k', 'quantile'],
            'weights': [0.5, 0.3, 0.2]
        },
        'max_with_support': {'support_ratio': 0.3},
        'exponential_decay': {'decay_rate': 0.5},
        'peak_detection': {'prominence_threshold': 0.1},
        'consensus_voting': {'vote_threshold': 0.6, 'min_voters': 3},
        'bayesian_fusion': {'prior_positive': 0.1},
        'outlier_robust_max': {'outlier_threshold': 2.0},
        'sigmoid_weighted': {'temperature': 2.0, 'shift': 0.5},
    }
    
    results = {}
    
    for method_name, method_kwargs in methods_to_test.items():
        print(f"\n🔍 Testing aggregation method: {method_name}")
        
        cls_labels = []
        loc_labels = []
        pred_cls_probs = []
        pred_loc_probs = []

        # Apply aggregation to each series
        for uid, pred_data in predictions_dict.items():
            cls_labels.append(pred_data['cls_label'])
            loc_labels.append(pred_data['loc_label'])
            
            # Map complex method names to base methods
            if method_name.startswith('weighted_mean'):
                base_method = 'weighted_mean'
            elif method_name.startswith('top_k'):
                base_method = 'top_k'
            elif method_name.startswith('attention'):
                base_method = 'attention'
            elif method_name.startswith('quantile'):
                base_method = 'quantile'
            elif method_name.startswith('ensemble'):
                base_method = 'ensemble'
            else:
                base_method = method_name
            
            # Apply aggregation method
            cls_prob, loc_prob = apply_aggregation_method(
                pred_data['cls_probs'], 
                pred_data['loc_probs'], 
                base_method,
                method_kwargs
            )
            
            pred_cls_probs.append(cls_prob)
            pred_loc_probs.append(loc_prob)

        # Calculate metrics
        cls_labels = np.array(cls_labels)
        pred_cls_probs = np.array(pred_cls_probs)
        loc_labels = np.stack(loc_labels)
        pred_loc_probs = np.stack(pred_loc_probs)

        # Calculate AUC scores
        try:
            cls_auc = roc_auc_score(cls_labels, pred_cls_probs)
            loc_auc_macro = roc_auc_score(loc_labels, pred_loc_probs, average="macro")
            kaggle_score = (cls_auc * 13 + loc_auc_macro * 1) / 14
        except ValueError as e:
            print(f"  ❌ Error calculating metrics for {method_name}: {e}")
            cls_auc = loc_auc_macro = kaggle_score = 0.0

        results[method_name] = {
            'cls_auc': cls_auc,
            'loc_auc_macro': loc_auc_macro,
            'kaggle_score': kaggle_score,
            'predictions': {
                'cls': pred_cls_probs,
                'loc': pred_loc_probs
            }
        }

        print(f"  📊 Results for {method_name}:")
        print(f"    Classification AUC: {cls_auc:.4f}")
        print(f"    Localization AUC (macro): {loc_auc_macro:.4f}")
        print(f"    Kaggle Score: {kaggle_score:.4f}")
    
    return results

@hydra.main(config_path="./configs", config_name="config", version_base=None)
def validation(cfg: DictConfig) -> None:

    print("✨ Configuration for this run: ✨")
    print(OmegaConf.to_yaml(cfg))

    pl.seed_everything(cfg.seed)

    checkpoint_path = "/home/sersasj/RSNA-IAD-Codebase/models/slice_based_efficientnet_v2_baseline_25d-epoch=119-val_kaggle_score=0.8410fold_id=0.ckpt"
    
    # Load slice-based model
    try:
        pl_model = LitTimmClassifier.load_from_checkpoint(checkpoint_path)
    except Exception as e:
        print(f"Failed to load from checkpoint: {e}")
        print("Using alternative loading method...")
        
        # Alternative: manually instantiate model and load state dict
        model = instantiate(cfg.model, pretrained=False)
        pl_model = LitTimmClassifier(model, cfg)
        
        # Load only the state dict
        checkpoint = torch.load(checkpoint_path)
        pl_model.load_state_dict(checkpoint['state_dict'])
    
    pl_model = pl_model.cuda()
    pl_model.eval()

    data_path = Path(cfg.data_dir)
    
    # Load data exactly like training
    slice_df = pd.read_csv(data_path / "slice_df.csv")
    train_df_slices = pd.read_csv(data_path / "train_df_slices.csv")
    label_df_slices = pd.read_csv(data_path / "label_df_slices.csv")

    # Get validation series UIDs (same as training)
    val_slice_df = slice_df[slice_df["fold_id"] == cfg.fold_id]
    val_series_uids = val_slice_df["series_uid"].unique()

    print(f"Validating on {len(val_series_uids)} series with {len(val_slice_df)} individual slices")

    # Generate or load cached predictions (expensive operation done once)
    force_regenerate = cfg.get('force_regenerate', False)
    predictions = generate_or_load_predictions(
        cfg, pl_model, data_path, val_series_uids, val_slice_df,
        train_df_slices, label_df_slices, force_regenerate
    )
    
    # Test all aggregation methods (fast operation using cached predictions)
    print("\n" + "="*70)
    print("🚀 TESTING ALL AGGREGATION METHODS ON CACHED PREDICTIONS")
    print("="*70)
    
    results = test_aggregation_methods(predictions)
    
    # Print summary of all results
    print("\n" + "="*70)
    print("🏆 SUMMARY OF ALL AGGREGATION METHODS")
    print("="*70)
    print(f"{'Method':<25} {'Cls AUC':<10} {'Loc AUC':<10} {'Kaggle':<10}")
    print("-"*70)
    
    best_method = None
    best_score = 0
    
    for method_name, scores in results.items():
        print(f"{method_name:<25} {scores['cls_auc']:<10.4f} {scores['loc_auc_macro']:<10.4f} {scores['kaggle_score']:<10.4f}")
        
        if scores['kaggle_score'] > best_score:
            best_score = scores['kaggle_score']
            best_method = method_name
    
    print("-"*70)
    print(f"BEST METHOD: {best_method} (Kaggle Score: {best_score:.4f})")
    print("="*70)
    
    # Save detailed results
    results_path = Path('results') / f"aggregation_results_fold_{cfg.fold_id}.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for method_name, method_results in results.items():
        json_results[method_name] = {
            'cls_auc': float(method_results['cls_auc']),
            'loc_auc_macro': float(method_results['loc_auc_macro']),
            'kaggle_score': float(method_results['kaggle_score'])
        }
    
    with open(results_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_path}")
    
    # Save results as CSV
    results_df = pd.DataFrame([
        {
            'method': method_name,
            'cls_auc': method_results['cls_auc'],
            'loc_auc_macro': method_results['loc_auc_macro'],
            'kaggle_score': method_results['kaggle_score']
        }
        for method_name, method_results in results.items()
    ])
    
    # Sort by kaggle_score descending
    results_df = results_df.sort_values('kaggle_score', ascending=False)
    
    
    return best_score


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
    
    LABELS = sorted(list(LABELS_TO_IDX.keys()))
    
    validation()