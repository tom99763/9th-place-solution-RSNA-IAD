import torch
import numpy as np
from torch.utils.data import Sampler
import random


class SimpleImbalanceSampler(Sampler):
    """
    Simple sampler for imbalanced data that maintains a target positive ratio.
    Randomly samples from positive and negative indices.
    """
    def __init__(self, dataset, pos_ratio=0.8, samples_per_epoch=None):
        """
        Args:
            dataset: The dataset
            pos_ratio: Target ratio of positive samples (e.g., 0.8 = 80% positive)
            samples_per_epoch: Total samples per epoch (if None, uses all data)
        """
        self.dataset = dataset
        self.pos_ratio = pos_ratio
        
        # Separate positive and negative indices
        self.positive_indices = []
        self.negative_indices = []
        
        for idx in range(len(dataset)):
            row = dataset.slice_df.iloc[idx]
            if row['has_aneurysm']:
                self.positive_indices.append(idx)
            else:
                self.negative_indices.append(idx)
        
        self.n_pos = len(self.positive_indices)
        self.n_neg = len(self.negative_indices)
        
        # Calculate samples per epoch
        if samples_per_epoch is None:
            samples_per_epoch = len(dataset)
        
        # Calculate number of positive and negative samples needed
        self.pos_samples = int(samples_per_epoch * pos_ratio)
        self.neg_samples = samples_per_epoch - self.pos_samples
        
        # Ensure we don't exceed available samples (will sample with replacement if needed)
        self.total_samples = samples_per_epoch
        
        print(f"SimpleImbalanceSampler: targeting {self.pos_samples} pos + {self.neg_samples} neg = {self.total_samples} total")
        print(f"Target positive ratio: {pos_ratio:.4f}")
        print(f"Available: {self.n_pos} pos, {self.n_neg} neg")
        
        # Check if we need replacement sampling
        if self.pos_samples > self.n_pos:
            print(f"Will oversample positives: need {self.pos_samples}, have {self.n_pos}")
        if self.neg_samples > self.n_neg:
            print(f"Will oversample negatives: need {self.neg_samples}, have {self.n_neg}")
    
    def __iter__(self):
        # Randomly sample from positive indices (with replacement if needed)
        pos_sampled = np.random.choice(
            self.positive_indices, 
            size=self.pos_samples, 
            replace=self.pos_samples > self.n_pos
        )
        
        # Randomly sample from negative indices (with replacement if needed)
        neg_sampled = np.random.choice(
            self.negative_indices, 
            size=self.neg_samples, 
            replace=self.neg_samples > self.n_neg
        )
        
        # Combine and shuffle all indices
        all_indices = np.concatenate([pos_sampled, neg_sampled])
        np.random.shuffle(all_indices)
        
        return iter(all_indices.tolist())
    
    def __len__(self):
        return self.total_samples