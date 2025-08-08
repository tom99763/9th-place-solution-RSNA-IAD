#!/usr/bin/env python3
"""
Benchmark script to demonstrate 2.5D performance improvement.
"""
import sys
sys.path.append('./src')

import time
import pandas as pd
from pathlib import Path
import numpy as np

def benchmark_slice_lookup():
    """Compare old vs new slice lookup performance."""
    
    print("🔬 Benchmarking 2.5D slice lookup performance...")
    
    # Load sample data
    data_path = Path("./data/processed")
    slice_df = pd.read_csv(data_path / "slice_df.csv")
    
    # Use smaller subset for benchmarking
    sample_df = slice_df.head(10000).copy()
    print(f"Using {len(sample_df)} slices for benchmark")
    
    # Build lookup cache (new method)
    print("\n1️⃣ Building fast lookup cache...")
    start_time = time.time()
    
    slice_lookup = {}
    for _, row in sample_df.iterrows():
        series_uid = row['series_uid']
        slice_idx = row['slice_idx_in_series']
        filename = row['slice_filename']
        
        if series_uid not in slice_lookup:
            slice_lookup[series_uid] = {}
        slice_lookup[series_uid][slice_idx] = filename
    
    cache_build_time = time.time() - start_time
    print(f"✅ Cache built in {cache_build_time:.3f} seconds")
    
    # Test lookups for adjacent slices
    num_lookups = 1000
    test_rows = sample_df.sample(num_lookups)
    
    print(f"\n2️⃣ Testing {num_lookups} lookups with OLD method (DataFrame filtering)...")
    
    start_time = time.time()
    found_old = 0
    
    for _, row in test_rows.iterrows():
        series_uid = row['series_uid']
        current_idx = row['slice_idx_in_series']
        
        # Test 3 adjacent slices (like 2.5D with num_adjacent_slices=1)
        for adj_idx in [current_idx-1, current_idx, current_idx+1]:
            # OLD METHOD: DataFrame filtering
            target_slice_row = sample_df[
                (sample_df['series_uid'] == series_uid) & 
                (sample_df['slice_idx_in_series'] == adj_idx)
            ]
            
            if not target_slice_row.empty:
                found_old += 1
    
    old_time = time.time() - start_time
    print(f"⏱️ OLD method: {old_time:.3f} seconds ({found_old} slices found)")
    
    print(f"\n3️⃣ Testing {num_lookups} lookups with NEW method (Dictionary lookup)...")
    
    start_time = time.time()
    found_new = 0
    
    for _, row in test_rows.iterrows():
        series_uid = row['series_uid']
        current_idx = row['slice_idx_in_series']
        
        # Test 3 adjacent slices
        for adj_idx in [current_idx-1, current_idx, current_idx+1]:
            # NEW METHOD: Dictionary lookup
            if (series_uid in slice_lookup and 
                adj_idx in slice_lookup[series_uid]):
                found_new += 1
    
    new_time = time.time() - start_time
    print(f"⚡ NEW method: {new_time:.3f} seconds ({found_new} slices found)")
    
    # Results
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Old method: {old_time:.3f}s")
    print(f"New method: {new_time:.3f}s")
    print(f"Speedup: {old_time/new_time:.1f}x faster")
    print(f"Time saved per lookup: {(old_time-new_time)/num_lookups*1000:.2f}ms")
    
    # Extrapolate to full training
    total_slices = len(slice_df)
    epochs = 200
    adj_slices = 3  # For 2.5D mode
    
    old_total_time = (old_time / num_lookups) * total_slices * epochs * adj_slices / 3600  # hours
    new_total_time = (new_time / num_lookups) * total_slices * epochs * adj_slices / 3600  # hours
    
    print(f"\n⏰ TRAINING TIME ESTIMATE (200 epochs, {total_slices} slices):")
    print(f"Old method: ~{old_total_time:.1f} hours just for slice lookups")
    print(f"New method: ~{new_total_time:.1f} hours just for slice lookups") 
    print(f"Time saved: ~{old_total_time - new_total_time:.1f} hours")
    
    if found_old != found_new:
        print(f"\n⚠️ WARNING: Different number of slices found! Check logic.")
    else:
        print(f"\n✅ Both methods found same number of slices - optimization is correct!")

if __name__ == "__main__":
    benchmark_slice_lookup()