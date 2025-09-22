#!/usr/bin/env python3
"""
Resize Experiments: Compare different methods for resizing 300x128x128 frames to 512x512
Methods tested:
1. For loop with OpenCV resize
2. Scipy ndimage zoom
3. CuPy resize
"""

import numpy as np
import time
import cv2
from scipy import ndimage
import cupy as cp

def generate_test_data(num_frames=300, height=128, width=128):
    """Generate random test data with shape (num_frames, height, width)"""
    return np.random.rand(num_frames, height, width).astype(np.float32)

def method1_forloop_opencv(frames, target_size=(512, 512)):
    """Method 1: For loop with OpenCV resize"""
    resized_frames = np.zeros((frames.shape[0], target_size[0], target_size[1]), dtype=frames.dtype)
    for i in range(frames.shape[0]):
        resized_frames[i] = cv2.resize(frames[i], target_size, interpolation=cv2.INTER_LINEAR)
    return resized_frames

def method2_scipy_zoom(frames, target_size=(512, 512)):
    """Method 2: Scipy ndimage zoom"""
    zoom_factors = (1.0, target_size[0] / frames.shape[1], target_size[1] / frames.shape[2])
    return ndimage.zoom(frames, zoom_factors, order=1, mode='nearest')

def method3_cupy_resize(frames, target_size=(512, 512)):
    """Method 3: CuPy resize using scipy.ndimage.zoom on GPU arrays"""
    # Convert to CuPy array
    frames_gpu = cp.asarray(frames)
    
    # Use CuPy's scipy-compatible ndimage module
    from cupyx.scipy import ndimage as cp_ndimage
    
    # Calculate zoom factors
    zoom_factors = (1.0, target_size[0] / frames.shape[1], target_size[1] / frames.shape[2])
    
    # Resize using CuPy's ndimage zoom
    resized_gpu = cp_ndimage.zoom(frames_gpu, zoom_factors, order=1, mode='nearest')
    
    # Convert back to CPU
    return cp.asnumpy(resized_gpu)

def run_experiment(method_func, frames, method_name, num_runs=10):
    """Run timing experiment for a specific method"""
    print(f"\n=== {method_name} ===")
    times = []
    
    for run in range(num_runs):
        start_time = time.time()
        result = method_func(frames)
        end_time = time.time()
        
        elapsed = end_time - start_time
        times.append(elapsed)
        print(f"Run {run + 1:2d}: {elapsed:.4f} seconds")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nStatistics for {method_name}:")
    print(f"  Average: {avg_time:.4f} seconds")
    print(f"  Std Dev: {std_time:.4f} seconds")
    print(f"  Min:     {min_time:.4f} seconds")
    print(f"  Max:     {max_time:.4f} seconds")
    
    return times, result

def verify_outputs(results):
    """Verify that all methods produce the same output shape"""
    print(f"\n=== Output Verification ===")
    for i, (method_name, result) in enumerate(results.items()):
        print(f"{method_name}: Shape {result.shape}, Dtype {result.dtype}")
    
    # Check if all outputs have the same shape
    shapes = [result.shape for result in results.values()]
    if len(set(shapes)) == 1:
        print("✓ All methods produce the same output shape")
    else:
        print("✗ Methods produce different output shapes")
        for method_name, shape in zip(results.keys(), shapes):
            print(f"  {method_name}: {shape}")

def main():
    print("Resize Experiments: 300x128x128 → 512x512")
    print("=" * 50)
    
    # Generate test data
    print("Generating test data...")
    frames = generate_test_data(num_frames=300, height=128, width=128)
    print(f"Input shape: {frames.shape}")
    print(f"Input dtype: {frames.dtype}")
    print(f"Input memory usage: {frames.nbytes / 1024**2:.2f} MB")
    
    # Check if CuPy is available
    try:
        cp.cuda.Device(0).use()
        cupy_available = True
        print("✓ CuPy is available")
    except:
        cupy_available = False
        print("✗ CuPy is not available - skipping CuPy method")
    
    # Run experiments
    results = {}
    
    # Method 1: For loop with OpenCV
    times1, result1 = run_experiment(method1_forloop_opencv, frames, "For Loop + OpenCV", num_runs=10)
    results["For Loop + OpenCV"] = result1
    
    # Method 2: Scipy ndimage zoom
    times2, result2 = run_experiment(method2_scipy_zoom, frames, "Scipy ndimage zoom", num_runs=10)
    results["Scipy ndimage zoom"] = result2
    
    # Method 3: CuPy (if available)
    if cupy_available:
        times3, result3 = run_experiment(method3_cupy_resize, frames, "CuPy resize", num_runs=10)
        results["CuPy resize"] = result3
    else:
        print("\n=== CuPy resize ===")
        print("Skipped - CuPy not available")
    
    # Verify outputs
    verify_outputs(results)
    
    # Summary comparison
    print(f"\n=== Performance Summary ===")
    all_times = [times1, times2]
    all_names = ["For Loop + OpenCV", "Scipy ndimage zoom"]
    
    if cupy_available:
        all_times.append(times3)
        all_names.append("CuPy resize")
    
    avg_times = [np.mean(times) for times in all_times]
    
    print("Average execution times:")
    for name, avg_time in zip(all_names, avg_times):
        print(f"  {name:20s}: {avg_time:.4f} seconds")
    
    # Find fastest method
    fastest_idx = np.argmin(avg_times)
    print(f"\nFastest method: {all_names[fastest_idx]} ({avg_times[fastest_idx]:.4f} seconds)")
    
    # Speedup comparison
    print(f"\nSpeedup relative to slowest method:")
    slowest_time = max(avg_times)
    for name, avg_time in zip(all_names, avg_times):
        speedup = slowest_time / avg_time
        print(f"  {name:20s}: {speedup:.2f}x")

if __name__ == "__main__":
    main()
