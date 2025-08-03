import pydicom
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from prepare_data import apply_dicom_windowing, get_windowing_params

def read_dicom(path):
    ds = pydicom.dcmread(path)
    return ds


def plot_dicom(ds): 
    img = ds.pixel_array
    window_center, window_width = get_windowing_params(ds.Modality)
    img_windowed = apply_dicom_windowing(img, window_center, window_width)
    #compare both images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(img_windowed, cmap=plt.cm.gray)
    plt.title("Windowed")
    plt.axis('off')
    plt.savefig("test.png")


if __name__ == "__main__":
    ds = read_dicom("processed_data/processed_data/data/processed/1.2.826.0.1.3680043.8.498.10004684224894397679901841656954650085.npz")
    plot_dicom(ds)