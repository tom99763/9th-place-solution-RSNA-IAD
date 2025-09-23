import numpy as np
import torch
import cv2
#from skimage.filters import frangi
from concurrent.futures import ThreadPoolExecutor, as_completed

class AneurysmVolumeProcessor3Planes:
    """
    Processes a 3D volume with YOLO-detected points.
    Extracts Axial, Sagittal, Coronal patches around each point.
    Returns Cartesian/Log-polar features with center slice and MIP only.
    Parallelized for speed.
    """

    def __init__(self, N=96, K_axial=5, K_sagittal=15, K_coronal=15,
                 Nr=64, Ntheta=128, augment=False, device='cpu', n_workers=4):
        self.N = N
        self.K_axial = K_axial
        self.K_sagittal = K_sagittal
        self.K_coronal = K_coronal
        self.Nr = Nr
        self.Ntheta = Ntheta
        self.augment = augment
        self.device = device
        self.n_workers = n_workers

    def __call__(self, volume, yolo_points):
        outputs = []

        def process_point(point):
            x, y, z = map(int, map(round, point))
            planes = {}

            # --- Axial ---
            K = self.K_axial
            z_min, z_max = max(0, z-K//2), min(volume.shape[0], z+K//2+1)
            y_min, y_max = max(0, y-self.N//2), min(volume.shape[1], y+self.N//2)
            x_min, x_max = max(0, x-self.N//2), min(volume.shape[2], x+self.N//2)
            axial_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            axial_patch = self._pad_patch(axial_patch, (K, self.N, self.N))
            planes['axial'] = axial_patch

            # --- Sagittal ---
            K = self.K_sagittal
            x_min, x_max = max(0, x-K//2), min(volume.shape[2], x+K//2+1)
            z_min, z_max = max(0, z-self.N//2), min(volume.shape[0], z+self.N//2)
            y_min, y_max = max(0, y-self.N//2), min(volume.shape[1], y+self.N//2)
            sag_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            sag_patch = np.transpose(sag_patch, (2, 0, 1))  # (x, z, y)
            sag_patch = self._pad_patch(sag_patch, (K, self.N, self.N))
            planes['sagittal'] = sag_patch

            # --- Coronal ---
            K = self.K_coronal
            y_min, y_max = max(0, y-K//2), min(volume.shape[1], y+K//2+1)
            z_min, z_max = max(0, z-self.N//2), min(volume.shape[0], z+self.N//2)
            x_min, x_max = max(0, x-self.N//2), min(volume.shape[2], x+self.N//2)
            cor_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            cor_patch = np.transpose(cor_patch, (1, 0, 2))  # (y, z, x)
            cor_patch = self._pad_patch(cor_patch, (K, self.N, self.N))
            planes['coronal'] = cor_patch

            # --- Cartesian & Log-Polar features ---
            cartesian_channels, logpolar_channels = [], []
            for plane_name, patch in planes.items():
                center_slice = patch[patch.shape[0] // 2]
                mip = np.max(patch, axis=0)

                # stack [center_slice, mip] only
                cartesian_channels.append(np.stack([center_slice, mip], axis=0))

                cx, cy = self.N / 2, self.N / 2
                logpolar_channels.append(np.stack([
                    self._logpolar(center_slice, cx, cy),
                    self._logpolar(mip, cx, cy)
                ], axis=0))

            return {
                'cartesian': torch.from_numpy(np.stack(cartesian_channels, axis=0)).float(),
                'logpolar': torch.from_numpy(np.stack(logpolar_channels, axis=0)).float(),
                'axial': torch.from_numpy(planes['axial']).float(),
                'sagittal': torch.from_numpy(planes['sagittal']).float(),
                'coronal': torch.from_numpy(planes['coronal']).float()
            }

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = [executor.submit(process_point, pt) for pt in yolo_points]
            for f in as_completed(futures):
                outputs.append(f.result())

        return outputs

    # ------------------ Helpers ------------------
    def _pad_patch(self, patch, shape):
        K, N, _ = shape
        padded = np.zeros(shape, dtype=patch.dtype)
        dz, dy, dx = patch.shape
        padded[:dz, :dy, :dx] = patch
        return padded

    def _logpolar(self, img, cx, cy):
        img = img.astype(np.float32)
        max_radius = np.sqrt(max(cx, img.shape[1]-cx)**2 + max(cy, img.shape[0]-cy)**2)
        logpolar_img = cv2.logPolar(
            img, center=(cx, cy),
            M=self.Nr / np.log(max_radius + 1e-6),
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
        )
        return cv2.resize(logpolar_img, (self.Ntheta, self.Nr))