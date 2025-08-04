from pathlib import Path

import numpy as np
import torch
from skimage.morphology import skeletonize, skeletonize_3d
from skimage.measure import euler_number, label
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import SimpleITK as sitk
from torch.utils.data import Dataset


class PretrainEvaluationDataset(Dataset):
    def __init__(self, data_path):
        data_dir = Path(data_path).resolve()
        self.val_data = {
            "deepvess": [
                torch.tensor(read_nifti(data_dir / "deepvess.nii"))[None],
                torch.tensor(read_nifti(data_dir / "deepvess_mask.nii"))[None],
            ],
            "deepvesselnet": [
                torch.tensor(read_nifti(data_dir / "deepvesselnet.nii"))[None],
                torch.tensor(read_nifti(data_dir / "deepvesselnet_mask.nii"))[None],
            ],
            "lightsheet": [
                torch.tensor(read_nifti(data_dir / "lightsheet.nii"))[None],
                torch.tensor(read_nifti(data_dir / "lightsheet_mask.nii"))[None],
            ],
            "minivess": [
                torch.tensor(read_nifti(data_dir / "minivess.nii"))[None],
                torch.tensor(read_nifti(data_dir / "minivess_mask.nii"))[None],
            ],
            "tubetk": [
                torch.tensor(read_nifti(data_dir / "tubetk.nii"))[None],
                torch.tensor(read_nifti(data_dir / "tubetk_mask.nii"))[None],
            ],
        }
        self._samples = list(self.val_data.keys())

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        name = self._samples[idx]
        image, mask = self.val_data[self._samples[idx]]
        return image, mask, name
        

class Evaluator:
    def extract_labels(self, gt_array, pred_array):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L18.
        """
        labels_gt = np.unique(gt_array)
        labels_pred = np.unique(pred_array)
        labels = list(set().union(labels_gt, labels_pred))
        labels = [int(x) for x in labels]
        return labels

    def betti_number_error(self, gt, pred):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L250.
        """
        labels = self.extract_labels(gt_array=gt, pred_array=pred)
        labels.remove(0)

        if len(labels) == 0:
            return 0, 0
        assert len(labels) == 1 and 1 in labels, "Invalid binary segmentatio.n"

        gt_betti_numbers = self.betti_number(gt)
        pred_betti_numbers = self.betti_number(pred)
        betti_0_error = abs(pred_betti_numbers[0] - gt_betti_numbers[0])
        betti_1_error = abs(pred_betti_numbers[1] - gt_betti_numbers[1])
        return betti_0_error, betti_1_error

    def betti_number(self, img):
        """
        Adapted from https://github.com/CoWBenchmark/TopCoW_Eval_Metrics/blob/master/metric_functions.py#L186.
        """
        assert img.ndim == 3
        N6 = 1
        N26 = 3

        padded = np.pad(img, pad_width=1)
        assert set(np.unique(padded)).issubset({0, 1})

        _, b0 = label(padded, return_num=True, connectivity=N26)
        euler_char_num = euler_number(padded, connectivity=N26)
        _, b2 = label(1 - padded, return_num=True, connectivity=N6)

        b2 -= 1
        b1 = b0 + b2 - euler_char_num
        return [b0, b1, b2]

    def cl_dice(self, v_p, v_l):
        """
        Adapted from https://github.com/jocpae/clDice/blob/master/cldice_metric/cldice.py.
        """
        def cl_score(v, s):
            return np.sum(v * s) / np.sum(s)

        if len(v_p.shape) == 2:
            tprec = cl_score(v_p, skeletonize(v_l))
            tsens = cl_score(v_l, skeletonize(v_p))
        elif len(v_p.shape) == 3:
            tprec = cl_score(v_p, skeletonize_3d(v_l))
            tsens = cl_score(v_l, skeletonize_3d(v_p))
        else:
            raise ValueError(f"Invalid shape for cl_dice: {v_p.shape}")
        return 2 * tprec * tsens / (tprec + tsens + np.finfo(float).eps)

    def estimate_metrics(self, pred_seg, gt_seg, threshold=0.5, fast=False):
        metrics = {}
        pred_seg_thresh = (pred_seg >= threshold).float().cpu()

        # estimate metrics
        tn, fp, fn, tp = confusion_matrix(
            gt_seg.flatten().cpu().clone().numpy(),
            pred_seg_thresh.flatten().cpu().clone().numpy(),
            labels=[0, 1],
        ).ravel()

        if fast:
            metrics["dice"] = (2 * tp) / (2 * tp + fp + fn)
            return metrics

        roc_auc = roc_auc_score(
            gt_seg.flatten().cpu().clone().detach().numpy(),
            pred_seg.flatten().cpu().clone().detach().numpy(),
        )

        pr_auc = average_precision_score(
            gt_seg.flatten().cpu().clone().detach().numpy(),
            pred_seg.flatten().cpu().clone().detach().numpy(),
        )

        cldice = self.cl_dice(
            pred_seg_thresh.squeeze().cpu().clone().detach().byte().numpy(),
            gt_seg.squeeze().cpu().clone().detach().byte().numpy(),
        )

        betti_0_error, betti_1_error = self.betti_number_error(
            gt_seg.squeeze().cpu().clone().detach().int().numpy(),
            pred_seg_thresh.squeeze().cpu().clone().detach().int().numpy(),
        )
        betti_0, betti_1, betti_2 = self.betti_number(
            pred_seg_thresh.squeeze().cpu().clone().detach().int().numpy()
        )

        metrics["recall_tpr_sensitivity"] = tp / (tp + fn)
        metrics["fpr"] = fp / (fp + tn)
        metrics["precision"] = tp / (tp + fp)
        metrics["specificity"] = tn / (tn + fp)
        metrics["jaccard_iou"] = tp / (tp + fp + fn)
        metrics["dice"] = (2 * tp) / (2 * tp + fp + fn)
        metrics["cldice"] = cldice
        metrics["accuracy"] = (tp + tn) / (tn + fp + tp + fn)
        metrics["roc_auc"] = roc_auc
        metrics["pr_auc_ap"] = pr_auc
        metrics["betti_0_error"] = betti_0_error
        metrics["betti_1_error"] = betti_1_error
        metrics["betti_0"] = betti_0
        metrics["betti_1"] = betti_1
        metrics["betti_2"] = betti_2
        return metrics


def read_nifti(path: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def calculate_mean_metrics(results, round_to=2):
    mean = {}
    for k in results[0].keys():
        numbers = [r[k] for r in results]
        numbers = [n for n in numbers if np.isnan(n) == False]
        mean[k] = np.mean(numbers)

        if "dice" in k:
            mean[k] = mean[k] * 100
        mean[k] = np.round(mean[k], round_to)
    return mean
