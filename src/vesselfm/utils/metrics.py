import torch


def volumetric_recall(pred_mask, gt_mask):
    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()
    tp = torch.logical_and(pred_mask, gt_mask).sum(dim=(-3, -2, -1))
    fn = torch.logical_and(~pred_mask, gt_mask).sum(dim=(-3, -2, -1))
    recall = tp.float() / (tp + fn).float().clamp(min=1)
    return recall.mean(), tp.sum(), fn.sum()