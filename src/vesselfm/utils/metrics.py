import torch


def volumetric_recall(pred_mask, gt_mask):
    if pred_mask.shape[1] == 1:
        pred_mask = pred_mask.squeeze(1)
        gt_mask = gt_mask.squeeze(1)

    pred_mask = pred_mask.bool()
    gt_mask = gt_mask.bool()

    tp = torch.logical_and(pred_mask, gt_mask).sum(dim=(-3, -2, -1))
    fn = torch.logical_and(~pred_mask, gt_mask).sum(dim=(-3, -2, -1))

    recall = tp.float() / (tp + fn).float().clamp(min=1)
    return recall.mean(), tp.sum(), fn.sum()