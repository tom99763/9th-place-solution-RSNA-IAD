import torch


def volumetric_recall(pred_mask, gt_mask, num_classes=13, ignore_index=None):
    """
    pred_mask: (B, C, D, H, W) raw logits or softmax probs
    gt_mask:   (B, D, H, W) with class indices [0..C-1]
    """
    # Convert predictions to class IDs
    pred_classes = pred_mask.softmax(dim=1).argmax(dim=1)  # (B, D, H, W)

    recalls = []
    tps, fns = [], []

    for c in range(num_classes):
        if ignore_index is not None and c == ignore_index:
            continue

        gt_c = (gt_mask == c)
        pred_c = (pred_classes == c)

        tp = torch.logical_and(pred_c, gt_c).sum(dim=(-3, -2, -1))
        fn = torch.logical_and(~pred_c, gt_c).sum(dim=(-3, -2, -1))

        recall_c = tp.float() / (tp + fn).float().clamp(min=1)
        recalls.append(recall_c.mean())
        tps.append(tp.sum())
        fns.append(fn.sum())

    mean_recall = torch.stack(recalls).mean()
    return mean_recall, torch.stack(tps), torch.stack(fns)