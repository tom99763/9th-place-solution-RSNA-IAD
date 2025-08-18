import os.path

import torch
from torch_cluster import knn_graph
import numpy as np
import gc


def sample_positive_points(segmentation, N):
    B, _, Z, Y, X = segmentation.shape
    points_list = []

    for b in range(B):
        pos_idx = torch.nonzero(segmentation[b, 0], as_tuple=False)  # (num_pos, 3)

        if len(pos_idx) == 0:
            rand_idx = torch.stack([
                torch.randint(0, Z, (N,)),
                torch.randint(0, Y, (N,)),
                torch.randint(0, X, (N,))
            ], dim=1)
            points_list.append(rand_idx)
            continue

        if len(pos_idx) >= N:
            # Uniformly spaced indices in the sorted list
            choice = torch.linspace(0, len(pos_idx) - 1, steps=N).long()
            choice = torch.clamp(choice, max=len(pos_idx) - 1)
            sampled = pos_idx[choice]
        else:
            # Repeat points if not enough
            repeats = (N + len(pos_idx) - 1) // len(pos_idx)
            repeated = pos_idx.repeat((repeats, 1))
            choice = torch.linspace(0, len(repeated) - 1, steps=N).long()
            sampled = repeated[choice]

        points_list.append(sampled)

    points = torch.stack(points_list, dim=0)  # (B, N, 3)
    return points


class StopForward(Exception): pass

def extract_features_with_hook(model, image, stop_at_layer):
    features = {}
    def hook_fn(module, input, output):
        features['feat'] = output
        raise StopForward()

    target_module = dict(model.named_modules())[stop_at_layer]
    handle = target_module.register_forward_hook(hook_fn)
    try:
        _ = model(image)
    except StopForward:
        pass
    finally:
        handle.remove()
    return features['feat']


def compute_slices(spatial_shape, roi_size, step):
    slices_list = []
    for start_d in range(0, spatial_shape[0], step[0]):
        end_d = start_d + roi_size[0]
        if end_d > spatial_shape[0]:
            start_d = spatial_shape[0] - roi_size[0]
            end_d = spatial_shape[0]

        for start_h in range(0, spatial_shape[1], step[1]):
            end_h = start_h + roi_size[1]
            if end_h > spatial_shape[1]:
                start_h = spatial_shape[1] - roi_size[1]
                end_h = spatial_shape[1]

            for start_w in range(0, spatial_shape[2], step[2]):
                end_w = start_w + roi_size[2]
                if end_w > spatial_shape[2]:
                    start_w = spatial_shape[2] - roi_size[2]
                    end_w = spatial_shape[2]

                slices_list.append((
                    (start_d, end_d),
                    (start_h, end_h),
                    (start_w, end_w)
                ))

    # Remove duplicates by converting to hashable tuples
    slices_list = list(dict.fromkeys(slices_list))

    # Convert tuples back to slices
    slices_list = [
        (slice(d[0], d[1]), slice(h[0], h[1]), slice(w[0], w[1]))
        for d, h, w in slices_list
    ]
    return slices_list


def sliding_window_patches(image, roi_size, device=None, overlap=0.5):
    batch_mode = (image.dim() == 5)
    if batch_mode:
        B = image.shape[0]
        C = image.shape[1]
        spatial_shape = image.shape[2:]
    else:
        B = 1
        C = image.shape[0]
        spatial_shape = image.shape[1:]
        image = image.unsqueeze(0)  # add batch dim

    step = [max(1, int(s * (1 - overlap))) for s in roi_size]
    slices = compute_slices(spatial_shape, roi_size, step)

    for sl in slices:
        patch = image[(slice(None), slice(None)) + sl]  # (B, C, D_roi, H_roi, W_roi)
        coord = (sl[0].start, sl[1].start, sl[2].start)
        if batch_mode:
            yield patch.to(device), coord
        else:
            yield patch[0].to(device), coord


def assign_point_features_with_sliding(model, image, points, stop_at="downsamples.3", roi_size=(128,128,128), overlap=0.5):
    B, N, _ = points.shape
    device = image.device
    point_feats_sum = None
    point_counts = torch.zeros((B, N), device=device)

    for patch, (start_z, start_y, start_x) in sliding_window_patches(image, roi_size, device=device, overlap=overlap):
        with torch.no_grad():
            feat_map = extract_features_with_hook(model, patch, stop_at)
        C_feat = feat_map.shape[1]

        if point_feats_sum is None:
            point_feats_sum = torch.zeros((B, N, C_feat), device=device)

        scale_z = feat_map.shape[2] / patch.shape[2]
        scale_y = feat_map.shape[3] / patch.shape[3]
        scale_x = feat_map.shape[4] / patch.shape[4]

        for b in range(B):
            mask_inside = (
                (points[b, :, 0] >= start_z) & (points[b, :, 0] < start_z + patch.shape[2]) &
                (points[b, :, 1] >= start_y) & (points[b, :, 1] < start_y + patch.shape[3]) &
                (points[b, :, 2] >= start_x) & (points[b, :, 2] < start_x + patch.shape[4])
            )
            inside_idx = torch.nonzero(mask_inside, as_tuple=False).squeeze(1)
            if len(inside_idx) == 0:
                continue

            local_z = ((points[b, inside_idx, 0] - start_z).float() * scale_z).long().clamp(0, feat_map.shape[2]-1)
            local_y = ((points[b, inside_idx, 1] - start_y).float() * scale_y).long().clamp(0, feat_map.shape[3]-1)
            local_x = ((points[b, inside_idx, 2] - start_x).float() * scale_x).long().clamp(0, feat_map.shape[4]-1)

            feats = feat_map[b, :, local_z, local_y, local_x].permute(1,0)  # (num_pts, C_feat)
            point_feats_sum[b, inside_idx] += feats
            point_counts[b, inside_idx] += 1

    point_feats = point_feats_sum / point_counts.clamp_min(1).unsqueeze(-1)
    return point_feats


def extract_and_save(uid, save_path, vol, segmap, model,
                         target_layer = 'downsamples.2',
                         N = 10000, threshold = 0.1):
    mask = segmap>threshold
    points = sample_positive_points(mask[None, None], N)
    point_feats = assign_point_features_with_sliding(model, vol, points, stop_at=target_layer)
    edge_index_k5 = knn_graph(points[0], k=5, loop=False)
    edge_index_k10 = knn_graph(points[0], k=10, loop=False)
    edge_index_k15 = knn_graph(points[0], k=15, loop=False)

    if not os.path.exists(f'{save_path}/{uid}'):
        os.makedirs(f'{save_path}/{uid}')

    np.save(f'{save_path}/{uid}/{uid}_points.npy', points[0].cpu())
    np.save(f'{save_path}/{uid}/{uid}_point_feats.npy', point_feats[0].cpu())
    np.save(f'{save_path}/{uid}/{uid}_edge_index_k5.npy', edge_index_k5)
    np.save(f'{save_path}/{uid}/{uid}_edge_index_k10.npy', edge_index_k10)
    np.save(f'{save_path}/{uid}/{uid}_edge_index_k15.npy', edge_index_k15)

    del points
    del point_feats
    del mask
    del segmap
    gc.collect()


