import torch


def chamfer_distance(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance for point clouds in shape [B, N, 3]."""
    distances = torch.cdist(pred_points, target_points, p=2)
    pred_to_target = distances.min(dim=2).values
    target_to_pred = distances.min(dim=1).values
    return pred_to_target.mean() + target_to_pred.mean()
