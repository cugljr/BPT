import torch


def chamfer_distance(pred_points: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """Symmetric Chamfer distance for point clouds in shape [B, N, 3]."""
    pred_points = pred_points.float()
    target_points = target_points.float()
    if not torch.isfinite(pred_points).all():
        raise FloatingPointError("pred_points contains NaN or Inf before Chamfer distance")
    if not torch.isfinite(target_points).all():
        raise FloatingPointError("target_points contains NaN or Inf before Chamfer distance")
    distances = torch.cdist(pred_points, target_points, p=2)
    if not torch.isfinite(distances).all():
        raise FloatingPointError("Chamfer distance matrix contains NaN or Inf")
    pred_to_target = distances.min(dim=2).values
    target_to_pred = distances.min(dim=1).values
    return pred_to_target.mean() + target_to_pred.mean()
