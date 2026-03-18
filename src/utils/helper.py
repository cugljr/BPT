import torch
from x_transformers.autoregressive_wrapper import top_p, top_k

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def first(it):
    return it[0]

def divisible_by(num, den):
    return (num % den) == 0


def accuracy(y_pred: torch.Tensor, y_gt: torch.Tensor, ignore_label=None) -> float:
    """计算分类准确率（支持 ignore_label 掩码）"""
    pred = y_pred.argmax(dim=-1)
    mask = (
        (y_gt != ignore_label)
        if ignore_label is not None
        else torch.ones_like(y_gt, dtype=torch.bool)
    )
    correct = (pred == y_gt) & mask
    return correct.sum() / mask.sum()


def joint_filter(logits, k=50, p=0.95):
    logits = top_k(logits, k=k)
    logits = top_p(logits, thres=p)
    return logits
