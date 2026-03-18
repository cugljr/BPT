import torch
from extensions.chamfer3D.dist_chamfer_3D import chamfer_3DDist_nograd
from tqdm import tqdm

cd_fun = chamfer_3DDist_nograd()


def compute_cd_batch(pred_points, gt_points, batch_size):
    N_sample = pred_points.shape[0]
    N_ref = gt_points.shape[0]
    all_cd = []
    for sample_b_start in tqdm(range(N_sample), desc="Compute CD Batch"):
        sample_batch = pred_points[sample_b_start]
        cd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = gt_points[ref_b_start:ref_b_end]
            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1
            )
            sample_batch_exp = sample_batch_exp.contiguous()
            dl, dr, _, _ = cd_fun(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))
        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)
    all_cd = torch.cat(all_cd, dim=0)
    return all_cd


def computed_mmd_cov_cd(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    _, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {"mmd": mmd, "cov": cov}


def compute_nna_cd(pred_points, gt_points, batch_size):
    results = {}
    M_rs_cd = compute_cd_batch(pred_points, gt_points, batch_size)
    res_cd = computed_mmd_cov_cd(M_rs_cd)
    results.update({"CD-%s" % k: v for k, v in res_cd.items()})
    return results
