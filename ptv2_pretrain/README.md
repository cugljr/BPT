# PTv2 Building Pretrain

这个子项目用于先预训练一个 `xyz-only` 的 `PointTransformerV2` 编码器，再把导出的 `encoder_best.pt` 直接给 BPT 当 conditioner 使用。

## 目录

- `preprocess.py`
  - 从建筑 mesh 预采样并归一化表面点云，生成训练缓存
- `datasets/building_point_npz_dataset.py`
  - 读取缓存点云，生成 `input_points` / `target_points`
- `models/ptv2_encoder.py`
  - 只保留 PTv2 的 `patch_embed + enc_stages`
  - 输出固定形状的 conditioner tokens: `[B, 1 + num_cond_tokens, C]`
- `models/autoencoder.py`
  - 预训练主模型：`PTV2 encoder + global point decoder`
- `train.py`
  - Lightning 训练入口

## 训练数据准备

先把建筑 mesh 预处理成缓存点云：

```bash
python ptv2_pretrain/preprocess.py \
  --mesh-dir /home/kemove/devdata1/ljr/dataset/Tallinn/meshes \
  --split-root /home/kemove/devdata1/ljr/dataset/Tallinn/split \
  --output-dir /home/kemove/devdata1/ljr/dataset/Tallinn/ptv2_pretrain_npz \
  --num-points 16384 \
  --jobs 8
```

每个样本会生成一个 `.npz`，包含：

- `surface_points`
- `center`
- `scale`

## 训练

编辑 [building_ptv2_ae.yaml](/D:/GitCode/BPT/ptv2_pretrain/configs/building_ptv2_ae.yaml) 里的路径，然后执行：

```bash
python ptv2_pretrain/train.py \
  --config ptv2_pretrain/configs/building_ptv2_ae.yaml
```

## 导出产物

训练结束后会在：

```text
runs/<experiment_name>/checkpoints/
```

生成：

- `best.ckpt`
- `last.ckpt`
- `encoder_best.pt`
- `encoder_last.pt`

其中 `encoder_best.pt` 是给 BPT 使用的核心产物。

## 给 BPT 使用

BPT 侧已经预留了 `PTV2Conditioner` 接口。后续在 `src/config/bpt.yaml` 中把：

- `conditioner_type: ptv2`
- `ptv2_repo_path`
- `ptv2_encoder_ckpt`

配好后，就能直接用这个 encoder checkpoint 训练 BPT。
