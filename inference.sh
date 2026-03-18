#!/bin/bash

# 配置变量
SCRIPT="inference.py"
EXP_NAME="bpt_worried-era"
CKPT_TYPE="best"  # "best" 或 "last"

# 输入文件，可以是 .obj 或 .xyz或 .txt
FILE_PATH="/home/kemove/devdata1/zyx/WireframeGPT/dataset/Zurich_P2B_Polygon/model/00000.obj"

python "$SCRIPT" \
    --exp_name "$EXP_NAME" \
    --ckpt_type "$CKPT_TYPE" \
    --file_path "$FILE_PATH" \
    --n_trial 5
    # --use_vertices \
    