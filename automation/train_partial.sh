#!/bin/bash

# Ensure conda env is active (optional if running via conda run)
# source activate shape_splat

echo "Generating partial dataset split based on available files..."
python scripts/dev/create_partial_split.py

echo "Starting training with partial dataset..."
torchrun --nproc_per_node=1 main.py \
    --config cfgs/pretrain/pretrain_enc_full_group_xyz_1k.yaml \
    --exp_name pretrain_shapesplat_partial \
    --total_bs 64 \
    --ckpts pretrain_shapesplat_partial_ckpt
