#!/bin/bash
# Activate conda environment and start training
source $(dirname $(which conda))/../etc/profile.d/conda.sh
conda activate shape_splat

echo "Starting training in screen session..."
torchrun --nproc_per_node=1 main.py \
    --config cfgs/pretrain/pretrain_enc_full_group_xyz_1k.yaml \
    --exp_name pretrain_shapesplat_full \
    --total_bs 64
