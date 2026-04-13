import torch
import torch.nn as nn
import os
import numpy as np
import argparse
from tools import builder
from utils import misc, dist_utils
from utils.logger import get_root_logger
from utils.config import get_config
from utils.gaussian import write_gaussian_feature_to_ply, unnormalize_gaussians

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    parser.add_argument('--ckpt', type=str, required=True, help='path to checkpoint')
    parser.add_argument('--exp_name', type=str, default='visualization', help='experiment name')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    args.use_gpu = torch.cuda.is_available()
    args.distributed = False
    args.launcher = 'none'
    args.experiment_path = os.path.dirname(args.ckpt)
    args.tfboard_path = args.experiment_path
    args.log_name = "visualization"
    args.resume = False
    args.use_wandb = False
    args.seed = 0
    args.deterministic = True
    args.launcher = 'none'
    args.num_workers = 4

    logger = get_root_logger(name=args.log_name)
    config = get_config(args, logger=logger)
    
    # Ensure config matches training parameters
    config.model.norm_attribute = config.dataset.train.others.norm_attribute
    
    # Set batch size
    if config.get('total_bs') is None:
        config.total_bs = 64
    config.dataset.val.others.bs = 1

    # Build model
    base_model = builder.model_builder(config.model)
    if args.use_gpu:
        base_model = base_model.cuda()
    
    # Load checkpoint
    builder.load_model(base_model, args.ckpt, logger=logger, strict_load=True)
    base_model.eval()

    # Build dataset (val set)
    _, test_dataloader = builder.dataset_builder(args, config.dataset.val)
    
    print("Dataset loaded. Running inference...")

    with torch.no_grad():
        for idx, (taxonomy_ids, model_ids, data, scale_c, scale_m) in enumerate(test_dataloader):
            points = data.cuda()
            
            # Sampling logic matches training
            npoints = config.npoints
            if config.npoints_fps:
                 points = misc.fps_gs(points, npoints, attribute=config.model.group_attribute)
            else:
                random_idx = np.random.choice(points.size(1), npoints, False)
                points = points[:, random_idx, :].contiguous()

            # Run model with save=True
            loss_dict, vis_gaussians, full_rebuild_gaussian, original_gaussians = base_model(points, save=True)

            # Unnormalize
            original_gaussians, vis_gaussians, full_rebuild_gaussian = unnormalize_gaussians(
                original_gaussians,
                vis_gaussians,
                full_rebuild_gaussian,
                scale_c,
                scale_m,
                config
            )

            # Save PLY files for the first object in the batch
            save_dir = os.path.join(args.experiment_path, "visualization_results")
            os.makedirs(save_dir, exist_ok=True)
            
            i = 0 # Just save the first one
            model_id = model_ids[i]
            
            vis_path = os.path.join(save_dir, f"{model_id}_masked_input.ply")
            recon_path = os.path.join(save_dir, f"{model_id}_reconstructed.ply")
            gt_path = os.path.join(save_dir, f"{model_id}_ground_truth.ply")
            
            write_gaussian_feature_to_ply(vis_gaussians[i], vis_path)
            write_gaussian_feature_to_ply(full_rebuild_gaussian[i], recon_path)
            write_gaussian_feature_to_ply(original_gaussians[i], gt_path)
            
            print(f"Saved visualization files to {save_dir}")
            print(f"Masked Input: {vis_path}")
            print(f"Reconstructed: {recon_path}")
            print(f"Ground Truth: {gt_path}")
            
            break # Only visualize one batch/sample

if __name__ == "__main__":
    main()
