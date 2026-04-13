import torch
import os
import sys
import argparse
import numpy as np
from plyfile import PlyData

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

from tools import builder
from utils import misc
from utils.config import get_config
from utils.gaussian import write_gaussian_feature_to_ply, unnormalize_gaussians

class DemoArgs:
    def __init__(self):
        self.config = "cfgs/pretrain/pretrain_enc_full_group_xyz_1k.yaml"
        self.experiment_path = "experiments/pretrain_enc_full_group_xyz_1k/pretrain/pretrain_shapesplat_full"
        self.ckpts = os.path.join(self.experiment_path, "ckpt-last.pth")
        self.use_gpu = True
        self.local_rank = 0
        self.distributed = False
        self.log_name = "demo"
        # Dummy values needed by builder
        self.launcher = 'none'
        self.num_workers = 0
        self.seed = 0
        self.exp_name = 'demo'
        self.loss = 'cd1'
        self.val_freq = 1
        self.resume = False
        self.start_ckpts = None
        self.test = False
        self.finetune_model = False
        self.scratch_model = False
        self.mode = None
        self.soft_knn = False
        self.total_bs = 1
        self.way = -1
        self.shot = -1
        self.fold = -1
        self.output_path = "demo_use_case/output"
        self.use_wandb = False
        self.deterministic = False
        self.sync_bn = False
        self.vote = False

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_and_preprocess(path, n_points=8192):
    print(f"Loading {path}...")
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    # 1. READ ATTRIBUTES (Logic from ShapeNet55Gaussian.read_gaussian_attribute)
    x = vertex["x"].astype(np.float32)
    y = vertex["y"].astype(np.float32)
    z = vertex["z"].astype(np.float32)
    data_xyz = np.stack((x, y, z), axis=-1)
    
    # Opacity (Stored as Logit -> Sigmoid -> [0,1])
    opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
    opacity = np_sigmoid(opacity)
    
    # Scale (Stored as Log -> Exp -> Linear)
    scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((data_xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = vertex[attr_name].astype(np.float32)
    scales = np.exp(scales)

    # Rotation (Normalize quaternion)
    rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rots = np.zeros((data_xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = vertex[attr_name].astype(np.float32)
    rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
    signs_vector = np.sign(rots[:, 0])
    rots = rots * signs_vector[:, None]

    # SH (Only DC components loaded in dataset!)
    features_dc = np.zeros((data_xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
    features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
    features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)
    feature_pc = features_dc.reshape(-1, 3)

    # Concatenate all (Order: xyz, opacity, scale, rotation, sh_dc)
    # Note: ShapeNet55Gaussian concatenates in order of checks.
    # It checks xyz, then opacity, then scale/rot, then sh.
    # So: xyz(3), opacity(1), scale(3), rot(4), sh(3) = 14 dims
    data = np.concatenate((data_xyz, opacity, scales, rots, feature_pc), axis=-1)

    # 2. NORMALIZE (Logic from ShapeNet55Gaussian.pc_norm_gs)
    # Normalize XYZ to unit sphere
    pc_xyz = data[..., :3]
    centroid = np.mean(pc_xyz, axis=0)
    pc_xyz = pc_xyz - centroid
    m = np.max(np.sqrt(np.sum(pc_xyz**2, axis=1)))
    pc_xyz = pc_xyz / m
    
    data[..., :3] = pc_xyz
    data[..., 4:7] = data[..., 4:7] / m  # Normalize scale by spatial scale

    # Normalize Opacity to [-1, 1]
    # Current range [0, 1] -> (x - 0)/(1-0)*2 - 1 = x*2 - 1
    data[..., 3] = data[..., 3] * 2 - 1

    # Normalize Scale
    # "s_center = np.mean... s_m = np.max..."
    # Warning: The dataset does scale normalization differently if 'scale' is not in norm_attribute?
    # Config says: norm_attribute: ['xyz','opacity','scale','rotation','sh']
    # So we MUST normalize scale.
    s_center = np.mean(data[..., 4:7], axis=0)
    data[..., 4:7] = data[..., 4:7] - s_center
    s_m = np.max(np.sqrt(np.sum(data[..., 4:7] ** 2, axis=1)))
    data[..., 4:7] = data[..., 4:7] / s_m
    
    # Normalize SH
    # sh = sh * 0.282... clip...
    sh = data[..., 11:14]
    sh = sh * 0.28209479177387814
    sh = np.clip(sh, -0.5, 0.5)
    sh = 2 * sh / np.sqrt(3) # math.sqrt(3)
    data[..., 11:14] = sh

    # Convert to tensor
    points = torch.from_numpy(data).float().unsqueeze(0)
    
    # Sampling
    if points.size(1) >= n_points:
        idx = np.random.choice(points.size(1), n_points, replace=False)
        points = points[:, idx, :]
    else:
        idx = np.random.choice(points.size(1), n_points, replace=True)
        points = points[:, idx, :]

    # Return data AND normalization params needed for un-normalization
    return points, torch.from_numpy(centroid).float(), torch.tensor(m).float(), torch.from_numpy(s_center).float(), torch.tensor(s_m).float()

def main():
    args = DemoArgs()
    os.makedirs(args.output_path, exist_ok=True)
    
    # Config and Model
    config = get_config(args, logger=None)
    config.total_bs = 1
    config.dataset.train.others.bs = 1
    config.dataset.val.others.bs = 1
    config.model.norm_attribute = config.dataset.train.others.norm_attribute
    
    print("Building model...")
    model = builder.model_builder(config.model)
    if args.use_gpu: model = model.cuda()
    
    print(f"Loading checkpoint from {args.ckpts}...")
    builder.load_model(model, args.ckpts, logger=None)
    
    # Load and Preprocess Data correctly
    input_ply = "demo_use_case/input_sample.ply"
    points, centroid, m, s_center, s_m = load_and_preprocess(input_ply, config.npoints)
    points = points.cuda()
    
    # We need to reshape normalization params for unnormalize_gaussians
    # It expects: scale_c [1, 3], scale_m [1] ?
    # Actually wait. In dataset: scale_c is s_center (3), scale_m is s_m (scalar).
    # But unnormalize_gaussians expects them to be tensors on device.
    # And there are TWO types of scaling:
    # 1. Spatial scaling (m) affecting XYZ and gaussian scale.
    # 2. Feature scaling (s_m) affecting gaussian scale.
    
    # utils/gaussian.py unnormalize_gaussians signatures:
    # (..., scale_c, scale_m, config)
    # AND lines 70-74:
    # original_gaussians[..., 4:7] * scale_m (...) + scale_c
    # It seems it only reverses the FEATURE scaling (s_m, s_center).
    # What about spatial scaling (m, centroid)?
    # unnormalize_gaussians DOES NOT seems to reverse spatial scaling (translation/scale of XYZ).
    # It only un-normalizes the attributes (opacity, scale feature, sh).
    # This means the output XYZ will be in the normalized unit sphere space!
    # For visualization, we might want to move it back to original space, OR just accept it's normalized.
    # The user complained it "looks weird". If it's normalized to unit sphere at origin, it should look fine but small/centered.
    # If the scale features are wrong, splats look huge or tiny.
    
    # Let's pass s_center/s_m as scale_c/scale_m to unnormalize_gaussians.
    scale_c = s_center.unsqueeze(0).cuda() # [1, 3]
    scale_m = s_m.unsqueeze(0).cuda()      # [1]
    
    # Run Inference
    print("Running inference...")
    model.eval()
    with torch.no_grad():
        loss_dict, vis_gaussians, full_rebuild_gaussian, original_gaussians = model(points, save=True)
        
    print(f"Reconstruction Loss (Chamfer): {loss_dict['cd'].item()}")
    
    # Un-normalize Attributes (Opacity, Scale, SH)
    orig_un, vis_un, rebuild_un = unnormalize_gaussians(
        original_gaussians, vis_gaussians, full_rebuild_gaussian,
        scale_c, scale_m, config
    )
    
    # Sanitize again just to be safe (clipping probabilities etc)
    for tensor in [orig_un, vis_un, rebuild_un]:
        tensor[..., 3] = torch.clamp(tensor[..., 3], 0.0001, 0.9999) # Opacity
        tensor[..., 4:7] = torch.clamp(torch.abs(tensor[..., 4:7]), min=1e-6) # Scale
        
    # NOTE: Output XYZ is still normalized. If we want original pose:
    # xyz = xyz * m + centroid
    # But write_gaussian_feature_to_ply just takes the tensor.
    # Let's verify if we should un-normalize XYZ.
    # Usually for viewing it doesn't matter as long as it's consistent.
    
    # Save
    print("Saving outputs...")
    write_gaussian_feature_to_ply(vis_un[0], os.path.join(args.output_path, "reconstructed_masked.ply"))
    write_gaussian_feature_to_ply(rebuild_un[0], os.path.join(args.output_path, "reconstructed_full.ply"))
    write_gaussian_feature_to_ply(orig_un[0], os.path.join(args.output_path, "original_input.ply"))
    
    print(f"Done! Results saved to {args.output_path}")

if __name__ == "__main__":
    main()
