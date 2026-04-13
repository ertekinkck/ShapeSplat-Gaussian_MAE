import torch
import os
import sys
import argparse
import numpy as np
from plyfile import PlyData

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

from tools import builder
from utils.config import get_config

# ModelNet40 Class Names
MODELNET40_CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl", "car", 
    "chair", "cone", "cup", "curtain", "desk", "door", "dresser", "flower_pot", 
    "glass_box", "guitar", "keyboard", "lamp", "laptop", "mantel", "monitor", 
    "night_stand", "person", "piano", "plant", "radio", "range_hood", "sink", 
    "sofa", "stairs", "stool", "table", "tent", "toilet", "tv_stand", "vase", 
    "wardrobe", "xbox"
]

class DemoArgs:
    def __init__(self):
        self.config = "cfgs/finetune/finetune_modelnet40_enc_full_group_xyz_4k.yaml"
        self.experiment_path = "experiments/demo_classification" 
        self.ckpts = "demo_classification/modelnet40_best.pth"
        self.use_gpu = True
        self.local_rank = 0
        self.distributed = False
        self.log_name = "demo_cls"
        self.num_workers = 0
        self.seed = 0
        self.exp_name = 'demo_cls'
        self.launcher = 'none'
        self.test = True # Classification usually runs in test mode
        self.resume = False
        self.mode = None
        self.output_path = "demo_classification/output"
        
        # Extra required fields
        self.loss = 'cd1'
        self.val_freq = 1
        self.start_ckpts = None
        self.finetune_model = False
        self.scratch_model = False
        self.soft_knn = False
        self.total_bs = 1
        self.way = -1
        self.shot = -1
        self.fold = -1
        self.use_wandb = False
        self.deterministic = False
        self.sync_bn = False
        self.vote = False

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def load_and_preprocess(path, norm_attribute, n_points=4096):
    print(f"Loading {path}...")
    plydata = PlyData.read(path)
    vertex = plydata['vertex']
    
    # 1. READ ATTRIBUTES (Always read all relevant)
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

    # SH (DC only)
    features_dc = np.zeros((data_xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
    features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
    features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)
    feature_pc = features_dc.reshape(-1, 3)

    data = np.concatenate((data_xyz, opacity, scales, rots, feature_pc), axis=-1)

    # 2. NORMALIZE (Conditionally based on norm_attribute)
    # Always normalize spatial XYZ and spatial scale of features
    pc_xyz = data[..., :3]
    centroid = np.mean(pc_xyz, axis=0)
    pc_xyz = pc_xyz - centroid
    m = np.max(np.sqrt(np.sum(pc_xyz**2, axis=1)))
    pc_xyz = pc_xyz / m
    
    data[..., :3] = pc_xyz
    data[..., 4:7] = data[..., 4:7] / m  # Normalize Gaussian scale by spatial scale

    if "opacity" in norm_attribute:
        # map [0,1] to [-1,1]
        data[..., 3] = data[..., 3] * 2 - 1

    if "scale" in norm_attribute:
        s_center = np.mean(data[..., 4:7], axis=0)
        data[..., 4:7] = data[..., 4:7] - s_center
        s_m = np.max(np.sqrt(np.sum(data[..., 4:7] ** 2, axis=1)))
        data[..., 4:7] = data[..., 4:7] / s_m
    
    if "sh" in norm_attribute:
        sh = data[..., 11:14]
        sh = sh * 0.28209479177387814
        sh = np.clip(sh, -0.5, 0.5)
        sh = 2 * sh / np.sqrt(3)
        data[..., 11:14] = sh

    # Convert to tensor
    points = torch.from_numpy(data).float().unsqueeze(0)
    
    # Sampling for 4k
    if points.size(1) >= n_points:
        idx = np.random.choice(points.size(1), n_points, replace=False)
        points = points[:, idx, :]
    else:
        idx = np.random.choice(points.size(1), n_points, replace=True)
        points = points[:, idx, :]

    return points

import glob

def main():
    args = DemoArgs()
    os.makedirs(args.output_path, exist_ok=True)
    
    # Config
    config = get_config(args, logger=None)
    config.npoints = 4096 
    config.model.group_size = 32
    config.model.num_group = 128
    config.model.norm_attribute = config.dataset.train.others.norm_attribute

    print("Building model (Classifier)...")
    model = builder.model_builder(config.model)
    if args.use_gpu: model = model.cuda()
        
    print(f"Loading classification checkpoint from {args.ckpts}...")
    builder.load_model(model, args.ckpts, logger=None)
    model.eval()

    # Find all samples
    sample_dir = "demo_classification/samples"
    files = glob.glob(os.path.join(sample_dir, "*.ply"))
    files.sort()
    
    if not files:
        print(f"No .ply files found in {sample_dir}")
        return

    print(f"Found {len(files)} samples. Running classification...\n")
    
    results = []

    for filepath in files:
        filename = os.path.basename(filepath)
        # Load Input
        try:
            points = load_and_preprocess(filepath, config.model.norm_attribute, n_points=4096)
            points = points.cuda()
            
            # Run Inference
            with torch.no_grad():
                logits = model(points)
            
            probs = torch.nn.functional.softmax(logits, dim=-1)
            top_p, top_class = probs.topk(1, dim=1)
            
            class_name = MODELNET40_CLASSES[top_class.item()]
            confidence = top_p.item() * 100
            
            results.append((filename, class_name, confidence))
            print(f"Processed {filename}: {class_name} ({confidence:.1f}%)")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            results.append((filename, "ERROR", 0.0))

    # Print Summary Table
    print("\n" + "="*65)
    print(f"{'FILENAME':<30} | {'PREDICTION':<15} | {'CONFIDENCE':<10}")
    print("-" * 65)
    for res in results:
        fname, pred, conf = res
        print(f"{fname:<30} | {pred.upper():<15} | {conf:.2f}%")
    print("="*65 + "\n")

if __name__ == "__main__":
    main()
