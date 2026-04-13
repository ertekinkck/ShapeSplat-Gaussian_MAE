import os
import torch
import torch.utils.data as data
import numpy as np
import glob

from .build import DATASETS
from utils.logger import *
from .io import IO
import math

def np_sigmoid(x):
    return 1 / (1 + np.exp(-x))

def read_gaussian_attribute(vertex, attribute):
    assert "xyz" in attribute, "At least need xyz attribute"
    if "xyz" in attribute:
        x = vertex["x"].astype(np.float32)
        y = vertex["y"].astype(np.float32)
        z = vertex["z"].astype(np.float32)
        data_arr = np.stack((x, y, z), axis=-1)  

    if "opacity" in attribute:
        opacity = vertex["opacity"].astype(np.float32).reshape(-1, 1)
        opacity = np_sigmoid(opacity)
        data_arr = np.concatenate((data_arr, opacity), axis=-1)

    if "scale" in attribute and "rotation" in attribute:
        scale_names = [p.name for p in vertex.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda p: int(p.split("_")[-1]))
        scales = np.zeros((data_arr.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = vertex[attr_name].astype(np.float32)

        scales = np.exp(scales)

        rot_names = [p.name for p in vertex.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda p: int(p.split("_")[-1]))
        rots = np.zeros((data_arr.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = vertex[attr_name].astype(np.float32)

        rots = rots / (np.linalg.norm(rots, axis=1, keepdims=True) + 1e-9)
        signs_vector = np.sign(rots[:, 0])
        rots = rots * signs_vector[:, None]

        data_arr = np.concatenate((data_arr, scales, rots), axis=-1)

    if "sh" in attribute:
        features_dc = np.zeros((data_arr.shape[0], 3, 1))
        features_dc[:, 0, 0] = vertex["f_dc_0"].astype(np.float32)
        features_dc[:, 1, 0] = vertex["f_dc_1"].astype(np.float32)
        features_dc[:, 2, 0] = vertex["f_dc_2"].astype(np.float32)
        feature_pc = features_dc.reshape(-1, 3)
        data_arr = np.concatenate((data_arr, feature_pc), axis=1)

    return data_arr


@DATASETS.register_module()
class LLFFGaussian(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.attribute = config.ATTRIBUTE
        self.subset = config.subset
        self.norm_attribute = config.norm_attribute
        self.sample_points_num = config.N_POINTS

        print_log(f"[DATASET] LLFF/MipNeRF Foundation Mode with Attributes {self.attribute}", logger="LLFFGaussian")
        
        # Orijinal ShapeNet gibi text okumak yerine direk klasör içi point_cloud okuma
        self.file_list = []
        if os.path.exists(self.data_root):
            ply_files = glob.glob(os.path.join(self.data_root, "**", "iteration_10000", "point_cloud.ply"), recursive=True)
            for ply in ply_files:
                self.file_list.append({"file_path": ply, "cls": 0})
                
        print_log(f"[DATASET] Loaded {len(self.file_list)} Unbounded Scenes for Pre-training!", logger="LLFFGaussian")

    def pc_norm_gs(self, pc, attribute=["xyz"]):
        # Ayrıntılı normalize (ModelNet ile aynı tutuyoruz model architecture bozmamak için)
        pc_xyz = pc[..., :3]
        centroid = np.mean(pc_xyz, axis=0)
        pc_xyz = pc_xyz - centroid
        m = np.max(np.sqrt(np.sum(pc_xyz**2, axis=1)))
        m = max(m, 1e-6)
        pc_xyz = pc_xyz / m
        pc[..., :3] = pc_xyz
        pc[..., 4:7] = pc[..., 4:7] / m

        if "opacity" in attribute:
            min_opacity, max_opacity = 0, 1
            pc[..., 3] = (pc[..., 3] - min_opacity) / (max_opacity - min_opacity) * 2 - 1

        if "scale" in attribute:
            s_center = np.mean(pc[..., 4:7], axis=0)
            pc[..., 4:7] = pc[..., 4:7] - s_center
            s_m = np.max(np.sqrt(np.sum(pc[..., 4:7] ** 2, axis=1)))
            s_m = max(s_m, 1e-6)
            pc[..., 4:7] = pc[..., 4:7] / s_m
        else:
            s_center = np.zeros(3)
            s_m = 1

        if "sh" in attribute:
            sh = pc[..., 11:14]
            sh = sh * 0.28209479177387814
            sh = np.clip(sh, -0.5, 0.5)
            sh = 2 * sh / math.sqrt(3)
            pc[..., 11:14] = sh

        return pc, s_center, s_m

    def __getitem__(self, idx):
        # Cycle over the real data scenes dynamically
        eff_idx = idx % len(self.file_list)
        sample = self.file_list[eff_idx]
        try:
            gs = IO.get(sample["file_path"])
            vertex = gs["vertex"]
            data_arr = read_gaussian_attribute(vertex, self.attribute)
            data_arr, scale_c, scale_m = self.pc_norm_gs(data_arr, self.norm_attribute)
            
            choice_gs = np.random.choice(len(data_arr), self.sample_points_num, replace=True)
            data_arr = data_arr[choice_gs, :]
            data_tensor = torch.from_numpy(data_arr).float()

            scale_c = torch.from_numpy(scale_c).float()
            scale_m = torch.tensor(scale_m).float()

            taxonomy_id = "LLFF"
            model_id = os.path.basename(sample["file_path"]).split('.')[0]

            return taxonomy_id, model_id, data_tensor, scale_c, scale_m
        except Exception as e:
            # Fallback for corrupted PLY
            data_tensor = torch.randn(self.sample_points_num, 14).float()
            scale_c = torch.zeros(3).float()
            scale_m = torch.tensor(1.0).float()
            return "LLFF", "corrupted", data_tensor, scale_c, scale_m

    def __len__(self):
        # Multiplier guarantees enough steps for the DataLoader epochs
        return max(len(self.file_list) * 200, 1000)
