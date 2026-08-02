"""
AssemblyHands Dataset Loader

Paper: "AssemblyHands:  Towards Egocentric Activity Understanding via 3D Hand Pose Estimation"
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from .transforms import separate_hands_2d_3d

class AssemblyHandsDataset(Dataset):
    def __init__(self, root_dir, split='train', seq_len=150, stride=None, min_valid_frames=None, augmentation=False, use_2d=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.min_valid_frames = min_valid_frames if min_valid_frames is not None else seq_len // 2
        self.use_2d = use_2d

        # MAPPING: Assembly (Tip->Base) -> MANO (Base->Tip)
        self.ASSEMBLY_TO_MANO = [
            20,                  # Wrist (becomes 0)
            3, 2, 1, 0,          # Thumb
            7, 6, 5, 4,          # Index
            11, 10, 9, 8,        # Middle
            15, 14, 13, 12,      # Ring
            19, 18, 17, 16       # Pinky
        ]

        # Cache mechanism for faster initialization
        cache_path = self.root_dir / f"cached_{split}_seq{self.seq_len}_str{self.stride}_min{self.min_valid_frames}_sequences.pt"

        if cache_path.exists():
            print(f"Loading cached dataset from {cache_path}...")
            self.sequences = torch.load(cache_path, weights_only=False)
        else:
            print(f"Parsing raw AssemblyHands {split} files...")
            self.sequences = self._collect_sequences()
            print(f"Saving cache to {cache_path}...")
            torch.save(self.sequences, cache_path)

        print(f"[AssemblyHands {split}] Total sequences: {len(self.sequences)}")
        
        self.augmentation = None
        if split == 'train' and augmentation:
            from ..augmentation import HandPoseAugmentation
            self.augmentation = HandPoseAugmentation(
                rotation_range=0,
                scale_range=(1.0, 1.0),
                noise_std=0.01,
                temporal_jitter=True,
                shear_prob=0.0,
                flip_prob=0.5,
                translate_range=0.0,
                dropout_prob=0.05,
            )

    def __len__(self):
        return len(self.sequences)

    def simulate_projection(self, joints_3d):
        """
        Simulates 2D projection from 3D joints.

        Returns: (joints_2d_norm, joints_3d_rotated)
        """
        if isinstance(joints_3d, np.ndarray):
            joints_3d = torch.from_numpy(joints_3d)
        
        joints_3d = joints_3d.float()
        
        root = joints_3d[:, 0:1, :] 
        joints_centered = joints_3d - root
        
        # Apply Rotation Augmentation
        if self.split == 'train' and self.augmentation is not None:
            az = np.deg2rad(np.random.uniform(-180, 180))
            el = np.deg2rad(np.random.uniform(-30, 30))
            
            Ry = torch.tensor([[np.cos(az), 0, np.sin(az)], [0, 1, 0], [-np.sin(az), 0, np.cos(az)]], dtype=torch.float32)
            Rx = torch.tensor([[1, 0, 0], [0, np.cos(el), -np.sin(el)], [0, np.sin(el), np.cos(el)]], dtype=torch.float32)
            
            R = Ry @ Rx
            joints_centered = joints_centered @ R.T
            
        z_offset = 600.0 # Canonical depth for visibility
        if self.split == 'train' and self.augmentation is not None:
             z_offset = np.random.uniform(400.0, 800.0)

        focal = 1000.0
        x, y, z = joints_centered[..., 0], joints_centered[..., 1], joints_centered[..., 2] + z_offset
        z = torch.clamp(z, min=1e-3)
        u, v = (x / z) * focal, (y / z) * focal
        
        # Normalize to [-1.1, 1.1] range based on bbox
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()
        scale = max(u_max - u_min, v_max - v_min) / 2.0 + 1e-6
        
        u_norm = (u - (u_min + u_max) / 2) / scale
        v_norm = (v - (v_min + v_max) / 2) / scale
        
        norm_2d = torch.stack([torch.clamp(u_norm, -1.1, 1.1), torch.clamp(v_norm, -1.1, 1.1)], dim=-1)
        
        return norm_2d, joints_centered

    def __getitem__(self, idx):
        frames = self.sequences[idx]['frames']
        
        def build_numpy(hand_key):
            valid = [f[hand_key] for f in frames if f.get(hand_key) is not None]
            if not valid: return None
            seq, last = [], valid[0]
            for f in frames:
                if f.get(hand_key) is not None: last = f[hand_key]
                seq.append(last)
            return np.array(seq, dtype=np.float32)
        
        left_3d = build_numpy('left_hand')
        right_3d = build_numpy('right_hand')
        
        if self.augmentation:
            if left_3d is not None: left_3d = self.augmentation(left_3d)
            if right_3d is not None: right_3d = self.augmentation(right_3d)
        
        left_2d, right_2d = None, None
        
        if self.use_2d: 
            if left_3d is not None: 
                left_2d, left_3d = self.simulate_projection(left_3d)
                left_2d = left_2d.numpy()
                left_3d = left_3d.numpy()
            
            if right_3d is not None: 
                right_2d, right_3d = self.simulate_projection(right_3d)
                right_2d = right_2d.numpy()
                right_3d = right_3d.numpy()
        else:
            # Zero-center if 2D is not used
            if left_3d is not None: left_3d -= left_3d[:, 0:1, :]
            if right_3d is not None: right_3d -= right_3d[:, 0:1, :]

        return separate_hands_2d_3d(
            input_2d={'left_hand': left_2d, 'right_hand': right_2d},
            target_3d={'left_hand': left_3d, 'right_hand': right_3d},
            seq_len=self.seq_len
        )

    def _collect_sequences(self):
        sequences = []
        split_dir = self.root_dir / self.split
        try:
            joint_3d_file = sorted(split_dir.glob("*_joint_3d*.json"))[0]
        except IndexError: return []

        print(f"Loading {joint_3d_file.name}...")
        with open(joint_3d_file) as f:
            data = json.load(f)
        if 'annotations' in data: data = data['annotations']

        seq_names = sorted(data.keys())
        for seq_name in tqdm(seq_names, desc=f"Parsing"):
            f_dict = data[seq_name]
            f_list = []
            
            for fid in sorted(f_dict.keys()):
                coord = np.array(f_dict[fid]['world_coord'], dtype=np.float32)
                valid = np.array(f_dict[fid]['joint_valid'], dtype=bool)
                
                rh = self._extract_and_map(coord[:21], valid[:21], is_left=False)
                lh = self._extract_and_map(coord[21:], valid[21:], is_left=True)
                
                if rh is not None or lh is not None:
                    f_list.append({'right_hand': rh, 'left_hand': lh})
            
            if len(f_list) < self.min_valid_frames: continue
            
            # Create overlapping sequences
            for start in range(0, len(f_list), self.stride):
                end = start + self.seq_len
                if end > len(f_list):
                    if len(f_list) >= self.min_valid_frames:
                        start = max(0, len(f_list) - self.seq_len); end = len(f_list)
                    else: break
                sequences.append({'frames': f_list[start:end]})
        return sequences

    def _extract_and_map(self, joints, valid, is_left=False):
        if np.sum(valid) < 10: return None
        
        mapped_joints = joints[self.ASSEMBLY_TO_MANO].copy()
        max_dist = np.linalg.norm(np.max(mapped_joints, axis=0) - np.min(mapped_joints, axis=0))
        if max_dist < 0.5: 
            mapped_joints *= 1000.0

        if is_left:
            mapped_joints[:, 0] *= -1.0
            
        return mapped_joints.astype(np.float32)