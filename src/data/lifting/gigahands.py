"""
GigaHands Dataset Loader

Paper: "GigaHands: A Massive Annotated Dataset of Bimanual Hand Activities"
"""

import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from .transforms import separate_hands_2d_3d

class GigaHandsDataset(Dataset):
    def __init__(self, root_dir, split='train', seq_len=150, stride=10, min_valid_frames=30, use_2d=True, augmentation=False):
        self.root_dir = Path(root_dir)
        self.split = split
        self.seq_len = seq_len
        self.stride = stride
        self.min_valid_frames = min_valid_frames
        self.use_2d = use_2d
        self.augmentation = None
        
        # Cache mechanism for faster initialization
        cache_filename = f"cached_{split}_len{seq_len}_str{stride}_min{min_valid_frames}_sequences.pt"
        cache_path = self.root_dir / cache_filename

        if cache_path.exists():
            print(f"Loading cached GigaHands {split} from {cache_path}...")
            self.all_sequences = torch.load(cache_path, weights_only=False)
        else:
            print(f"Parsing raw GigaHands {split} files (No cache found)...")
            self.all_sequences = []
            self._load_data(split)
            print(f"Saving GigaHands cache to {cache_path}...")
            torch.save(self.all_sequences, cache_path)
        
        print(f"[GigaHands {split}] Loaded {len(self.all_sequences)} sequences.")

        if split == 'train' and augmentation:
            from ..augmentation import HandPoseAugmentation
            self.augmentation = HandPoseAugmentation(
                rotation_range=15,
                scale_range=(1.0, 1.0),
                noise_std=0.01,
                temporal_jitter=True,
                shear_prob=0.0,
                translate_range=0.0,
                dropout_prob=0.05,
            )

    def _load_data(self, split):
        if not self.root_dir.exists():
            print(f"Warning: {self.root_dir} not found.")
            return

        participant_dirs = sorted([d for d in os.listdir(self.root_dir) if d.startswith('p') and (self.root_dir / d).is_dir()])
        
        for p_dir in tqdm(participant_dirs, desc=f"Parsing GigaHands {split}"):
            try:
                p_id = int(p_dir.split('-')[0][1:])
            except: continue

            is_train = p_id <= 35
            if split == 'train' and not is_train: continue
            if split == 'val' and is_train: continue
            
            mano_dir = self.root_dir / p_dir / 'keypoints_3d_mano'
            if not mano_dir.exists(): continue
                
            json_files = sorted(list(mano_dir.glob("*.json")))
            
            for json_path in json_files:
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    
                    num_frames = len(data)
                    # Sliding Window
                    for start_idx in range(0, num_frames - self.seq_len + 1, self.stride):
                        self.all_sequences.append({
                            'path': str(json_path),
                            'start_idx': start_idx,
                            'end_idx': start_idx + self.seq_len
                        })
                except Exception as e:
                    print(f"Skipping {json_path}: {e}")

    def simulate_projection(self, joints_3d):
        if isinstance(joints_3d, np.ndarray):
            joints_3d = torch.from_numpy(joints_3d)
        
        joints_3d = joints_3d.float()

        root = joints_3d[:, 0:1, :]
        joints_centered = joints_3d - root

        z_offset = 600.0

        focal = 1000.0
        x, y, z = joints_centered[..., 0], joints_centered[..., 1], joints_centered[..., 2] + z_offset
        z = torch.clamp(z, min=1e-3)

        u = (x / z) * focal
        v = (y / z) * focal

        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()
        scale = max(u_max - u_min, v_max - v_min) / 2.0 + 1e-6

        u_norm = (u - (u_min + u_max) / 2) / scale
        v_norm = (v - (v_min + v_max) / 2) / scale

        norm_2d = torch.stack([torch.clamp(u_norm, -1.1, 1.1), torch.clamp(v_norm, -1.1, 1.1)], dim=-1)

        return norm_2d.float().numpy(), joints_centered.float().numpy()

    def __len__(self):
        return len(self.all_sequences)

    def __getitem__(self, idx):
        info = self.all_sequences[idx]
        
        with open(info['path'], 'r') as f:
            full_seq = json.load(f)
            
        clip = full_seq[info['start_idx'] : info['end_idx']]
        clip = np.array(clip, dtype=np.float32)
        
        try:
            # GigaHands: 21 Left + 21 Right (Meters -> mm)
            seq_both_hands = clip.reshape(self.seq_len, 42, 3) * 1000.0

            left_hand_3d = seq_both_hands[:, :21, :]
            right_hand_3d = seq_both_hands[:, 21:, :] 

            left_hand_3d[:, :, 0] *= -1.0
            
        except ValueError:
            return None

        if self.augmentation is not None:
            left_hand_3d = self.augmentation(left_hand_3d)
            right_hand_3d = self.augmentation(right_hand_3d)

        left_2d, right_2d = None, None
        
        if self.use_2d:
            left_2d, left_hand_3d = self.simulate_projection(left_hand_3d)
            right_2d, right_hand_3d = self.simulate_projection(right_hand_3d)
        else:
            # Zero-center if no projection needed
            left_hand_3d -= left_hand_3d[:, 0:1, :]
            right_hand_3d -= right_hand_3d[:, 0:1, :]

        return separate_hands_2d_3d(
            input_2d={'left_hand': left_2d, 'right_hand': right_2d},
            target_3d={'left_hand': left_hand_3d, 'right_hand': right_hand_3d},
            seq_len=self.seq_len
        )