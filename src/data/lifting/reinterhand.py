"""
Re:InterHand Dataset Loader

Paper: "A Dataset of Relighted 3D Interacting Hands"
"""

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from .transforms import separate_hands_2d_3d

class ReInterHandDataset(Dataset):
    
    def __init__(self, root_dir, split='train', seq_len=150, stride=None, min_valid_frames=None, augmentation=False, use_2d=True):
        self.root_dir = Path(root_dir)
        self.split = split
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.min_valid_frames = min_valid_frames if min_valid_frames is not None else seq_len // 2
        self.use_2d = use_2d

        # Cache mechanism for faster initialization
        cache_path = self.root_dir / f"cached_{split}_seq{self.seq_len}_str{self.stride}_min{self.min_valid_frames}_sequences.pt"

        if cache_path.exists():
            print(f"Loading cached dataset structure from {cache_path}...")
            self.sequences = torch.load(cache_path, weights_only=False)
        else:
            print("Parsing raw files (3D only)...")
            self.sequences = self._collect_sequences()
            print(f"Saving cache to {cache_path}...")
            torch.save(self.sequences, cache_path)

        print(f"[ReInterHand {split}] Total sequences: {len(self.sequences)}")
        
        self.augmentation = None
        if split == 'train' and augmentation:
            from ..augmentation import HandPoseAugmentation
            self.augmentation = HandPoseAugmentation(
                rotation_range=0,
                scale_range=(1.0, 1.0),
                noise_std=0.01,
                temporal_jitter=True,
                translate_range=0.0,
                dropout_prob=0.05,
            )

    def __len__(self):
        return len(self.sequences)
        
    def simulate_projection(self, joints_3d):
        """
        3D -> 2D projection + bbox normalization.
        RETURNS: (2D_Normalized, 3D_Rotated_Centered)
        """
        if isinstance(joints_3d, np.ndarray):
            joints_3d = torch.from_numpy(joints_3d)
        
        joints_3d = joints_3d.float()

        root = joints_3d[:, 0:1, :]
        joints_centered = joints_3d - root
        
        # Random rotation
        z_offset = 600.0

        if self.split == 'train' and self.augmentation is not None:
            az = np.deg2rad(np.random.uniform(-180, 180))
            el = np.deg2rad(np.random.uniform(-30, 30))
            
            Ry = torch.tensor([
                [np.cos(az), 0, np.sin(az)],
                [0, 1, 0],
                [-np.sin(az), 0, np.cos(az)]
            ], dtype=torch.float32)
            
            Rx = torch.tensor([
                [1, 0, 0],
                [0, np.cos(el), -np.sin(el)],
                [0, np.sin(el), np.cos(el)]
            ], dtype=torch.float32)
            
            R = Ry @ Rx

            joints_centered = joints_centered @ R.T

            z_offset = np.random.uniform(400.0, 800.0)
        
        # Pinhole projection
        focal = 1000.0
        x = joints_centered[..., 0]
        y = joints_centered[..., 1]
        z = joints_centered[..., 2] + z_offset
        z = torch.clamp(z, min=1e-3)
        
        u = (x / z) * focal
        v = (y / z) * focal
        
        # Dynamic bbox normalization
        u_min, u_max = u.min(), u.max()
        v_min, v_max = v.min(), v.max()
        
        center_x = (u_min + u_max) / 2
        center_y = (v_min + v_max) / 2
        
        width = u_max - u_min
        height = v_max - v_min
        scale = max(width, height) / 2.0 + 1e-6
        
        u_norm = (u - center_x) / scale
        v_norm = (v - center_y) / scale
        
        norm_2d = torch.stack([torch.clamp(u_norm, -1.1, 1.1), torch.clamp(v_norm, -1.1, 1.1)], dim=-1)

        return norm_2d, joints_centered
    
    def __getitem__(self, idx):
        seq_data = self.sequences[idx]
        frames = seq_data['frames']

        left_seq_3d = [f.get('left_hand_3d') for f in frames]
        right_seq_3d = [f.get('right_hand_3d') for f in frames]
        
        def build_numpy(hand_list):
            valid = [h for h in hand_list if h is not None]
            if not valid:  return None
            seq, last = [], valid[0]
            for h in hand_list:
                if h is not None:  last = h
                seq.append(last)
            return np.array(seq)
        
        left_3d = build_numpy(left_seq_3d)
        right_3d = build_numpy(right_seq_3d)
        
        #  Augmentation on 3D (Scaling/Noise)
        if self.augmentation:
            if left_3d is not None: left_3d = self.augmentation(left_3d)
            if right_3d is not None: right_3d = self.augmentation(right_3d)
        
        # Project to 2D & Get Rotated 3D
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
            if left_3d is not None: left_3d -= left_3d[:, 0:1, :]
            if right_3d is not None: right_3d -= right_3d[:, 0:1, :]

        result = separate_hands_2d_3d(
            input_2d={'left_hand': left_2d, 'right_hand': right_2d},
            target_3d={'left_hand': left_3d, 'right_hand': right_3d},
            seq_len=self.seq_len
        )
        
        return result

    def _collect_sequences(self):
        """Only 3d data (orig_fits)."""
        sequences = []
        captures = sorted([d for d in self.root_dir.iterdir() if d.is_dir() and d.name.startswith('m--')])
        
        train_captures = [
            'm--20210701--1058--0000000--pilot--relightablehandsy--participant0--two-hands',
            'm--20220628--1327--BKS383--pilot--ProjectGoliath--ContinuousHandsy--two-hands',
            'm--20221007--1215--HIR112--pilot--ProjectGoliathScript--Hands--two-hands',
            'm--20221110--1033--TQH976--pilot--ProjectGoliathScript--Hands--two-hands',
            'm--20221111--0944--JFQ550--pilot--ProjectGoliathScript--Hands--two-hands',
            'm--20230313--1433--TXB805--pilot--ProjectGoliath--Hands--two-hands',
            'm--20230317--1433--TRO760--pilot--ProjectGoliath--Hands--two-hands'
        ]
        test_captures = ['m--20221215--0949--RNS217--pilot--ProjectGoliathScript--Hands--two-hands', 'm--20221216--0953--NKC880--pilot--ProjectGoliathScript--Hands--two-hands', 'm--20230317--1130--QZX685--pilot--ProjectGoliath--Hands--two-hands']
        
        if self.split == 'train': captures = [c for c in captures if c.name in train_captures]
        elif self.split in ['val', 'test']: captures = [c for c in captures if c.name in test_captures]

        for capture_dir in tqdm(captures, desc=f"Processing {self.split}"):
            frame_list_path = capture_dir / 'frame_list.txt'
            if not frame_list_path.exists(): continue
            
            frame_indices = []
            with open(frame_list_path) as f:
                for line in f:
                    p = line.strip().split()
                    if p: 
                        try: frame_indices.append(int(p[-1] if len(p)>=2 else p[0]))
                        except: continue
            
            orig_fits_right = capture_dir / 'orig_fits' / 'right' / 'Keypoints'
            orig_fits_left = capture_dir / 'orig_fits' / 'left' / 'Keypoints'
            
            json_map = {}
            for side, path in [('right', orig_fits_right), ('left', orig_fits_left)]:
                if path.exists():
                    for jf in sorted(path.glob('keypoint-*.json')):
                        try: json_map.setdefault(int(jf.stem.replace('keypoint-','')), {})[side] = jf
                        except: continue

            frames = []
            for idx in frame_indices:
                if idx not in json_map: continue
                rh, lh = None, None
                
                if 'right' in json_map[idx]: 
                    with open(json_map[idx]['right']) as f: rh = self._load_keypoints(json.load(f))
                if 'left' in json_map[idx]:
                    with open(json_map[idx]['left']) as f: lh = self._load_keypoints(json.load(f))
                
                if rh is None and lh is None: continue
                
                frames.append({
                    'frame_idx': idx, 
                    'left_hand_3d': lh, 
                    'right_hand_3d': rh
                })
            
            if not frames: continue
            frames.sort(key=lambda x: x['frame_idx'])
            
            for start in range(0, len(frames), self.stride):
                end = start + self.seq_len
                if end > len(frames): 
                    if len(frames) >= self.min_valid_frames: start = max(0, len(frames)-self.seq_len); end = len(frames)
                    else: break
                seq = frames[start:end]
                if len(seq) >= self.min_valid_frames:
                    sequences.append({'frames': seq})
        
        return sequences

    def _load_keypoints(self, data):
        if isinstance(data, list): j = np.array(data, dtype=np.float32)
        elif isinstance(data, dict): j = np.array(data['joints'], dtype=np.float32)
        else: return None
        if j.shape[0] < 21: return None
        if np.max(np.abs(j)) < 10.0: j *= 1000.0 
        if j.shape[0] >= 24: j = j[[0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22]]
        return j