"""
EgoGesture Dataset Loader

Paper: "EgoGesture: A New Dataset and Benchmark for Egocentric Hand Gesture Recognition"
"""

import pickle
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# EgoGesture has 83 classes (from 1 to 83).
NUM_CLASSES = 83

def _parse_subject_int(subject_str):
    """Extraction of the subject integer from the subject string (e.g., '01' -> 1, '50' -> 50)"""
    m = re.match(r'^(\d+)$', str(subject_str).strip())
    if not m:
        m = re.match(r'^Subject(\d+)$', str(subject_str).strip(), re.IGNORECASE)
    return int(m.group(1)) if m else None

def _get_egogesture_split(subject_idx):
    """Official split for EgoGesture"""
    if subject_idx is None:
        return 'train'
    
    train_subjects = [3, 4, 5, 6, 8, 10, 15, 16, 17, 20, 21, 22, 23, 25, 26, 27, 30, 32, 36, 38, 39, 40, 42, 43, 44, 45, 46, 48, 49, 50]
    val_subjects = [1, 7, 12, 13, 24, 29, 33, 34, 35, 37]
    test_subjects = [2, 9, 11, 14, 18, 19, 28, 31, 41, 47]
    
    if subject_idx in val_subjects:
        return 'val'
    elif subject_idx in test_subjects:
        return 'test'
    else:
        return 'train'


class EgoGestureDataset(Dataset):
    def __init__(
        self,
        keypoints_dir,
        split='train',
        seq_len=64,             
        min_valid_frames=10,    
        min_valid_ratio=0.30,  
        augmentation=True, 
    ):
        self.keypoints_dir = Path(keypoints_dir)
        self.split = split
        self.seq_len = seq_len
        self.min_valid_frames = min_valid_frames
        self.min_valid_ratio = min_valid_ratio

        if self.split == 'train' and augmentation:
            try:
                from ..augmentation import HandPoseAugmentation
                self.augmenter = HandPoseAugmentation(
                    rotation_range=10,
                    scale_range=(0.88, 1.12),
                    noise_std=0.012,
                    translate_range=0.06,
                    flip_prob=0.0,
                    dropout_prob=0.0,
                    temporal_jitter=True,
                )
            except ImportError:
                print("WARNING: Augmentation module not found. Augmentation disabled.")
                self.augmenter = None
        else:
            self.augmenter = None

        ratio_tag = str(self.min_valid_ratio).replace('.', 'p')
        cache_filename = f"cached_EgoGesture_{split}_seq{self.seq_len}_min{self.min_valid_frames}_ratio{ratio_tag}_samples.pt"
        cache_path = self.keypoints_dir / cache_filename

        if cache_path.exists():
            print(f"Loading EgoGesture {split} cache from {cache_path}...")
            self.samples = torch.load(cache_path, weights_only=False)
        else:
            print(f"Parsing raw PKL+CSV files for EgoGesture {split} (No cache found)...")
            self.samples = self._collect_samples()
            print(f"Saving cache to {cache_path}...")
            torch.save(self.samples, cache_path)
            
        print(f"[EgoGesture {split}] Loaded {len(self.samples)} sequences (gestures) valid.")

    @property
    def num_classes(self):
        return NUM_CLASSES

    def _collect_samples(self):
        samples = []
        pkl_files = sorted(self.keypoints_dir.glob('*.pkl'))

        for pkl_path in pkl_files:
            try:
                with open(pkl_path, 'rb') as f:
                    data = pickle.load(f)
            except Exception:
                continue

            subject = data.get('subject', '00')
            subject_idx = _parse_subject_int(subject)
            if _get_egogesture_split(subject_idx) != self.split:
                continue

            keypoints = np.asarray(data.get('keypoints', []), dtype=np.float32)
            valid_mask = np.asarray(data.get('valid_mask', []), dtype=bool)
            csv_path = data.get('csv_path')
            
            if keypoints.ndim != 3 or keypoints.shape[1:] != (21, 2):
                continue
            
            total_frames = keypoints.shape[0]

            if not csv_path or not Path(csv_path).exists():
                continue
                
            with open(csv_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) != 3:
                        continue
                        
                    try:
                        class_id = int(parts[0]) - 1
                        start_f = int(parts[1])
                        end_f = int(parts[2])
                    except ValueError:
                        continue
                    
                    start_f = max(0, start_f)
                    end_f = min(total_frames, end_f)
                    
                    if end_f <= start_f:
                        continue

                    kps_slice = keypoints[start_f:end_f]
                    valid_slice = valid_mask[start_f:end_f]
                    
                    t_slice = kps_slice.shape[0]
                    if t_slice < self.min_valid_frames:
                        continue
                        
                    valid_ratio = float(valid_slice.mean())
                    if valid_ratio < self.min_valid_ratio:
                        continue
                        
                    samples.append({
                        'keypoints': kps_slice,
                        'valid_mask': valid_slice,
                        'label': class_id,
                        'subject': subject,
                        'scene': data.get('scene', ''),
                        'video_num': data.get('video_num', ''),
                        'original_video_id': data.get('video_id', pkl_path.stem),
                        'frame_range': (start_f, end_f)
                    })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        keypoints_np = sample['keypoints'].copy()

        # Data Augmentation
        if self.augmenter is not None:
            keypoints_np = self.augmenter(keypoints_np)
            
        keypoints = torch.from_numpy(keypoints_np).float()
        valid_mask = torch.from_numpy(sample['valid_mask']).float()

        t = keypoints.shape[0]
        if t < self.seq_len:
            pad = self.seq_len - t
            keypoints = torch.cat([keypoints, keypoints[-1:].repeat(pad, 1, 1)], dim=0)
            valid_mask = torch.cat([valid_mask, valid_mask[-1:].repeat(pad)], dim=0)
        elif t > self.seq_len:
            indices = torch.linspace(0, t - 1, self.seq_len).long()
            keypoints = keypoints[indices]
            valid_mask = valid_mask[indices]

        keypoints = self._normalize_keypoints(keypoints)

        return {
            'input': keypoints,
            'valid_mask': valid_mask,
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'video_id': sample['original_video_id'],
            'subject': sample['subject'],
            'frame_range': sample['frame_range']
        }

    def _normalize_keypoints(self, keypoints):
        """Centering and scaling of coordinates"""
        x_min, x_max = keypoints[:, :, 0].min(), keypoints[:, :, 0].max()
        y_min, y_max = keypoints[:, :, 1].min(), keypoints[:, :, 1].max()
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        scale = max(x_max - x_min, y_max - y_min) / 2.0 + 1e-6
        keypoints[:, :, 0] = (keypoints[:, :, 0] - center_x) / scale
        keypoints[:, :, 1] = (keypoints[:, :, 1] - center_y) / scale
        return torch.clamp(keypoints, -1.1, 1.1)


def collate_fn_egogesture(batch):
    """Collate function for Pytorch DataLoader"""
    return {
        'input': torch.stack([x['input'] for x in batch]),
        'valid_mask': torch.stack([x['valid_mask'] for x in batch]),
        'label': torch.stack([x['label'] for x in batch]),
        'video_id': [x['video_id'] for x in batch],
        'subject': [x['subject'] for x in batch],
    }