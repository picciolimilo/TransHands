"""
Jester Dataset Loader

Paper: "The Jester Dataset: A Large-Scale Video Dataset of Human Gestures"
"""

import json
import csv
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

class JesterDataset(Dataset):
    
    GESTURE_LABELS = [
        "Doing other things",           # 0
        "Drumming Fingers",             # 1
        "No gesture",                   # 2
        "Pulling Hand In",              # 3
        "Pulling Two Fingers In",       # 4
        "Pushing Hand Away",            # 5
        "Pushing Two Fingers Away",     # 6
        "Rolling Hand Backward",        # 7
        "Rolling Hand Forward",         # 8
        "Shaking Hand",                 # 9
        "Sliding Two Fingers Down",     # 10
        "Sliding Two Fingers Left",     # 11
        "Sliding Two Fingers Right",    # 12
        "Sliding Two Fingers Up",       # 13
        "Stop Sign",                    # 14
        "Swiping Down",                 # 15
        "Swiping Left",                 # 16
        "Swiping Right",                # 17
        "Swiping Up",                   # 18
        "Thumb Down",                   # 19
        "Thumb Up",                     # 20
        "Turning Hand Clockwise",       # 21
        "Turning Hand Counterclockwise",# 22
        "Zooming In With Full Hand",    # 23
        "Zooming In With Two Fingers",  # 24
        "Zooming Out With Full Hand",   # 25
        "Zooming Out With Two Fingers", # 26
    ]
    
    def __init__(
        self,
        root_dir,
        keypoints_dir=None,
        split='train',
        seq_len=150,
        stride=None,
        min_valid_frames=None,
        augmentation=False,
        num_classes=27,
        labels_file=None,
        annotation_file=None
    ):
        """
        Args:
            stride: sequence stride for the sliding window (default: seq_len, i.e. non-overlapping)
            labels_file: CSV mapping gesture names to IDs ("id;name" per line, or plain names)
            annotation_file: CSV of "video_id;label_name" pairs
        """
        self.root_dir = Path(root_dir)
        self.keypoints_dir = Path(keypoints_dir) if keypoints_dir else None
        self.split = split
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.min_valid_frames = min_valid_frames if min_valid_frames is not None else seq_len // 2
        self.num_classes = num_classes

        self.label_to_id = self._load_label_mapping(labels_file)
        self.annotations = self._load_annotations(annotation_file)
        
        # Cache mechanism
        cache_suffix = f"_kps" if keypoints_dir else "_raw"
        labels_tag = Path(labels_file).stem if labels_file else "nolabels"
        annotations_tag = Path(annotation_file).stem if annotation_file else "noann"
        cache_path = self.root_dir / (
            f"cached_{split}{cache_suffix}_seq{self.seq_len}_str{self.stride}_"
            f"min{self.min_valid_frames}_{labels_tag}_{annotations_tag}_sequences.pt"
        )
        
        if cache_path.exists():
            print(f"Loading cached Jester dataset from {cache_path}...")
            self.samples = torch.load(cache_path, weights_only=False)
        else:
            print(f"Building Jester {split} dataset...")
            self.samples = self._collect_samples()
            print(f"Saving cache to {cache_path}...")
            torch.save(self.samples, cache_path)
        
        print(f"[Jester {split}] Total samples: {len(self.samples)}")

        self.augmentation = None
        if split == 'train' and augmentation:
            from ..augmentation import HandPoseAugmentation
            self.augmentation = HandPoseAugmentation(
                rotation_range=0,
                scale_range=(0.95, 1.05),
                noise_std=0.005,
                translate_range=0.03,
                flip_prob=0.0,
                dropout_prob=0.0,
                temporal_jitter=True,
            )
    
    def _load_label_mapping(self, labels_file):
        """Load gesture name -> ID mapping."""
        if labels_file and Path(labels_file).exists():
            label_map = {}
            with open(labels_file) as f:
                lines = f.read().strip().split('\n')
                for idx, line in enumerate(lines):
                    # Semicolon-separated format first (id;name)
                    if ';' in line:
                        parts = line.split(';')
                        if len(parts) >= 2:
                            label_id, label_name = int(parts[0]), parts[1].strip()
                            label_map[label_name] = label_id
                    else:
                        # Plain text format
                        label_name = line.strip()
                        if label_name:
                            label_map[label_name] = idx
            
            if label_map:
                print(f"Loaded {len(label_map)} gesture labels")
                return label_map
        
        # Default mapping
        print("Using default gesture label mapping")
        return {name: idx for idx, name in enumerate(self.GESTURE_LABELS)}
    
    def _load_annotations(self, annotation_file):
        """Load video_id -> label mapping."""
        annotations = {}
        
        if annotation_file and Path(annotation_file).exists():
            print(f"Loading annotations from: {annotation_file}")
            with open(annotation_file) as f:
                reader = csv.reader(f, delimiter=';')
                for row in reader:
                    if len(row) >= 2:
                        video_id, label_name = row[0], row[1]
                        label_id = self.label_to_id.get(label_name, 0)
                        annotations[video_id] = label_id
            print(f"Loaded {len(annotations)} annotations")
        else:
            print(f"Warning: No annotation file found at {annotation_file}. Labels will be set to 0.")
        
        return annotations
    
    def _collect_samples(self):
        """Collect all valid video samples."""
        samples = []
        skipped = 0
        loaded = 0
        
        if self.keypoints_dir and self.keypoints_dir.exists():
            # Load from pre-extracted keypoints
            pkl_files = sorted(self.keypoints_dir.glob('*.pkl'))
            
            for pkl_path in tqdm(pkl_files, desc=f"Loading {self.split} keypoints"):
                video_id = pkl_path.stem

                if video_id not in self.annotations:
                    skipped += 1
                    continue
                
                loaded += 1
                label = self.annotations[video_id]
                
                try:
                    with open(pkl_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Extract keypoints sequence
                    if isinstance(data, dict):
                        keypoints = data.get('keypoints', None)
                        if keypoints is None:
                            continue
                    elif isinstance(data, np.ndarray):
                        keypoints = data
                    else:
                        continue
                    
                    # keypoints shape: (T, 21, 2) or (T, 21, 3)
                    if len(keypoints) < self.min_valid_frames:
                        continue
                    
                    # Create sequences with sliding window
                    for start in range(0, len(keypoints), self.stride):
                        end = start + self.seq_len
                        if end > len(keypoints):
                            if len(keypoints) >= self.min_valid_frames:
                                start = max(0, len(keypoints) - self.seq_len)
                                end = len(keypoints)
                            else:
                                break
                        
                        seq = keypoints[start:end]
                        samples.append({
                            'video_id': video_id,
                            'keypoints': seq,
                            'label': label
                        })
                except Exception as e:
                    print(f"Error loading {pkl_path}: {e}")
                    continue
        else:
            # List videos (keypoints must be extracted separately)
            videos_dir = self.root_dir / 'extracted_frames'
            if not videos_dir.exists():
                videos_dir = self.root_dir / '20bnjester-v1-00'
            
            video_dirs = sorted([d for d in videos_dir.iterdir() if d.is_dir()])
            
            for video_dir in tqdm(video_dirs[:500], desc=f"Listing {self.split} videos"):  # Limit for speed
                video_id = video_dir.name
                label = self.annotations.get(video_id, 0)
                
                # Count frames
                frames = sorted(video_dir.glob('*.jpg'))
                if len(frames) < self.min_valid_frames:
                    continue
                
                samples.append({
                    'video_id': video_id,
                    'video_path': str(video_dir),
                    'num_frames': len(frames),
                    'label': label
                })
        
        if self.keypoints_dir and self.keypoints_dir.exists():
            print(f"Loaded {loaded} videos, skipped {skipped} (not in annotations)")
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        if 'keypoints' in sample:
            # Load from pre-extracted keypoints
            keypoints = sample['keypoints']
            
            if isinstance(keypoints, np.ndarray):
                keypoints = torch.from_numpy(keypoints).float()
            
            # Ensure shape (T, 21, C)
            T, J, C = keypoints.shape
            
            # Pad if sequence too short
            if T < self.seq_len:
                pad_len = self.seq_len - T
                keypoints = torch.cat([
                    keypoints,
                    keypoints[-1:].repeat(pad_len, 1, 1)
                ], dim=0)
            
            # Augmentation
            if self.augmentation and C >= 2:
                keypoints_2d = keypoints[:, :, :2].numpy()
                keypoints_2d = self.augmentation(keypoints_2d)
                keypoints[:, :, :2] = torch.from_numpy(keypoints_2d)
            
            if keypoints.abs().max() > 2.0:
                keypoints[:, :, :2] = self._normalize_keypoints(keypoints[:, :, :2])
            
            label = sample['label']
            
            return {
                'input': keypoints,  # (T, 21, C)
                'label': torch.tensor(label, dtype=torch.long),
                'video_id': sample['video_id']
            }
        else:
            # List videos (keypoints must be extracted separately)
            raise NotImplementedError(
                "Direct frame loading not implemented. "
                "Please extract keypoints using RTMPose first."
            )
    
    def _normalize_keypoints(self, keypoints):
        """
        Normalize keypoints to [-1, 1] range per sequence.
        
        Args:
            keypoints: (T, 21, 2) tensor
        Returns:
            normalized: (T, 21, 2) tensor in [-1, 1]
        """
        # Per-sequence normalization
        x_min, x_max = keypoints[:, :, 0].min(), keypoints[:, :, 0].max()
        y_min, y_max = keypoints[:, :, 1].min(), keypoints[:, :, 1].max()
        
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        
        scale = max(x_max - x_min, y_max - y_min) / 2.0 + 1e-6
        
        keypoints[:, :, 0] = (keypoints[:, :, 0] - center_x) / scale
        keypoints[:, :, 1] = (keypoints[:, :, 1] - center_y) / scale
        
        return torch.clamp(keypoints, -1.1, 1.1)


def collate_fn_gesture(batch):
    """Stack a list of already-padded per-sample dicts into batched tensors."""
    inputs = torch.stack([item['input'] for item in batch])  # (B, T, 21, C)
    labels = torch.stack([item['label'] for item in batch])  # (B,)
    video_ids = [item['video_id'] for item in batch]
    
    return {
        'input': inputs,
        'label': labels,
        'video_id': video_ids
    }