"""Data preprocessing transforms for 3D hand pose sequences."""

import numpy as np
import torch


def custom_collate_fn(batch):
    batch = [s for s in batch if s is not None]

    if not batch:
        return None

    first_sample = batch[0]
    
    if "poses" in first_sample:
        valid_samples = [s for s in batch if s.get("poses") is not None]
        if not valid_samples:
            return None
        
        return {
            "left_hand":  torch.stack([s["poses"] for s in valid_samples]),
            "left_scales": [float(s.get("scale", 120.0)) for s in valid_samples],
            "gesture_names": [s.get("gesture_name", "") for s in valid_samples]
        }
    
    # Lifting 2D -> 3D task
    elif "left_hand_input" in first_sample or "right_hand_input" in first_sample:
        left_inputs = []
        left_targets = []
        left_scales = []
        right_inputs = []
        right_targets = []
        right_scales = []
        origins = []

        for s in batch:
            if s.get("left_hand_input") is not None:
                left_inputs.append(s["left_hand_input"])
                left_targets.append(s["left_hand_target"])
                left_scales.append(s.get("left_scale_3d", 1.0))

            if s.get("right_hand_input") is not None:
                right_inputs.append(s["right_hand_input"])
                right_targets.append(s["right_hand_target"])
                right_scales.append(s.get("right_scale_3d", 1.0))

            if "dataset_origin" in s:
                origins.append(s["dataset_origin"])

        result = {}
        if left_inputs:
            result["left_hand_input"] = torch.stack(left_inputs)    # Input 2D
            result["left_hand_target"] = torch.stack(left_targets)  # GT 3D
            result["left_scales"] = left_scales

        if right_inputs:
            result["right_hand_input"] = torch.stack(right_inputs)
            result["right_hand_target"] = torch.stack(right_targets)
            result["right_scales"] = right_scales

        if origins:
            result["dataset_origins"] = origins

        return result if result else None

def center_at_wrist(pose):
    """
    Center pose at wrist (joint 0).
    Args: (T, J, 3) numpy array
    """
    if pose is None or len(pose) == 0:
        return pose
    
    wrist = pose[:, 0:1, :]  # (T, 1, 3)
    return pose - wrist

def separate_hands_2d_3d(input_2d, target_3d, seq_len=150):
    """
    2D -> 3D Lifting Task.
    
    Args:
        input_2d:  dict with 'left_hand'/'right_hand' (T, 21, 2)
        target_3d: dict with 'left_hand'/'right_hand' (T, 21, 3) in mm
        seq_len: target sequence length
    Returns:
        dict with normalized torch tensors ready for the model.
    """
    result = {}
    
    if input_2d.get('left_hand') is not None and target_3d.get('left_hand') is not None:
        input_2d_lh = input_2d['left_hand']  # (T, 21, 2)
        target_3d_lh = target_3d['left_hand']  # (T, 21, 3) mm

        target_3d_lh = center_at_wrist(target_3d_lh)
        
        # Compute robust scale (wrist is at origin after centering)
        middle_mcp = target_3d_lh[:, 9, :]

        bone_lengths = np.linalg.norm(middle_mcp, axis=-1)
        bone_length = np.median(bone_lengths)
        
        # Filter outlier
        if bone_length < 20.0 or bone_length > 300.0:
            # print(f"--> bone_length={bone_length:.1f}mm")
            q25, q75 = np.percentile(bone_lengths, [25, 75])
            valid_lengths = bone_lengths[(bone_lengths >= q25) & (bone_lengths <= q75)]
            if len(valid_lengths) > 0:
                bone_length = valid_lengths.mean()
            else:
                bone_length = 65.0

        target_3d_lh_norm = target_3d_lh / bone_length
        input_2d_lh_norm = input_2d_lh
        
        # Pad/Truncate
        T = len(input_2d_lh_norm)
        if T < seq_len: 
            pad_len = seq_len - T
            input_2d_lh_norm = np.pad(input_2d_lh_norm, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
            target_3d_lh_norm = np.pad(target_3d_lh_norm, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
        elif T > seq_len:
            input_2d_lh_norm = input_2d_lh_norm[:seq_len]
            target_3d_lh_norm = target_3d_lh_norm[:seq_len]
        
        result['left_hand_input'] = torch.from_numpy(input_2d_lh_norm).float()
        result['left_hand_target'] = torch.from_numpy(target_3d_lh_norm).float()
        result['left_scale_3d'] = bone_length
    
    if input_2d.get('right_hand') is not None and target_3d.get('right_hand') is not None:
        input_2d_rh = input_2d['right_hand']
        target_3d_rh = target_3d['right_hand']
        
        target_3d_rh = center_at_wrist(target_3d_rh)

        middle_mcp = target_3d_rh[:, 9, :]
        
        bone_lengths = np.linalg.norm(middle_mcp, axis=-1)
        bone_length = np.median(bone_lengths)
        
        if bone_length < 20.0 or bone_length > 300.0:
            q25, q75 = np.percentile(bone_lengths, [25, 75])
            valid_lengths = bone_lengths[(bone_lengths >= q25) & (bone_lengths <= q75)]
            if len(valid_lengths) > 0:
                bone_length = valid_lengths.mean()
            else:
                bone_length = 65.0
        
        target_3d_rh_norm = target_3d_rh / bone_length
        input_2d_rh_norm = input_2d_rh
        
        T = len(input_2d_rh_norm)
        if T < seq_len:
            pad_len = seq_len - T
            input_2d_rh_norm = np.pad(input_2d_rh_norm, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
            target_3d_rh_norm = np.pad(target_3d_rh_norm, ((0, pad_len), (0, 0), (0, 0)), mode='edge')
        elif T > seq_len:
            input_2d_rh_norm = input_2d_rh_norm[:seq_len]
            target_3d_rh_norm = target_3d_rh_norm[:seq_len]
        
        result['right_hand_input'] = torch.from_numpy(input_2d_rh_norm).float()
        result['right_hand_target'] = torch.from_numpy(target_3d_rh_norm).float()
        result['right_scale_3d'] = bone_length
    
    return result