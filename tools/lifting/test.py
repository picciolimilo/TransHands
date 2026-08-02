#!/usr/bin/env python3
"""Evaluation Script."""

import torch
import yaml
import argparse
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transhands import TransHands
from src.data.lifting.transforms import custom_collate_fn
from torch.utils.data import DataLoader

from src.data.lifting.reinterhand import ReInterHandDataset
from src.data.lifting.assemblyhands import AssemblyHandsDataset
from src.data.lifting.gigahands import GigaHandsDataset

# Default Paths for datasets
DEFAULT_PATHS = {
    'reinterhand': 'dataset/ReInterHand',
    'assemblyhands': 'dataset/AssemblyHands',
    'gigahands': 'dataset/GigaHands'
}

def add_2d_noise(keypoints_2d, noise_std_pixels=8.0, image_size=256):
    """Add Gaussian noise to 2D keypoints to simulate detector errors."""
    noise_std_norm = noise_std_pixels / (image_size / 2.0)
    noise = torch.randn_like(keypoints_2d) * noise_std_norm
    noisy_keypoints_2d = keypoints_2d + noise
    noisy_keypoints_2d = torch.clamp(noisy_keypoints_2d, -1.2, 1.2)
    return noisy_keypoints_2d

def load_config(config_path):
    with open(config_path) as f:
        try:
            return yaml.safe_load(f)
        except yaml.YAMLError:
            return yaml.full_load(f)

def compute_mpjpe_mm(pred, target, scales):
    """Compute MPJPE in mm (denormalized)."""
    if isinstance(scales, list):
        scales = [float(s) if s is not None else 120.0 for s in scales]
    scales_tensor = torch.tensor(scales, device=pred.device).view(-1, 1, 1, 1)
    pred_mm = pred * scales_tensor
    target_mm = target * scales_tensor
    return float(torch.mean(torch.norm(pred_mm - target_mm, dim=-1)).item())

def compute_pck(pred, target, scales, threshold=20.0):
    """Percentage of Correct Keypoints within threshold (mm)."""
    scales_tensor = torch.tensor(scales, device=pred.device).view(-1, 1, 1, 1)
    pred_mm = pred * scales_tensor
    target_mm = target * scales_tensor
    errors = torch.norm(pred_mm - target_mm, dim=-1)
    correct = (errors < threshold).float()
    return float(correct.mean().item() * 100)

def compute_pa_mpjpe(pred, target, scales):
    """ Procrustes-Aligned MPJPE (PA-MPJPE)."""
    if isinstance(scales, list):
        scales = [float(s) if s is not None else 120.0 for s in scales]
    
    scales_tensor = torch.tensor(scales, device=pred.device).view(-1, 1, 1, 1)
    p = (pred * scales_tensor).reshape(-1, 21, 3)
    t = (target * scales_tensor).reshape(-1, 21, 3)
    
    mu_p = p.mean(dim=1, keepdim=True)
    mu_t = t.mean(dim=1, keepdim=True)
    p_centered = p - mu_p
    t_centered = t - mu_t
    
    # H = P^T * T
    H = torch.matmul(p_centered.transpose(1, 2), t_centered)
    U, S, Vh = torch.linalg.svd(H)
    
    # R = V * U^T
    V = Vh.transpose(1, 2)
    
    R = torch.matmul(V, U.transpose(1, 2))
    
    det = torch.det(R)
    for b in range(p.shape[0]):
        if det[b] < 0:
            S_mat = torch.eye(3, device=pred.device)
            S_mat[2, 2] = -1
            R[b] = torch.matmul(V[b], torch.matmul(S_mat, U[b].transpose(0, 1)))

    p_aligned = torch.matmul(p_centered, R.transpose(1, 2))
    error = torch.norm(p_aligned - t_centered, dim=-1).mean()
    
    return float(error.item())

def get_val_dataset(dataset_name, config, args, split_name='val'):
    dataset_name = dataset_name.lower()
    seq_len = config['dataset']['seq_len']
    stride = seq_len
    min_valid = seq_len // 2

    def get_smart_root(target_name):
        if args.data_root: return args.data_root
        if 'datasets' in config['dataset']:
            for ds in config['dataset']['datasets']:
                if ds['name'].lower() == target_name: return ds['root']
        config_ds_name = config['dataset'].get('name', '').lower()
        if config_ds_name == target_name: return config['dataset'].get('root')
        if target_name in DEFAULT_PATHS: return DEFAULT_PATHS[target_name]
        return f'dataset/{target_name}'

    root = get_smart_root(dataset_name)
    print(f"Loading Dataset: {dataset_name} (Split: {split_name}) from: {root}")

    if dataset_name == 'reinterhand':
        return ReInterHandDataset(root_dir=root, split=split_name, seq_len=seq_len, stride=stride, min_valid_frames=min_valid, augmentation=False, use_2d=True)
    elif dataset_name == 'assemblyhands':
        return AssemblyHandsDataset(root_dir=root, split=split_name, seq_len=seq_len, stride=stride, min_valid_frames=min_valid, augmentation=False)
    elif dataset_name == 'gigahands':
        return GigaHandsDataset(root_dir=root, split=split_name, seq_len=seq_len, stride=stride, min_valid_frames=min_valid, use_2d=True, augmentation=False)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for evaluation.")

def evaluate(model, loader, device, add_noise=False, noise_std=8.0):
    model.eval()
    if len(loader) == 0: return 0.0, 0.0, 0.0

    total_mpjpe, total_pck, total_pa = 0.0, 0.0, 0.0
    num_batches = 0

    print(f"Starting evaluation{' with 2D noise' if add_noise else ''} on {len(loader.dataset)} sequences...")
    
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            
            batch_mpjpe, batch_pck, batch_pa = 0.0, 0.0, 0.0
            num_hands = 0
            
            for hand_side in ['left', 'right']:
                if f"{hand_side}_hand_input" in batch:
                    input_2d = batch[f"{hand_side}_hand_input"].to(device)
                    target_3d = batch[f"{hand_side}_hand_target"].to(device)
                    scales = batch[f"{hand_side}_scales"]

                    if add_noise:
                        input_2d = add_2d_noise(input_2d, noise_std_pixels=noise_std)

                    pred_3d = model(input_2d)

                    if pred_3d.shape[1] != target_3d.shape[1]:
                        if pred_3d.shape[1] == 1:
                            mid = target_3d.shape[1] // 2
                            target_3d = target_3d[:, mid:mid+1]
                    
                    batch_mpjpe += compute_mpjpe_mm(pred_3d, target_3d, scales)
                    batch_pck += compute_pck(pred_3d, target_3d, scales)
                    batch_pa += compute_pa_mpjpe(pred_3d, target_3d, scales)
                    num_hands += 1
            
            if num_hands > 0:
                total_mpjpe += batch_mpjpe / num_hands
                total_pck += batch_pck / num_hands
                total_pa += batch_pa / num_hands
                num_batches += 1
    
    if num_batches == 0: return 0.0, 0.0, 0.0

    return total_mpjpe / num_batches, total_pck / num_batches, total_pa / num_batches

def main():
    parser = argparse.ArgumentParser(description="TransHands Evaluation")
    parser.add_argument('--config', required=True, help='Path to model config file')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--dataset', default=None, help='Override test dataset (e.g. multi)')
    parser.add_argument('--split', default='val', help='Dataset split to evaluate')
    parser.add_argument('--data-root', default=None, help='Override dataset root path (optional)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--add-noise', action='store_true', help='Add Gaussian noise to 2D input (robustness test)')
    parser.add_argument('--noise-std', type=float, nargs='+', default=[8.0], help='Noise std deviation(s) in pixels (e.g., 0 5 8 10 15)')
    args = parser.parse_args()

    config = load_config(args.config)
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"=== Evaluating TransHands ===")
    print(f"Config: {args.config} | Checkpoint: {args.checkpoint} | Split: {args.split}")
    if args.add_noise:
        print(f"2D Noise Enabled: Testing noise levels: {args.noise_std} pixels")

    model = TransHands(
        num_hand_joints=config['model']['num_hand_joints'],
        num_body_joints=config['model']['num_body_joints'],
        freeze_encoder=config['model']['freeze_encoder'],
        weights_path=None,
        encoder_type=config['model'].get('encoder_type', 'motionbert'),
        seq_len=config['dataset']['seq_len'],
        adapter_type=config['model'].get('adapter_type', 'ode'),
        projection_type=config['model'].get('projection_type', 'retnet')
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)
    
    target_dataset = args.dataset if args.dataset else config['dataset']['name']
    
    if target_dataset == 'multi':
        datasets_to_test = ['reinterhand', 'assemblyhands', 'gigahands']
    else:
        datasets_to_test = [target_dataset]

    all_results = {}

    if args.add_noise and len(args.noise_std) > 1:
        print(f"\n" + "="*60)
        print("NOISE ROBUSTNESS EVALUATION")
        print("="*60)
        
        for ds_name in datasets_to_test:
            print(f"Model: {args.checkpoint}")
            print(f"Dataset: {ds_name} ({args.split})")
            
            # Load dataset once
            dataset = get_val_dataset(ds_name, config, args, split_name=args.split)
            print(f"Sequences: {len(dataset)}")
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=custom_collate_fn, num_workers=2, pin_memory=True)
            
            noise_results = {}

            for noise_std in args.noise_std:
                print(f"\n>>> Testing noise std = {noise_std:.1f} pixels")
                mpjpe, pck, pa = evaluate(model, loader, device, add_noise=True, noise_std=noise_std)
                noise_results[noise_std] = {"mpjpe": mpjpe, "pck": pck, "pa": pa}

            print(f"\n{'Noise (px)':<12} {'MPJPE (mm)':<12} {'PCK (%)':<12} {'PA-MPJPE (mm)':<15}")
            print("-"*60)
            for noise_std in sorted(noise_results.keys()):
                res = noise_results[noise_std]
                print(f"{noise_std:<12.1f} {res['mpjpe']:<12.2f} {res['pck']:<12.2f} {res['pa']:<15.2f}")
            print("="*60)
            
            all_results[ds_name] = noise_results

            plot_path = Path(config['paths']['plot'])
            plot_path.mkdir(parents=True, exist_ok=True)
            
            save_path = plot_path / f"noise_robustness_{ds_name}_{args.split}.txt"
            with open(save_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("NOISE ROBUSTNESS EVALUATION\n")
                f.write("="*60 + "\n")
                f.write(f"Model: {args.checkpoint}\n")
                f.write(f"Dataset: {ds_name} ({args.split})\n")
                f.write(f"Sequences: {len(dataset)}\n\n")
                f.write(f"{'Noise (px)':<12} {'MPJPE (mm)':<12} {'PCK (%)':<12} {'PA-MPJPE (mm)':<15}\n")
                f.write("-"*60 + "\n")
                for noise_std in sorted(noise_results.keys()):
                    res = noise_results[noise_std]
                    f.write(f"{noise_std:<12.1f} {res['mpjpe']:<12.2f} {res['pck']:<12.2f} {res['pa']:<15.2f}\n")
                f.write("="*60 + "\n")
            
            print(f"\nResults saved to: {save_path}")
    
    # Standard evaluation (single noise level or no noise)
    else:
        noise_std = args.noise_std[0] if args.add_noise else 0.0
        
        for ds_name in datasets_to_test:
            print(f"\n>>> Running Evaluation on: {ds_name.upper()}")
            try:
                dataset = get_val_dataset(ds_name, config, args, split_name=args.split)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, 
                                    collate_fn=custom_collate_fn, num_workers=2, pin_memory=True)
                
                mpjpe, pck, pa = evaluate(model, loader, device, 
                                         add_noise=args.add_noise, 
                                         noise_std=noise_std)
                
                all_results[ds_name] = {"mpjpe": mpjpe, "pck": pck, "pa": pa}
                print(f"--> {ds_name}: MPJPE={mpjpe:.2f}, PCK={pck:.2f}")

            except Exception as e:
                print(f"ERROR evaluating {ds_name}: {e}")
                all_results[ds_name] = {"mpjpe": 0.0, "pck": 0.0, "pa": 0.0}

    # Print summary only for standard evaluation (not multi-noise)
    if not (args.add_noise and len(args.noise_std) > 1):
        valid_results = [res for res in all_results.values() if isinstance(res, dict) and "mpjpe" in res and res["mpjpe"] > 0]
        
        if len(valid_results) > 0:
            final_mpjpe = np.mean([res["mpjpe"] for res in valid_results])
            final_pck = np.mean([res["pck"] for res in valid_results])
            final_pa = np.mean([res["pa"] for res in valid_results])
        else:
            final_mpjpe, final_pck, final_pa = 0.0, 0.0, 0.0

        if len(datasets_to_test) > 1:
            print("\n" + "="*40)
            print("MACRO-AVERAGE RESULTS")
            for ds in datasets_to_test:
                val = all_results[ds]['mpjpe']
                if val > 0:
                    print(f"{ds:15}: MPJPE {val:5.2f} mm")
                else:
                     print(f"{ds:15}: FAILED / EMPTY")
        
        print("\n" + "="*40)
        print(f"OVERALL RESULTS ({target_dataset.upper()})")
        print(f"MPJPE: {final_mpjpe:.2f} mm | PCK: {final_pck:.2f} % | PA-MPJPE: {final_pa:.2f} mm")
        print("="*40 + "\n")

        plot_path = Path(config['paths']['plot'])
        plot_path.mkdir(parents=True, exist_ok=True)
        
        results_dict = {
            "model": config['model'].get('encoder_type', 'unknown'),
            "dataset": target_dataset,
            "split": args.split,
            "metrics": {
                "mpjpe": float(final_mpjpe), 
                "pck": float(final_pck), 
                "pa_mpjpe": float(final_pa)
            },
            "detailed": all_results if len(all_results) > 1 else None
        }
        
        save_path = plot_path / f"summary_{target_dataset}_{args.split}.yaml"
        with open(save_path, 'w') as f:
            yaml.dump(results_dict, f)
        print(f"Metrics saved to: {save_path}")

if __name__ == '__main__':
    main()