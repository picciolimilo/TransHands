#!/usr/bin/env python3
"""Visualization and Overall Statistics Script for TransHands."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import yaml
import sys
import warnings
from collections import defaultdict
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transhands import TransHands

# Global Constants for Plotting
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),   # Thumb
    (0,5),(5,6),(6,7),(7,8),   # Index
    (0,9),(9,10),(10,11),(11,12), # Middle
    (0,13),(13,14),(14,15),(15,16), # Ring
    (0,17),(17,18),(18,19),(19,20)  # Pinky
]
JOINT_NAMES = [
    'Wrist', 'Th1', 'Th2', 'Th3', 'Th4', 
    'Id1', 'Id2', 'Id3', 'Id4',
    'Md1', 'Md2', 'Md3', 'Md4', 
    'Rg1', 'Rg2', 'Rg3', 'Rg4',
    'Pk1', 'Pk2', 'Pk3', 'Pk4'
]

def get_final_pred(output):
    """Extract final output if model returns list"""
    return output[-1] if isinstance(output, (list, tuple)) else output

def get_visualization_frames(pred_3d, target_3d, input_2d=None):
    """
    Return individual frames centered at root.
    """
    mid_target = target_3d.shape[1] // 2
    
    if pred_3d.shape[1] == 1:
        pred_frame = pred_3d[0, 0]
    else:
        pred_frame = pred_3d[0, mid_target]

    target_frame = target_3d[0, mid_target]
    
    # Root centering (Wrist at 0,0,0)
    pred_frame = pred_frame - pred_frame[0:1, :]       
    target_frame = target_frame - target_frame[0:1, :] 

    input_frame = None
    if input_2d is not None:
        input_frame = input_2d[0, mid_target]
        input_frame = input_frame - input_frame[0:1, :] 
        
    return pred_frame, target_frame, input_frame

def procrustes_alignment(pred, gt):
    # Centering
    mu_pred = pred.mean(0)
    mu_gt = gt.mean(0)
    pred0 = pred - mu_pred
    gt0 = gt - mu_gt
    
    # SVD
    H = pred0.T @ gt0
    U, S, Vt = np.linalg.svd(H)
    R = U @ Vt
    
    # Reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    return (pred0 @ R) + mu_gt

def get_balanced_indices(dataset, num_samples):
    if hasattr(dataset, 'cumulative_lengths') and hasattr(dataset, 'lengths'):
        num_datasets = len(dataset.lengths)
        base_count = num_samples // num_datasets
        remainder = num_samples % num_datasets
        indices = []
        
        print(f"\n[Balanced Sampling] Requesting {num_samples} samples across {num_datasets} datasets:")
        
        for i in range(num_datasets):
            count = base_count + (1 if i < remainder else 0)
            if count == 0: continue
            
            start_global = dataset.cumulative_lengths[i]
            length_ds = dataset.lengths[i]
            
            # Random choice within this sub-dataset
            local_indices = np.random.choice(length_ds, min(count, length_ds), replace=False)
            global_indices = local_indices + start_global
            indices.extend(global_indices)
            
            ds_name = dataset.datasets[i].__class__.__name__.replace('Dataset', '')
            print(f"  - {ds_name}: {len(local_indices)} samples")
            
        return np.array(indices)
    else:
        # Standard random sampling for single dataset
        return np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

def get_dataset(config, split='test'):
    """Load dataset based on config"""
    dataset_config = config['dataset']
    dataset_name = dataset_config['name'].lower()
    
    if dataset_name == 'multi': 
        from src.data.lifting.multi_dataset import MultiHandDataset
        datasets = []
        global_seq_len = dataset_config.get('seq_len')
        global_stride = dataset_config.get('stride', 10)
        global_min_valid = dataset_config.get('min_valid_frames', 75)
        
        for ds_cfg in dataset_config['datasets']: 
            temp_config = {'dataset': ds_cfg.copy()}
            if 'seq_len' not in temp_config['dataset']: temp_config['dataset']['seq_len'] = global_seq_len
            if 'stride' not in temp_config['dataset']: temp_config['dataset']['stride'] = global_stride
            if 'min_valid_frames' not in temp_config['dataset']: temp_config['dataset']['min_valid_frames'] = global_min_valid
            datasets.append(get_dataset(temp_config, split))
        return MultiHandDataset(datasets)

    if 'root' not in dataset_config:
        raise KeyError(f"Dataset '{dataset_name}' config is missing 'root' path.")
        
    root_dir = dataset_config['root']
    seq_len = dataset_config['seq_len']
    stride = dataset_config.get('stride', None)
    min_valid_frames = dataset_config.get('min_valid_frames', None)
    use_2d = dataset_config.get('use_2d', True)
    augmentation = dataset_config.get('augmentation', False)

    if dataset_name == 'reinterhand':
        from src.data.lifting.reinterhand import ReInterHandDataset
        return ReInterHandDataset(root_dir, split, seq_len, stride, min_valid_frames, use_2d=use_2d, augmentation=augmentation)
    elif dataset_name == 'assemblyhands':
        from src.data.lifting.assemblyhands import AssemblyHandsDataset
        return AssemblyHandsDataset(root_dir, split, seq_len, stride, min_valid_frames, use_2d=use_2d, augmentation=augmentation)
    elif dataset_name == 'gigahands':
        from src.data.lifting.gigahands import GigaHandsDataset
        return GigaHandsDataset(root_dir, split, seq_len, stride, min_valid_frames, use_2d=use_2d, augmentation=augmentation)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def plot_predictions_3d(model, dataset, device, indices, save_path='predictions_3d.png'):
    """Generate 3D plots (Pred vs GT)"""
    model.eval()
    ncols = 5
    nrows = (len(indices) + ncols - 1) // ncols
    fig = plt.figure(figsize=(ncols * 4, nrows * 7))

    with torch.no_grad():
        for i, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            if sample is None or 'left_hand_input' not in sample: continue

            input_2d = sample['left_hand_input'].unsqueeze(0).to(device)
            target_3d = sample['left_hand_target'].unsqueeze(0).to(device)
            scale_3d = sample.get('left_scale_3d', 100.0)

            pred_3d = get_final_pred(model(input_2d))
            pred_frame, target_frame, _ = get_visualization_frames(pred_3d, target_3d)

            gt_vis = (target_frame * scale_3d).cpu().numpy()
            pred_vis = (pred_frame * scale_3d).cpu().numpy()

            mpjpe = np.mean(np.linalg.norm(gt_vis - pred_vis, axis=-1))
            pred_aligned = procrustes_alignment(pred_vis, gt_vis)
            pa_mpjpe = np.mean(np.linalg.norm(pred_aligned - gt_vis, axis=-1))

            views = [{'elev': 15, 'azim': 45, 'label': 'Front'}, {'elev': 15, 'azim': 90, 'label': 'Side'}]
            
            for view_idx, view in enumerate(views):
                subplot_idx = i * 2 + view_idx + 1
                ax = fig.add_subplot(nrows * 2, ncols, subplot_idx, projection='3d')
                for c in HAND_CONNECTIONS:
                    ax.plot(gt_vis[c, 0], gt_vis[c, 1], gt_vis[c, 2], c='blue', lw=2, alpha=0.6, zorder=1)
                    ax.plot(pred_vis[c, 0], pred_vis[c, 1], pred_vis[c, 2], c='red', ls='--', lw=1.5, alpha=0.6, zorder=1)
                ax.scatter(gt_vis[:,0], gt_vis[:,1], gt_vis[:,2], c='blue', s=20, alpha=0.7); ax.scatter(pred_vis[:,0], pred_vis[:,1], pred_vis[:,2], c='red', s=20, alpha=0.7)

                origin = sample.get('dataset_origin')
                title_prefix = f"[{origin}] " if origin else ""
                title = f"{title_prefix}S:{sample_idx}\nMPJPE:{mpjpe:.1f} | PA:{pa_mpjpe:.1f}"
                ax.set_title(title, fontsize=9)
                ax.view_init(elev=view['elev'], azim=view['azim'])
                
                all_pts = np.concatenate([gt_vis, pred_vis])
                mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0); centers = (mins + maxs) / 2
                max_range = (maxs - mins).max() * 0.6
                ax.set_xlim(centers[0]-max_range, centers[0]+max_range); ax.set_ylim(centers[1]-max_range, centers[1]+max_range); ax.set_zlim(centers[2]-max_range, centers[2]+max_range)
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.tick_params(labelsize=7)

    plt.tight_layout(pad=1.5)
    plt.savefig(save_path, dpi=220, bbox_inches='tight')
    print(f"- Saved 3D plot: {save_path}")
    plt.close()

def get_best_worst_indices(model, dataset, device, num_samples=5):
    """Mines hard examples for qualitative analysis"""
    model.eval()
    errors = []

    print("Mining hard examples...")
    indices = get_balanced_indices(dataset, 1000) if len(dataset) > 1000 else range(len(dataset))
    
    with torch.no_grad():
        for i in indices:
            sample = dataset[i]
            if sample is not None and 'left_hand_input' in sample and sample['left_hand_input'] is not None:
                input_2d = sample['left_hand_input'].unsqueeze(0).to(device)
                target_3d = sample['left_hand_target'].unsqueeze(0).to(device)
                scale = sample.get('left_scale_3d', 100.0)

                pred_3d = get_final_pred(model(input_2d))
                pred_frame, target_frame, _ = get_visualization_frames(pred_3d, target_3d)

                gt_mm = (target_frame * scale).cpu().numpy()
                pred_mm = (pred_frame * scale).cpu().numpy()
                
                error = np.mean(np.linalg.norm(pred_mm - gt_mm, axis=-1))
                errors.append((i, error))
    
    errors.sort(key=lambda x: x[1])
    best_indices = [x[0] for x in errors[:num_samples]]
    worst_indices = [x[0] for x in errors[-num_samples:]]
    
    return best_indices, worst_indices

def plot_predictions_3d_authentic(model, dataset, device, indices, save_path, title_prefix=""):
    """High-quality 3D plots. Includes Aligned pose."""
    model.eval()
    fig = plt.figure(figsize=(10, 4 * len(indices)))

    with torch.no_grad():
        for row, sample_idx in enumerate(indices):
            sample = dataset[sample_idx]
            if sample is None or 'left_hand_input' not in sample: continue

            input_2d = sample['left_hand_input'].unsqueeze(0).to(device)
            target_3d = sample['left_hand_target'].unsqueeze(0).to(device)
            scale = sample.get('left_scale_3d', 100.0)
            
            pred_3d = get_final_pred(model(input_2d))
            pred_frame, target_frame, _ = get_visualization_frames(pred_3d, target_3d)
            gt_mm = (target_frame * scale).cpu().numpy()
            pred_mm = (pred_frame * scale).cpu().numpy()
            pred_aligned = procrustes_alignment(pred_mm, gt_mm)
            
            mpjpe = np.mean(np.linalg.norm(pred_mm - gt_mm, axis=-1))
            pampjpe = np.mean(np.linalg.norm(pred_aligned - gt_mm, axis=-1))

            views = [(15, 45, "Front"), (15, 120, "Side")]
            for col, (elev, azim, v_name) in enumerate(views):
                ax = fig.add_subplot(len(indices), 2, 2*row + col + 1, projection='3d')
                for p1, p2 in HAND_CONNECTIONS:
                    ax.plot([gt_mm[p1,0], gt_mm[p2,0]], [gt_mm[p1,1], gt_mm[p2,1]], [gt_mm[p1,2], gt_mm[p2,2]], c='blue', lw=2, alpha=0.6)
                    ax.plot([pred_mm[p1,0], pred_mm[p2,0]], [pred_mm[p1,1], pred_mm[p2,1]], [pred_mm[p1,2], pred_mm[p2,2]], c='red', ls='--', lw=1.5, alpha=0.6)
                    ax.plot([pred_aligned[p1,0], pred_aligned[p2,0]], [pred_aligned[p1,1], pred_aligned[p2,1]], [pred_aligned[p1,2], pred_aligned[p2,2]], c='gray', ls=':', lw=1, alpha=0.4)
                
                ax.scatter(gt_mm[:,0], gt_mm[:,1], gt_mm[:,2], c='blue', s=20, label='GT')
                ax.scatter(pred_mm[:,0], pred_mm[:,1], pred_mm[:,2], c='red', s=20, label='Pred')
                ax.scatter(pred_aligned[:,0], pred_aligned[:,1], pred_aligned[:,2], c='gray', s=10, label='Aligned', alpha=0.5)

                origin = sample.get('dataset_origin')
                ds_label = f"[{origin}] " if origin else ""
                ax.set_title(f"{ds_label}{title_prefix} S:{sample_idx}\nMPJPE: {mpjpe:.1f} | PA: {pampjpe:.1f} ({v_name})", fontsize=10)
                ax.view_init(elev=elev, azim=azim)
                
                all_pts = np.concatenate([gt_mm, pred_mm, pred_aligned], axis=0)
                mins = all_pts.min(axis=0); maxs = all_pts.max(axis=0); centers = (mins + maxs) / 2
                max_range = (maxs - mins).max() * 0.6
                ax.set_xlim(centers[0]-max_range, centers[0]+max_range); ax.set_ylim(centers[1]-max_range, centers[1]+max_range); ax.set_zlim(centers[2]-max_range, centers[2]+max_range)
                ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z'); ax.tick_params(labelsize=6)
                if col == 0 and row == 0: ax.legend(fontsize=6, loc='upper left')
                
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path.with_suffix('.pdf'), dpi=300)
    print(f"- Saved Authentic 3D Plot: {save_path.with_suffix('.pdf')}")
    plt.close()

def print_statistics(model, dataset, device, dataset_name='', save_path=None, full_eval=False):
    """Statistics with confidence intervals, PA-MPJPE, and Dataset Breakdown."""
    model.eval()
    
    print("\n" + "="*60)
    print(f"COMPUTING STATISTICS FOR:  {dataset_name}")
    print("="*60)

    all_mpjpe = []
    all_pa_mpjpe = []
    all_pck_20 = []
    all_pck_30 = []
    all_pck_50 = []
    per_joint_errors = []
    stats_by_dataset = defaultdict(list)
    
    if full_eval:
        print("Running FULL EVALUATION on all samples...")
        indices = range(len(dataset))
    else:
        indices = get_balanced_indices(dataset, 2000) if len(dataset) > 2000 else range(len(dataset))
    
    with torch.no_grad():
        for i in indices:
            sample = dataset[i]

            if sample is not None and 'left_hand_input' in sample and sample['left_hand_input'] is not None:
                input_2d = sample['left_hand_input'].unsqueeze(0).to(device)
                target_3d = sample['left_hand_target'].unsqueeze(0).to(device)
                scale_3d = sample.get('left_scale_3d', 100.0) or 100.0
                origin = sample.get('dataset_origin', 'Total')
                
                pred_3d = model(input_2d)
                pred_3d = get_final_pred(pred_3d)
                
                # Get frames in mm
                pred_frame, target_frame, _ = get_visualization_frames(pred_3d, target_3d)
                gt_mm = (target_frame * scale_3d).cpu().numpy()
                pred_mm = (pred_frame * scale_3d).cpu().numpy()

                errors = np.linalg.norm(pred_mm - gt_mm, axis=-1) # (21,)
                mpjpe = errors.mean()
                
                all_mpjpe.append(mpjpe)
                per_joint_errors.append(errors)

                pred_aligned = procrustes_alignment(pred_mm, gt_mm)
                pa_mpjpe = np.linalg.norm(pred_aligned - gt_mm, axis=-1).mean()
                all_pa_mpjpe.append(pa_mpjpe)

                all_pck_20.append((errors < 20).mean() * 100)
                all_pck_30.append((errors < 30).mean() * 100)
                all_pck_50.append((errors < 50).mean() * 100)
                
                stats_by_dataset[origin].append(mpjpe)
                stats_by_dataset['Total (Aggregated)'].append(mpjpe)
    
    if len(all_mpjpe) == 0:
        print("No samples processed.")
        return

    # Convert to numpy
    all_mpjpe = np.array(all_mpjpe)
    all_pa_mpjpe = np.array(all_pa_mpjpe)
    per_joint_errors = np.array(per_joint_errors)
    
    # Confidence intervals helper
    def ci_95(data):
        if len(data) < 2: return (0, 0)
        return scipy_stats.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=scipy_stats.sem(data))
    
    mpjpe_ci = ci_95(all_mpjpe)
    pa_mpjpe_ci = ci_95(all_pa_mpjpe)
    
    report = []

    report.append(f"\n{'='*60}")
    report.append(f"BREAKDOWN BY DATASET")
    report.append(f"{'='*60}")
    for name, errors in stats_by_dataset.items():
        if not errors: continue
        errors = np.array(errors)
        mean_err = errors.mean()
        std_err = errors.std()
        ci = ci_95(errors)
        report.append(f"{name:20s}: {mean_err:.2f} ± {std_err:.2f} mm  (N={len(errors)})")
        report.append(f"                      95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        report.append("-" * 40)

    report.append(f"\n{'='*60}")
    report.append(f"GLOBAL METRICS")
    report.append(f"{'='*60}")
    report.append(f"Samples evaluated: {len(all_mpjpe)}")
    report.append(f"")
    report.append(f"OVERALL:")
    report.append(f"  MPJPE:         {all_mpjpe.mean():.2f} ± {all_mpjpe.std():.2f} mm")
    report.append(f"                95% CI: [{mpjpe_ci[0]:.2f}, {mpjpe_ci[1]:.2f}]")
    report.append(f"  PA-MPJPE:     {all_pa_mpjpe.mean():.2f} ± {all_pa_mpjpe.std():.2f} mm")
    report.append(f"                95% CI: [{pa_mpjpe_ci[0]:.2f}, {pa_mpjpe_ci[1]:.2f}]")
    report.append(f"  Median Error: {np.median(all_mpjpe):.2f} mm")
    report.append(f"")
    report.append(f"PCK (Percentage of Correct Keypoints):")
    report.append(f"  PCK@20mm:      {np.mean(all_pck_20):.2f}%")
    report.append(f"  PCK@30mm:     {np.mean(all_pck_30):.2f}%")
    report.append(f"  PCK@50mm:     {np.mean(all_pck_50):.2f}%")
    report.append(f"")
    report.append(f"PER-JOINT ERRORS (mm):")
    joint_mean = per_joint_errors.mean(axis=0)
    for name, err in zip(JOINT_NAMES, joint_mean):
        report.append(f"  {name:8s}: {err:.2f} mm")
    report.append(f"{'='*60}\n")
    
    final_text = "\n".join(report)
    print(final_text)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(final_text)
        print(f"- Saved statistics:  {save_path}")

def main():
    parser = argparse.ArgumentParser(description='TransHands Visualization & Statistics')
    parser.add_argument('--config', type=str, required=True, help='Config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'])
    parser.add_argument('--full-eval', action='store_true', help='Evaluate on the entire dataset for accurate stats')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Force augmentation off for evaluation
    if 'dataset' in config and 'augmentation' in config['dataset']:
        config['dataset']['augmentation'] = False
    if 'dataset' in config and 'datasets' in config['dataset']:
        for ds_cfg in config['dataset']['datasets']:
            ds_cfg['augmentation'] = False
    
    # Checkpoint path
    if args.checkpoint is None:
        args.checkpoint = str(Path(config['paths']['checkpoint']) / 'best.pth')
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Dataset & Model
    dataset = get_dataset(config, split=args.split)
    print(f"Loaded {len(dataset)} samples from {args.split} split")
    
    print(f"Loading model from: {checkpoint_path}")

    model = TransHands(
        num_hand_joints=config['model']['num_hand_joints'],
        num_body_joints=config['model']['num_body_joints'],
        freeze_encoder=config['model']['freeze_encoder'],
        weights_path=config['model']['weights_path'],
        encoder_type=config['model'].get('encoder_type', 'motionbert'),
        seq_len=config['dataset']['seq_len'],
        adapter_type=config['model'].get('adapter_type', 'ode'),
        projection_type=config['model'].get('projection_type', 'retnet')
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    output_dir = Path(config['paths']['plot'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualizations
    print("\n" + "="*60 + "\nVISUALIZATIONS\n" + "="*60)
    indices = get_balanced_indices(dataset, args.num_samples)
    plot_predictions_3d(model, dataset, device, indices=indices, save_path=output_dir / 'predictions_3d.png')
    
    # Minining best & worst samples
    print("\n" + "="*60 + "\nMINING BEST & WORST SAMPLES\n" + "="*60)
    best_idx, worst_idx = get_best_worst_indices(model, dataset, device, num_samples=3)
    
    plot_predictions_3d_authentic(model, dataset, device, best_idx, output_dir / 'analysis_best', title_prefix="BEST")
    plot_predictions_3d_authentic(model, dataset, device, worst_idx, output_dir / 'analysis_failure_cases', title_prefix="WORST")
    plot_predictions_3d_authentic(model, dataset, device, indices[:3], output_dir / 'analysis_random', title_prefix="RANDOM")
    
    # Statistics
    print_statistics(model, dataset, device, dataset_name=f"{config['dataset']['name']} ({args.split})", save_path=output_dir / f'statistics_{args.split}.txt', full_eval=args.full_eval)
    
    print("\n" + "="*60 + f"\nAll visualizations saved to: {output_dir}\n" + "="*60 + "\n")

if __name__ == '__main__':
    main()