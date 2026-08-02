#!/usr/bin/env python3
"""Evaluation Script for Gesture Recognition (Jester & EgoGesture)."""

import torch
import yaml
import argparse
import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transhands import TransHands
from src.models.gesture_head import create_gesture_head
from src.models.transhands_gesture import TransHandsGesture
from src.data.gesture.jester import JesterDataset, collate_fn_gesture
from src.data.gesture.egogesture import EgoGestureDataset, collate_fn_egogesture
from torch.utils.data import DataLoader

# Default Paths for datasets
DEFAULT_PATHS = {
    'jester': 'dataset/Jester',
    'egogesture': 'dataset/EgoGesture',
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

def compute_top_k_accuracy(logits, labels, k=5):
    _, top_k_pred = logits.topk(k, dim=1)
    correct = top_k_pred.eq(labels.view(-1, 1).expand_as(top_k_pred))
    return float(correct.any(dim=1).float().mean().item() * 100)

def get_val_dataset(dataset_name, config, args, split_name='val'):
    dataset_name = dataset_name.lower()
    ds_cfg = config['dataset']
    seq_len = ds_cfg['seq_len']

    def get_smart_root(target_name):
        if args.data_root:
            return args.data_root
        config_ds_name = ds_cfg.get('name', '').lower()
        if config_ds_name == target_name:
            return ds_cfg.get('root')
        if target_name in DEFAULT_PATHS:
            return DEFAULT_PATHS[target_name]
        return f'dataset/{target_name}'

    root = get_smart_root(dataset_name)
    print(f"Loading Dataset: {dataset_name} (Split: {split_name}) from: {root}")

    if dataset_name == 'jester':
        return JesterDataset(
            root_dir=root,
            keypoints_dir=ds_cfg.get('keypoints_dir'),
            split=split_name,
            seq_len=seq_len,
            augmentation=False,
            labels_file=ds_cfg.get('labels_file'),
            annotation_file=ds_cfg.get(f'{split_name}_annotation')
        )
    elif dataset_name == 'egogesture':
        return EgoGestureDataset(
            keypoints_dir=ds_cfg.get('keypoints_dir'),
            split=split_name,
            seq_len=seq_len,
            min_valid_frames=ds_cfg.get('min_valid_frames', 10),
            min_valid_ratio=ds_cfg.get('min_valid_ratio', 0.3)
        )
    else:
        raise ValueError(f"Dataset {dataset_name} not supported for evaluation.")

def get_collate_fn(dataset_name):
    dataset_name = dataset_name.lower()
    if dataset_name == 'jester':
        return collate_fn_gesture
    elif dataset_name == 'egogesture':
        return collate_fn_egogesture
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

def evaluate(model, loader, device, num_classes, add_noise=False, noise_std=8.0):
    model.eval()
    if len(loader) == 0:
        return {'accuracy': 0.0, 'top5_accuracy': 0.0, 'per_class_accuracy': {},
                'predictions': np.array([]), 'labels': np.array([])}

    all_preds = []
    all_labels = []
    all_logits = []

    print(f"Starting evaluation{' with 2D noise' if add_noise else ''} on {len(loader.dataset)} sequences...")

    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None:
                continue

            inputs = batch['input'].to(device)
            labels = batch['label'].to(device)

            if add_noise:
                inputs = add_2d_noise(inputs, noise_std_pixels=noise_std)

            logits = model(inputs)
            if logits.dtype == torch.float16:
                logits = logits.float()

            _, predicted = logits.max(dim=1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(logits.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)

    accuracy = float(100.0 * (all_preds == all_labels).sum() / max(1, len(all_labels)))
    top5_acc = compute_top_k_accuracy(all_logits, torch.tensor(all_labels), k=min(5, num_classes))

    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)
    for pred, label in zip(all_preds, all_labels):
        class_total[label] += 1
        if pred == label:
            class_correct[label] += 1

    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[i] = float(100.0 * class_correct[i] / class_total[i])
        else:
            per_class_acc[i] = 0.0

    return {
        'accuracy': accuracy,
        'top5_accuracy': top5_acc,
        'per_class_accuracy': per_class_acc,
        'predictions': all_preds,
        'labels': all_labels
    }

def build_model(config, device):
    transhands = TransHands(
        num_hand_joints=config['model']['num_hand_joints'],
        num_body_joints=config['model']['num_body_joints'],
        freeze_encoder=config['model']['freeze_encoder'],
        encoder_type=config['model'].get('encoder_type', 'motionbert'),
        seq_len=config['dataset']['seq_len'],
        adapter_type=config['model'].get('adapter_type', 'ode'),
        projection_type=config['model'].get('projection_type', 'retnet')
    ).to(device)

    head_type = config['gesture']['head_type']
    head_kwargs = {
        'input_dim': config['gesture']['feature_dim'],
        'num_classes': config['gesture']['num_classes'],
        'hidden_dim': config['gesture'].get('hidden_dim', 512),
        'dropout': config['gesture'].get('dropout', 0.5),
        'num_layers': config['gesture'].get('num_layers', 2),
        'kernel_size': config['gesture'].get('kernel_size', 3),
    }

    gesture_head = create_gesture_head(head_type=head_type, **head_kwargs).to(device)

    return TransHandsGesture(
        transhands_model=transhands,
        gesture_head=gesture_head,
        extraction_point=config['gesture'].get('extraction_point', 'geometric'),
        freeze_transhands=config['gesture'].get('freeze_transhands', True),
        use_velocity=config['gesture'].get('use_velocity', False)
    ).to(device)

def label_name_for(dataset_name, idx):
    if dataset_name == 'jester' and idx < len(JesterDataset.GESTURE_LABELS):
        return JesterDataset.GESTURE_LABELS[idx]
    return f"Gesture_Class_{idx + 1}"

def main():
    parser = argparse.ArgumentParser(description="Gesture Recognition Evaluation")
    parser.add_argument('--config', required=True, help='Path to model config file')
    parser.add_argument('--checkpoint', required=True, help='Path to .pth checkpoint')
    parser.add_argument('--dataset', default=None, help='Override test dataset (jester / egogesture)')
    parser.add_argument('--split', default='val', help='Dataset split to evaluate')
    parser.add_argument('--data-root', default=None, help='Override dataset root path (optional)')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--save-dir', default=None, help='Directory to save results (overrides config paths.plot)')
    parser.add_argument('--add-noise', action='store_true', help='Add Gaussian noise to 2D input (robustness test)')
    parser.add_argument('--noise-std', type=float, nargs='+', default=[8.0], help='Noise std deviation(s) in pixels (e.g., 0 5 8 10 15)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    num_classes = config['gesture']['num_classes']

    print(f"=== Evaluating Gesture Recognition ===")
    print(f"Config: {args.config} | Checkpoint: {args.checkpoint} | Split: {args.split}")
    print(f"Extract From: {config['gesture'].get('extraction_point', 'geometric')} | "
          f"Use Velocity: {config['gesture'].get('use_velocity', False)}")
    if args.add_noise:
        print(f"2D Noise Enabled: Testing noise levels: {args.noise_std} pixels")

    # Build & load model
    model = build_model(config, device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict, strict=True)

    target_dataset = (args.dataset if args.dataset else config['dataset']['name']).lower()
    datasets_to_test = [target_dataset]

    save_dir = Path(args.save_dir) if args.save_dir else Path(config['paths']['plot'])
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Multi-noise robustness evaluation
    if args.add_noise and len(args.noise_std) > 1:
        print(f"\n" + "="*60)
        print("NOISE ROBUSTNESS EVALUATION")
        print("="*60)

        for ds_name in datasets_to_test:
            print(f"Model: {args.checkpoint}")
            print(f"Dataset: {ds_name} ({args.split})")

            dataset = get_val_dataset(ds_name, config, args, split_name=args.split)
            collate_func = get_collate_fn(ds_name)
            print(f"Sequences: {len(dataset)}")
            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_func, num_workers=2, pin_memory=True)

            noise_results = {}
            for noise_std in args.noise_std:
                print(f"\n>>> Testing noise std = {noise_std:.1f} pixels")
                results = evaluate(model, loader, device, num_classes, add_noise=True, noise_std=noise_std)
                noise_results[noise_std] = {
                    "accuracy": results['accuracy'],
                    "top5_accuracy": results['top5_accuracy'],
                }

            print(f"\n{'Noise (px)':<12} {'Top-1 (%)':<12} {'Top-5 (%)':<12}")
            print("-"*60)
            for noise_std in sorted(noise_results.keys()):
                res = noise_results[noise_std]
                print(f"{noise_std:<12.1f} {res['accuracy']:<12.2f} {res['top5_accuracy']:<12.2f}")
            print("="*60)

            all_results[ds_name] = noise_results

            save_path = save_dir / f"noise_robustness_{ds_name}_{args.split}.txt"
            with open(save_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("NOISE ROBUSTNESS EVALUATION\n")
                f.write("="*60 + "\n")
                f.write(f"Model: {args.checkpoint}\n")
                f.write(f"Dataset: {ds_name} ({args.split})\n")
                f.write(f"Sequences: {len(dataset)}\n\n")
                f.write(f"{'Noise (px)':<12} {'Top-1 (%)':<12} {'Top-5 (%)':<12}\n")
                f.write("-"*60 + "\n")
                for noise_std in sorted(noise_results.keys()):
                    res = noise_results[noise_std]
                    f.write(f"{noise_std:<12.1f} {res['accuracy']:<12.2f} {res['top5_accuracy']:<12.2f}\n")
                f.write("="*60 + "\n")

            print(f"\nResults saved to: {save_path}")

    # Standard evaluation
    else:
        noise_std = args.noise_std[0] if args.add_noise else 0.0

        for ds_name in datasets_to_test:
            print(f"\n>>> Running Evaluation on: {ds_name.upper()}")
            try:
                dataset = get_val_dataset(ds_name, config, args, split_name=args.split)
                collate_func = get_collate_fn(ds_name)
                loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                    collate_fn=collate_func, num_workers=2, pin_memory=True)

                results = evaluate(model, loader, device, num_classes,
                                   add_noise=args.add_noise, noise_std=noise_std)
                all_results[ds_name] = results
                print(f"--> {ds_name}: Top-1={results['accuracy']:.2f}%, Top-5={results['top5_accuracy']:.2f}%")

            except Exception as e:
                print(f"ERROR evaluating {ds_name}: {e}")
                all_results[ds_name] = {'accuracy': 0.0, 'top5_accuracy': 0.0,
                                        'per_class_accuracy': {},
                                        'predictions': np.array([]), 'labels': np.array([])}

        # Print and save standard results
        for ds_name in datasets_to_test:
            results = all_results[ds_name]

            print("\n" + "="*60)
            print(f"EVALUATION RESULTS: {ds_name.upper()}")
            print("="*60)
            print(f"Overall Accuracy: {results['accuracy']:.2f}%")
            print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
            print("="*60)

            print("\nPer-Class Accuracy:")
            print("-"*60)
            for i in range(num_classes):
                acc = results['per_class_accuracy'].get(i, 0.0)
                print(f"{label_name_for(ds_name, i):40s} | Accuracy: {acc:6.2f}%")

            report_path = save_dir / f'results_{ds_name}_{args.split}.txt'
            with open(report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write(f"Gesture Recognition Evaluation Results: {ds_name.upper()}\n")
                f.write("="*60 + "\n")
                f.write(f"Model: {config['model'].get('encoder_type', 'unknown')}\n")
                f.write(f"Head: {config['gesture']['head_type']}\n")
                f.write(f"Config: {args.config}\n")
                f.write(f"Checkpoint: {args.checkpoint}\n")
                f.write(f"Split: {args.split}\n")
                f.write("="*60 + "\n\n")
                f.write(f"Overall Accuracy: {results['accuracy']:.2f}%\n")
                f.write(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%\n\n")
                f.write("Per-Class Accuracy:\n")
                f.write("-"*60 + "\n")
                for i in range(num_classes):
                    acc = results['per_class_accuracy'].get(i, 0.0)
                    f.write(f"{label_name_for(ds_name, i):40s} | Accuracy: {acc:6.2f}%\n")
                f.write("="*60 + "\n")
            print(f"\nResults saved to: {report_path}")

            np.savez(
                save_dir / f'predictions_{ds_name}_{args.split}.npz',
                predictions=results['predictions'],
                labels=results['labels']
            )
            print(f"Predictions saved to: {save_dir / f'predictions_{ds_name}_{args.split}.npz'}")

            results_dict = {
                "model": config['model'].get('encoder_type', 'unknown'),
                "head": config['gesture']['head_type'],
                "dataset": ds_name,
                "split": args.split,
                "metrics": {
                    "accuracy": float(results['accuracy']),
                    "top5_accuracy": float(results['top5_accuracy']),
                },
                "per_class_accuracy": {int(k): float(v) for k, v in results['per_class_accuracy'].items()}
            }
            yaml_path = save_dir / f"summary_{ds_name}_{args.split}.yaml"
            with open(yaml_path, 'w') as f:
                yaml.dump(results_dict, f)
            print(f"Metrics saved to: {yaml_path}")

if __name__ == '__main__':
    main()