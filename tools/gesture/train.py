#!/usr/bin/env python3
"""Training script for Gesture Recognition (Jester & EgoGesture) with TransHands."""

import torch
import yaml
from pathlib import Path
import argparse
import sys
import numpy as np
import wandb
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transhands import TransHands
from src.models.gesture_head import create_gesture_head
from src.models.transhands_gesture import TransHandsGesture, _load_transhands_checkpoint_compatible
from src.training.losses import GestureClassificationLoss
from src.training.schedulers import WarmupCosineLR
from src.data.gesture.jester import JesterDataset, collate_fn_gesture
from src.data.gesture.egogesture import EgoGestureDataset, collate_fn_egogesture
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from src.training.reproducibility import set_seed, seed_worker


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

### Metrics ###
def compute_top_k_accuracy(logits, labels, k=5):
    _, top_k_pred = logits.topk(k, dim=1)
    correct = top_k_pred.eq(labels.view(-1, 1).expand_as(top_k_pred))
    return correct.any(dim=1).float().mean().item() * 100

### Training Loop ###
def _mixup_batch(inputs, labels, alpha):
    """Sample-pair mixup on inputs; return mixed inputs, label_a, label_b, lam."""
    lam = float(np.random.beta(alpha, alpha))
    lam = max(lam, 1.0 - lam) 
    perm = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1.0 - lam) * inputs[perm]
    return mixed, labels, labels[perm], lam


def train_epoch(model, loader, optimizer, criterion, device, scaler,
                grad_clip=1.0, mixup_alpha=0.0):
    model.train()
    total_loss, total_correct, n = 0.0, 0, 0
    use_mixup = mixup_alpha > 0.0

    for batch in loader:
        if batch is None:
            continue

        inputs = batch['input'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        if use_mixup:
            inputs, labels_a, labels_b, lam = _mixup_batch(inputs, labels, mixup_alpha)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda'):
            logits = model(inputs)
            if use_mixup:
                loss = lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        _, predicted = logits.max(dim=1)
        ref_labels = labels_a if use_mixup else labels
        total_correct += (predicted == ref_labels).sum().item()
        total_loss += loss.item()
        n += ref_labels.size(0)

    return total_loss / max(1, len(loader)), 100.0 * total_correct / max(1, n)

def validate(model, loader, criterion, device, num_classes):
    model.eval()
    total_loss, total_correct, n = 0.0, 0, 0
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            inputs = batch['input'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            with autocast('cuda'):
                logits = model(inputs)
                loss = criterion(logits, labels)

            if logits.dtype == torch.float16:
                logits = logits.float()

            _, predicted = logits.max(dim=1)
            total_correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            n += labels.size(0)

            for label, pred in zip(labels.cpu(), predicted.cpu()):
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    top5_acc = compute_top_k_accuracy(all_logits, all_labels, k=min(5, num_classes))

    per_class_acc = {}
    for i in range(num_classes):
        if class_total[i] > 0:
            per_class_acc[f'class_{i}'] = 100.0 * class_correct[i].item() / class_total[i].item()

    return (total_loss / max(1, len(loader)),
            100.0 * total_correct / max(1, n),
            top5_acc,
            per_class_acc)

def save_checkpoint(path, epoch, model, optimizer, best_acc, scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    target_path = Path(path)
    temp_path = target_path.with_suffix('.tmp')

    try:
        torch.save(checkpoint, temp_path)
        temp_path.replace(target_path)
    except Exception as e:
        print(f">>> CRITICAL I/O ERROR: Failed saving {target_path.name}: {e}")
        if temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass

### Dataset Loader ###
def get_dataset(config, split):
    dataset_name = config['dataset']['name'].lower()
    ds_cfg = config['dataset']
    seq_len = ds_cfg['seq_len']
    is_train = (split == 'train')

    augmentation = is_train and ds_cfg.get('augmentation', True)

    if dataset_name == 'jester':
        return JesterDataset(
            root_dir=ds_cfg['root'],
            keypoints_dir=ds_cfg.get('keypoints_dir'),
            split=split,
            seq_len=seq_len,
            stride=ds_cfg.get('stride') if is_train else None,
            augmentation=augmentation,
            labels_file=ds_cfg.get('labels_file'),
            annotation_file=ds_cfg.get('train_annotation' if is_train else 'val_annotation')
        )
    elif dataset_name == 'egogesture':
        return EgoGestureDataset(
            keypoints_dir=ds_cfg.get('keypoints_dir'),
            split=split,
            seq_len=seq_len,
            min_valid_frames=ds_cfg.get('min_valid_frames', 10),
            min_valid_ratio=ds_cfg.get('min_valid_ratio', 0.3),
            augmentation=augmentation
        )
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized!")

def get_collate_fn(config):
    dataset_name = config['dataset']['name'].lower()
    if dataset_name == 'jester':
        return collate_fn_gesture
    elif dataset_name == 'egogesture':
        return collate_fn_egogesture
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized!")

def main():
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--resume', default=None, type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--pretrained-checkpoint', default=None, type=str, help='Override TransHands pretrained checkpoint path')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--wandb-project', default=None, type=str, help='WandB project name (overrides config)')
    parser.add_argument('--wandb-name', default=None, type=str, help='WandB run name (overrides config)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config['training'].get('device', 'cuda') if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    dataset_name = config['dataset']['name'].lower()
    num_classes = config['gesture']['num_classes']
    extraction_point = config['gesture'].get('extraction_point', 'geometric')
    use_velocity = config['gesture'].get('use_velocity', False)
    freeze_transhands = config['gesture'].get('freeze_transhands', config['model'].get('freeze_encoder', True))

    print("=== Gesture Recognition Training ===")
    print(f"Dataset: {dataset_name.upper()} | Classes: {num_classes}")
    print(f"Extract From: {extraction_point} | Use Velocity: {use_velocity} | Freeze Backbone: {freeze_transhands}")

    if not args.no_wandb:
        run_name = args.wandb_name or config.get('experiment_name', f"{dataset_name}_{config['training']['epochs']}ep")
        wandb.init(
            project=args.wandb_project or config.get('wandb_project', f'TransHands-{dataset_name.upper()}'),
            name=run_name,
            config=config,
            tags=[dataset_name]
        )

    train_dataset = get_dataset(config, split='train')
    val_dataset = get_dataset(config, split='val')
    collate_func = get_collate_fn(config)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_func,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_func,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Model Init (TransHands backbone + Gesture Head)
    transhands = TransHands(
        num_hand_joints=config['model']['num_hand_joints'],
        num_body_joints=config['model']['num_body_joints'],
        freeze_encoder=config['model']['freeze_encoder'],
        encoder_type=config['model'].get('encoder_type', 'motionbert'),
        seq_len=config['dataset']['seq_len'],
        adapter_type=config['model'].get('adapter_type', 'ode'),
        projection_type=config['model'].get('projection_type', 'retnet')
    ).to(device)

    pretrained_checkpoint = args.pretrained_checkpoint or config['gesture'].get('pretrained_checkpoint')
    if pretrained_checkpoint:
        print(f"Loading TransHands checkpoint: {pretrained_checkpoint}")
        _load_transhands_checkpoint_compatible(transhands, pretrained_checkpoint, device)

    head_type = config['gesture']['head_type']
    head_kwargs = {
        'input_dim': config['gesture']['feature_dim'],
        'num_classes': num_classes,
        'hidden_dim': config['gesture'].get('hidden_dim', 512),
        'dropout': config['gesture'].get('dropout', 0.5),
        'num_layers': config['gesture'].get('num_layers', 2),
        'kernel_size': config['gesture'].get('kernel_size', 3),
    }

    gesture_head = create_gesture_head(head_type=head_type, **head_kwargs).to(device)

    model = TransHandsGesture(
        transhands_model=transhands,
        gesture_head=gesture_head,
        extraction_point=extraction_point,
        freeze_transhands=freeze_transhands,
        use_velocity=use_velocity
    ).to(device)

    # Optimizer: gesture training keeps the TransHands backbone frozen by design.
    # Only adapter_in (input adapter) and the gesture head are trained.
    print(">>> TransHands Frozen (EXCEPT adapter_in).")
    trainable_params = []
    for name, p in model.transhands.named_parameters():
        if 'adapter_in' in name:
            p.requires_grad = True
            trainable_params.append(p)
        else:
            p.requires_grad = False
    for p in model.gesture_head.parameters():
        p.requires_grad = True
        trainable_params.append(p)
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )

    params = model.transhands.count_parameters() if hasattr(model.transhands, 'count_parameters') else None
    if params is not None:
        print(f"Backbone Parameters: {params['trainable']:,} / {params['total']:,} trainable")

    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=config['training'].get('warmup_epochs', 5),
        max_epochs=config['training']['epochs'],
        min_lr=1e-6
    )

    loss_cfg = config['training'].get('loss', {})
    criterion = GestureClassificationLoss(
        alpha=loss_cfg.get('alpha', 1.0),
        gamma=loss_cfg.get('gamma', 2.0),
        smoothing=loss_cfg.get('smoothing', 0.1)
    ).to(device)
    print(">>> Using GestureClassificationLoss (Focal + Label Smoothing).")

    best_acc = 0.0
    start_epoch = 1
    checkpoint_dir = Path(config['paths']['checkpoint'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Resume from checkpoint
    if args.resume is not None:
        print(f"Loading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)

        pretrained_dict = checkpoint['model_state_dict']
        model_dict = model.state_dict()

        filtered_dict = {k: v for k, v in pretrained_dict.items()
                         if k in model_dict and v.shape == model_dict[k].shape}

        dropped_keys = [k for k in pretrained_dict.keys() if k not in filtered_dict]

        if len(dropped_keys) > 0:
            print(f">>> SMART LOAD: Dropped {len(dropped_keys)} layers due to shape mismatch.")

        model.load_state_dict(filtered_dict, strict=False)

        if len(filtered_dict) == 0:
            raise RuntimeError(">>> ERROR: No weight was loaded, the dictionaries do not match.")

        resume_path = Path(args.resume).resolve()
        current_out_dir = Path(config['paths']['checkpoint']).resolve()

        is_new_stage = current_out_dir not in resume_path.parents

        if is_new_stage:
            print(f">>> NEW STAGE DETECTED: Resume path is different from current output.")
            print(">>> Resetting Best Acc to 0 and starting from Epoch 1.")
            best_acc = 0.0
            start_epoch = 1
        else:
            best_acc = checkpoint.get('best_acc', 0.0)
            start_epoch = checkpoint.get('epoch', 0) + 1
            print(f">>> RESUMING SAME RUN: Keeping previous Best Acc: {best_acc:.2f}%")

            optimizer_loaded = False
            if len(dropped_keys) > 0:
                print(">>> CHANGE DETECTED: Skipping Optimizer loading to prevent shape mismatch crash.")
            elif 'optimizer_state_dict' in checkpoint:
                try:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    optimizer_loaded = True
                    print("Successfully loaded optimizer state.")
                except (ValueError, RuntimeError) as e:
                    print(f">>> WARNING: Failed to load optimizer state: {e}")

            if 'scheduler_state_dict' in checkpoint and optimizer_loaded:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    print("Successfully loaded scheduler state.")
                except Exception as e:
                    print(f"Could not load scheduler: {e}")

    # Training loop
    print("\nStarting Loop...")
    grad_clip = config['training'].get('grad_clip', 1.0)
    mixup_alpha = config['training'].get('mixup_alpha', 0.0)
    if mixup_alpha > 0:
        print(f">>> Mixup enabled (alpha={mixup_alpha}).")

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler,
            grad_clip=grad_clip, mixup_alpha=mixup_alpha,
        )
        val_loss, val_acc, val_top5, per_class_acc = validate(model, val_loader, criterion, device, num_classes)

        if not args.no_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/accuracy': train_acc,
                'val/loss': val_loss,
                'val/accuracy': val_acc,
                'val/top5_accuracy': val_top5,
                'lr': optimizer.param_groups[0]['lr']
            }
            for k, v in per_class_acc.items():
                log_dict[f'val/{k}'] = v
            wandb.log(log_dict)

        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(checkpoint_dir / 'best.pth', epoch, model, optimizer, best_acc, scheduler)

        print(f"[Epoch {epoch:3d}] | TrLoss {train_loss:.4f} | TrAcc {train_acc:.2f}% || VaLoss {val_loss:.4f} | VaAcc {val_acc:.2f}% | Top5 {val_top5:.2f}% | LR: {optimizer.param_groups[0]['lr']:.6f}")

        scheduler.step()

        save_checkpoint(checkpoint_dir / 'latest.pth', epoch, model, optimizer, best_acc, scheduler)

    if not args.no_wandb:
        wandb.finish()

    print(f"Training complete!")
    print(f"Best Val Accuracy: {best_acc:.2f}%")

if __name__ == '__main__':
    main()