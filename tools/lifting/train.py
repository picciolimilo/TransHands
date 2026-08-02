#!/usr/bin/env python3
"""Training script for TransHands."""

import torch
import yaml
from pathlib import Path
import argparse
import sys
import math
import numpy as np
import wandb
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.transhands import TransHands
from src.training.losses import WeightedMPJPE, LiftingMaskedModelingLoss
from src.data.lifting.transforms import custom_collate_fn
from torch.utils.data import DataLoader
from src.training.schedulers import WarmupCosineLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from src.training.reproducibility import set_seed, seed_worker


def load_config(config_path):
    with open(config_path) as f:
        return yaml.safe_load(f)

### Metrics ###
def compute_mpjpe_mm(pred, target, scales):
    if isinstance(scales, list):
        scales = [float(s) if s is not None else 120.0 for s in scales]
    scales_tensor = torch.tensor(scales, device=pred.device).view(-1, 1, 1, 1)
    pred_mm = pred * scales_tensor
    target_mm = target * scales_tensor
    return torch.mean(torch.norm(pred_mm - target_mm, dim=-1)).item()

def compute_pa_mpjpe(pred, target, scales):
    """Procustes Alignment MPJPE"""
    scales_tensor = torch.tensor(scales, device=pred.device).view(-1, 1, 1, 1)
    p = (pred * scales_tensor).reshape(-1, 21, 3)
    t = (target * scales_tensor).reshape(-1, 21, 3)

    p_centered = p - p.mean(dim=1, keepdim=True)
    t_centered = t - t.mean(dim=1, keepdim=True)

    # Kabsch Algorithm
    H = torch.matmul(p_centered.transpose(1, 2), t_centered)
    U, S, V = torch.linalg.svd(H)
    R = torch.matmul(V.transpose(1, 2), U.transpose(1, 2))

    p_aligned = torch.matmul(p_centered, R.transpose(1, 2))
    error = torch.norm(p_aligned - t_centered, dim=-1).mean()
    
    return error.item()

def compute_pck(pred, target, scales, threshold=20.0):
    scales_tensor = torch.tensor(scales, device=pred.device).view(-1, 1, 1, 1)
    pred_mm = pred * scales_tensor
    target_mm = target * scales_tensor
    errors = torch.norm(pred_mm - target_mm, dim=-1)
    correct = (errors < threshold).float()
    return correct.mean().item() * 100 

def compute_auc(pred, target, scales, max_threshold=50.0):
    """Area Under Curve for PCK over multiple thresholds."""
    scales_tensor = torch.tensor(scales, device=pred.device).view(-1, 1, 1, 1)
    pred_mm = pred * scales_tensor
    target_mm = target * scales_tensor
    errors = torch.norm(pred_mm - target_mm, dim=-1)
    thresholds = np.arange(0, max_threshold + 1, 1.0)
    pck_values = []
    for thresh in thresholds:
        pck = (errors < thresh).float().mean().item()
        pck_values.append(pck)
    auc = np.trapz(pck_values, thresholds) / max_threshold
    return auc

### Training Loop ###
def train_epoch(model, loader, optimizer, criterion_self, device, scaler, loss_mode='supervised', mmm_weight=0.1, grad_clip=0.2):
    model.train()
    total_loss, total_mpjpe, total_pck = 0, 0, 0
    num_batches = 0

    criterion_lift = WeightedMPJPE().to(device)

    target_batch_size = 32
    physical_batch_size = loader.batch_size
    grad_accum_steps = max(1, target_batch_size // physical_batch_size)
    
    for batch_idx, batch in enumerate(loader):
        if batch is None:  
            continue
        
        if batch_idx % grad_accum_steps == 0:
            optimizer.zero_grad(set_to_none=True)
        
        loss_log_val = 0
        batch_mpjpe = 0
        batch_pck = 0
        num_hands = 0

        # Process both hands
        for side in ['left', 'right']:
            input_key = f"{side}_hand_input"
            target_key = f"{side}_hand_target"
            scale_key = f"{side}_scales"
            
            if input_key in batch:
                input_2d = batch[input_key].to(device, non_blocking=True)
                target_3d = batch[target_key].to(device, non_blocking=True)
                
                # Supervised Lifting Loss
                with autocast('cuda'):
                    pred_3d = model(input_2d)

                    # Handle sequence length mismatch (e.g. PoseFormerV2 outputting 1 frame)
                    if pred_3d.shape[1] != target_3d.shape[1]:
                         if pred_3d.shape[1] == 1:
                             mid = target_3d.shape[1] // 2
                             target_3d = target_3d[:, mid:mid+1]

                    loss_lift = criterion_lift(pred_3d, target_3d) / grad_accum_steps
                
                scaler.scale(loss_lift).backward()
                loss_log_val += loss_lift.item() * grad_accum_steps

                # Self-Supervised L-MMM Loss
                if loss_mode == 'lmmm':
                    with autocast('cuda'):
                        loss_self = criterion_self(model, input_2d, target_3d)
                        loss_self = (loss_self * mmm_weight) / grad_accum_steps
                    
                    scaler.scale(loss_self).backward()
                    loss_log_val += loss_self.item() * grad_accum_steps

                with torch.no_grad():
                    scales = batch[scale_key]
                    batch_mpjpe += compute_mpjpe_mm(pred_3d, target_3d, scales)
                    batch_pck += compute_pck(pred_3d, target_3d, scales, threshold=20.0)
                    num_hands += 1

        # Optimizer step
        if (batch_idx + 1) % grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
 
        total_loss += loss_log_val
        total_mpjpe += batch_mpjpe / num_hands if num_hands > 0 else 0
        total_pck += batch_pck / num_hands if num_hands > 0 else 0
        num_batches += 1

    if num_batches == 0:
        return 0.0, 0.0, 0.0

    return total_loss / num_batches, total_mpjpe / num_batches, total_pck / num_batches

def validate(model, loader, device, compute_expensive_metrics=False):
    model.eval()
    criterion_lift = WeightedMPJPE().to(device)

    total_loss, total_mpjpe, total_pa, total_pck, total_auc = 0.0, 0.0, 0.0, 0.0, 0.0
    num_batches = 0

    # Per-dataset tracking for multi-dataset runs
    per_dataset_mpjpe = {}
    per_dataset_count = {}

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue

            loss, b_mpjpe, b_pa, b_pck, b_auc, n_hands = 0.0, 0.0, 0.0, 0.0, 0.0, 0
            origins = batch.get("dataset_origins", None)

            for side in ['left', 'right']:
                if f"{side}_hand_input" in batch:
                    input_2d = batch[f"{side}_hand_input"].to(device, non_blocking=True)
                    target_3d = batch[f"{side}_hand_target"].to(device, non_blocking=True)
                    scales = batch[f"{side}_scales"]

                    pred_3d = model(input_2d)

                    if pred_3d.shape[1] != target_3d.shape[1]:
                        if pred_3d.shape[1] == 1:
                            mid = target_3d.shape[1] // 2
                            target_3d = target_3d[:, mid:mid+1]

                    loss += criterion_lift(pred_3d, target_3d)

                    mpjpe_val = float(compute_mpjpe_mm(pred_3d, target_3d, scales))
                    b_mpjpe += mpjpe_val
                    b_pck += float(compute_pck(pred_3d, target_3d, scales, threshold=20.0))

                    if compute_expensive_metrics:
                        b_pa += float(compute_pa_mpjpe(pred_3d, target_3d, scales))
                        b_auc += float(compute_auc(pred_3d, target_3d, scales))

                    # Track per-dataset metrics
                    if origins:
                        origin = origins[0] if len(set(origins)) == 1 else "Mixed"
                        per_dataset_mpjpe[origin] = per_dataset_mpjpe.get(origin, 0.0) + mpjpe_val
                        per_dataset_count[origin] = per_dataset_count.get(origin, 0) + 1

                    n_hands += 1

            total_loss += loss.item() if hasattr(loss, 'item') else loss

            if n_hands > 0:
                total_mpjpe += b_mpjpe / n_hands
                total_pck += b_pck / n_hands
                if compute_expensive_metrics:
                    total_pa += b_pa / n_hands
                    total_auc += b_auc / n_hands
            num_batches += 1

    # Compute per-dataset averages
    per_dataset_avg = {}
    for ds, total in per_dataset_mpjpe.items():
        per_dataset_avg[ds] = total / per_dataset_count[ds]

    if num_batches == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, per_dataset_avg

    return (total_loss / num_batches,
            total_mpjpe / num_batches,
            total_pa / num_batches if compute_expensive_metrics else 0.0,
            total_pck / num_batches,
            total_auc / num_batches if compute_expensive_metrics else 0.0,
            per_dataset_avg)

def save_checkpoint(path, epoch, model, optimizer, best_val, scheduler=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_mpjpe': best_val
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
def get_subset_deterministic(dataset, fraction, seed=42):
    if fraction >= 1.0:
        return dataset
    num_samples = int(len(dataset) * fraction)
    g = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=g)[:num_samples].tolist()
    return torch.utils.data.Subset(dataset, indices)

def get_dataset(config, split):
    dataset_name = config['dataset']['name'].lower()
    root_dir = config['dataset']['root']
    seq_len = config['dataset']['seq_len']
    fraction = config['dataset'].get('fraction', 1.0)

    if dataset_name == 'reinterhand':
        from src.data.lifting.reinterhand import ReInterHandDataset
        stride = config['dataset'].get('stride', seq_len)
        min_valid_frames = config['dataset'].get('min_valid_frames', seq_len // 2)
        augmentation = config['dataset'].get('augmentation', False)
        use_2d = config['dataset'].get('use_2d', True)
        ds = ReInterHandDataset(root_dir=root_dir, split=split, seq_len=seq_len, stride=stride, min_valid_frames=min_valid_frames, augmentation=augmentation, use_2d=use_2d)
    elif dataset_name == 'assemblyhands':
        from src.data.lifting.assemblyhands import AssemblyHandsDataset
        stride = config['dataset'].get('stride', seq_len)
        min_valid_frames = config['dataset'].get('min_valid_frames', seq_len // 2)
        augmentation = config['dataset'].get('augmentation', False)
        ds = AssemblyHandsDataset(root_dir=root_dir, split=split, seq_len=seq_len, stride=stride, min_valid_frames=min_valid_frames, augmentation=augmentation)
    elif dataset_name == 'gigahands':
        from src.data.lifting.gigahands import GigaHandsDataset
        stride = config['dataset'].get('stride', seq_len)
        min_valid_frames = config['dataset'].get('min_valid_frames', seq_len // 2)
        use_2d = config['dataset'].get('use_2d', True)
        augmentation = config['dataset'].get('augmentation', False)
        ds = GigaHandsDataset(root_dir=root_dir, split=split, seq_len=seq_len, stride=stride, min_valid_frames=min_valid_frames, use_2d=use_2d, augmentation=augmentation)
    else:
        raise ValueError(f"Dataset '{dataset_name}' not recognized!")
        
    if split == 'train':
        return get_subset_deterministic(ds, fraction)
    return ds

def get_multi_dataset(config, split):
    """
    Load multiple datasets and combine them.
    
    Config format:
        dataset:  
          name: multi
          datasets:
            - name: reinterhand
              root: dataset/ReInterHand
              weight: 1.0
            - name: assemblyhands
              root: dataset/AssemblyHands
              weight: 1.0
            - name: gigahands
              root: dataset/GigaHands
              weight: 1.0
    
    Returns: 
        (dataset, weights) -> Tupla separata per il Sampler
    """
    from src.data.lifting.multi_dataset import MultiHandDataset
    
    datasets_config = config['dataset']['datasets']
    datasets = []
    weights = []
    
    seq_len = config['dataset']['seq_len']
    stride = config['dataset'].get('stride', seq_len)
    min_valid = config['dataset'].get('min_valid_frames', seq_len // 2)
    aug = config['dataset'].get('augmentation', False)
    
    print(f"Loading Multi-Dataset ({split})...")
    
    for ds_cfg in datasets_config:
        ds_name = ds_cfg['name'].lower()
        root = ds_cfg['root']
        w = ds_cfg.get('weight', 1.0)
        use_2d = ds_cfg.get('use_2d', True)
        
        if ds_name == 'reinterhand':
            from src.data.lifting.reinterhand import ReInterHandDataset
            ds = ReInterHandDataset(root, split, seq_len, stride, min_valid, aug, use_2d)
        elif ds_name == 'assemblyhands':
            from src.data.lifting.assemblyhands import AssemblyHandsDataset
            ds = AssemblyHandsDataset(root, split, seq_len, stride, min_valid, aug, use_2d)
        elif ds_name == 'gigahands':
            from src.data.lifting.gigahands import GigaHandsDataset
            ds = GigaHandsDataset(root, split, seq_len, stride, min_valid, use_2d, aug)
        else:
            raise ValueError(f"Unknown: {ds_name}")
            
        fraction = ds_cfg.get('fraction', config['dataset'].get('fraction', 1.0))
        if split == 'train':
            ds = get_subset_deterministic(ds, fraction)
            
        datasets.append(ds)
        weights.append(w)

    return MultiHandDataset(datasets), weights

def main():
    set_seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/lifting/MotionBERT/motionbert_stage1.yaml')
    parser.add_argument('--resume', default=None, type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--use-lmmm', action='store_true', help='Use L-MMM (2D masking)')
    parser.add_argument('--no-wandb', action='store_true', help='Disable WandB logging')
    parser.add_argument('--wandb-project', default='TransHands-Pretrain', type=str, help='WandB project name')
    parser.add_argument('--wandb-name', default=None, type=str, help='WandB run name')
    parser.add_argument('--balanced-sampling', action='store_true', help='Use balanced sampling for multi-dataset')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler()

    print("=== TransHands Training ===")

    if not args.no_wandb:
        run_name = args.wandb_name or f"{config['dataset']['name']}_{config['training']['epochs']}ep"    
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=config,
            tags=[config['dataset']['name']]
        )

    dataset_name = config['dataset']['name'].lower()
    
    if dataset_name == 'multi':  
        train_dataset, train_weights = get_multi_dataset(config, split='train')
        val_dataset, _ = get_multi_dataset(config, split='val')
        train_sampler = None
        if args.balanced_sampling:
            from src.data.lifting.multi_dataset import BalancedMultiDatasetSampler
            epoch_size = config['training'].get('epoch_size', len(train_dataset))
            print(f"Initializing BalancedMultiDatasetSampler with weights: {train_weights}")
            train_sampler = BalancedMultiDatasetSampler(train_dataset, weights=train_weights, epoch_size=epoch_size)
        val_sampler = None
    else:  
        train_dataset = get_dataset(config, split='train')
        val_dataset = get_dataset(config, split='val')
        train_sampler = None
        val_sampler = None

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    g = torch.Generator()
    g.manual_seed(42)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=custom_collate_fn,
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
        sampler=val_sampler,
        collate_fn=custom_collate_fn,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
        worker_init_fn=seed_worker,
        generator=g
    )

    # Model Init
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

    # Optimizer & Unfreezing Logic
    if config['model']['freeze_encoder']:
        print(">>> Encoder Frozen.")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['training']['lr'],
            weight_decay=0.01
        )
    else:
        print(">>> Encoder Partially Unfrozen.")
        backbone_params = []
        head_params = []

        for name, param in model.named_parameters():
            if any(x in name for x in ['head', 'projection', 'adapter', 'embed']):
                if 'embedding' in name and 'encoder' in name:
                     param.requires_grad = False
                else:
                     param.requires_grad = True
                     head_params.append(param)
                     print(f"Unfrozen (Head/Adapter): {name}")
            elif 'encoder' in name:
                if config['model']['encoder_type'] == 'motionbert':
                    if 'blocks_st.4' in name or 'blocks_ts.4' in name:
                        param.requires_grad = True
                        backbone_params.append(param)
                        print(f"Unfrozen Backbone: {name}")
                    else:
                        param.requires_grad = False
                elif config['model']['encoder_type'] == 'mixste':
                    if 'blocks.6' in name or 'blocks.7' in name:
                        param.requires_grad = True
                        backbone_params.append(param)
                        print(f"Unfrozen Backbone: {name}")
                    else:
                        param.requires_grad = False
                elif config['model']['encoder_type'] == 'poseformerv2':
                    if '.3.' in name and not any(idx in name for idx in ['.0.', '.1.', '.2.']):
                        param.requires_grad = True
                        backbone_params.append(param)
                        print(f"Unfrozen Backbone: {name}")
                    else:
                        param.requires_grad = False
                elif config['model']['encoder_type'] == 'stgcn':
                    if 'st_gcn_networks.8' in name or 'st_gcn_networks.9' in name:
                         param.requires_grad = True
                         backbone_params.append(param)
                         print(f"Unfrozen Backbone: {name}")
                    else:
                         param.requires_grad = False
                else:
                    raise ValueError(f"Unknown encoder type: {config['model']['encoder_type']}")
            else:
                param.requires_grad = True
                head_params.append(param)
        
        lr_backbone = config['training'].get('lr_backbone', 1e-6)
        print(f"    Backbone LR: {lr_backbone} | Head LR: {config['training']['lr']}")

        param_groups = [
            {'params': backbone_params, 'lr': lr_backbone},
            {'params': head_params, 'lr': config['training']['lr']}
        ]
        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=0.01
        )
    
    params = model.count_parameters()
    print(f"Parameters: {params['trainable']: ,} / {params['total']:,} trainable")
    
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=config['training']['warmup_epochs'],
        max_epochs=config['training']['epochs'],
        min_lr=1e-6
    )

    loss_mode = 'supervised'  # default
    criterion_self = None
    if args.use_lmmm:
        criterion_self = LiftingMaskedModelingLoss().to(device)
        loss_mode = 'lmmm'
        print(">>> Using L-MMM (Self-Supervised).")
    
    else:
        print(">>> Using Supervised Lifting Only.")
    
    best_val_mpjpe = float('inf')
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
            print(">>> Resetting Best Val MPJPE to Infinity and starting from Epoch 1.")
            best_val_mpjpe = float('inf')
            start_epoch = 1
        else:
            best_val_mpjpe = checkpoint.get('best_val_mpjpe', float('inf'))
            start_epoch = checkpoint['epoch'] + 1
            print(f">>> RESUMING SAME RUN: Keeping previous Best Val MPJPE: {best_val_mpjpe:.2f} mm")
            
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
    mmm_weight = config['training'].get('mmm_weight', 0.25)
    grad_clip = config['training'].get('grad_clip', 0.2)

    # Early Stopping Configuration
    # early_stopping_patience = config['training'].get('early_stopping_patience', None)
    # patience_counter = 0

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        train_loss, train_mpjpe, train_pck = train_epoch(model, train_loader, optimizer, criterion_self, device, scaler, loss_mode=loss_mode, mmm_weight=mmm_weight, grad_clip=grad_clip)

        compute_expensive = (epoch % 10 == 0) or (epoch == config['training']['epochs'])
        val_loss, val_mpjpe, val_pa_mpjpe, val_pck, val_auc, per_ds = validate(model, val_loader, device, compute_expensive_metrics=compute_expensive)

        if not args.no_wandb:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_loss,
                'train/mpjpe_mm': train_mpjpe,
                'val/loss': val_loss,
                'val/mpjpe_mm': val_mpjpe,
                'val/pck': val_pck,
                'lr': optimizer.param_groups[0]['lr']
            }
            if compute_expensive:
                log_dict['val/pa_mpjpe_mm'] = val_pa_mpjpe
                log_dict['val/auc'] = val_auc
            for ds_name, ds_mpjpe in per_ds.items():
                log_dict[f'val/{ds_name}_mpjpe'] = ds_mpjpe
            wandb.log(log_dict)

        if val_mpjpe < best_val_mpjpe:
            best_val_mpjpe = val_mpjpe
            # patience_counter = 0
            save_checkpoint(checkpoint_dir / 'best.pth', epoch, model, optimizer, best_val_mpjpe, scheduler)
        # else:
        #     if early_stopping_patience is not None:
        #         patience_counter += 1

        ds_str = " | ".join(f"{k}: {v:.2f}" for k, v in per_ds.items()) if per_ds else ""
        print(f"[Epoch {epoch:3d}] | Tr MPJPE: {train_mpjpe:.2f} | Val MPJPE: {val_mpjpe:.2f} | Val PCK: {val_pck:.2f} | LR: {optimizer.param_groups[0]['lr']:.6f}")
        if ds_str:
            print(f"    >>> Per-Dataset: {ds_str}")
        if compute_expensive:
            print(f"    >>> PA-MPJPE: {val_pa_mpjpe:.2f} mm | AUC: {val_auc:.4f}")

        scheduler.step()

        save_checkpoint(checkpoint_dir / 'latest.pth', epoch, model, optimizer, best_val_mpjpe, scheduler)

        # Optional: save checkpoint every 10 epochs
        # if epoch % 10 == 0:
        #     save_checkpoint(checkpoint_dir / f'epoch_{epoch}.pth', epoch, model, optimizer, best_val_mpjpe, scheduler)

        # Early stopping check
        # if early_stopping_patience is not None and patience_counter >= early_stopping_patience:
        #     print(f"\n>>> Early stopping triggered at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
        #     break

    if not args.no_wandb:
        wandb.finish()

    print(f"Training complete!")
    print(f"Best Val MPJPE: {best_val_mpjpe:.2f} mm")

if __name__ == '__main__':
    main()