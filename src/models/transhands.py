"""
TransHands: Repurposing Human Pose Encoders as Hand Pose Encoders.

Reference:
    - Model: MotionBERT
    Paper: "MotionBERT: A Unified Perspective on Learning Human Motion Representations"
    - Model: MixSTE -> neeeds seq_len (not flexible as MotionBERT)
    Paper: "MixSTE: Seq2seq Mixed Spatio-Temporal Encoder for 3D Human Pose Estimation in Video"
    - Model: PoseFormerV2
    Paper: "PoseFormerV2: Exploring Frequency Domain for Efficient and Robust 3D Human Pose Estimation"
    - Model: ST-GCN
    Paper: "Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition"

Architecture: Input Adapter -> Encoder -> Projection -> Output Adapter
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from types import SimpleNamespace

from .adapters import HandAdapter, HandMLPAdapter, HandNeuralODEAdapter
from .projection import RetNetProjection, LinearProjection


class STGCN_Wrapper(nn.Module):
    """Wrapper for ST-GCN that handles feature extraction and temporal upsampling."""
    def __init__(self, stgcn_model, upsample_temporal=True):
        super().__init__()
        self.stgcn = stgcn_model
        self.upsample = upsample_temporal
        
    def forward(self, x):
        # x input shape: (N, C, T, V, M)
        
        # Data preparation
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.stgcn.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # Forward through GCN layers
        for gcn, importance, adaptive in zip(self.stgcn.st_gcn_networks, self.stgcn.edge_importance, self.stgcn.adaptive_graph):
            A_adapted = (self.stgcn.A + adaptive) * importance
            x, _ = gcn(x, A_adapted)
        
        # Output here is (N*M, 256, reduced_T, V)
        
        # Temporal Upsampling
        if self.upsample and T != x.shape[2]:
            x = torch.nn.functional.interpolate(
                x, 
                size=(T, V), 
                mode='bilinear', 
                align_corners=False
            )
        
        # Reshape for TransHands: (N, T, V, C) 
        x = x.permute(0, 2, 3, 1).contiguous()
        
        return x

class TransHands(nn.Module):
    """
    Main Model Wrapper.
    Adapts Body Pose Estimators (MotionBERT, MixSTE, etc.) for Hand Pose Lifting.
    """
    
    def __init__(
        self,
        num_hand_joints=21,
        num_body_joints=17,
        freeze_encoder=True,
        weights_path=None,
        encoder_type='motionbert',
        seq_len=150,
        **kwargs
    ):
        super().__init__()

        self.encoder_type = encoder_type.lower()
        self.num_hand_joints = num_hand_joints
        self.num_body_joints = num_body_joints
        self.freeze_encoder = freeze_encoder
        adapter_type = kwargs.get('adapter_type', 'ode')
        external_root = Path(__file__).resolve().parent.parent.parent / 'external'

        if self.encoder_type == 'motionbert':
            self.input_channels = 3  # (x, y, confidence)
        elif self.encoder_type == 'mixste' or self.encoder_type == 'poseformerv2':
            self.input_channels = 2  # (x, y) only
        elif self.encoder_type == 'stgcn':
            self.input_channels = 3  # (x, y, confidence)
            if num_body_joints != 18:
                num_body_joints = 18  # ST-GCN Kinetics uses 18 joints
            self.num_body_joints = num_body_joints
        else:
            raise ValueError(f"Unknown encoder:  {self.encoder_type}")

        # 21 hand joints -> N body joints
        if adapter_type == 'mlp':
            self.adapter_in = HandMLPAdapter(
                in_channels=self.input_channels,
                num_hand_joints=num_hand_joints,
                num_body_joints=num_body_joints,
                embed_dim=128
            )
        else:
            self.adapter_in = HandNeuralODEAdapter(
                in_channels=self.input_channels,
                num_hand_joints=num_hand_joints,
                num_body_joints=num_body_joints,
                embed_dim=128
            )

        # Encoder initialization       
        if self.encoder_type == 'motionbert':
            print("Encoder: MotionBERT")
            mb_root = external_root / 'MotionBERT'
            if str(mb_root) not in sys.path: sys.path.insert(0, str(mb_root))

            from lib.model.DSTformer import DSTformer
            self.encoder = DSTformer(
                dim_in=3, 
                dim_out=3, 
                dim_feat=512, 
                dim_rep=512
            )
            proj_in_dim = num_body_joints * 3

        elif self.encoder_type == 'mixste':
            print(f"Encoder: MixSTE (seq_len={seq_len})")
            ms_root = external_root / 'MixSTE'
            if str(ms_root) not in sys.path: sys.path.insert(0, str(ms_root))
            
            from common.model_cross import MixSTE2
            self.encoder = MixSTE2(
                num_frame=seq_len,
                num_joints=num_body_joints,
                in_chans=2,
                embed_dim_ratio=512,
                depth=8, num_heads=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_path_rate=0.1
            )
            proj_in_dim = num_body_joints * 3

        elif self.encoder_type == 'poseformerv2':
            print(f"Encoder: PoseFormerV2 (seq_len={seq_len})")
            pf_root = external_root / 'PoseFormerV2'
            if str(pf_root) not in sys.path: sys.path.insert(0, str(pf_root))
            
            from common.model_poseformer import PoseTransformerV2
            kept_frames = 27
            print(f"PoseFormerV2: Setting kept_frames to {kept_frames} for cuFFT compatibility")

            pf_args = SimpleNamespace(
                layers=4,
                depth=4, 
                embed_dim_ratio=32,
                frames=seq_len,
                num_joints=num_body_joints,
                out_joints=num_body_joints,
                number_of_kept_frames=kept_frames,
                number_of_kept_coeffs=kept_frames,
                n_heads=8,
                channel=2,
                down_rate=1,
                subset_list=[1]
            )
            
            self.encoder = PoseTransformerV2(
                num_frame=seq_len,
                num_joints=num_body_joints,
                in_chans=2,
                num_heads=8,
                args=pf_args 
            )
            proj_in_dim = num_body_joints * 3

        elif self.encoder_type == 'stgcn':
            print(f"Encoder: ST-GCN ")
            stgcn_root = external_root / 'st-gcn'
            if str(stgcn_root) not in sys.path: sys.path.insert(0, str(stgcn_root))

            from net.st_gcn import Model as STGCN           
            graph_args = {'layout': 'openpose', 'strategy': 'spatial'}
            raw_stgcn = STGCN (
                in_channels=3,
                num_class=400,
                graph_args=graph_args,
                edge_importance_weighting=True
            )
            self.encoder = STGCN_Wrapper(raw_stgcn, upsample_temporal=True)
            proj_in_dim = num_body_joints * 256

        else:
            raise ValueError(f"Encoder '{self.encoder_type}' not supported.")
        
        if weights_path:
            print(f"Loading {self.encoder_type} weights from {weights_path}")
            self._load_encoder_weights(weights_path)
        
        if freeze_encoder:
            for name, param in self.encoder.named_parameters():
                if self.encoder_type == 'stgcn':
                    if any(key in name for key in ['.gcn', 'edge_importance', 'adaptive_graph']):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
                else:
                    param.requires_grad = False

        projection_type = kwargs.get('projection_type', 'retnet')
        if projection_type == 'linear':
            self.projection = LinearProjection(in_dim=proj_in_dim, out_dim=256)
        else:
            self.projection = RetNetProjection(in_dim=proj_in_dim, out_dim=256, num_heads=4)
        self.adapter_out = HandAdapter(dim=256, hidden=512, num_joints=num_hand_joints)
    
    def _load_encoder_weights(self, path):
        ckpt_path = Path(path)
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            
            if 'model_state_dict' in ckpt:
                state_dict = ckpt['model_state_dict']
            elif 'model_pos' in ckpt:
                state_dict = ckpt['model_pos']
            elif 'model' in ckpt:
                state_dict = ckpt['model']
            elif 'state_dict' in ckpt:
                state_dict = ckpt['state_dict']
            else:
                state_dict = ckpt

            if list(state_dict.keys())[0].startswith('module.'):
                 state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            model_dict = self.encoder.state_dict()
            new_state_dict = {}

            for k, v in state_dict.items():
                if self.encoder_type == 'stgcn':
                    new_key = 'stgcn.' + k if not k.startswith('stgcn.') else k

                    if '.gcn' in new_key or 'edge_importance' in new_key:
                        continue
                else:
                    new_key = k.replace('encoder.', '').replace('backbone.', '').replace('dstformer.', '')
                
                new_state_dict[new_key] = v

            filtered_dict = {k: v for k, v in new_state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
            
            self.encoder.load_state_dict(filtered_dict, strict=False)
            
            loaded_pct = (len(filtered_dict) / len(model_dict)) * 100 if len(model_dict) > 0 else 0
            print(f"Loaded Encoder weights ({loaded_pct:.1f}% layers matched)")
            
            if loaded_pct < 10.0:
                 print(f">>> WARNING: Very low weight match for {self.encoder_type}. Check model dimensions!")
        else:
            print(f"Checkpoint not found at {ckpt_path}")

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': trainable,
            'trainable_pct': 100 * trainable / total if total > 0 else 0
        }
    
    def forward(self, x):
        """
        Forward pass for Lifting (2D -> 3D).
        
        Args:
            x: (B, T, 21, 2) Input 2D Keypoints
        Returns:
            out: (B, T, 21, 3) Predicted 3D Pose
        """
        B, T = x.shape[:2]

        # Handle input channels (2 vs 3)
        if x.shape[-1] == 2:
            if self.encoder_type in ['motionbert', 'stgcn']:
                confidence = torch.ones(B, T, 21, 1, device=x.device, dtype=x.dtype)
                x = torch.cat([x, confidence], dim=-1)  # (B, T, 21, 3)
            # MixSTE and PoseFormer keep (x, y) as is
        
        # Structured adapter: Keep (B, T, Joints, C) format
        x_body = self.adapter_in(x)
        
        # Encoder Forward
        if self.encoder_type == 'poseformerv2':
            with torch.amp.autocast('cuda', enabled=False):
                emb = self.encoder(x_body.float())
        elif self.encoder_type == 'stgcn':
            # Prepare format (N, C, T, V, M)
            feat_in = x_body.permute(0, 3, 1, 2).unsqueeze(-1)
            emb = self.encoder(feat_in)
        else:
            emb = self.encoder(x_body)
        
        # Handle cases where encoder returns list (intermediate layers)
        if isinstance(emb, (list, tuple)):
            emb = emb[-1]
        
        # Projection and Output Adapter
        curr_T = emb.shape[1]
        emb_flat = emb.view(B, curr_T, -1)
        
        proj = self.projection(emb_flat)
        out = self.adapter_out(proj)
        
        return out