import torch
import torch.nn as nn
import torch.nn.functional as F


def _load_transhands_checkpoint_compatible(transhands, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model_state = transhands.state_dict()

    ckpt_tpe_key = 'encoder.Temporal_pos_embed'
    if ckpt_tpe_key in state_dict and ckpt_tpe_key in model_state:
        src = state_dict[ckpt_tpe_key]
        dst = model_state[ckpt_tpe_key]
        if src.shape != dst.shape and src.dim() == 3 and dst.dim() == 3 and src.shape[0] == dst.shape[0] and src.shape[2] == dst.shape[2]:
            src_t = src.permute(0, 2, 1)
            src_t = F.interpolate(src_t, size=dst.shape[1], mode='linear', align_corners=False)
            state_dict[ckpt_tpe_key] = src_t.permute(0, 2, 1).to(dtype=dst.dtype)
            print(f"[Checkpoint] Interpolated {ckpt_tpe_key}: {tuple(src.shape)} -> {tuple(dst.shape)}")

    filtered = {}
    dropped = []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape != v.shape:
            dropped.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        filtered[k] = v

    if dropped:
        print("[Checkpoint] Dropping incompatible keys:")
        for k, s1, s2 in dropped:
            print(f"  - {k}: ckpt{s1} vs model{s2}")

    transhands.load_state_dict(filtered, strict=False)


class TransHandsGesture(nn.Module):
    """TransHands + Gesture Recognition Head."""
    def __init__(self, transhands_model, gesture_head, extraction_point='encoder', freeze_transhands=True, use_velocity=False):
        super().__init__()
        self.transhands = transhands_model
        self.gesture_head = gesture_head
        self.extraction_point = extraction_point
        self.use_velocity = use_velocity
        
        if freeze_transhands:
            for name, param in self.transhands.named_parameters():
                if 'adapter_in' not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            print("[Gesture] TransHands backbone frozen (EXCEPT adapter_in)")
            
    def extract_features(self, x):
        transhands_trainable = any(p.requires_grad for p in self.transhands.parameters())
        
        if self.extraction_point == 'encoder':
            with torch.set_grad_enabled(transhands_trainable):
                B, T = x.shape[:2]

                if x.shape[-1] == 2 and self.transhands.encoder_type in ['motionbert', 'stgcn']:
                    confidence = torch.ones(B, T, 21, 1, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, confidence], dim=-1)

                x_body = self.transhands.adapter_in(x)

                if self.transhands.encoder_type == 'motionbert':
                    emb = self.transhands.encoder(x_body, return_rep=True)
                    encoder_features = emb.mean(dim=2)
                elif self.transhands.encoder_type == 'poseformerv2':
                    with torch.amp.autocast('cuda', enabled=False):
                        emb = self.transhands.encoder(x_body.float())
                    if isinstance(emb, (list, tuple)): emb = emb[-1]
                    encoder_features = emb.view(B, emb.shape[1], -1)
                elif self.transhands.encoder_type == 'stgcn':
                    feat_in = x_body.permute(0, 3, 1, 2).unsqueeze(-1)
                    emb = self.transhands.encoder(feat_in)
                    if isinstance(emb, (list, tuple)): emb = emb[-1]
                    encoder_features = emb.view(B, emb.shape[1], -1)
                else:
                    emb = self.transhands.encoder(x_body)
                    if isinstance(emb, (list, tuple)): emb = emb[-1]
                    encoder_features = emb.view(B, emb.shape[1], -1)

            return encoder_features
        
        elif self.extraction_point == 'latent':
            with torch.set_grad_enabled(transhands_trainable):
                B, T = x.shape[:2]

                if x.shape[-1] == 2 and self.transhands.encoder_type in ['motionbert', 'stgcn']:
                    confidence = torch.ones(B, T, 21, 1, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, confidence], dim=-1)

                x_body = self.transhands.adapter_in(x)

                if self.transhands.encoder_type == 'poseformerv2':
                    with torch.amp.autocast('cuda', enabled=False):
                        emb = self.transhands.encoder(x_body.float())
                elif self.transhands.encoder_type == 'stgcn':
                    feat_in = x_body.permute(0, 3, 1, 2).unsqueeze(-1)
                    emb = self.transhands.encoder(feat_in)
                else:
                    emb = self.transhands.encoder(x_body)

                if isinstance(emb, (list, tuple)): emb = emb[-1]

                curr_T = emb.shape[1]
                emb_flat = emb.view(B, curr_T, -1)
                latent_features = self.transhands.projection(emb_flat)

            if self.use_velocity:
                velocity = torch.zeros_like(latent_features)
                velocity[:, 1:, :] = latent_features[:, 1:, :] - latent_features[:, :-1, :]
                latent_features = torch.cat([latent_features, velocity], dim=-1)
            
            return latent_features
        
        elif self.extraction_point == 'geometric':
            with torch.set_grad_enabled(transhands_trainable):
                output_3d = self.transhands(x)

            B, T = output_3d.shape[:2]
            output_3d = output_3d.view(B, T, -1)

            if self.use_velocity:
                velocity = torch.zeros_like(output_3d)
                velocity[:, 1:, :] = output_3d[:, 1:, :] - output_3d[:, :-1, :]
                output_3d = torch.cat([output_3d, velocity], dim=-1)
            
            return output_3d
        else:
            raise ValueError(f"Unknown extraction point: {self.extraction_point}")

    def forward(self, x, x_sub2=None, domain_knowledge=None):
        """
        x: Input of the Master camera (or Single-View). Shape: (B, T, 21, 2)
        x_sub2: Optional input from the Sub2 camera. Shape: (B, T, 21, 2)
        """
        features_master = self.extract_features(x)
        
        if x_sub2 is not None:
            features_sub2 = self.extract_features(x_sub2)
            features = (features_master + features_sub2) / 2.0
        else:
            features = features_master
        
        if hasattr(self.gesture_head, 'distance_tcn') or hasattr(self.gesture_head, 'domain_tcn'):
            logits = self.gesture_head(features, domain_knowledge=domain_knowledge) 
        else:
            logits = self.gesture_head(features)
            
        return logits