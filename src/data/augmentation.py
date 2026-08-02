"""Data augmentation for hand pose sequences."""

import numpy as np


class HandPoseAugmentation: 
    
    def __init__(self, 
                 rotation_range=30,
                 scale_range=(0.8, 1.2),
                 noise_std=0.015,
                 temporal_jitter=True,
                 shear_prob=0.3,
                 shear_range=0.1,
                 flip_prob=0.5,
                 translate_range=0.1,
                 dropout_prob=0.0):
        
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std
        self.temporal_jitter = temporal_jitter
        self.shear_prob = shear_prob
        self.shear_range = shear_range
        self.flip_prob = flip_prob
        self.translate_range = translate_range
        self.dropout_prob = dropout_prob
    
    def __call__(self, pose):
        """
        Apply augmentation based on input dimensionality.
        
        Args:
            pose:  (T, 21, 2) for 2D or (T, 21, 3) for 3D
        Returns:
            augmented:  same shape as input
        """
        pose = pose.copy()

        is_2d = (pose.shape[-1] == 2)
        
        if is_2d:
            return self._augment_2d(pose)
        else:
            return self._augment_3d(pose)
    
    def _augment_2d(self, pose):
        """Augmentation for 2D pose (T, 21, 2)"""

        if np.random.rand() < 0.6:
            pose = self._random_rotation_2d(pose)
        if np.random.rand() < 0.6:
            scale = np.random.uniform(*self.scale_range)
            pose = pose * scale
        if np.random.rand() < 0.5:
            pose = self._random_translation_2d(pose)
        if np.random.rand() < 0.6:
            pose = self._add_noise_2d(pose)
        if self.temporal_jitter and np.random.rand() < 0.4:
            pose = self._temporal_jitter(pose)
        if self.dropout_prob > 0 and np.random.rand() < 0.3:
            pose = self._joint_dropout(pose)
        if np.random.rand() < self.flip_prob:
            pose = self._flip_horizontal_2d(pose)

        return pose
    
    def _augment_3d(self, pose):
        """Augmentation for 3D pose (T, 21, 3)"""

        if np.random.rand() < 0.6:
            pose = self._random_rotation_3d(pose)

        if np.random.rand() < 0.6:
            scale = np.random.uniform(*self.scale_range)
            pose = pose * scale

        if np.random.rand() < self.shear_prob:
            pose = self._shear_transform_3d(pose)

        if np.random.rand() < 0.6:
            pose = self._add_smooth_noise_3d(pose)

        if self.temporal_jitter and np.random.rand() < 0.4:
            pose = self._temporal_jitter(pose)

        if np.random.rand() < self.flip_prob:
            pose[:, :, 0] = -pose[:, :, 0]

        return pose

    # 2D Transformations
    def _random_rotation_2d(self, pose):
        """2D planar rotation (simulates camera roll)."""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.radians(angle)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # 2D rotation matrix
        R = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        
        return pose @ R.T
    
    def _random_translation_2d(self, pose):
        """Random translation (simulates bbox jitter)."""
        shift = np.random.uniform(-self.translate_range, self.translate_range, size=(2,))
        return pose + shift
    
    def _add_noise_2d(self, pose):
        """Add Gaussian noise, spatially correlated per finger so nearby joints move together."""
        T, J, D = pose.shape
        
        # Spatially correlated noise (same noise for nearby joints)
        noise_base = np.random.randn(T, 5, D) * self.noise_std  # 5 fingers
        
        noise = np.zeros((T, J, D))
        
        # Wrist
        noise[:, 0] = noise_base[:, 0]
        # Thumb (1-4)
        noise[:, 1:5] = noise_base[:, 0: 1] * 0.8 + np.random.randn(T, 4, D) * self.noise_std * 0.2
        # Index (5-8)
        noise[:, 5:9] = noise_base[:, 1:2] * 0.8 + np.random.randn(T, 4, D) * self.noise_std * 0.2
        # Middle (9-12)
        noise[:, 9:13] = noise_base[:, 2:3] * 0.8 + np.random.randn(T, 4, D) * self.noise_std * 0.2
        # Ring (13-16)
        noise[:, 13:17] = noise_base[:, 3:4] * 0.8 + np.random.randn(T, 4, D) * self.noise_std * 0.2
        # Pinky (17-20)
        noise[:, 17:21] = noise_base[:, 4:5] * 0.8 + np.random.randn(T, 4, D) * self.noise_std * 0.2
        
        return pose + noise
    
    def _flip_horizontal_2d(self, pose):
        """Horizontal flip (negates x only; does not remap joint indices)."""
        flipped = pose.copy()
        
        # Flip X coordinate
        flipped[:, : , 0] = -flipped[:, :, 0]
        
        return flipped
        
    def _joint_dropout(self, pose):
        """Randomly zero out joints to simulate occlusion (wrist is always kept)."""
        T, J = pose.shape[: 2]
        
        # Create dropout mask (always keep wrist visible)
        mask = np.random.rand(T, J) > self.dropout_prob
        mask[: , 0] = True  # Never drop wrist
        
        pose_dropped = pose.copy()
        pose_dropped[~mask] = 0.0
        
        return pose_dropped
    
    # 3D Transformations    
    def _random_rotation_3d(self, pose):
        """3D rotation around the Y-axis."""
        angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        angle_rad = np.radians(angle)
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        R = np.array([
            [cos_a, 0, sin_a],
            [0, 1, 0],
            [-sin_a, 0, cos_a]
        ])
        
        return pose @ R.T
    
    def _shear_transform_3d(self, pose):
        """3D shear transform (simulates viewpoint changes)."""
        shear = np.random.uniform(-self.shear_range, self.shear_range, size=2)

        shear_matrix = np.array([
            [1, shear[0], 0],
            [shear[1], 1, 0],
            [0, 0, 1]
        ])

        return pose @ shear_matrix.T
    
    def _add_smooth_noise_3d(self, pose):
        """Add smooth low-frequency noise plus a smaller high-frequency component."""
        T, J, D = pose.shape
        
        # Low-frequency component (smooth over time)
        t = np.linspace(0, 2*np.pi, T)
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi, size=(J, D))
        
        noise_smooth = np.zeros((T, J, D))
        for j in range(J):
            for d in range(D):
                noise_smooth[: , j, d] = np.sin(freq * t + phase[j, d]) * self.noise_std
        
        # High-frequency component
        noise_high = np.random.randn(T, J, D) * (self.noise_std / 2)
        
        return pose + noise_smooth + noise_high
    
    # Shared Transformation (works for both 2D and 3D)
    def _temporal_jitter(self, pose):
        """Temporal jitter via random-offset linear interpolation between frames."""
        T = len(pose)

        # Generate perturbed time indices
        indices_float = np.arange(T, dtype=np.float32)
        jitter = np.random.uniform(-0.5, 0.5, size=T)
        indices_float = np.clip(indices_float + jitter, 0, T - 1)
        
        # Linear interpolation
        pose_jittered = np.zeros_like(pose)
        for i, idx in enumerate(indices_float):
            idx_low = int(np.floor(idx))
            idx_high = min(int(np.ceil(idx)), T - 1)
            alpha = idx - idx_low
            
            if idx_low == idx_high:
                pose_jittered[i] = pose[idx_low]
            else:
                pose_jittered[i] = (1 - alpha) * pose[idx_low] + alpha * pose[idx_high]
        
        return pose_jittered