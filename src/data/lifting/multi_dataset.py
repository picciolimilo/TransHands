"""Multi-Dataset Wrapper and Balanced Sampler."""
import torch
import numpy as np
from torch.utils.data import Dataset, Sampler

class MultiHandDataset(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.total_length = sum(self.lengths)
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.dataset_names = [ds.__class__.__name__ for ds in datasets]
        
        print(f"\n[MultiDataset] Combined {len(datasets)} datasets (Total samples: {self.total_length:,})")
        for name, length in zip(self.dataset_names, self.lengths):
            print(f"  - {name}: {length:,}")

    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Map global index to specific dataset
        dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
        local_idx = idx - self.cumulative_lengths[dataset_idx]

        max_retries = 10
        for attempt in range(max_retries):
            data = self.datasets[dataset_idx][local_idx]
            
            if data is None:
                # Try next index
                idx = (idx + 1) % self.total_length
                dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
                local_idx = idx - self.cumulative_lengths[dataset_idx]
                continue
            
            valid_sides = []
            if data.get('left_hand_input') is not None: valid_sides.append('left')
            if data.get('right_hand_input') is not None: valid_sides.append('right')
            
            if not valid_sides: 
                idx = (idx + 1) % self.total_length
                dataset_idx = np.searchsorted(self.cumulative_lengths[1:], idx, side='right')
                local_idx = idx - self.cumulative_lengths[dataset_idx]
                continue
                
            # Valid sample found
            side = np.random.choice(valid_sides)
            dataset_name = self.datasets[dataset_idx].__class__.__name__.replace('Dataset', '')
            
            return {
                'left_hand_input': data[f'{side}_hand_input'],   
                'left_hand_target': data[f'{side}_hand_target'], 
                'left_scale_3d': data.get(f'{side}_scale_3d', 100.0),
                'dataset_origin': dataset_name,               
                'side_orig': side
            }

        return None

class BalancedMultiDatasetSampler(Sampler):
    def __init__(self, multi_dataset, weights=None, epoch_size=None):
        self.multi_dataset = multi_dataset
        self.num_datasets = len(multi_dataset.datasets)
        
        # If weights are not specified, use equal balancing (1/N)
        if weights is None:
            self.weights = np.ones(self.num_datasets) / self.num_datasets
        else:
            self.weights = np.array(weights) / sum(weights)
            
        self.epoch_size = epoch_size if epoch_size else len(multi_dataset)
        
        # How many samples to draw from each dataset to form an epoch
        self.samples_per_dataset = (self.weights * self.epoch_size).astype(int)
        
        print(f"\n[BalancedSampler] Epoch Size: {self.epoch_size:,}")
        for i, count in enumerate(self.samples_per_dataset):
            name = multi_dataset.dataset_names[i]
            print(f"  - {name}: {count:,} samples/epoch")

    def __iter__(self):
        indices = []
        for i in range(self.num_datasets):
            length = self.multi_dataset.lengths[i]
            start_offset = self.multi_dataset.cumulative_lengths[i]
            count = self.samples_per_dataset[i]
            
            # Oversampling / Undersampling logic
            replace = count > length
            local_indices = np.random.choice(length, count, replace=replace)
            
            global_indices = local_indices + start_offset
            indices.extend(global_indices)
        
        indices_tensor = torch.tensor(indices)
        shuffled_idx = torch.randperm(len(indices_tensor)).tolist()
        indices = indices_tensor[shuffled_idx].tolist()

        return iter(indices)

    def __len__(self):
        return self.epoch_size