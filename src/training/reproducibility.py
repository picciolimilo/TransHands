import os
import random
import numpy as np
import torch

def set_seed(seed=42):
    """Determinism on all libraries to ensure reproducibility."""
    random.seed(seed)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)

def seed_worker(worker_id):
    """Derive each worker's seed from its own initial_seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)