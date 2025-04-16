import os
import glob
import torch
from torch_geometric.data import Dataset, Data
from typing import Optional, List, Callable

class WDNLeakDataset(Dataset):
    """
    Custom PyG Dataset for loading processed leak localization scenarios.

    All configuration options (data directories, file patterns, indices, transforms, etc.)
    are passed as class constructor parameters for modularity and testability.

    Parameters
    ----------
    root : str
        Root data directory (default: './data')
    processed_dir : str
        Subdirectory for processed `.pt` files (default: 'processed')
    file_pattern : str
        Glob pattern for selecting `.pt` files (default: 'scenario_*.pt')
    indices : Optional[List[int]]
        Optional list of indices or file names to restrict the dataset (for cross-validation, debugging, etc.)
    transform : Optional[Callable]
        Optional transform function for data augmentation or preprocessing
    pre_transform : Optional[Callable]
        Optional pre-transform function (not used here)
    """
    def __init__(
        self,
        root: str = './data',
        processed_dir: str = 'processed',
        file_pattern: str = 'scenario_*.pt',
        indices: Optional[List[int]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        super().__init__(root, transform, pre_transform)
        # Normalize processed_dir to avoid trailing slash/backslash issues
        processed_dir_norm = processed_dir.rstrip("/\\")
        self._custom_processed_dir = os.path.join(root, processed_dir_norm) if processed_dir_norm else root
        self._custom_file_pattern = file_pattern
        self._custom_indices = indices

    def len(self):
        files = sorted(glob.glob(os.path.join(self._custom_processed_dir, self._custom_file_pattern)))
        if self._custom_indices is not None:
            files = [files[i] for i in self._custom_indices]
        return len(files)

    def get(self, idx):
        pattern = os.path.normpath(os.path.join(self._custom_processed_dir, self._custom_file_pattern))
        #print(f"[DEBUG] Using glob pattern: {pattern}")
        files = sorted(glob.glob(pattern))
        if self._custom_indices is not None:
            files = [files[i] for i in self._custom_indices]
        if not files:
            raise RuntimeError(
                f"No files found in directory: {self._custom_processed_dir} with pattern: {self._custom_file_pattern}\n"
                f"Resolved path: {pattern}"
            )
        #print(f"[DEBUG] Files found for dataset: {files}")
        data = torch.load(files[idx], weights_only=False)
        return data

    def __len__(self):
        return self.len()

    def __getitem__(self, idx):
        return self.get(idx)

    @property
    def processed_file_names(self):
        # Return a pattern-based list to avoid relying on __init__ attributes
        return ["scenario_*.pt"]

    @property
    def raw_file_names(self):
        # Not used, but required by PyG Dataset API
        return []

    def process(self):
        # Processing is handled externally (see process_data.py)
        pass

# Example usage (programmatic, not via command-line):
# from src.datasets import WDNLeakDataset
# dataset = WDNLeakDataset(root='./data', processed_dir='processed', file_pattern='scenario_*.pt')
# print(len(dataset))
# data = dataset[0]
