"""
Data loading utilities for PyTorch RED-PAN training.

This module implements memory-efficient data loading for large-scale seismic datasets.
Key features:
- HDF5-based lazy loading (never load entire dataset into memory)
- TFRecord reading support (for backward compatibility)
- Streaming shuffle for large datasets
- Efficient prefetching and pinned memory

The Golden Rule: Stream from disk, process on-the-fly!
"""

import os
import json
import logging
import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Callable, Union

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

logger = logging.getLogger(__name__)


class HDF5SeismicDataset(Dataset):
    """
    Memory-efficient PyTorch Dataset that streams data from HDF5 files.
    
    This dataset NEVER loads all data into memory. Instead, it:
    1. Keeps only the index in memory (sample ID -> file location mapping)
    2. Opens HDF5 files lazily when samples are requested
    3. Uses HDF5's chunked reading for efficient partial file access
    
    Expected HDF5 structure:
        /waveforms: (N, 3, 6000) float32 - ENZ waveform data
        /labels: (N, 6000, 3) float32 - P/S/Noise probabilities  
        /masks: (N, 6000, 2) float32 - Mask/Unmask probabilities
        /indices: (N,) string - Sample identifiers (optional)
    
    Example:
        dataset = HDF5SeismicDataset(
            data_dir='/path/to/hdf5_shards/',
            transform=MyAugmentation()
        )
        loader = DataLoader(
            dataset,
            batch_size=32,
            num_workers=4,
            pin_memory=True,
            prefetch_factor=2,
        )
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        index_file: Optional[str] = None,
        transform: Optional[Callable] = None,
        data_length: int = 6000,
        channels: int = 3,
        label_channels: int = 3,
        mask_channels: int = 2,
    ):
        """
        Args:
            data_dir: Directory containing HDF5 shard files
            index_file: Optional JSON file with sample->shard mapping.
                        If None, will scan data_dir for .h5 files.
            transform: Optional transform to apply to (waveform, label, mask)
            data_length: Expected waveform length (samples)
            channels: Number of waveform channels (3 for ENZ)
            label_channels: Number of label channels (3 for P/S/Noise)
            mask_channels: Number of mask channels (2 for Mask/Unmask)
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.data_length = data_length
        self.channels = channels
        self.label_channels = label_channels
        self.mask_channels = mask_channels
        
        # Build index: list of (shard_path, local_idx) for each sample
        self.index = self._build_index(index_file)
        
        # Lazy file handles (opened on first access)
        self._file_handles: Dict[str, h5py.File] = {}
        
        logger.info(f"HDF5SeismicDataset: {len(self.index)} samples from {self.data_dir}")
    
    
    
    def _build_index(self, index_file: Optional[str]) -> List[Dict]:
        """
        Build index mapping global idx -> metadata.
        Returns list of dicts with file path, indices, and arrival info.
        """
        index = []
        
        # Scan directory for HDF5 files
        if index_file:
             with open(index_file, 'r') as f: return json.load(f)

        h5_files = sorted(self.data_dir.glob('*.h5')) + sorted(self.data_dir.glob('*.hdf5'))
        
        for h5_path in h5_files:
            try:
                # Try loading metadata
                meta_path = str(h5_path).replace('.h5', '_metadata.csv')
                if os.path.exists(meta_path):
                    df = pd.read_csv(meta_path)
                    
                    with h5py.File(h5_path, 'r') as f:
                        for _, row in df.iterrows():
                            category = row.get('category', 'unknown')
                            split = row.get('split', 'train')
                            
                            # Determine group path
                            if f"/{category}/{split}" in f:
                                group_path = f"/{category}/{split}"
                            elif f"/{category}" in f:
                                group_path = f"/{category}"
                            else:
                                group_path = "/" # Root (legacy)

                            try:
                                p_peaks = json.loads(str(row.get('p_arrival_sample', '[]')))
                                if isinstance(p_peaks, int): p_peaks = [p_peaks]
                            except:
                                p_peaks = []
                            
                            try:
                                s_peaks = json.loads(str(row.get('s_arrival_sample', '[]')))
                                if isinstance(s_peaks, int): s_peaks = [s_peaks]
                            except:
                                s_peaks = []

                            cat_lower = str(category).lower()
                            if 'noise' in cat_lower:
                                # Noise samples should not carry P/S labels
                                p_peaks = []
                                s_peaks = []

                            index.append({
                                'path': str(h5_path),
                                'idx': int(row['hdf5_index']),
                                'p_peaks': p_peaks,
                                's_peaks': s_peaks,
                                'group': group_path
                            })
                else:
                    # Fallback for files without metadata (no labels)
                    with h5py.File(h5_path, 'r') as f:
                        # Assume flat if no metadata, or scan?
                        # If flat:
                        if 'waveforms' in f:
                             n = f['waveforms'].shape[0]
                             for i in range(n):
                                 index.append({'path': str(h5_path), 'idx': i, 'p_peaks': [], 's_peaks': [], 'group': '/'})
                        else:
                             # Try to scan groups? Too complex for fallback.
                             pass
                            
            except Exception as e:
                logger.warning(f"Failed to scan {h5_path}: {e}")
        
        return index

    def _get_file_handle(self, path: str) -> h5py.File:
        """Get or open HDF5 file handle (cached)."""
        if path not in self._file_handles:
            self._file_handles[path] = h5py.File(path, 'r', swmr=True)
        return self._file_handles[path]
    
    def __len__(self) -> int:
        return len(self.index)


    def _generate_targets(self, length: int, p_idxs: List[int], s_idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        # Use original-style fixed windows matching P12 TFRecord generation:
        # err_win is the full window (half_win * 2), so P=±0.2s, S=±0.3s
        dt = 0.01
        err_win_p = 0.4
        err_win_s = 0.6
        err_win_npts_p = int(np.round(err_win_p / dt))
        err_win_npts_s = int(np.round(err_win_s / dt))

        def _gen_tar_func(data_length, point, mask_window):
            target = np.zeros(data_length, dtype=np.float32)
            half_win = mask_window // 2
            if half_win <= 0:
                return target
            gaus = np.exp(-((np.arange(-half_win, half_win + 1)) ** 2) / (2 * (max(half_win // 2, 1) ** 2)))
            gaus_first_half = gaus[: mask_window // 2]
            gaus_second_half = gaus[mask_window // 2 + 1 :]
            target[point] = gaus.max()
            if point < half_win:
                reduce_pts = half_win - point
                start_pt = 0
                gaus_first_half = gaus_first_half[reduce_pts:]
            else:
                start_pt = point - half_win
            target[start_pt:point] = gaus_first_half
            target[point + 1 : point + half_win + 1] = gaus_second_half[
                : len(target[point + 1 : point + half_win + 1])
            ]
            return target

        label_p = np.zeros(length, dtype=np.float32)
        label_s = np.zeros(length, dtype=np.float32)

        for p in p_idxs:
            if 0 <= p < length:
                label_p += _gen_tar_func(length, int(p), err_win_npts_p)
        for s in s_idxs:
            if 0 <= s < length:
                label_s += _gen_tar_func(length, int(s), err_win_npts_s)

        label_p = np.clip(label_p, 0.0, 1.0)
        label_s = np.clip(label_s, 0.0, 1.0)
        labels = np.stack([np.zeros(length, dtype=np.float32), label_p, label_s], axis=-1)

        # Keep explicit P/S arrays for mask generation (independent of any label order)
        label_p = labels[:, 1].copy()
        label_s = labels[:, 2].copy()

        # Noise as complement of P+S (sum=1 per timestep)
        labels[:, 0] = 1.0 - (label_p + label_s)
        labels[:, 0] = np.clip(labels[:, 0], 0.0, 1.0)

        # Renormalize to ensure sum=1 at each timestep
        denom = labels.sum(axis=1, keepdims=True)
        denom = np.where(denom == 0.0, 1.0, denom)
        labels = labels / denom
        
        # --- Generate Masks with P-S wrapping (match mosaic_tar_func_detect) ---
        sig_mask = (label_p + label_s).astype(np.float32)

        # Sort indices
        p_sorted = sorted([int(p) for p in p_idxs if 0 <= p < length])
        s_sorted = sorted([int(s) for s in s_idxs if 0 <= s < length])

        if len(p_sorted) == len(s_sorted) > 1:
            if min(s_sorted) < min(p_sorted):
                sig_mask[: int(s_sorted[0]) + 1] = 1.0
                sig_mask[int(p_sorted[-1]) :] = 1.0
                for arr_pt in range(len(p_sorted) - 1):
                    sig_mask[
                        int(p_sorted[arr_pt]) : int(s_sorted[arr_pt + 1]) + 1
                    ] = 1.0
            else:
                for arr_pt in range(len(p_sorted)):
                    sig_mask[
                        int(p_sorted[arr_pt]) : int(s_sorted[arr_pt]) + 1
                    ] = 1.0
        elif len(p_sorted) == len(s_sorted) == 1:
            if s_sorted[0] < p_sorted[0]:
                sig_mask[: int(s_sorted[0])] = 1.0
                sig_mask[int(p_sorted[0]) :] = 1.0
            else:
                sig_mask[
                    int(p_sorted[0]) : int(s_sorted[0]) + 1
                ] = 1.0
        elif len(p_sorted) > len(s_sorted):
            sig_mask[int(p_sorted[-1]) + 1 :] = 1.0
            for arr_pt in range(len(s_sorted)):
                sig_mask[
                    int(p_sorted[arr_pt]) : int(s_sorted[arr_pt]) + 1
                ] = 1.0
        elif len(p_sorted) < len(s_sorted):
            sig_mask[: int(s_sorted[0]) + 1] = 1.0
            for arr_pt in range(len(p_sorted)):
                sig_mask[
                    int(p_sorted[arr_pt]) : int(s_sorted[arr_pt + 1]) + 1
                ] = 1.0
        
        sig_mask = np.clip(sig_mask, 0.0, 1.0)
        noise_mask = 1.0 - sig_mask
        masks = np.stack([sig_mask, noise_mask], axis=-1)
        
        return labels, masks
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample with on-the-fly target generation."""
        # sample_info is now a dict
        sample_info = self.index[idx]
        shard_path = sample_info['path']
        local_idx = sample_info['idx']
        
        h5_file = self._get_file_handle(shard_path)
        
        # Read the waveform. A sample indexed with its group path, the layout
        # the conversion script writes (/category/split/waveforms), is read from
        # that group; otherwise 'waveforms' is expected at the file root.
        if 'group' not in sample_info:
             ds = h5_file['waveforms']
        else:
             ds = h5_file[sample_info['group']]['waveforms']

        waveform = ds[local_idx]
        if waveform.shape[0] != self.channels: waveform = waveform.T
        
        length = waveform.shape[-1]
        label, mask = self._generate_targets(length, sample_info['p_peaks'], sample_info['s_peaks'])
        
        if self.transform is not None:
            waveform, label, mask = self.transform(waveform, label, mask)
        
        return {
            'waveform': torch.from_numpy(waveform.copy()).float(),
            'label': torch.from_numpy(label.copy()).float(),
            'mask': torch.from_numpy(mask.copy()).float(),
        }
    
    def __del__(self):
        """Close all file handles on deletion."""
        for handle in self._file_handles.values():
            try:
                handle.close()
            except:
                pass


class TFRecordSeismicDataset(IterableDataset):
    """
    PyTorch IterableDataset that reads from TensorFlow TFRecord files.
    
    This provides backward compatibility with existing TFRecord datasets.
    Uses tfrecord package (pip install tfrecord) for reading.
    
    Note: For new projects, HDF5SeismicDataset is recommended.
    """
    
    def __init__(
        self,
        file_list: List[str],
        data_length: int = 6000,
        transform: Optional[Callable] = None,
        shuffle: bool = True,
        buffer_size: int = 1000,
    ):
        """
        Args:
            file_list: List of TFRecord file paths
            data_length: Expected waveform length
            transform: Optional transform function
            shuffle: Whether to shuffle files and samples
            buffer_size: Size of shuffle buffer
        """
        self.file_list = file_list
        self.data_length = data_length
        self.transform = transform
        self.shuffle = shuffle
        self.buffer_size = buffer_size
    
    def _parse_tfrecord(self, record):
        """Parse a single TFRecord example matching TensorFlow format."""
        import tfrecord
        
        # Feature description of the original RED-PAN TFRecords
        features = {
            'trc_data': np.zeros(self.data_length * 3, dtype=np.float32),
            'label_data': np.zeros(self.data_length * 3, dtype=np.float32),
            'mask': np.zeros(self.data_length * 2, dtype=np.float32),
        }
        
        for key, value in record.items():
            if key in features:
                features[key] = np.array(value, dtype=np.float32)
        
        # Reshape
        waveform = features['trc_data'].reshape(self.data_length, 3).T  # -> (3, T)
        label = features['label_data'].reshape(self.data_length, 3)     # (T, 3)
        mask = features['mask'].reshape(self.data_length, 2)            # (T, 2)
        
        return waveform, label, mask
    
    def __iter__(self):
        try:
            from tfrecord.torch.dataset import TFRecordDataset
        except ImportError:
            raise ImportError(
                "TFRecord support requires tfrecord package: pip install tfrecord"
            )
        
        # Shuffle file list
        files = self.file_list.copy()
        if self.shuffle:
            np.random.shuffle(files)
        
        for tfrecord_path in files:
            try:
                dataset = TFRecordDataset(tfrecord_path, None)
                
                for record in dataset:
                    waveform, label, mask = self._parse_tfrecord(record)
                    
                    if self.transform is not None:
                        waveform, label, mask = self.transform(waveform, label, mask)
                    
                    yield {
                        'waveform': torch.from_numpy(waveform).float(),
                        'label': torch.from_numpy(label).float(),
                        'mask': torch.from_numpy(mask).float(),
                    }
            except Exception as e:
                logger.warning(f"Error reading {tfrecord_path}: {e}")
                continue


class NumpySeismicDataset(Dataset):
    """
    Simple dataset that loads from NumPy arrays.
    
    Useful for small datasets or validation sets that fit in memory.
    """
    
    def __init__(
        self,
        waveforms: np.ndarray,
        labels: np.ndarray,
        masks: np.ndarray,
        transform: Optional[Callable] = None,
    ):
        """
        Args:
            waveforms: (N, C, T) array of waveform data
            labels: (N, T, 3) array of labels
            masks: (N, T, 2) array of masks
            transform: Optional transform function
        """
        assert len(waveforms) == len(labels) == len(masks)
        
        self.waveforms = waveforms
        self.labels = labels
        self.masks = masks
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.waveforms)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        waveform = self.waveforms[idx]
        label = self.labels[idx]
        mask = self.masks[idx]
        
        if self.transform is not None:
            waveform, label, mask = self.transform(waveform, label, mask)
        
        return {
            'waveform': torch.from_numpy(waveform.copy()).float(),
            'label': torch.from_numpy(label.copy()).float(),
            'mask': torch.from_numpy(mask.copy()).float(),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
    drop_last: bool = True,
    persistent_workers: bool = True,
) -> DataLoader:
    """
    Create an optimized DataLoader for seismic data.
    
    This configures the DataLoader with best practices for GPU training:
    - Multiple workers for parallel data loading
    - Pinned memory for faster CPU->GPU transfer
    - Prefetching to overlap data loading and training
    - Persistent workers to avoid worker restart overhead
    
    Args:
        dataset: PyTorch Dataset instance
        batch_size: Samples per batch
        shuffle: Whether to shuffle (use False for IterableDataset)
        num_workers: Number of parallel data loading workers
        pin_memory: Use pinned memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
        drop_last: Drop incomplete final batch
        persistent_workers: Keep workers alive between epochs
    
    Returns:
        Configured DataLoader
    """
    # For IterableDataset, shuffle must be False
    if isinstance(dataset, IterableDataset):
        shuffle = False
        persistent_workers = False  # Not compatible with IterableDataset
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=drop_last,
        persistent_workers=persistent_workers and num_workers > 0,
    )


if __name__ == '__main__':
    # Quick test with synthetic data
    import tempfile
    
    # Create test HDF5 file
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = os.path.join(tmpdir, 'test.h5')
        
        n_samples = 100
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('waveforms', data=np.random.randn(n_samples, 3, 6000).astype(np.float32))
            f.create_dataset('labels', data=np.random.randn(n_samples, 6000, 3).astype(np.float32))
            f.create_dataset('masks', data=np.random.randn(n_samples, 6000, 2).astype(np.float32))
        
        # Test dataset
        dataset = HDF5SeismicDataset(tmpdir)
        print(f"Dataset size: {len(dataset)}")
        
        sample = dataset[0]
        print(f"Waveform shape: {sample['waveform'].shape}")
        print(f"Label shape: {sample['label'].shape}")
        print(f"Mask shape: {sample['mask'].shape}")
        
        # Test dataloader
        loader = create_dataloader(dataset, batch_size=8, num_workers=0)
        batch = next(iter(loader))
        print(f"\nBatch waveform: {batch['waveform'].shape}")
        print(f"Batch label: {batch['label'].shape}")
        
        print("\n✓ Dataset test passed!")
