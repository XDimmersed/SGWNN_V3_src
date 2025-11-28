"""
ModelNet40 Dataset Loader for Point Cloud Classification
Supports HDF5 format with efficient loading and preprocessing
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import json
from typing import Tuple, List, Optional
import warnings
import random


class ModelNet40Dataset(Dataset):
    """
    ModelNet40 point cloud dataset loader
    
    Supports HDF5 format with efficient batch loading and data augmentation
    """
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 num_points: int = 1024,
                 normalize: bool = True,
                 augmentation: bool = True,
                 cache_data: bool = False):
        """
        Initialize ModelNet40 dataset
        
        Args:
            data_root: path to ModelNet40 data directory
            split: 'train' or 'test'
            num_points: number of points to sample from each shape
            normalize: whether to normalize point clouds
            augmentation: whether to apply data augmentation (for training)
            cache_data: whether to cache all data in memory
        """
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.augmentation = augmentation and (split == 'train')
        self.cache_data = cache_data
        
        # Load class names
        with open(os.path.join(data_root, 'shape_names.txt'), 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
        # Load file list
        file_list_path = os.path.join(data_root, f'{split}_files.txt')
        with open(file_list_path, 'r') as f:
            self.h5_files = [line.strip() for line in f.readlines()]
        
        # Load all data from H5 files
        self.data, self.labels = self._load_h5_data()
        
        # Cache data in memory if requested
        if self.cache_data:
            print(f"Caching {len(self.data)} samples in memory...")
            self.cached_data = []
            for i in range(len(self.data)):
                self.cached_data.append(self._process_sample(i))
        else:
            self.cached_data = None
            
        print(f"Loaded {len(self.data)} samples from ModelNet40 {split} set")
        
    def _load_h5_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load data and labels from H5 files"""
        all_data = []
        all_labels = []
        
        for h5_file in self.h5_files:
            # Handle different path formats
            if h5_file.startswith('data/'):
                # File list contains absolute path from project root
                file_path = h5_file
            elif os.path.isabs(h5_file):
                # File list contains system absolute path
                file_path = h5_file
            else:
                # File list contains relative path
                file_path = os.path.join(self.data_root, h5_file)
            
            if not os.path.exists(file_path):
                print(f"Warning: H5 file not found: {file_path}")
                continue
                
            try:
                with h5py.File(file_path, 'r') as f:
                    data = f['data'][:]  # [N, num_points, 3]
                    labels = f['label'][:]  # [N]
                    
                    all_data.append(data)
                    all_labels.append(labels)
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
                
        if not all_data:
            raise ValueError(f"No valid data found in {self.data_root}. Check if H5 files exist.")
            
        # Concatenate all data
        all_data = np.concatenate(all_data, axis=0)  # [Total_N, num_points, 3]
        all_labels = np.concatenate(all_labels, axis=0)  # [Total_N]
        
        return all_data, all_labels
    
    def _process_sample(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Process a single sample with preprocessing and augmentation"""
        points = self.data[idx].copy()  # [num_points, 3]
        label = int(self.labels[idx])
        
        # Random point sampling
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < self.num_points:
            # Upsample by repeating points
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[indices]
        
        # Normalization
        if self.normalize:
            # Center to origin
            points = points - np.mean(points, axis=0, keepdims=True)
            # Scale to unit sphere
            max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
            if max_dist > 0:
                points = points / max_dist
        
        # Data augmentation for training
        if self.augmentation:
            points = self._augment_points(points)
        
        # Convert to tensor
        points = torch.from_numpy(points).float()  # [num_points, 3]
        
        return points, label
    
    def _augment_points(self, points: np.ndarray) -> np.ndarray:
        """Apply data augmentation to point cloud"""
        # Random rotation around Y-axis (up axis)
        if np.random.random() > 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])
            points = points @ rotation_matrix.T
        
        # Random scaling
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            points = points * scale
        
        # Random jittering
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.02, points.shape)
            points = points + noise
        
        # Random point dropout
        if np.random.random() > 0.5:
            dropout_ratio = np.random.uniform(0, 0.1)
            num_dropout = int(dropout_ratio * points.shape[0])
            if num_dropout > 0:
                dropout_indices = np.random.choice(points.shape[0], num_dropout, replace=False)
                keep_indices = np.setdiff1d(np.arange(points.shape[0]), dropout_indices)
                if len(keep_indices) > 0:
                    # Duplicate random points to maintain point count
                    duplicate_indices = np.random.choice(keep_indices, num_dropout, replace=True)
                    points[dropout_indices] = points[duplicate_indices]
        
        return points
    
    def __len__(self) -> int:
        """Get dataset size"""
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single sample"""
        if self.cached_data is not None:
            return self.cached_data[idx]
        else:
            return self._process_sample(idx)
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        
        # Inverse frequency weighting
        weights = np.zeros(self.num_classes)
        for label, count in zip(unique_labels, counts):
            weights[label] = total_samples / (self.num_classes * count)
        
        return torch.from_numpy(weights).float()
    
    def get_data_statistics(self) -> dict:
        """Get dataset statistics"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        # Sample a few point clouds for coordinate statistics
        sample_indices = np.random.choice(len(self.data), min(1000, len(self.data)), replace=False)
        sample_points = self.data[sample_indices]  # [N, num_points, 3]
        
        stats = {
            'num_samples': len(self.data),
            'num_classes': self.num_classes,
            'samples_per_class': dict(zip(self.class_names, counts)),
            'points_per_sample': self.data.shape[1],
            'coordinate_mean': np.mean(sample_points, axis=(0, 1)),
            'coordinate_std': np.std(sample_points, axis=(0, 1)),
            'coordinate_min': np.min(sample_points, axis=(0, 1)),
            'coordinate_max': np.max(sample_points, axis=(0, 1)),
        }
        
        return stats


def create_dataloaders(data_root: str,
                      batch_size: int = 32,
                      num_points: int = 1024,
                      num_workers: int = 4,
                      normalize: bool = True,
                      augmentation: bool = True,
                      cache_data: bool = False) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders for ModelNet40
    
    Args:
        data_root: path to ModelNet40 data directory
        batch_size: batch size for training
        num_points: number of points per sample
        num_workers: number of worker processes for data loading
        normalize: whether to normalize point clouds
        augmentation: whether to apply data augmentation
        cache_data: whether to cache data in memory
        
    Returns:
        train_loader: training data loader
        test_loader: test data loader
    """
    # Create datasets
    train_dataset = ModelNet40Dataset(
        data_root=data_root,
        split='train',
        num_points=num_points,
        normalize=normalize,
        augmentation=augmentation,
        cache_data=cache_data
    )
    
    test_dataset = ModelNet40Dataset(
        data_root=data_root,
        split='test',
        num_points=num_points,
        normalize=normalize,
        augmentation=False,  # No augmentation for test
        cache_data=cache_data
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, test_loader


def test_dataset_loading():
    """Test dataset loading functionality"""
    data_root = "data/modelnet40_ply_hdf5_2048"
    
    print("Testing ModelNet40 dataset loading...")
    
    # Create dataset
    dataset = ModelNet40Dataset(
        data_root=data_root,
        split='train',
        num_points=1024,
        normalize=True,
        augmentation=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names: {dataset.class_names[:5]}...")  # First 5 classes
    
    # Test a sample
    points, label = dataset[0]
    print(f"Sample shape: {points.shape}")
    print(f"Sample label: {label} ({dataset.class_names[label]})")
    print(f"Point range: [{points.min():.3f}, {points.max():.3f}]")
    
    # Test data loader
    train_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=0  # For testing
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Test a batch
    batch_points, batch_labels = next(iter(train_loader))
    print(f"Batch points shape: {batch_points.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")
    
    # Get statistics
    stats = dataset.get_data_statistics()
    print(f"Dataset statistics: {stats}")
    
    print("✓ Dataset loading test passed!")


if __name__ == "__main__":
    test_dataset_loading() 