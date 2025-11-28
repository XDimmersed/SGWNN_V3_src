"""ModelNet40 点云分类数据集加载器（支持 HDF5，含预处理与增强）。

本文件为点云数据加载的入口，职责涵盖：
1. 解析官方提供的 ``train_files.txt``/``test_files.txt`` 列表，逐个读取 H5。 
2. 在 ``__getitem__`` 中完成固定点数采样、归一化与数据增强，将 numpy 数据转为 torch 张量。 
3. 提供 ``create_dataloaders`` 快捷函数便于脚本直接构造 DataLoader，以及 ``test_dataset_loading`` 自检入口。

所有注释均为中文，并对原有的英文提示做了直译与补充解释，方便快速理解数据流。"""

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
    """ModelNet40 点云数据集加载类，支持 HDF5 文件的批量读取与数据增强。"""
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 num_points: int = 1024,
                 normalize: bool = True,
                 augmentation: bool = True,
                 cache_data: bool = False):
        """初始化数据集并完成列表加载、可选缓存。

        参数：
            data_root: 数据根目录。
            split: 数据划分（train/test）。
            num_points: 每个样本采样的点数。
            normalize: 是否归一化到单位球体。
            augmentation: 是否启用训练增强。
            cache_data: 是否将全部数据缓存到内存，换取更快迭代。
        """
        self.data_root = data_root
        self.split = split
        self.num_points = num_points
        self.normalize = normalize
        self.augmentation = augmentation and (split == 'train')
        self.cache_data = cache_data
        
        # 读取类别名称列表（原英文注释“Load class names”）
        with open(os.path.join(data_root, 'shape_names.txt'), 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        self.num_classes = len(self.class_names)
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

        # 读取划分列表文件（原英文注释“Load file list”）：里面是多个 h5 路径
        file_list_path = os.path.join(data_root, f'{split}_files.txt')
        with open(file_list_path, 'r') as f:
            self.h5_files = [line.strip() for line in f.readlines()]

        # 将所有 H5 文件中的数据一次性读入内存（原英文注释“Load all data from H5 files”）
        self.data, self.labels = self._load_h5_data()

        # 如配置开启则将预处理后的样本缓存到内存，避免每次 __getitem__ 再做采样/增强
        if self.cache_data:
            print(f"Caching {len(self.data)} samples in memory...")
            self.cached_data = []
            for i in range(len(self.data)):
                self.cached_data.append(self._process_sample(i))
        else:
            self.cached_data = None
            
        print(f"Loaded {len(self.data)} samples from ModelNet40 {split} set")  # 保留英文输出，便于与官方示例对齐
        
    def _load_h5_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """从 H5 文件批量加载点云与标签。"""
        all_data = []
        all_labels = []
        
        for h5_file in self.h5_files:
            # 兼容不同的路径格式
            if h5_file.startswith('data/'):
                # 列表包含相对项目根目录的路径
                file_path = h5_file
            elif os.path.isabs(h5_file):
                # 已经是系统绝对路径
                file_path = h5_file
            else:
                # 相对 data_root 的路径
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
            
        # 将各个 H5 文件拼接成完整数组
        all_data = np.concatenate(all_data, axis=0)  # [Total_N, num_points, 3]
        all_labels = np.concatenate(all_labels, axis=0)  # [Total_N]
        
        return all_data, all_labels
    
    def _process_sample(self, idx: int) -> Tuple[torch.Tensor, int]:
        """处理单个样本：采样/归一化/增强并转为张量。"""
        points = self.data[idx].copy()  # [num_points, 3]
        label = int(self.labels[idx])
        
        # 随机采样或重复，保证点数固定
        if points.shape[0] > self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
            points = points[indices]
        elif points.shape[0] < self.num_points:
            # 若点数不足则通过重复点进行上采样（原英文注释“Upsample by repeating points”）
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
            points = points[indices]
        
        # 归一化：中心化并缩放到单位球
        if self.normalize:
            # 将质心移到原点（原英文注释“Center to origin”）
            points = points - np.mean(points, axis=0, keepdims=True)
            # 按最远点距离缩放到单位球（原英文注释“Scale to unit sphere”）
            max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
            if max_dist > 0:
                points = points / max_dist
        
        # 训练阶段可选数据增强
        if self.augmentation:
            points = self._augment_points(points)

        # 将 numpy 数组转换为 torch 张量（原英文注释“Convert to tensor”）
        points = torch.from_numpy(points).float()  # [num_points, 3]
        
        return points, label
    
    def _augment_points(self, points: np.ndarray) -> np.ndarray:
        """随机旋转/缩放/抖动，并执行丢点补点增强。"""
        # 围绕 Y 轴随机旋转，保持上下方向不变
        if np.random.random() > 0.5:
            theta = np.random.uniform(0, 2 * np.pi)
            cos_theta, sin_theta = np.cos(theta), np.sin(theta)
            rotation_matrix = np.array([
                [cos_theta, 0, sin_theta],
                [0, 1, 0],
                [-sin_theta, 0, cos_theta]
            ])
            points = points @ rotation_matrix.T

        # 随机缩放
        if np.random.random() > 0.5:
            scale = np.random.uniform(0.8, 1.2)
            points = points * scale

        # 随机抖动
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.02, points.shape)
            points = points + noise

        # 随机丢点并以保留点补齐数量
        if np.random.random() > 0.5:
            dropout_ratio = np.random.uniform(0, 0.1)
            num_dropout = int(dropout_ratio * points.shape[0])
            if num_dropout > 0:
                dropout_indices = np.random.choice(points.shape[0], num_dropout, replace=False)
                keep_indices = np.setdiff1d(np.arange(points.shape[0]), dropout_indices)
                if len(keep_indices) > 0:
                    # 复制保留点补齐被丢弃的部分
                    duplicate_indices = np.random.choice(keep_indices, num_dropout, replace=True)
                    points[dropout_indices] = points[duplicate_indices]
        
        return points
    
    def __len__(self) -> int:
        """返回数据集中样本总数。"""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取单个样本的点云张量与标签。"""
        if self.cached_data is not None:
            return self.cached_data[idx]
        else:
            return self._process_sample(idx)
    
    def get_class_weights(self) -> torch.Tensor:
        """计算类别权重以平衡训练。"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        total_samples = len(self.labels)
        
        # 反频率权重（原英文注释“Inverse frequency weighting”）
        weights = np.zeros(self.num_classes)
        for label, count in zip(unique_labels, counts):
            weights[label] = total_samples / (self.num_classes * count)
        
        return torch.from_numpy(weights).float()
    
    def get_data_statistics(self) -> dict:
        """统计数据集规模与坐标分布。"""
        unique_labels, counts = np.unique(self.labels, return_counts=True)
        
        # 随机抽取部分点云做坐标统计（原英文注释“Sample a few point clouds for coordinate statistics”）
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
    """构建训练/测试 DataLoader，封装常用参数。"""
    # 创建训练/测试数据集（原英文注释“Create datasets”）
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
        augmentation=False,  # 测试集不做数据增强（原英文注释“No augmentation for test”）
        cache_data=cache_data
    )

    # 构建 DataLoader（原英文注释“Create data loaders”）
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
    """简单测试数据读取流程（仅限本地调试）。"""
    data_root = "data/modelnet40_ply_hdf5_2048"

    print("Testing ModelNet40 dataset loading...")

    # 创建训练集并检查基本信息（原英文注释“Create dataset”）
    dataset = ModelNet40Dataset(
        data_root=data_root,
        split='train',
        num_points=1024,
        normalize=True,
        augmentation=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Class names: {dataset.class_names[:5]}...")  # 展示前 5 个类别（原英文注释“First 5 classes”）

    # 抽取一个样本验证形状与标签（原英文注释“Test a sample”）
    points, label = dataset[0]
    print(f"Sample shape: {points.shape}")
    print(f"Sample label: {label} ({dataset.class_names[label]})")
    print(f"Point range: [{points.min():.3f}, {points.max():.3f}]")

    # 测试 DataLoader 是否能正常迭代（原英文注释“Test data loader”）
    train_loader, test_loader = create_dataloaders(
        data_root=data_root,
        batch_size=4,
        num_workers=0  # 关闭多进程方便本地调试（原英文注释“For testing”）
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # 取出一个 batch 检查张量维度（原英文注释“Test a batch”）
    batch_points, batch_labels = next(iter(train_loader))
    print(f"Batch points shape: {batch_points.shape}")
    print(f"Batch labels shape: {batch_labels.shape}")

    # 打印数据统计信息（原英文注释“Get statistics”）
    stats = dataset.get_data_statistics()
    print(f"Dataset statistics: {stats}")
    
    print("✓ Dataset loading test passed!")


if __name__ == "__main__":
    test_dataset_loading() 