"""
Evaluation metrics for SGWCN
"""

import torch
"""训练/验证期间使用的评估指标与辅助统计方法。"""

import numpy as np
import torch
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    top_k_accuracy_score
)


def compute_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> Dict[str, float]:
    """计算分类常用指标（精度/精确率/召回/F1 及分类别表现）。"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro'),
        'f1': f1_score(targets, predictions, average='macro')
    }
    
    # 逐类别计算精确率/召回/F1（原英文注释“Compute per-class metrics”）
    for i in range(num_classes):
        class_pred = (predictions == i)
        class_true = (targets == i)
        
        metrics[f'class_{i}_precision'] = precision_score(
            class_true, class_pred, zero_division=0
        )
        metrics[f'class_{i}_recall'] = recall_score(
            class_true, class_pred, zero_division=0
        )
        metrics[f'class_{i}_f1'] = f1_score(
            class_true, class_pred, zero_division=0
        )
    
    return metrics


def compute_top_k_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    k: int = 5
) -> float:
    """计算 Top-k 精度，用于多分类评估。"""
    logits = logits.cpu().numpy()
    targets = targets.cpu().numpy()
    
    return top_k_accuracy_score(targets, logits, k=k)


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """生成混淆矩阵，查看各类别的混淆情况。"""
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    return confusion_matrix(targets, predictions, labels=range(num_classes))


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    method: str = 'balanced'
) -> torch.Tensor:
    """计算类别权重，用于类别不均衡场景。"""
    targets = targets.cpu().numpy()
    class_counts = np.bincount(targets, minlength=num_classes)
    
    if method == 'balanced':
        # 均衡权重：n_samples / (n_classes * n_samples_per_class)
        weights = len(targets) / (num_classes * class_counts)
    elif method == 'inverse':
        # 反频率权重（样本越少权重越大）
        weights = 1.0 / (class_counts + 1e-6)
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    # 归一化权重，保证权重和为 1
    weights = weights / weights.sum()
    
    return torch.FloatTensor(weights)


def compute_energy_metrics(
    spike_counts: torch.Tensor,
    num_neurons: int
) -> Dict[str, float]:
    """根据脉冲计数估算能耗相关指标。"""
    spike_counts = spike_counts.cpu().numpy()
    
    metrics = {
        'total_spikes': np.sum(spike_counts),
        'avg_spikes_per_neuron': np.mean(spike_counts),
        'max_spikes_per_neuron': np.max(spike_counts),
        'spike_rate': np.sum(spike_counts) / (spike_counts.shape[0] * num_neurons)
    }
    
    return metrics 