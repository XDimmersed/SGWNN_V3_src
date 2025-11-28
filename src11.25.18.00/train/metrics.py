"""
Evaluation metrics for SGWCN
"""

import torch
import numpy as np
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
    """
    Compute classification metrics
    
    Args:
        predictions: Model predictions (N,)
        targets: Ground truth labels (N,)
        num_classes: Number of classes
    
    Returns:
        Dictionary of metrics
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    metrics = {
        'accuracy': accuracy_score(targets, predictions),
        'precision': precision_score(targets, predictions, average='macro'),
        'recall': recall_score(targets, predictions, average='macro'),
        'f1': f1_score(targets, predictions, average='macro')
    }
    
    # Compute per-class metrics
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
    """
    Compute top-k accuracy
    
    Args:
        logits: Model logits (N, C)
        targets: Ground truth labels (N,)
        k: Number of top predictions to consider
    
    Returns:
        Top-k accuracy
    """
    logits = logits.cpu().numpy()
    targets = targets.cpu().numpy()
    
    return top_k_accuracy_score(targets, logits, k=k)


def compute_confusion_matrix(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix
    
    Args:
        predictions: Model predictions (N,)
        targets: Ground truth labels (N,)
        num_classes: Number of classes
    
    Returns:
        Confusion matrix (num_classes, num_classes)
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()
    
    return confusion_matrix(targets, predictions, labels=range(num_classes))


def compute_class_weights(
    targets: torch.Tensor,
    num_classes: int,
    method: str = 'balanced'
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets
    
    Args:
        targets: Ground truth labels (N,)
        num_classes: Number of classes
        method: Weighting method ('balanced' or 'inverse')
    
    Returns:
        Class weights (num_classes,)
    """
    targets = targets.cpu().numpy()
    class_counts = np.bincount(targets, minlength=num_classes)
    
    if method == 'balanced':
        # Balanced weights: n_samples / (n_classes * n_samples_per_class)
        weights = len(targets) / (num_classes * class_counts)
    elif method == 'inverse':
        # Inverse frequency weights
        weights = 1.0 / (class_counts + 1e-6)
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights
    weights = weights / weights.sum()
    
    return torch.FloatTensor(weights)


def compute_energy_metrics(
    spike_counts: torch.Tensor,
    num_neurons: int
) -> Dict[str, float]:
    """
    Compute energy consumption metrics
    
    Args:
        spike_counts: Number of spikes per neuron (N, num_neurons)
        num_neurons: Total number of neurons
    
    Returns:
        Dictionary of energy metrics
    """
    spike_counts = spike_counts.cpu().numpy()
    
    metrics = {
        'total_spikes': np.sum(spike_counts),
        'avg_spikes_per_neuron': np.mean(spike_counts),
        'max_spikes_per_neuron': np.max(spike_counts),
        'spike_rate': np.sum(spike_counts) / (spike_counts.shape[0] * num_neurons)
    }
    
    return metrics 