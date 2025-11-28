"""
Spiking neural network utilities including surrogate gradients and encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpikeFunction(torch.autograd.Function):
    """
    Differentiable spike function with surrogate gradient
    Supports both positive and negative thresholds for bipolar LIF neurons
    """
    
    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor, 
                threshold: float, 
                is_negative: bool = False) -> torch.Tensor:
        """
        Forward pass of spike function
        
        Args:
            membrane_potential: [B, T, N, F] membrane potentials
            threshold: spike threshold (positive or negative)
            is_negative: whether this is a negative threshold
            
        Returns:
            spikes: [B, T, N, F] binary spike tensor
        """
        # Save for backward pass
        if isinstance(threshold, torch.Tensor):
            threshold_tensor = threshold.clone().detach()
        else:
            threshold_tensor = torch.tensor(threshold, device=membrane_potential.device, dtype=membrane_potential.dtype)
        ctx.save_for_backward(membrane_potential, threshold_tensor)
        ctx.is_negative = is_negative
        
        # Generate spikes
        if is_negative:
            # Negative threshold: spike when V <= threshold
            spikes = (membrane_potential <= threshold).float()
        else:
            # Positive threshold: spike when V >= threshold  
            spikes = (membrane_potential >= threshold).float()
        
        return spikes
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass using surrogate gradient with numerical stability
        
        Args:
            grad_output: gradient from subsequent layers
            
        Returns:
            grad_input: gradient w.r.t. membrane potential
        """
        membrane_potential, threshold_tensor = ctx.saved_tensors
        is_negative = ctx.is_negative
        threshold = threshold_tensor.item()
        
        # Check for NaN in grad_output
        if torch.isnan(grad_output).any():
            print("WARNING: NaN detected in grad_output!")
            grad_output = torch.where(torch.isnan(grad_output), torch.zeros_like(grad_output), grad_output)
        
        # Surrogate gradient: derivative of sigmoid function with stability
        # For positive threshold: gradient is positive around threshold
        # For negative threshold: gradient is negative around threshold  
        gamma = 0.5  # controls smoothness of surrogate gradient
        
        # Clamp membrane potential to prevent extreme values
        membrane_potential = torch.clamp(membrane_potential, -10.0, 10.0)
        
        if is_negative:
            # For negative spikes, use negative surrogate gradient
            diff = threshold - membrane_potential
            diff = torch.clamp(diff, -10.0, 10.0)  # Clamp for numerical stability
            surrogate_grad = -gamma * torch.sigmoid(gamma * diff) * \
                            torch.sigmoid(gamma * (-diff))
        else:
            # For positive spikes, use positive surrogate gradient
            diff = membrane_potential - threshold
            diff = torch.clamp(diff, -10.0, 10.0)  # Clamp for numerical stability
            surrogate_grad = gamma * torch.sigmoid(gamma * diff) * \
                           torch.sigmoid(gamma * (-diff))
        
        # Clamp surrogate gradient to prevent extreme values
        surrogate_grad = torch.clamp(surrogate_grad, -1.0, 1.0)
        
        # Apply chain rule
        grad_input = grad_output * surrogate_grad
        
        # Final NaN check
        if torch.isnan(grad_input).any():
            print("WARNING: NaN detected in grad_input after surrogate gradient!")
            grad_input = torch.where(torch.isnan(grad_input), torch.zeros_like(grad_input), grad_input)
        
        return grad_input, None, None


def poisson_encoding(x: torch.Tensor, 
                    T: int, 
                    max_rate: float = 1.0,
                    normalize_per_sample: bool = True) -> torch.Tensor:
    """
    Encode continuous values as Poisson spike trains
    
    Args:
        x: [B, N, F] input features to encode
        T: number of time steps
        max_rate: maximum firing rate (spikes per time step)
        normalize_per_sample: whether to normalize each sample individually
        
    Returns:
        spike_trains: [B, T, N, F] Poisson spike trains
    """
    B, N, F = x.shape
    device = x.device
    
    # Normalize features to [0, max_rate]
    if normalize_per_sample:
        # Per-sample normalization
        x_min = x.amin(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        x_max = x.amax(dim=(1, 2), keepdim=True)  # [B, 1, 1]
    else:
        # Global normalization
        x_min = x.min()
        x_max = x.max()
    
    # Avoid division by zero
    x_range = x_max - x_min
    x_range = torch.where(x_range > 1e-8, x_range, torch.ones_like(x_range))
    
    # Normalize to [0, max_rate]
    rates = (x - x_min) / x_range * max_rate  # [B, N, F]
    
    # Generate Poisson spike trains
    # Expand for time dimension
    rates_expanded = rates.unsqueeze(1).expand(B, T, N, F)  # [B, T, N, F]
    
    # Generate random numbers and compare with rates
    random_vals = torch.rand_like(rates_expanded)
    spike_trains = (random_vals < rates_expanded).float()
    
    return spike_trains


def rate_encoding(x: torch.Tensor, 
                 T: int,
                 normalize_per_sample: bool = True) -> torch.Tensor:
    """
    Simple rate encoding: convert continuous values to constant firing rates
    
    Args:
        x: [B, N, F] input features
        T: number of time steps
        normalize_per_sample: whether to normalize each sample individually
        
    Returns:
        spike_trains: [B, T, N, F] rate-encoded spike trains
    """
    B, N, F = x.shape
    
    # Normalize to [0, 1]
    if normalize_per_sample:
        x_min = x.amin(dim=(1, 2), keepdim=True)
        x_max = x.amax(dim=(1, 2), keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()
    
    x_range = x_max - x_min
    x_range = torch.where(x_range > 1e-8, x_range, torch.ones_like(x_range))
    
    rates = (x - x_min) / x_range  # [B, N, F]
    
    # Expand to time dimension and threshold
    rates_expanded = rates.unsqueeze(1).expand(B, T, N, F)
    thresholds = torch.linspace(0, 1, T+1, device=x.device)[:-1].view(1, T, 1, 1)
    
    spike_trains = (rates_expanded > thresholds).float()
    
    return spike_trains


def spike_count_decoding(spike_trains: torch.Tensor, 
                        dim: int = 1) -> torch.Tensor:
    """
    Decode spike trains by counting spikes
    
    Args:
        spike_trains: [..., T, ...] spike trains with time dimension
        dim: time dimension to sum over
        
    Returns:
        decoded: spike counts with time dimension removed
    """
    return spike_trains.sum(dim=dim)


def spike_rate_decoding(spike_trains: torch.Tensor, 
                       dim: int = 1) -> torch.Tensor:
    """
    Decode spike trains by computing firing rates
    
    Args:
        spike_trains: [..., T, ...] spike trains with time dimension
        dim: time dimension to average over
        
    Returns:
        decoded: firing rates with time dimension removed
    """
    return spike_trains.mean(dim=dim)


def temporal_coding_loss(spike_trains: torch.Tensor, 
                        targets: torch.Tensor,
                        method: str = 'mse') -> torch.Tensor:
    """
    Compute loss for temporally coded spike trains
    
    Args:
        spike_trains: [B, T, N, F] predicted spike trains
        targets: [B, N, F] target values
        method: loss computation method ('mse', 'rate_mse', 'count_mse')
        
    Returns:
        loss: scalar loss value
    """
    if method == 'mse':
        # Direct MSE on spike patterns
        targets_expanded = targets.unsqueeze(1).expand_as(spike_trains)
        return F.mse_loss(spike_trains, targets_expanded)
    
    elif method == 'rate_mse':
        # MSE on firing rates
        predicted_rates = spike_rate_decoding(spike_trains, dim=1)
        return F.mse_loss(predicted_rates, targets)
    
    elif method == 'count_mse':
        # MSE on spike counts (normalized by time steps)
        predicted_counts = spike_count_decoding(spike_trains, dim=1) / spike_trains.shape[1]
        return F.mse_loss(predicted_counts, targets)
    
    else:
        raise ValueError(f"Unknown temporal coding loss method: {method}")


def compute_spike_statistics(spike_trains: torch.Tensor, 
                           dim: int = 1) -> dict:
    """
    Compute statistics of spike trains
    
    Args:
        spike_trains: spike trains tensor with time dimension
        dim: time dimension
        
    Returns:
        stats: dictionary of spike statistics
    """
    # Move time dimension to last position for easier computation
    spike_trains_transposed = spike_trains.transpose(dim, -1)
    T = spike_trains_transposed.shape[-1]
    
    # Compute statistics
    firing_rates = spike_trains_transposed.mean(dim=-1)  # Average firing rate
    spike_counts = spike_trains_transposed.sum(dim=-1)   # Total spike count
    
    # Inter-spike intervals (for non-zero firing neurons)
    isi_means = []
    for batch_idx in range(spike_trains_transposed.shape[0]):
        batch_spikes = spike_trains_transposed[batch_idx]  # [N, F, T]
        batch_isi = []
        for n in range(batch_spikes.shape[0]):
            for f in range(batch_spikes.shape[1]):
                spike_times = torch.nonzero(batch_spikes[n, f], as_tuple=False).squeeze(-1)
                if len(spike_times) > 1:
                    isis = torch.diff(spike_times.float())
                    batch_isi.append(isis.mean().item())
        if batch_isi:
            isi_means.append(torch.tensor(batch_isi).mean().item())
        else:
            isi_means.append(float('inf'))
    
    stats = {
        'mean_firing_rate': firing_rates.mean().item(),
        'max_firing_rate': firing_rates.max().item(),
        'min_firing_rate': firing_rates.min().item(),
        'std_firing_rate': firing_rates.std().item(),
        'total_spikes': spike_counts.sum().item(),
        'sparsity': (spike_trains == 0).float().mean().item(),
        'mean_isi': torch.tensor(isi_means).mean().item() if isi_means else float('inf'),
        'active_neurons_ratio': (firing_rates > 0).float().mean().item(),
    }
    
    return stats 