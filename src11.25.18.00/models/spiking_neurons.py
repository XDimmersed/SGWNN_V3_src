"""
Bipolar LIF Neuron with Positive and Negative Thresholds
Implements the core innovation of convex/concave feature encoding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ..utils.spike_utils import SpikeFunction


class BipolarLIFNeuron(nn.Module):
    """
    Bipolar Leaky Integrate-and-Fire neuron with dual thresholds
    
    Core innovations:
    1. Dual thresholds: θ_pos > 0 (excitatory), θ_neg < 0 (inhibitory)
    2. Positive spikes encode convex features, negative spikes encode concave features
    3. One neuron outputs 2×F channels, improving feature utilization
    4. Correct reset: V -= pos_spikes*θ_pos - neg_spikes*θ_neg
    """
    
    def __init__(self, 
                 membrane_dim: int,
                 tau_mem: float = 20.0,
                 theta_pos: float = 1.0,
                 theta_neg: float = -1.0,
                 reset_mode: str = 'subtract',
                 dt: float = 1.0):
        """
        Initialize bipolar LIF neuron
        
        Args:
            membrane_dim: dimension of membrane potential
            tau_mem: membrane time constant
            theta_pos: positive threshold (> 0)
            theta_neg: negative threshold (< 0)
            reset_mode: reset mode ('subtract' or 'zero')
            dt: time step
        """
        super().__init__()
        self.membrane_dim = membrane_dim
        self.tau_mem = tau_mem
        self.theta_pos = theta_pos
        self.theta_neg = theta_neg
        self.reset_mode = reset_mode
        self.dt = dt
        
        # Membrane decay factor
        self.alpha = torch.exp(torch.tensor(-dt / tau_mem))
        
        # Register membrane potential as buffer for BPTT
        self.register_buffer('membrane_potential', torch.zeros(1, 1, 1, membrane_dim))
        
        # Learnable parameters
        self.learnable_threshold = False
        if self.learnable_threshold:
            self.theta_pos_param = nn.Parameter(torch.tensor(theta_pos))
            self.theta_neg_param = nn.Parameter(torch.tensor(theta_neg))
    
    def reset_state(self, batch_size: int, num_nodes: int, device: torch.device):
        """
        Reset membrane potential state
        
        Args:
            batch_size: batch size
            num_nodes: number of nodes  
            device: device to create tensors on
        """
        self.membrane_potential = torch.zeros(
            batch_size, 1, num_nodes, self.membrane_dim, 
            device=device, dtype=torch.float32
        )
    
    def forward(self, 
                input_current: torch.Tensor,
                return_membrane: bool = False) -> torch.Tensor:
        """
        Forward pass of bipolar LIF neuron
        
        Args:
            input_current: [B, T, N, F] input current
            return_membrane: whether to return membrane potential
            
        Returns:
            spike_output: [B, T, N, 2*F] bipolar spike output
            membrane_potential: [B, T, N, F] membrane potential (optional)
        """
        B, T, N, F = input_current.shape
        device = input_current.device
        
        # Initialize membrane potential if needed
        if self.membrane_potential.shape != (B, 1, N, F):
            self.reset_state(B, N, device)
        
        # Get current thresholds
        if self.learnable_threshold:
            theta_pos = self.theta_pos_param
            theta_neg = self.theta_neg_param
        else:
            theta_pos = self.theta_pos
            theta_neg = self.theta_neg
        
        # Ensure thresholds have correct signs
        if not isinstance(theta_pos, torch.Tensor):
            theta_pos = torch.tensor(theta_pos, device=device, dtype=torch.float32)
        if not isinstance(theta_neg, torch.Tensor):
            theta_neg = torch.tensor(theta_neg, device=device, dtype=torch.float32)
            
        theta_pos = torch.abs(theta_pos)  # Force positive
        theta_neg = -torch.abs(theta_neg)  # Force negative
        
        spike_output_list = []
        membrane_history = []
        
        for t in range(T):
            # Membrane dynamics: V[t] = α*V[t-1] + I[t]
            # Clamp input current to prevent extreme values
            current_input = torch.clamp(input_current[:, t:t+1, :, :], -5.0, 5.0)
            
            self.membrane_potential = (
                self.alpha.to(device) * self.membrane_potential + 
                current_input  # [B, 1, N, F]
            )
            
            # Clamp membrane potential to prevent extreme values
            self.membrane_potential = torch.clamp(self.membrane_potential, -20.0, 20.0)
            
            # Generate positive and negative spikes
            pos_spikes = SpikeFunction.apply(
                self.membrane_potential, theta_pos, False
            ).squeeze(1)  # [B, N, F]
            
            neg_spikes = SpikeFunction.apply(
                self.membrane_potential, theta_neg, True  
            ).squeeze(1)  # [B, N, F]
            
            # Reset membrane potential after spiking
            if self.reset_mode == 'subtract':
                # Correct bipolar reset: V -= pos_spikes*θ_pos - neg_spikes*θ_neg
                reset_amount = (
                    pos_spikes * theta_pos - 
                    neg_spikes * theta_neg
                ).unsqueeze(1)  # [B, 1, N, F]
                self.membrane_potential = self.membrane_potential - reset_amount
                
            elif self.reset_mode == 'zero':
                # Reset to zero where any spike occurred
                spike_mask = (pos_spikes + neg_spikes > 0).unsqueeze(1)  # [B, 1, N, F]
                self.membrane_potential = self.membrane_potential * (~spike_mask).float()
            
            # Concatenate positive and negative spikes: [B, N, 2*F]
            bipolar_spikes = torch.cat([pos_spikes, neg_spikes], dim=-1)
            spike_output_list.append(bipolar_spikes)
            
            if return_membrane:
                membrane_history.append(self.membrane_potential.squeeze(1))  # [B, N, F]
        
        # Stack time dimension: [B, T, N, 2*F]
        spike_output = torch.stack(spike_output_list, dim=1)
        
        if return_membrane:
            membrane_potential = torch.stack(membrane_history, dim=1)  # [B, T, N, F]
            return spike_output, membrane_potential
        else:
            return spike_output
    
    def compute_firing_rates(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute positive and negative firing rates
        
        Args:
            spikes: [B, T, N, 2*F] bipolar spikes
            
        Returns:
            pos_rates: [B, N, F] positive firing rates
            neg_rates: [B, N, F] negative firing rates
        """
        F = spikes.shape[-1] // 2
        pos_spikes = spikes[:, :, :, :F]      # [B, T, N, F]
        neg_spikes = spikes[:, :, :, F:]      # [B, T, N, F]
        
        pos_rates = pos_spikes.mean(dim=1)    # [B, N, F]
        neg_rates = neg_spikes.mean(dim=1)    # [B, N, F]
        
        return pos_rates, neg_rates
    
    def get_neuron_statistics(self, 
                            spikes: torch.Tensor,
                            membrane_potential: Optional[torch.Tensor] = None) -> dict:
        """
        Compute neuron statistics
        
        Args:
            spikes: [B, T, N, 2*F] bipolar spikes
            membrane_potential: [B, T, N, F] membrane potential (optional)
            
        Returns:
            stats: neuron statistics
        """
        B, T, N, total_F = spikes.shape
        F = total_F // 2
        
        pos_spikes = spikes[:, :, :, :F]
        neg_spikes = spikes[:, :, :, F:]
        
        # Firing rate statistics
        pos_rates = pos_spikes.mean(dim=1)  # [B, N, F]
        neg_rates = neg_spikes.mean(dim=1)  # [B, N, F]
        
        stats = {
            'pos_firing_rate_mean': pos_rates.mean().item(),
            'pos_firing_rate_std': pos_rates.std().item(),
            'neg_firing_rate_mean': neg_rates.mean().item(),
            'neg_firing_rate_std': neg_rates.std().item(),
            'total_pos_spikes': pos_spikes.sum().item(),
            'total_neg_spikes': neg_spikes.sum().item(),
            'spike_balance': (pos_spikes.sum() / (neg_spikes.sum() + 1e-8)).item(),
            'sparsity': (spikes == 0).float().mean().item(),
        }
        
        # Membrane potential statistics
        if membrane_potential is not None:
            stats.update({
                'membrane_mean': membrane_potential.mean().item(),
                'membrane_std': membrane_potential.std().item(),
                'membrane_min': membrane_potential.min().item(),
                'membrane_max': membrane_potential.max().item(),
                'above_pos_threshold': (membrane_potential > self.theta_pos).float().mean().item(),
                'below_neg_threshold': (membrane_potential < self.theta_neg).float().mean().item(),
            })
        
        return stats


class AdaptiveBipolarLIFNeuron(BipolarLIFNeuron):
    """
    Adaptive bipolar LIF with learnable thresholds and time constants
    """
    
    def __init__(self, 
                 membrane_dim: int,
                 tau_mem: float = 20.0,
                 theta_pos: float = 1.0,
                 theta_neg: float = -1.0,
                 learnable_tau: bool = True,
                 learnable_threshold: bool = True,
                 reset_mode: str = 'subtract',
                 dt: float = 1.0):
        """
        Initialize adaptive bipolar LIF neuron
        
        Args:
            membrane_dim: dimension of membrane potential
            tau_mem: initial membrane time constant
            theta_pos: initial positive threshold
            theta_neg: initial negative threshold
            learnable_tau: whether tau is learnable
            learnable_threshold: whether thresholds are learnable
            reset_mode: reset mode
            dt: time step
        """
        super().__init__(membrane_dim, tau_mem, theta_pos, theta_neg, reset_mode, dt)
        
        # Learnable parameters
        if learnable_tau:
            self.log_tau_mem = nn.Parameter(torch.log(torch.tensor(tau_mem)))
        else:
            self.register_buffer('log_tau_mem', torch.log(torch.tensor(tau_mem)))
            
        if learnable_threshold:
            self.theta_pos_param = nn.Parameter(torch.tensor(theta_pos))
            self.theta_neg_param = nn.Parameter(torch.tensor(theta_neg))
            self.learnable_threshold = True
        else:
            self.learnable_threshold = False
    
    @property
    def tau_mem_current(self):
        """Current membrane time constant"""
        return torch.exp(self.log_tau_mem)
    
    @property 
    def alpha_current(self):
        """Current decay factor"""
        return torch.exp(-self.dt / self.tau_mem_current)
    
    def forward(self, 
                input_current: torch.Tensor,
                return_membrane: bool = False) -> torch.Tensor:
        """
        Forward pass with adaptive parameters
        
        Args:
            input_current: [B, T, N, F] input current
            return_membrane: whether to return membrane potential
            
        Returns:
            spike_output: [B, T, N, 2*F] bipolar spike output
            membrane_potential: [B, T, N, F] membrane potential (optional)
        """
        B, T, N, F = input_current.shape
        device = input_current.device
        
        # Initialize membrane potential if needed
        if self.membrane_potential.shape != (B, 1, N, F):
            self.reset_state(B, N, device)
        
        # Get current parameters
        alpha = self.alpha_current.to(device)
        
        if self.learnable_threshold:
            theta_pos = torch.abs(self.theta_pos_param)  # Force positive
            theta_neg = -torch.abs(self.theta_neg_param)  # Force negative
        else:
            theta_pos = self.theta_pos
            theta_neg = self.theta_neg
        
        spike_output_list = []
        membrane_history = []
        
        for t in range(T):
            # Membrane dynamics with adaptive tau
            self.membrane_potential = (
                alpha * self.membrane_potential + 
                input_current[:, t:t+1, :, :]
            )
            
            # Generate spikes
            pos_spikes = SpikeFunction.apply(
                self.membrane_potential, theta_pos, False
            ).squeeze(1)
            
            neg_spikes = SpikeFunction.apply(
                self.membrane_potential, theta_neg, True
            ).squeeze(1)
            
            # Reset membrane potential
            if self.reset_mode == 'subtract':
                reset_amount = (
                    pos_spikes * theta_pos - 
                    neg_spikes * theta_neg
                ).unsqueeze(1)
                self.membrane_potential = self.membrane_potential - reset_amount
                
            elif self.reset_mode == 'zero':
                spike_mask = (pos_spikes + neg_spikes > 0).unsqueeze(1)
                self.membrane_potential = self.membrane_potential * (~spike_mask).float()
            
            # Concatenate bipolar spikes
            bipolar_spikes = torch.cat([pos_spikes, neg_spikes], dim=-1)
            spike_output_list.append(bipolar_spikes)
            
            if return_membrane:
                membrane_history.append(self.membrane_potential.squeeze(1))
        
        spike_output = torch.stack(spike_output_list, dim=1)
        
        if return_membrane:
            membrane_potential = torch.stack(membrane_history, dim=1)
            return spike_output, membrane_potential
        else:
            return spike_output


class SpikingReadout(nn.Module):
    """
    Readout layer for spiking networks with rate-based decoding
    """
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 readout_mode: str = 'rate',
                 dropout: float = 0.5):
        """
        Initialize spiking readout layer
        
        Args:
            input_dim: input dimension (usually 2*F for bipolar)
            output_dim: output dimension (number of classes)
            readout_mode: decoding mode ('rate', 'count', 'last')
            dropout: dropout rate for regularization
        """
        super().__init__()
        self.readout_mode = readout_mode
        
        # Linear projection with dropout
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        Decode spike trains to output logits
        
        Args:
            spike_trains: [B, T, N, F] spike trains
            
        Returns:
            logits: [B, output_dim] classification logits
        """
        if self.readout_mode == 'rate':
            # Rate decoding: average over time
            rates = spike_trains.mean(dim=1)  # [B, N, F]
            
        elif self.readout_mode == 'count':
            # Count decoding: sum over time
            rates = spike_trains.sum(dim=1)  # [B, N, F]
            
        elif self.readout_mode == 'last':
            # Last time step
            rates = spike_trains[:, -1, :, :]  # [B, N, F]
            
        else:
            raise ValueError(f"Unknown readout mode: {self.readout_mode}")
        
        # Global pooling over nodes
        pooled = rates.mean(dim=1)  # [B, F]
        
        # Apply dropout for regularization
        pooled = self.dropout(pooled)
        
        # Linear classification
        logits = self.linear(pooled)  # [B, output_dim]
        
        return logits 