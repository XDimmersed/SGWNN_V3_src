"""
Spiking Graph Wavelet Convolution Network (SGWCN)
Complete integration of all core innovations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

from .graph_builder import SparseGraphBuilder
from .wavelet_conv import AdaptiveGraphWaveletConv
from .spiking_neurons import BipolarLIFNeuron, SpikingReadout
from ..utils.spike_utils import poisson_encoding


class SpikingGraphWaveletNet(nn.Module):
    """
    Complete Spiking Graph Wavelet Convolution Network
    
    Architecture:
    [B,N,3] → Graph → [B,N,k] edges → Wavelet Conv → [B,T,N,F] spikes → Classification
    
    Core innovations:
    1. Local density adaptive graph construction
    2. Node-level adaptive graph wavelet convolution  
    3. Bipolar LIF neurons with dual thresholds
    """
    
    def __init__(self,
                 input_dim: int = 3,
                 hidden_dims: List[int] = [64, 128, 256],
                 num_classes: int = 40,
                 num_time_steps: int = 10,
                 k_neighbors: int = 20,
                 chebyshev_order: int = 3,
                 # Graph construction parameters
                 beta: float = 1.0,
                 lambda_param: float = 1.0,
                 epsilon: float = 1e-6,
                 # Spiking neuron parameters
                 tau_mem: float = 20.0,
                 theta_pos: float = 1.0,
                 theta_neg: float = -1.0,
                 # Training parameters
                 dropout: float = 0.1,
                 use_faiss: bool = True,
                 readout_mode: str = 'rate'):
        """
        Initialize SGWCN
        
        Args:
            input_dim: input feature dimension (3 for xyz coordinates)
            hidden_dims: list of hidden layer dimensions
            num_classes: number of output classes
            num_time_steps: number of spiking time steps
            k_neighbors: number of nearest neighbors for graph
            chebyshev_order: order of Chebyshev approximation
            beta: scaling factor for adaptive σ
            lambda_param: scaling factor for diffusion scales
            epsilon: small value for numerical stability
            tau_mem: membrane time constant
            theta_pos: positive spike threshold
            theta_neg: negative spike threshold
            dropout: dropout rate
            use_faiss: whether to use FAISS for kNN
            readout_mode: spike decoding mode
        """
        super().__init__()
        
        self.num_time_steps = num_time_steps
        self.num_classes = num_classes
        
        # Graph construction
        self.graph_builder = SparseGraphBuilder(
            k=k_neighbors,
            beta=beta,
            lambda_param=lambda_param,
            epsilon=epsilon,
            use_faiss=use_faiss
        )
        
        # Feature encoding (optional: learn initial features from coordinates)
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Graph wavelet convolution layers
        self.conv_layers = nn.ModuleList()
        self.spiking_layers = nn.ModuleList()
        
        # Build layers
        layer_dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(layer_dims) - 1):
            # Wavelet convolution
            conv_layer = AdaptiveGraphWaveletConv(
                F_in=layer_dims[i] * 2 if i > 0 else layer_dims[i],  # *2 for bipolar spikes
                F_out=layer_dims[i+1],
                K=chebyshev_order,
                dropout=dropout
            )
            self.conv_layers.append(conv_layer)
            
            # Bipolar LIF neuron
            spiking_layer = BipolarLIFNeuron(
                membrane_dim=layer_dims[i+1],
                tau_mem=tau_mem,
                theta_pos=theta_pos,
                theta_neg=theta_neg
            )
            self.spiking_layers.append(spiking_layer)
        
        # Final readout with dropout
        final_dim = hidden_dims[-1] * 2  # *2 for bipolar output
        self.readout = SpikingReadout(
            input_dim=final_dim,
            output_dim=num_classes,
            readout_mode=readout_mode,
            dropout=dropout
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of SGWCN
        
        Args:
            point_cloud: [B, N, 3] input point cloud coordinates
            
        Returns:
            logits: [B, num_classes] classification logits
        """
        B, N, _ = point_cloud.shape
        device = point_cloud.device
        
        # Step 1: Build adaptive graphs
        edge_indices, edge_attrs, s_local = self.graph_builder(point_cloud)
        
        # Step 2: Encode initial features
        x = self.feature_encoder(point_cloud)  # [B, N, hidden_dims[0]]
        
        # Step 3: Convert to spike trains
        spike_trains = poisson_encoding(x, self.num_time_steps)  # [B, T, N, F]
        
        # Reset all spiking neuron states
        for layer in self.spiking_layers:
            layer.reset_state(B, N, device)
        
        # Step 4: Process through graph wavelet convolution layers
        for i, (conv_layer, spike_layer) in enumerate(zip(self.conv_layers, self.spiking_layers)):
            layer_outputs = []
            
            # Process each time step
            for t in range(self.num_time_steps):
                # Current time step features
                x_t = spike_trains[:, t, :, :]  # [B, N, F]
                
                # Graph wavelet convolution
                conv_out = conv_layer(x_t, edge_indices, edge_attrs, s_local)  # [B, N, F_out]
                layer_outputs.append(conv_out)
            
            # Stack time dimension and apply spiking neuron
            conv_output = torch.stack(layer_outputs, dim=1)  # [B, T, N, F_out]
            spike_trains = spike_layer(conv_output)  # [B, T, N, 2*F_out]
        
        # Step 5: Readout classification
        logits = self.readout(spike_trains)  # [B, num_classes]
        
        return logits
    
    def forward_with_analysis(self, point_cloud: torch.Tensor) -> Dict:
        """
        Forward pass with detailed analysis for debugging/visualization
        
        Args:
            point_cloud: [B, N, 3] input point cloud
            
        Returns:
            analysis: dictionary with intermediate results and statistics
        """
        B, N, _ = point_cloud.shape
        device = point_cloud.device
        
        analysis = {
            'layer_outputs': [],
            'spike_statistics': [],
            'graph_statistics': {},
            'energy_consumption': 0.0
        }
        
        # Build graphs with statistics
        edge_indices, edge_attrs, s_local = self.graph_builder(point_cloud)

        if isinstance(edge_indices, list):
            total_edges = sum(edge_idx.shape[1] for edge_idx in edge_indices)
            # Use first batch for statistics (single batch analysis)
            analysis['graph_statistics'] = self.graph_builder.get_graph_statistics(
                edge_indices[0], edge_attrs[0], N
            )
        else:
            total_edges = edge_indices.shape[1]
            # For concatenated edge indices, we need to extract first batch
            # Edge indices are in range [0, B*N-1], we need [0, N-1] for first batch
            batch_mask = (edge_indices[0] < N) & (edge_indices[1] < N)
            first_batch_edges = edge_indices[:, batch_mask]
            first_batch_attrs = edge_attrs[batch_mask]
            
            analysis['graph_statistics'] = self.graph_builder.get_graph_statistics(
                first_batch_edges, first_batch_attrs, N
            )
        
        # Initial encoding
        x = self.feature_encoder(point_cloud)
        spike_trains = poisson_encoding(x, self.num_time_steps)
        
        # Reset states
        for layer in self.spiking_layers:
            layer.reset_state(B, N, device)
        
        # Process layers with analysis
        for i, (conv_layer, spike_layer) in enumerate(zip(self.conv_layers, self.spiking_layers)):
            layer_outputs = []
            
            for t in range(self.num_time_steps):
                x_t = spike_trains[:, t, :, :]
                conv_out = conv_layer(x_t, edge_indices, edge_attrs, s_local)
                layer_outputs.append(conv_out)
            
            conv_output = torch.stack(layer_outputs, dim=1)
            spike_trains, membrane_potential = spike_layer(conv_output, return_membrane=True)
            
            # Collect statistics
            layer_stats = spike_layer.get_neuron_statistics(spike_trains, membrane_potential)
            analysis['spike_statistics'].append(layer_stats)
            analysis['layer_outputs'].append({
                'conv_output': conv_output.detach(),
                'spike_output': spike_trains.detach(),
                'membrane_potential': membrane_potential.detach()
            })
            
            # Estimate energy consumption (spikes * energy_per_spike)
            total_spikes = spike_trains.sum().item()
            analysis['energy_consumption'] += total_spikes * 1e-12  # pJ per spike
        
        # Final classification
        logits = self.readout(spike_trains)
        analysis['logits'] = logits.detach()
        
        return analysis
    
    def get_model_statistics(self) -> Dict:
        """Get comprehensive model statistics"""
        total_params = sum(p.numel() for p in self.parameters())
        
        stats = {
            'total_parameters': total_params,
            'num_layers': len(self.conv_layers),
            'num_time_steps': self.num_time_steps,
            'graph_builder_params': sum(p.numel() for p in self.graph_builder.parameters()),
            'conv_layer_params': [sum(p.numel() for p in layer.parameters()) for layer in self.conv_layers],
            'spiking_layer_params': [sum(p.numel() for p in layer.parameters()) for layer in self.spiking_layers],
            'readout_params': sum(p.numel() for p in self.readout.parameters()),
        }
        
        return stats
    
    def estimate_energy_consumption(self, num_samples: int, points_per_sample: int) -> Dict:
        """
        Estimate energy consumption compared to traditional ANNs
        
        Args:
            num_samples: number of input samples
            points_per_sample: number of points per sample
            
        Returns:
            energy_stats: energy consumption estimates
        """
        # Rough estimates based on literature
        ENERGY_PER_SPIKE = 1e-12  # 1 pJ per spike
        ENERGY_PER_FLOP = 1e-15   # 1 fJ per FLOP (for comparison)
        
        # Estimate spikes per forward pass
        estimated_spikes_per_layer = []
        for i, layer in enumerate(self.spiking_layers):
            # Assume ~10% firing rate for bipolar neurons
            layer_dim = layer.membrane_dim * 2  # bipolar output
            spikes_per_timestep = num_samples * points_per_sample * layer_dim * 0.1
            total_spikes = spikes_per_timestep * self.num_time_steps
            estimated_spikes_per_layer.append(total_spikes)
        
        total_spikes = sum(estimated_spikes_per_layer)
        snn_energy = total_spikes * ENERGY_PER_SPIKE
        
        # Compare with equivalent ANN (rough estimate)
        total_flops = num_samples * points_per_sample * sum(
            layer.count_parameters() for layer in self.conv_layers
        ) * 2  # 2 FLOPs per MAC
        ann_energy = total_flops * ENERGY_PER_FLOP
        
        energy_stats = {
            'total_spikes': total_spikes,
            'spikes_per_layer': estimated_spikes_per_layer,
            'snn_energy_joules': snn_energy,
            'estimated_ann_energy_joules': ann_energy,
            'energy_reduction_factor': ann_energy / snn_energy if snn_energy > 0 else float('inf'),
            'energy_per_sample_nj': snn_energy / num_samples * 1e9,  # nanojoules
        }
        
        return energy_stats


class SGWCNClassifier(SpikingGraphWaveletNet):
    """
    SGWCN specifically configured for point cloud classification
    Pre-configured for common datasets like ModelNet40
    """
    
    def __init__(self,
                 num_classes: int = 40,
                 num_points: int = 1024,
                 **kwargs):
        """
        Initialize classifier with sensible defaults
        
        Args:
            num_classes: number of classes (40 for ModelNet40)
            num_points: number of points per sample
            **kwargs: additional arguments for SpikingGraphWaveletNet
        """
        # Default configuration optimized for point cloud classification
        defaults = {
            'hidden_dims': [64, 128, 256],
            'num_time_steps': 8,
            'k_neighbors': 20,
            'chebyshev_order': 3,
            'beta': 1.0,
            'lambda_param': 1.0,
            'tau_mem': 20.0,
            'theta_pos': 1.0,
            'theta_neg': -1.0,
            'dropout': 0.1,
            'readout_mode': 'rate'
        }
        
        # Update with user-provided arguments
        defaults.update(kwargs)
        
        super().__init__(
            num_classes=num_classes,
            **defaults
        )
        
        self.num_points = num_points
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass optimized for classification
        
        Args:
            data: [B, N, 3] or [B, N, C] point cloud data
            
        Returns:
            logits: [B, num_classes] classification logits
        """
        # Handle different input formats
        if data.shape[-1] > 3:
            # If more than 3 channels, use only xyz coordinates
            point_cloud = data[:, :, :3]
        else:
            point_cloud = data
        
        # Subsample if too many points
        if point_cloud.shape[1] > self.num_points:
            indices = torch.randperm(point_cloud.shape[1])[:self.num_points]
            point_cloud = point_cloud[:, indices, :]
        
        return super().forward(point_cloud) 