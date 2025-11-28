"""
Node-level Adaptive Graph Wavelet Convolution Layer
Implements the core innovation of density-aware wavelet convolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from ..utils.graph_utils import sparse_message_passing


class AdaptiveGraphWaveletConv(nn.Module):
    """
    Node-level adaptive graph wavelet convolution with Chebyshev approximation
    
    Core innovations:
    1. Node-level Chebyshev coefficients: α_k(s_i) = α_k0 + α_k1·s_i
    2. Full convolution kernel expressivity: [K+1, F_in, F_out]
    3. Efficient recursive computation without storing [B,N,N] matrices
    4. Single einsum acceleration for speed
    """
    
    def __init__(self, 
                 F_in: int,
                 F_out: int, 
                 K: int = 3,
                 bias: bool = True,
                 dropout: float = 0.0):
        """
        Initialize adaptive graph wavelet convolution layer
        
        Args:
            F_in: input feature dimension
            F_out: output feature dimension
            K: order of Chebyshev approximation
            bias: whether to use bias
            dropout: dropout rate
        """
        super().__init__()
        self.F_in = F_in
        self.F_out = F_out
        self.K = K
        
        # Node-level adaptive Chebyshev parameters
        # Theta_k(s_i) = Theta0_k + Theta1_k * s_i
        self.Theta0 = nn.Parameter(torch.randn(K+1, F_in, F_out))
        self.Theta1 = nn.Parameter(torch.randn(K+1, F_in, F_out))
        
        # Bias and regularization
        if bias:
            self.bias = nn.Parameter(torch.zeros(F_out))
        else:
            self.register_parameter('bias', None)
            
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters using Xavier initialization"""
        # Xavier initialization for Theta matrices
        gain = nn.init.calculate_gain('relu')
        
        nn.init.xavier_uniform_(self.Theta0, gain=gain)
        nn.init.xavier_uniform_(self.Theta1, gain=gain * 0.1)  # Smaller scale for adaptive part
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, 
                x: torch.Tensor,
                edge_indices: Union[torch.Tensor, List[torch.Tensor]], 
                edge_attrs: Union[torch.Tensor, List[torch.Tensor]],
                s_local: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of adaptive graph wavelet convolution (Memory-Efficient)
        
        Optimized implementation that avoids constructing the large Theta_local tensor.
        Instead, it computes contributions from Theta0 and Theta1 separately and combines
        them with node-specific scaling, reducing memory from O(B×N×K×F²) to O(K×N×F).
        
        Args:
            x: [B, N, F_in] input node features
            edge_indices: [2, E] edge indices or list of edge indices per batch
            edge_attrs: [E] edge weights or list of edge weights per batch  
            s_local: [B, N] local diffusion scales
            
        Returns:
            output: [B, N, F_out] convolved features
        """
        B, N, F_in = x.shape
        device = x.device
        
        # Process each batch - compute Theta_local on-the-fly to avoid large tensor
        output_list = []
        for b in range(B):
            # Get edges for batch b
            if isinstance(edge_indices, list):
                edge_idx = edge_indices[b]
                edge_attr = edge_attrs[b]
            else:
                # Extract edges for batch b
                batch_mask = (edge_indices[0] >= b*N) & (edge_indices[0] < (b+1)*N)
                edge_idx = edge_indices[:, batch_mask].clone()
                edge_attr = edge_attrs[batch_mask].clone()
                # Adjust node indices to be relative to current batch
                edge_idx[0] -= b*N
                edge_idx[1] -= b*N
            
            # Chebyshev polynomial basis: compute T_k * x for k=0...K
            Tx0 = x[b]  # T_0 * x (identity) -> shape [N, F_in]
            Tx = [Tx0]
            
            if self.K >= 1:
                # T_1 * x = L̃ * x
                Tx1 = sparse_message_passing(Tx0, edge_idx, edge_attr)  # [N, F_in]
                Tx.append(Tx1)
                
                # Recursive computation for higher orders
                for k in range(2, self.K + 1):
                    Txk = 2 * sparse_message_passing(Tx1, edge_idx, edge_attr) - Tx0  # T_k * x
                    Tx.append(Txk)
                    Tx0, Tx1 = Tx1, Txk
            
            # Stack Chebyshev outputs for all k: x_cheb has shape [K+1, N, F_in]
            x_cheb = torch.stack(Tx, dim=0)  # [K+1, N, F_in]
            
            # Compute contributions from Theta0 and Theta1 without building Theta_local
            # A = sum_{k=0}^K (Tx_k @ Theta0_k),  B = sum_{k=0}^K (Tx_k @ Theta1_k)
            A = torch.einsum('knf,kfo->no', x_cheb, self.Theta0)   # [N, F_out]
            B = torch.einsum('knf,kfo->no', x_cheb, self.Theta1)   # [N, F_out]
            
            # Combine contributions with node-specific scale s_local[b]
            # output[n,o] = A[n,o] + s_local[n] * B[n,o]
            batch_out = A + s_local[b].unsqueeze(-1) * B  # [N, F_out]
            
            output_list.append(batch_out)
        
        output = torch.stack(output_list, dim=0)  # [B, N, F_out]
        
        # Apply bias and dropout
        if self.bias is not None:
            output = output + self.bias
            
        output = self.dropout(output)
        
        return output
    
    def _process_single_batch(self,
                            x: torch.Tensor,
                            edge_index: torch.Tensor,
                            edge_attr: torch.Tensor, 
                            Theta_local: torch.Tensor) -> torch.Tensor:
        """
        Process single batch with recursive Chebyshev computation
        
        Args:
            x: [N, F_in] single batch features
            edge_index: [2, E] edge indices for this batch
            edge_attr: [E] edge weights for this batch
            Theta_local: [N, K+1, F_in, F_out] local parameters
            
        Returns:
            output: [N, F_out] convolved features
        """
        N, F_in = x.shape
        
        # Recursive Chebyshev computation: T_k(L̃)
        # T_0(L̃) = I, T_1(L̃) = L̃, T_k(L̃) = 2*L̃*T_{k-1} - T_{k-2}
        Tx = []  # Store T_k(L̃) * x for each k
        
        # T_0 * x = x
        Tx0 = x  # [N, F_in]
        Tx.append(Tx0)
        
        if self.K >= 1:
            # T_1 * x = L̃ * x
            Tx1 = sparse_message_passing(Tx0, edge_index, edge_attr)  # [N, F_in]
            Tx.append(Tx1)
            
            # Recursive computation for higher orders
            for k in range(2, self.K + 1):
                Txk = 2 * sparse_message_passing(Tx1, edge_index, edge_attr) - Tx0
                Tx.append(Txk)
                Tx0, Tx1 = Tx1, Txk
        
        # Stack all T_k * x: [K+1, N, F_in]
        Tx_stacked = torch.stack(Tx, dim=0)
        
        # Apply node-local convolution using einsum for efficiency
        # Tx_stacked: [K+1, N, F_in]
        # Theta_local: [N, K+1, F_in, F_out]  
        # Want: sum over k and F_in dimensions
        output = torch.einsum('knf,nkfo->no', Tx_stacked, Theta_local)  # [N, F_out]
        
        return output
    
    def get_computational_stats(self, 
                              B: int, 
                              N: int, 
                              num_edges: int) -> dict:
        """
        Compute computational complexity statistics
        
        Args:
            B: batch size
            N: number of nodes
            num_edges: total number of edges
            
        Returns:
            stats: computational complexity statistics
        """
        # FLOPs estimation
        message_passing_flops = num_edges * self.F_in  # Message aggregation
        chebyshev_recursion_flops = self.K * message_passing_flops  # K recursive steps
        convolution_flops = B * N * (self.K + 1) * self.F_in * self.F_out  # Final convolution
        
        total_flops = chebyshev_recursion_flops + convolution_flops
        
        # Memory estimation (in number of tensors of size [N, F])
        chebyshev_memory = 2 * N * self.F_in  # Store Tx0, Tx1
        parameter_memory = (self.K + 1) * self.F_in * self.F_out * 2  # Theta0, Theta1
        
        stats = {
            'total_flops': total_flops,
            'message_passing_flops': message_passing_flops,
            'chebyshev_recursion_flops': chebyshev_recursion_flops, 
            'convolution_flops': convolution_flops,
            'chebyshev_memory': chebyshev_memory,
            'parameter_memory': parameter_memory,
            'parameters': self.count_parameters(),
        }
        
        return stats
    
    def count_parameters(self) -> int:
        """Count total number of parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def visualize_adaptive_coefficients(self, 
                                      s_local: torch.Tensor,
                                      sample_idx: int = 0) -> dict:
        """
        Extract adaptive coefficients for visualization
        
        Args:
            s_local: [B, N] local diffusion scales
            sample_idx: which sample to visualize
            
        Returns:
            vis_data: visualization data
        """
        with torch.no_grad():
            s_sample = s_local[sample_idx]  # [N]
            
            # Compute local coefficients for this sample
            s_expanded = s_sample.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
            Theta_local = self.Theta0.unsqueeze(0) + self.Theta1.unsqueeze(0) * s_expanded  # [N, K+1, F_in, F_out]
            
            # Extract statistics
            coeff_norms = torch.norm(Theta_local, dim=(2, 3))  # [N, K+1]
            coeff_means = Theta_local.mean(dim=(2, 3))  # [N, K+1]
            coeff_stds = Theta_local.std(dim=(2, 3))   # [N, K+1]
            
            vis_data = {
                'scales': s_sample.cpu().numpy(),
                'coeff_norms': coeff_norms.cpu().numpy(),
                'coeff_means': coeff_means.cpu().numpy(), 
                'coeff_stds': coeff_stds.cpu().numpy(),
                'theta0_norm': torch.norm(self.Theta0, dim=(1, 2)).cpu().numpy(),
                'theta1_norm': torch.norm(self.Theta1, dim=(1, 2)).cpu().numpy(),
            }
            
            return vis_data


class MultiScaleGraphWaveletConv(nn.Module):
    """
    Multi-scale graph wavelet convolution with multiple fixed scales
    Alternative implementation for comparison with adaptive version
    """
    
    def __init__(self, 
                 F_in: int,
                 F_out: int,
                 scales: List[float] = [0.1, 0.5, 1.0, 2.0],
                 K: int = 3,
                 bias: bool = True):
        """
        Initialize multi-scale convolution
        
        Args:
            F_in: input feature dimension
            F_out: output feature dimension  
            scales: list of fixed diffusion scales
            K: order of Chebyshev approximation
            bias: whether to use bias
        """
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # One convolution layer per scale
        self.conv_layers = nn.ModuleList([
            AdaptiveGraphWaveletConv(F_in, F_out // self.num_scales, K, bias=False)
            for _ in range(self.num_scales)
        ])
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(F_out))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, 
                x: torch.Tensor,
                edge_indices: Union[torch.Tensor, List[torch.Tensor]],
                edge_attrs: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass with multiple fixed scales
        
        Args:
            x: [B, N, F_in] input features
            edge_indices: edge indices
            edge_attrs: edge weights
            
        Returns:
            output: [B, N, F_out] multi-scale convolved features
        """
        B, N, _ = x.shape
        
        outputs = []
        for i, (scale, conv_layer) in enumerate(zip(self.scales, self.conv_layers)):
            # Create constant scale tensor
            s_local = torch.full((B, N), scale, device=x.device, dtype=x.dtype)
            
            # Apply convolution at this scale
            scale_output = conv_layer(x, edge_indices, edge_attrs, s_local)
            outputs.append(scale_output)
        
        # Concatenate multi-scale features
        output = torch.cat(outputs, dim=-1)  # [B, N, F_out]
        
        if self.bias is not None:
            output = output + self.bias
            
        return output 