"""
Sparse Graph Builder with Local Density Adaptation
Implements the core innovation of density-aware graph construction
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..utils.graph_utils import knn_search, compute_local_density


class SparseGraphBuilder(nn.Module):
    """
    Build adaptive sparse graphs from point clouds with local density awareness
    
    Core innovations:
    1. Local adaptive σ: σ_i = β·d_i + ε (density-aware edge weights)
    2. Diffusion scale: s_i = λ·d_i² (density-aware wavelet scales)
    3. Standard Gaussian kernel: exp(-dist²/(2σ_i²))
    """
    
    def __init__(self, 
                 k: int = 20,
                 beta: float = 1.0,
                 lambda_param: float = 1.0,
                 epsilon: float = 1e-6,
                 density_method: str = 'kth_neighbor',
                 use_faiss: bool = True):
        """
        Initialize sparse graph builder
        
        Args:
            k: number of nearest neighbors
            beta: scaling factor for adaptive σ
            lambda_param: scaling factor for diffusion scale s_i
            epsilon: small value to avoid division by zero
            density_method: method to compute local density
            use_faiss: whether to use FAISS acceleration
        """
        super().__init__()
        self.k = k
        self.beta = beta
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.density_method = density_method
        self.use_faiss = use_faiss
        
        # Cache for avoiding repeated computation during training
        self._cached_graphs = {}
        
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build adaptive graphs from point clouds
        
        Args:
            point_cloud: [B, N, 3] point coordinates
            
        Returns:
            edge_indices: list of [2, E_b] edge indices for each batch
            edge_attrs: list of [E_b] edge weights for each batch  
            s_local: [B, N] local diffusion scales
        """
        B, N, D = point_cloud.shape
        device = point_cloud.device
        
        # Step 1: k-NN search
        knn_indices, knn_distances = knn_search(point_cloud, self.k, self.use_faiss)
        
        # Step 2: Compute local density indicators
        d_i = compute_local_density(knn_distances, self.density_method)  # [B, N]
        
        # Step 3: Compute adaptive parameters
        # Local σ for edge weights: σ_i = β·d_i + ε
        sigma_local = self.beta * d_i + self.epsilon  # [B, N]
        
        # Local diffusion scales: s_i = λ·d_i²
        s_local = self.lambda_param * (d_i ** 2)  # [B, N]
        
        # Step 4: Build edges and compute adaptive weights
        if B * N * self.k < 100000:  # Use batch processing for small graphs
            edge_indices, edge_attrs = self._batch_build_edges(
                point_cloud, knn_indices, knn_distances, sigma_local
            )
        else:  # Process each batch separately for large graphs
            edge_indices = []
            edge_attrs = []
            for b in range(B):
                edge_idx, edge_attr = self._build_single_batch_edges(
                    point_cloud[b], knn_indices[b], knn_distances[b], sigma_local[b]
                )
                edge_indices.append(edge_idx)
                edge_attrs.append(edge_attr)
        
        return edge_indices, edge_attrs, s_local
    
    def _batch_build_edges(self, 
                          point_cloud: torch.Tensor,
                          knn_indices: torch.Tensor, 
                          knn_distances: torch.Tensor,
                          sigma_local: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges for entire batch simultaneously
        
        Args:
            point_cloud: [B, N, 3] coordinates
            knn_indices: [B, N, k] neighbor indices
            knn_distances: [B, N, k] neighbor distances  
            sigma_local: [B, N] local sigma values
            
        Returns:
            edge_indices: [2, total_edges] concatenated edge indices
            edge_attrs: [total_edges] concatenated edge weights
        """
        B, N, k = knn_indices.shape
        device = point_cloud.device
        
        edge_indices_list = []
        edge_attrs_list = []
        
        for b in range(B):
            # Source nodes for this batch
            source_nodes = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            target_nodes = knn_indices[b].reshape(-1)
            distances = knn_distances[b].reshape(-1)
            
            # Adaptive sigma for edge weights
            sigma_expanded = sigma_local[b].unsqueeze(1).expand(-1, k).reshape(-1)
            
            # Compute edge weights using standard Gaussian kernel
            edge_weights = torch.exp(-distances ** 2 / (2 * sigma_expanded ** 2))
            
            # Add batch offset to node indices
            batch_offset = b * N
            source_batch = source_nodes + batch_offset
            target_batch = target_nodes + batch_offset
            
            # Store edges
            edge_idx = torch.stack([source_batch, target_batch], dim=0)
            edge_indices_list.append(edge_idx)
            edge_attrs_list.append(edge_weights)
        
        # Concatenate all batches
        edge_indices = torch.cat(edge_indices_list, dim=1)
        edge_attrs = torch.cat(edge_attrs_list, dim=0)
        
        return edge_indices, edge_attrs
    
    def _build_single_batch_edges(self,
                                 points: torch.Tensor,
                                 knn_idx: torch.Tensor,
                                 knn_dist: torch.Tensor, 
                                 sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build edges for a single batch
        
        Args:
            points: [N, 3] point coordinates
            knn_idx: [N, k] neighbor indices
            knn_dist: [N, k] neighbor distances
            sigma: [N] local sigma values
            
        Returns:
            edge_index: [2, E] edge indices
            edge_attr: [E] edge weights
        """
        N, k = knn_idx.shape
        device = points.device
        
        # Source and target nodes
        source_nodes = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        target_nodes = knn_idx.reshape(-1)
        distances = knn_dist.reshape(-1)
        
        # Expand sigma for all edges from each source node
        sigma_expanded = sigma.unsqueeze(1).expand(-1, k).reshape(-1)
        
        # Compute adaptive edge weights: exp(-dist²/(2σ_i²))
        edge_weights = torch.exp(-distances ** 2 / (2 * sigma_expanded ** 2))
        
        # Create edge index
        edge_index = torch.stack([source_nodes, target_nodes], dim=0).long()
        
        return edge_index, edge_weights
    
    def clear_cache(self):
        """Clear cached graphs to free memory"""
        self._cached_graphs.clear()
    
    def get_graph_statistics(self, 
                           edge_indices: torch.Tensor,
                           edge_attrs: torch.Tensor,
                           num_nodes: int) -> dict:
        """
        Compute statistics of the constructed graph
        
        Args:
            edge_indices: [2, E] edge indices
            edge_attrs: [E] edge weights
            num_nodes: number of nodes
            
        Returns:
            stats: dictionary of graph statistics
        """
        from ..utils.graph_utils import compute_graph_statistics
        return compute_graph_statistics(edge_indices, edge_attrs, num_nodes)
    
    def visualize_local_adaptivity(self, 
                                  point_cloud: torch.Tensor,
                                  s_local: torch.Tensor,
                                  sample_idx: int = 0) -> dict:
        """
        Extract data for visualizing local adaptivity
        
        Args:
            point_cloud: [B, N, 3] point coordinates
            s_local: [B, N] local diffusion scales
            sample_idx: which sample to visualize
            
        Returns:
            vis_data: dictionary containing visualization data
        """
        points = point_cloud[sample_idx].detach().cpu().numpy()  # [N, 3]
        scales = s_local[sample_idx].detach().cpu().numpy()      # [N]
        
        # Compute local density for coloring
        knn_indices, knn_distances = knn_search(point_cloud[sample_idx:sample_idx+1], self.k, False)
        d_i = compute_local_density(knn_distances, self.density_method)
        densities = d_i[0].detach().cpu().numpy()  # [N]
        
        vis_data = {
            'points': points,
            'scales': scales,
            'densities': densities,
            'sigma_values': self.beta * densities + self.epsilon,
        }
        
        return vis_data 