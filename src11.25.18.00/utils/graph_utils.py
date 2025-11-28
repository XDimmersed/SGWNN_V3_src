"""
Graph construction utilities for point cloud processing
"""

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import Tuple, Optional
import numpy as np


def knn_search(point_cloud: torch.Tensor, k: int, 
               use_faiss: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    K-nearest neighbor search for point clouds
    
    Args:
        point_cloud: [B, N, 3] point coordinates
        k: number of nearest neighbors
        use_faiss: whether to use FAISS for acceleration
        
    Returns:
        knn_indices: [B, N, k] neighbor indices
        knn_distances: [B, N, k] neighbor distances
    """
    B, N, D = point_cloud.shape
    device = point_cloud.device
    
    if use_faiss and D == 3:
        try:
            return _knn_search_faiss(point_cloud, k)
        except ImportError:
            print("FAISS not available, falling back to PyTorch implementation")
    
    return _knn_search_pytorch(point_cloud, k)


def _knn_search_pytorch(point_cloud: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch implementation of kNN search"""
    B, N, D = point_cloud.shape
    device = point_cloud.device
    
    # Compute pairwise distances: [B, N, N]
    point_cloud_expanded = point_cloud.unsqueeze(2)  # [B, N, 1, 3]
    point_cloud_repeated = point_cloud.unsqueeze(1)   # [B, 1, N, 3]
    
    # Euclidean distance
    distances = torch.norm(point_cloud_expanded - point_cloud_repeated, dim=-1)  # [B, N, N]
    
    # Find k+1 nearest neighbors (including self)
    knn_distances, knn_indices = torch.topk(distances, k+1, dim=-1, largest=False)
    
    # Remove self (index 0)
    knn_indices = knn_indices[:, :, 1:]  # [B, N, k]
    knn_distances = knn_distances[:, :, 1:]  # [B, N, k]
    
    return knn_indices, knn_distances


def _knn_search_faiss(point_cloud: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """FAISS implementation for faster kNN search"""
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS not installed. Install with: pip install faiss-gpu")
    
    B, N, D = point_cloud.shape
    device = point_cloud.device
    
    knn_indices_list = []
    knn_distances_list = []
    
    for b in range(B):
        # Convert to numpy and build FAISS index
        points_np = point_cloud[b].detach().cpu().numpy().astype(np.float32)
        
        # Build index
        index = faiss.IndexFlatL2(D)
        index.add(points_np)
        
        # Search k+1 neighbors (including self)
        distances, indices = index.search(points_np, k+1)
        
        # Remove self (first column)
        distances = distances[:, 1:]  # [N, k]
        indices = indices[:, 1:]      # [N, k]
        
        # Convert back to torch tensors
        knn_distances_list.append(torch.from_numpy(distances).to(device))
        knn_indices_list.append(torch.from_numpy(indices).long().to(device))
    
    knn_distances = torch.stack(knn_distances_list, dim=0)  # [B, N, k]
    knn_indices = torch.stack(knn_indices_list, dim=0)      # [B, N, k]
    
    return knn_indices, knn_distances


def compute_local_density(knn_distances: torch.Tensor, 
                         density_method: str = 'kth_neighbor') -> torch.Tensor:
    """
    Compute local density indicator for each point
    
    Args:
        knn_distances: [B, N, k] k-nearest neighbor distances
        density_method: method to compute density ('kth_neighbor', 'mean', 'std')
        
    Returns:
        local_density: [B, N] density indicator (larger = sparser)
    """
    if density_method == 'kth_neighbor':
        # Use k-th neighbor distance as density indicator
        local_density = knn_distances[:, :, -1]  # [B, N]
    elif density_method == 'mean':
        # Use mean distance to k neighbors
        local_density = knn_distances.mean(dim=-1)  # [B, N]
    elif density_method == 'std':
        # Use standard deviation of distances
        local_density = knn_distances.std(dim=-1)  # [B, N]
    else:
        raise ValueError(f"Unknown density method: {density_method}")
    
    return local_density


def sparse_message_passing(x: torch.Tensor, 
                          edge_index: torch.Tensor, 
                          edge_attr: torch.Tensor,
                          epsilon: float = 1e-8) -> torch.Tensor:
    """
    Sparse message passing: L̃x = x - Âx where Â is normalized adjacency
    
    Args:
        x: [N, F] node features
        edge_index: [2, E] edge indices (source, target)
        edge_attr: [E] edge weights
        epsilon: small value to avoid division by zero
        
    Returns:
        output: [N, F] transformed features L̃x
    """
    N, F = x.shape
    source, target = edge_index
    
    # Compute node degrees
    deg = scatter_add(edge_attr, target, dim=0, dim_size=N)
    deg_inv_sqrt = torch.pow(deg + epsilon, -0.5)
    
    # Normalize edge weights: D^{-1/2} A D^{-1/2}
    norm_edge_attr = deg_inv_sqrt[source] * edge_attr * deg_inv_sqrt[target]
    
    # Message passing: Âx
    messages = x[source] * norm_edge_attr.unsqueeze(-1)  # [E, F]
    aggregated = scatter_add(messages, target, dim=0, dim_size=N)  # [N, F]
    
    # L̃x = x - Âx (normalized Laplacian)
    return x - aggregated


def build_symmetric_adjacency(knn_indices: torch.Tensor, 
                             edge_weights: torch.Tensor,
                             N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build symmetric adjacency matrix from k-NN graph
    
    Args:
        knn_indices: [N, k] neighbor indices for each node
        edge_weights: [N, k] edge weights
        N: number of nodes
        
    Returns:
        edge_index: [2, E] symmetric edge indices
        edge_attr: [E] symmetric edge weights
    """
    device = knn_indices.device
    k = knn_indices.shape[1]
    
    # Source and target nodes
    source_nodes = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    target_nodes = knn_indices.reshape(-1)
    edge_weights_flat = edge_weights.reshape(-1)
    
    # Create bidirectional edges
    edge_index = torch.stack([
        torch.cat([source_nodes, target_nodes]),
        torch.cat([target_nodes, source_nodes])
    ], dim=0).long()
    
    edge_attr = torch.cat([edge_weights_flat, edge_weights_flat])
    
    return edge_index, edge_attr


def compute_graph_statistics(edge_index: torch.Tensor, 
                            edge_attr: torch.Tensor, 
                            num_nodes: int) -> dict:
    """
    Compute graph connectivity statistics
    
    Args:
        edge_index: [2, E] edge indices
        edge_attr: [E] edge weights
        num_nodes: number of nodes
        
    Returns:
        stats: dictionary of graph statistics
    """
    from torch_scatter import scatter_add
    
    # Node degrees
    degrees = scatter_add(edge_attr, edge_index[1], dim=0, dim_size=num_nodes)
    
    # Edge weight statistics
    stats = {
        'num_nodes': num_nodes,
        'num_edges': edge_index.shape[1],
        'avg_degree': degrees.mean().item(),
        'max_degree': degrees.max().item(),
        'min_degree': degrees.min().item(),
        'avg_edge_weight': edge_attr.mean().item(),
        'max_edge_weight': edge_attr.max().item(),
        'min_edge_weight': edge_attr.min().item(),
    }
    
    return stats 