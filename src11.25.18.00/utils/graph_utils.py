"""点云图构建与稀疏图算子工具。

模块亮点：
1. 提供 PyTorch/FAISS 的 k 近邻搜索接口，便于根据点云生成稀疏邻接。
2. 依据邻居距离估计局部密度，可直接用于自适应边权或扩散尺度。
3. 实现归一化拉普拉斯的稀疏消息传递，以及对称邻接与图统计分析。
"""

import torch
import torch.nn.functional as F
from torch_scatter import scatter_add
from typing import Tuple, Optional
import numpy as np


def knn_search(point_cloud: torch.Tensor, k: int,
               use_faiss: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """基于点云的 k 近邻搜索。

    参数：
        point_cloud: [B, N, 3] 点坐标。
        k: 近邻数量。
        use_faiss: 是否优先使用 FAISS 加速。

    返回：
        knn_indices: [B, N, k] 邻居索引。
        knn_distances: [B, N, k] 邻居距离。
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
    """纯 PyTorch 的 kNN 实现，适用于小批量或无 FAISS 环境。"""
    B, N, D = point_cloud.shape
    device = point_cloud.device
    
    # 计算两两欧氏距离矩阵 [B, N, N]
    point_cloud_expanded = point_cloud.unsqueeze(2)  # [B, N, 1, 3]
    point_cloud_repeated = point_cloud.unsqueeze(1)   # [B, 1, N, 3]

    # 欧氏距离范数
    distances = torch.norm(point_cloud_expanded - point_cloud_repeated, dim=-1)  # [B, N, N]

    # 取前 k+1（含自身）
    knn_distances, knn_indices = torch.topk(distances, k+1, dim=-1, largest=False)

    # 去掉自身（top1）
    knn_indices = knn_indices[:, :, 1:]  # [B, N, k]
    knn_distances = knn_distances[:, :, 1:]  # [B, N, k]
    
    return knn_indices, knn_distances


def _knn_search_faiss(point_cloud: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """FAISS 加速实现，适合大规模点云。"""
    try:
        import faiss
    except ImportError:
        raise ImportError("FAISS not installed. Install with: pip install faiss-gpu")
    
    B, N, D = point_cloud.shape
    device = point_cloud.device
    
    knn_indices_list = []
    knn_distances_list = []
    
    for b in range(B):
        # 转为 numpy 并构建 FAISS 索引
        points_np = point_cloud[b].detach().cpu().numpy().astype(np.float32)

        # 建立 L2 索引（欧式距离）
        index = faiss.IndexFlatL2(D)
        index.add(points_np)

        # 搜索 k+1 邻居（含自身）
        distances, indices = index.search(points_np, k+1)

        # 去掉自身对应的列
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
    """根据 kNN 距离估计局部密度，数值越大表示越稀疏。"""
    if density_method == 'kth_neighbor':
        # 第 k 个邻居距离
        local_density = knn_distances[:, :, -1]  # [B, N]
    elif density_method == 'mean':
        # 平均邻居距离
        local_density = knn_distances.mean(dim=-1)  # [B, N]
    elif density_method == 'std':
        # 邻居距离标准差
        local_density = knn_distances.std(dim=-1)  # [B, N]
    else:
        raise ValueError(f"Unknown density method: {density_method}")
    
    return local_density


def sparse_message_passing(x: torch.Tensor,
                          edge_index: torch.Tensor,
                          edge_attr: torch.Tensor,
                          epsilon: float = 1e-8) -> torch.Tensor:
    """稀疏消息传递：计算归一化拉普拉斯 L̃x = x - Âx。"""
    N, F = x.shape
    source, target = edge_index

    # 计算节点度并做归一化
    deg = scatter_add(edge_attr, target, dim=0, dim_size=N)
    deg_inv_sqrt = torch.pow(deg + epsilon, -0.5)

    # D^{-1/2} A D^{-1/2}
    norm_edge_attr = deg_inv_sqrt[source] * edge_attr * deg_inv_sqrt[target]

    # 消息聚合：Âx
    messages = x[source] * norm_edge_attr.unsqueeze(-1)  # [E, F]
    aggregated = scatter_add(messages, target, dim=0, dim_size=N)  # [N, F]

    # 拉普拉斯输出
    return x - aggregated


def build_symmetric_adjacency(knn_indices: torch.Tensor,
                             edge_weights: torch.Tensor,
                             N: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """依据 kNN 结果生成对称邻接，方便无向图卷积。"""
    device = knn_indices.device
    k = knn_indices.shape[1]
    
    # Source and target nodes
    source_nodes = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
    target_nodes = knn_indices.reshape(-1)
    edge_weights_flat = edge_weights.reshape(-1)
    
    # 创建双向边保持对称性
    edge_index = torch.stack([
        torch.cat([source_nodes, target_nodes]),
        torch.cat([target_nodes, source_nodes])
    ], dim=0).long()
    
    edge_attr = torch.cat([edge_weights_flat, edge_weights_flat])
    
    return edge_index, edge_attr


def compute_graph_statistics(edge_index: torch.Tensor,
                            edge_attr: torch.Tensor,
                            num_nodes: int) -> dict:
    """统计图的连通性与边权分布，用于分析构图质量。"""
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