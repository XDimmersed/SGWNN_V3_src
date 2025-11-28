"""
带局部密度自适应的稀疏图构建器
---------------------------------
负责根据点云构建稀疏邻接并计算与密度相关的边权/扩散尺度。注释全面改为中文，
强调每一步的几何含义，便于理解后续小波卷积的输入。
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..utils.graph_utils import knn_search, compute_local_density


class SparseGraphBuilder(nn.Module):
    """
    从点云构建带局部密度感知的稀疏图。

    核心思路：
    1. 自适应 σ：σ_i = β·d_i + ε，用于平滑边权。
    2. 扩散尺度：s_i = λ·d_i²，为后续小波卷积提供节点局部尺度。
    3. 边权采用高斯核 exp(-dist²/(2σ_i²))，兼顾距离与密度。
    """
    
    def __init__(self, 
                 k: int = 20,
                 beta: float = 1.0,
                 lambda_param: float = 1.0,
                 epsilon: float = 1e-6,
                 density_method: str = 'kth_neighbor',
                 use_faiss: bool = True):
        """
        初始化稀疏图构建器。

        参数：
            k: kNN 的邻居数。
            beta: 自适应 σ 的缩放因子。
            lambda_param: 扩散尺度 s_i 的缩放因子。
            epsilon: 防止除零的稳定项。
            density_method: 计算局部密度的方式。
            use_faiss: 是否使用 FAISS 加速近邻搜索。
        """
        super().__init__()
        self.k = k
        self.beta = beta
        self.lambda_param = lambda_param
        self.epsilon = epsilon
        self.density_method = density_method
        self.use_faiss = use_faiss
        
        # 训练中可复用的缓存，避免重复构图
        self._cached_graphs = {}
        
    def forward(self, point_cloud: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据点云坐标构建自适应稀疏图。

        参数：
            point_cloud: [B, N, 3] 点云坐标。

        返回：
            edge_indices: 每个 batch 的边索引列表或拼接矩阵。
            edge_attrs: 对应的边权。
            s_local: 每个节点的局部扩散尺度。
        """
        B, N, D = point_cloud.shape
        device = point_cloud.device

        # 第一步：kNN 搜索获取邻居索引与距离
        knn_indices, knn_distances = knn_search(point_cloud, self.k, self.use_faiss)

        # 第二步：计算局部密度指标 d_i
        d_i = compute_local_density(knn_distances, self.density_method)  # [B, N]

        # 第三步：根据密度得到自适应参数
        # 本地 σ（影响边权）：σ_i = β·d_i + ε
        sigma_local = self.beta * d_i + self.epsilon  # [B, N]

        # 本地扩散尺度：s_i = λ·d_i²
        s_local = self.lambda_param * (d_i ** 2)  # [B, N]

        # 第四步：构建边并计算自适应权重
        if B * N * self.k < 100000:  # 图较小时可一次性拼接
            edge_indices, edge_attrs = self._batch_build_edges(
                point_cloud, knn_indices, knn_distances, sigma_local
            )
        else:  # 图较大时逐 batch 处理，降低显存占用
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
        在一个批次内整体构建边。

        参数：
            point_cloud: [B, N, 3] 坐标。
            knn_indices: [B, N, k] 邻居索引。
            knn_distances: [B, N, k] 邻居距离。
            sigma_local: [B, N] 本地 σ。

        返回：
            edge_indices: [2, total_edges] 拼接后的边索引。
            edge_attrs: [total_edges] 对应边权。
        """
        B, N, k = knn_indices.shape
        device = point_cloud.device
        
        edge_indices_list = []
        edge_attrs_list = []
        
        for b in range(B):
            # 本 batch 的源节点索引
            source_nodes = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
            target_nodes = knn_indices[b].reshape(-1)
            distances = knn_distances[b].reshape(-1)

            # 将 σ 扩展到每条边
            sigma_expanded = sigma_local[b].unsqueeze(1).expand(-1, k).reshape(-1)

            # 使用高斯核计算边权
            edge_weights = torch.exp(-distances ** 2 / (2 * sigma_expanded ** 2))

            # 为节点索引添加 batch 偏移，保证唯一性
            batch_offset = b * N
            source_batch = source_nodes + batch_offset
            target_batch = target_nodes + batch_offset

            # 收集边索引与权重
            edge_idx = torch.stack([source_batch, target_batch], dim=0)
            edge_indices_list.append(edge_idx)
            edge_attrs_list.append(edge_weights)

        # 拼接所有 batch 的边
        edge_indices = torch.cat(edge_indices_list, dim=1)
        edge_attrs = torch.cat(edge_attrs_list, dim=0)
        
        return edge_indices, edge_attrs
    
    def _build_single_batch_edges(self,
                                 points: torch.Tensor,
                                 knn_idx: torch.Tensor,
                                 knn_dist: torch.Tensor,
                                 sigma: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        针对单个 batch 构建边，逻辑与批量版本一致。

        参数与返回值与批量版本对应。
        """
        N, k = knn_idx.shape
        device = points.device

        # 源节点与目标节点
        source_nodes = torch.arange(N, device=device).unsqueeze(1).expand(-1, k).reshape(-1)
        target_nodes = knn_idx.reshape(-1)
        distances = knn_dist.reshape(-1)

        # 为每个源节点的 k 条边复制 σ
        sigma_expanded = sigma.unsqueeze(1).expand(-1, k).reshape(-1)

        # 计算自适应边权：exp(-dist²/(2σ_i²))
        edge_weights = torch.exp(-distances ** 2 / (2 * sigma_expanded ** 2))

        # 生成边索引
        edge_index = torch.stack([source_nodes, target_nodes], dim=0).long()
        
        return edge_index, edge_weights
    
    def clear_cache(self):
        """清空缓存以释放内存。"""
        self._cached_graphs.clear()
    
    def get_graph_statistics(self,
                           edge_indices: torch.Tensor,
                           edge_attrs: torch.Tensor,
                           num_nodes: int) -> dict:
        """
        统计构图结果，便于分析稀疏性和权值分布。

        参数：
            edge_indices: [2, E] 边索引。
            edge_attrs: [E] 边权。
            num_nodes: 节点数量。

        返回：
            stats: 图结构的统计字典。
        """
        from ..utils.graph_utils import compute_graph_statistics
        return compute_graph_statistics(edge_indices, edge_attrs, num_nodes)
    
    def visualize_local_adaptivity(self,
                                  point_cloud: torch.Tensor,
                                  s_local: torch.Tensor,
                                  sample_idx: int = 0) -> dict:
        """
        提取可视化局部自适应性的所需数据。

        参数：
            point_cloud: [B, N, 3] 点云坐标。
            s_local: [B, N] 节点扩散尺度。
            sample_idx: 选择展示的样本索引。

        返回：
            vis_data: 包含点坐标、尺度与密度的字典。
        """
        points = point_cloud[sample_idx].detach().cpu().numpy()  # [N, 3]
        scales = s_local[sample_idx].detach().cpu().numpy()      # [N]

        # 计算局部密度，用于可视化上色
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