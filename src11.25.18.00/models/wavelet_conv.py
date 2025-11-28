"""节点级自适应图小波卷积层，实现密度感知的小波核。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union
from ..utils.graph_utils import sparse_message_passing


class AdaptiveGraphWaveletConv(nn.Module):
    """节点自适应的图小波卷积（Chebyshev 近似）。

    关键设计：
    1. 节点级切比雪夫系数 α_k(s_i) = α_k0 + α_k1·s_i，随局部尺度自适应。
    2. 完整核表达力：[K+1, F_in, F_out] 的权重矩阵保持通道灵活性。
    3. 递归计算 Chebyshev 多项式，避免显式构造 [B,N,N] 大矩阵。
    4. 单次 einsum 完成卷积累积，兼顾速度与显存。
    """
    
    def __init__(self, 
                 F_in: int,
                 F_out: int,
                 K: int = 3,
                 bias: bool = True,
                 dropout: float = 0.0):
        """初始化自适应小波卷积层。

        参数：
            F_in: 输入通道数。
            F_out: 输出通道数。
            K: 切比雪夫近似阶数。
            bias: 是否使用偏置。
            dropout: dropout 比例，用于正则化。
        """
        super().__init__()
        self.F_in = F_in
        self.F_out = F_out
        self.K = K
        
        # 节点级自适应切比雪夫系数（原英文注释“Node-level adaptive Chebyshev parameters”）
        # Theta_k(s_i) = Theta0_k + Theta1_k * s_i（原公式说明保留便于对照英文文献）
        self.Theta0 = nn.Parameter(torch.randn(K+1, F_in, F_out))
        self.Theta1 = nn.Parameter(torch.randn(K+1, F_in, F_out))
        
        # 偏置与正则化配置（原英文注释“Bias and regularization”）
        if bias:
            self.bias = nn.Parameter(torch.zeros(F_out))
        else:
            self.register_parameter('bias', None)

        self.dropout = nn.Dropout(dropout)

        # 参数初始化（原英文注释“Initialize parameters”）
        self.reset_parameters()
    
    def reset_parameters(self):
        """使用 Xavier 初始化权重，减小梯度不稳定风险。"""
        # 对 Theta 矩阵使用 Xavier 初始化（原英文注释“Xavier initialization for Theta matrices”）
        gain = nn.init.calculate_gain('relu')
        
        nn.init.xavier_uniform_(self.Theta0, gain=gain)
        nn.init.xavier_uniform_(self.Theta1, gain=gain * 0.1)  # 自适应部分缩小初始化尺度（原英文注释“Smaller scale for adaptive part”）
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, 
                x: torch.Tensor,
                edge_indices: Union[torch.Tensor, List[torch.Tensor]],
                edge_attrs: Union[torch.Tensor, List[torch.Tensor]],
                s_local: torch.Tensor) -> torch.Tensor:
        """前向计算：显存友好的节点自适应小波卷积。

        核心思路：不显式构造 ``Theta_local``，而是分别计算 Theta0/Theta1 的贡献，
        再按节点尺度 ``s_local`` 组合，显存从 O(B·N·K·F²) 降到 O(K·N·F)。

        参数：
            x: [B, N, F_in] 节点特征。
            edge_indices: 边索引（[2, E] 或批次列表）。
            edge_attrs: 边权（[E] 或批次列表）。
            s_local: [B, N] 局部扩散尺度。

        返回：
            output: [B, N, F_out] 卷积后的特征。
        """
        B, N, F_in = x.shape
        device = x.device
        
        # 逐 batch 处理：按需计算 Theta_local，避免创建过大张量（原英文注释翻译）
        output_list = []
        for b in range(B):
            # 取出第 b 个样本的边（原英文注释“Get edges for batch b”）
            if isinstance(edge_indices, list):
                edge_idx = edge_indices[b]
                edge_attr = edge_attrs[b]
            else:
                # 对拼接索引掩码筛选出第 b 个 batch 的边（原英文注释“Extract edges for batch b”）
                batch_mask = (edge_indices[0] >= b*N) & (edge_indices[0] < (b+1)*N)
                edge_idx = edge_indices[:, batch_mask].clone()
                edge_attr = edge_attrs[batch_mask].clone()
                # 将全局节点编号转换为当前 batch 内局部编号（原英文注释）
                edge_idx[0] -= b*N
                edge_idx[1] -= b*N
            
            # 生成切比雪夫多项式基：计算 T_k * x（原英文注释翻译）
            Tx0 = x[b]  # T_0 * x (identity) -> shape [N, F_in]
            Tx = [Tx0]

            if self.K >= 1:
                # T_1 * x = L̃ * x（原英文注释翻译）
                Tx1 = sparse_message_passing(Tx0, edge_idx, edge_attr)  # [N, F_in]
                Tx.append(Tx1)

                # Recursive computation for higher orders
                for k in range(2, self.K + 1):
                    Txk = 2 * sparse_message_passing(Tx1, edge_idx, edge_attr) - Tx0  # T_k * x
                    Tx.append(Txk)
                    Tx0, Tx1 = Tx1, Txk
            
            # 堆叠所有阶的切比雪夫输出，得到 [K+1, N, F_in]（原英文注释翻译）
            x_cheb = torch.stack(Tx, dim=0)  # [K+1, N, F_in]

            # 直接用 Theta0/Theta1 分别计算贡献，避免显式构建 Theta_local
            # A = sum_{k=0}^K (Tx_k @ Theta0_k),  B = sum_{k=0}^K (Tx_k @ Theta1_k)（原英文注释翻译）
            A = torch.einsum('knf,kfo->no', x_cheb, self.Theta0)   # [N, F_out]
            B = torch.einsum('knf,kfo->no', x_cheb, self.Theta1)   # [N, F_out]

            # 按节点尺度 s_local[b] 组合两部分：output[n,o] = A[n,o] + s_local[n] * B[n,o]
            batch_out = A + s_local[b].unsqueeze(-1) * B  # [N, F_out]
            
            output_list.append(batch_out)
        
        output = torch.stack(output_list, dim=0)  # [B, N, F_out]
        
        # 加上偏置并应用 dropout（原英文注释翻译）
        if self.bias is not None:
            output = output + self.bias
            
        output = self.dropout(output)
        
        return output
    
    def _process_single_batch(self,
                            x: torch.Tensor,
                            edge_index: torch.Tensor,
                            edge_attr: torch.Tensor,
                            Theta_local: torch.Tensor) -> torch.Tensor:
        """单批次 Chebyshev 递归计算的参考实现。"""
        N, F_in = x.shape
        
        # Chebyshev 递推：T_0=I，T_1=L̃，T_k=2L̃T_{k-1}-T_{k-2}
        Tx = []  # 存储各阶 T_k(L̃) * x（原英文注释“Store T_k(L̃) * x for each k”）
        
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
        
        # 将所有 T_k * x 堆叠为 [K+1, N, F_in]（原英文注释翻译）
        Tx_stacked = torch.stack(Tx, dim=0)
        
        # 使用 einsum 聚合节点局部卷积核，避免显式 for 循环
        output = torch.einsum('knf,nkfo->no', Tx_stacked, Theta_local)  # [N, F_out]
        
        return output
    
    def get_computational_stats(self,
                              B: int,
                              N: int,
                              num_edges: int) -> dict:
        """估算算力与参数量，便于对比不同配置的开销。"""
        # FLOPs 估算
        message_passing_flops = num_edges * self.F_in  # 消息聚合 FLOPs（原英文注释“Message aggregation”）
        chebyshev_recursion_flops = self.K * message_passing_flops  # 递归 K 次的计算量
        convolution_flops = B * N * (self.K + 1) * self.F_in * self.F_out  # 最终卷积计算量
        
        total_flops = chebyshev_recursion_flops + convolution_flops
        
        # 显存估计（近似以 [N,F] 张量数计）
        chebyshev_memory = 2 * N * self.F_in  # 需要缓存 Tx0 与 Tx1（原英文注释“Store Tx0, Tx1”）
        parameter_memory = (self.K + 1) * self.F_in * self.F_out * 2  # 两套 Theta0/Theta1 参数
        
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
        """返回参数总量。"""
        return sum(p.numel() for p in self.parameters())
    
    def visualize_adaptive_coefficients(self,
                                      s_local: torch.Tensor,
                                      sample_idx: int = 0) -> dict:
        """提取节点自适应系数，方便可视化与分析。"""
        with torch.no_grad():
            s_sample = s_local[sample_idx]  # [N]
            
            # 计算该样本下的局部系数（原英文注释“Compute local coefficients for this sample”）
            s_expanded = s_sample.unsqueeze(-1).unsqueeze(-1)  # [N, 1, 1]
            Theta_local = self.Theta0.unsqueeze(0) + self.Theta1.unsqueeze(0) * s_expanded  # [N, K+1, F_in, F_out]

            # 提取统计量（原英文注释“Extract statistics”）
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
    """多尺度固定扩散系数的图小波卷积，用于与自适应版本对比。"""
    
    def __init__(self, 
                 F_in: int,
                 F_out: int,
                 scales: List[float] = [0.1, 0.5, 1.0, 2.0],
                 K: int = 3,
                 bias: bool = True):
        """初始化多尺度卷积层。

        参数：
            F_in: 输入通道数。
            F_out: 输出通道数。
            scales: 固定扩散尺度列表。
            K: 切比雪夫阶数。
            bias: 是否使用偏置。
        """
        super().__init__()
        self.scales = scales
        self.num_scales = len(scales)
        
        # 为每个尺度各自构建一个卷积层（原英文注释“One convolution layer per scale”）
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
        """前向计算：在多组固定尺度上分别卷积后拼接。"""
        B, N, _ = x.shape
        
        outputs = []
        for i, (scale, conv_layer) in enumerate(zip(self.scales, self.conv_layers)):
            # 创建固定尺度的张量（原英文注释“Create constant scale tensor”）
            s_local = torch.full((B, N), scale, device=x.device, dtype=x.dtype)

            # 在该尺度上执行卷积（原英文注释“Apply convolution at this scale”）
            scale_output = conv_layer(x, edge_indices, edge_attrs, s_local)
            outputs.append(scale_output)
        
        # 拼接多尺度特征
        output = torch.cat(outputs, dim=-1)  # [B, N, F_out]
        
        if self.bias is not None:
            output = output + self.bias
            
        return output 