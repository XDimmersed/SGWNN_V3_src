"""
脉冲图小波卷积网络（SGWCN）
--------------------------------
本文件实现论文中的核心创新，并将其组装成完整的分类模型。代码中所有注释均以
中文详细说明设计意图，方便快速理解。主要包含两部分：
1. 基于局部密度的自适应稀疏图构建（SparseGraphBuilder）
2. 支持双极脉冲的自适应图小波卷积堆栈与读出层
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
    完整的脉冲图小波卷积网络实现。

    计算流程：
    [B,N,3] 输入点云 → 构图 → [B,N,k] 稀疏邻接 → 小波卷积 → [B,T,N,F] 双极脉冲 → 读出分类。

    核心特性：
    1. 结合局部密度的自适应构图，保证稀疏但信息充分的邻域。
    2. 节点级自适应的小波卷积核，提升几何细节捕获能力。
    3. 带双阈值的双极 LIF 神经元，同时编码凸/凹特征，输出翻倍通道数。
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
        初始化 SGWCN。

        参数说明：
            input_dim: 输入特征维度（点云默认 xyz=3）。
            hidden_dims: 每一层小波卷积的输出维度列表。
            num_classes: 分类类别数。
            num_time_steps: 脉冲序列的时间步数。
            k_neighbors: 构图时选择的近邻数量。
            chebyshev_order: 切比雪夫近似阶数，用于小波卷积。
            beta: 自适应 σ 的缩放因子。
            lambda_param: 扩散尺度的缩放因子。
            epsilon: 数值稳定性的小常数。
            tau_mem: 膜电位时间常数。
            theta_pos: 正阈值（激发阈值）。
            theta_neg: 负阈值（抑制阈值）。
            dropout: 随机失活比例，用于正则化。
            use_faiss: 是否使用 FAISS 加速 kNN。
            readout_mode: 脉冲解码方式（rate/count/last）。
        """
        super().__init__()
        
        self.num_time_steps = num_time_steps
        self.num_classes = num_classes
        
        # 图构建模块：根据局部密度产生稀疏邻接，后续层重复使用
        self.graph_builder = SparseGraphBuilder(
            k=k_neighbors,
            beta=beta,
            lambda_param=lambda_param,
            epsilon=epsilon,
            use_faiss=use_faiss
        )

        # 特征编码：将原始坐标映射到高维，便于卷积学习
        self.feature_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 图小波卷积层与对应的脉冲神经元层
        self.conv_layers = nn.ModuleList()
        self.spiking_layers = nn.ModuleList()

        # 构建多层卷积与 LIF 神经元，除第一层外输入通道均为双极输出
        layer_dims = [hidden_dims[0]] + hidden_dims
        for i in range(len(layer_dims) - 1):
            # 小波卷积：i>0 时输入为上层双极脉冲，通道数翻倍
            conv_layer = AdaptiveGraphWaveletConv(
                F_in=layer_dims[i] * 2 if i > 0 else layer_dims[i],  # *2 for bipolar spikes
                F_out=layer_dims[i+1],
                K=chebyshev_order,
                dropout=dropout
            )
            self.conv_layers.append(conv_layer)

            # 双极 LIF 神经元：将连续输出转为正/负脉冲
            spiking_layer = BipolarLIFNeuron(
                membrane_dim=layer_dims[i+1],
                tau_mem=tau_mem,
                theta_pos=theta_pos,
                theta_neg=theta_neg
            )
            self.spiking_layers.append(spiking_layer)

        # 读出层：接收最后一层的双极脉冲（通道翻倍）并完成分类
        final_dim = hidden_dims[-1] * 2  # *2 表示包含正负脉冲
        self.readout = SpikingReadout(
            input_dim=final_dim,
            output_dim=num_classes,
            readout_mode=readout_mode,
            dropout=dropout
        )

        # 预留的全局池化，可用于替换读出策略
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        模型前向传播。

        参数：
            point_cloud: [B, N, 3] 点云坐标。

        返回：
            logits: [B, num_classes] 分类预测。
        """
        B, N, _ = point_cloud.shape
        device = point_cloud.device

        # 步骤 1：基于局部密度构建自适应稀疏图
        edge_indices, edge_attrs, s_local = self.graph_builder(point_cloud)

        # 步骤 2：编码初始特征，输出 [B, N, hidden_dim]
        x = self.feature_encoder(point_cloud)  # [B, N, hidden_dims[0]]

        # 步骤 3：泊松编码，将连续值转为脉冲序列
        spike_trains = poisson_encoding(x, self.num_time_steps)  # [B, T, N, F]

        # 重置所有脉冲神经元的膜电位，避免跨样本污染
        for layer in self.spiking_layers:
            layer.reset_state(B, N, device)

        # 步骤 4：依次通过小波卷积和 LIF 神经元
        for i, (conv_layer, spike_layer) in enumerate(zip(self.conv_layers, self.spiking_layers)):
            layer_outputs = []

            # 按时间维度循环，逐步处理脉冲序列
            for t in range(self.num_time_steps):
                # 取出当前时间步的节点特征
                x_t = spike_trains[:, t, :, :]  # [B, N, F]

                # 进行图小波卷积，保持节点维度不变
                conv_out = conv_layer(x_t, edge_indices, edge_attrs, s_local)  # [B, N, F_out]
                layer_outputs.append(conv_out)

            # 将时间维堆叠再交由 LIF 神经元生成双极脉冲
            conv_output = torch.stack(layer_outputs, dim=1)  # [B, T, N, F_out]
            spike_trains = spike_layer(conv_output)  # [B, T, N, 2*F_out]

        # 步骤 5：读出层进行分类
        logits = self.readout(spike_trains)  # [B, num_classes]

        return logits
    
    def forward_with_analysis(self, point_cloud: torch.Tensor) -> Dict:
        """
        带详细统计信息的前向过程，便于调试或可视化。

        参数：
            point_cloud: [B, N, 3] 点云输入。

        返回：
            analysis: 包含中间张量与统计指标的字典。
        """
        B, N, _ = point_cloud.shape
        device = point_cloud.device

        analysis = {
            'layer_outputs': [],
            'spike_statistics': [],
            'graph_statistics': {},
            'energy_consumption': 0.0
        }

        # 构建图并记录统计数据
        edge_indices, edge_attrs, s_local = self.graph_builder(point_cloud)

        if isinstance(edge_indices, list):
            total_edges = sum(edge_idx.shape[1] for edge_idx in edge_indices)
            # 仅取首个 batch 统计（原英文注释“Use first batch for statistics”）
            analysis['graph_statistics'] = self.graph_builder.get_graph_statistics(
                edge_indices[0], edge_attrs[0], N
            )
        else:
            total_edges = edge_indices.shape[1]
            # 对拼接后的边索引，需要过滤出第一个 batch 的节点
            # 边索引取值范围为 [0, B*N-1]，此处只保留 [0, N-1]
            batch_mask = (edge_indices[0] < N) & (edge_indices[1] < N)
            first_batch_edges = edge_indices[:, batch_mask]
            first_batch_attrs = edge_attrs[batch_mask]
            
            analysis['graph_statistics'] = self.graph_builder.get_graph_statistics(
                first_batch_edges, first_batch_attrs, N
            )
        
        # 初始特征编码与泊松脉冲生成
        x = self.feature_encoder(point_cloud)
        spike_trains = poisson_encoding(x, self.num_time_steps)

        # 重置状态以防止跨调用干扰
        for layer in self.spiking_layers:
            layer.reset_state(B, N, device)

        # 逐层处理并收集统计信息
        for i, (conv_layer, spike_layer) in enumerate(zip(self.conv_layers, self.spiking_layers)):
            layer_outputs = []
            
            for t in range(self.num_time_steps):
                x_t = spike_trains[:, t, :, :]
                conv_out = conv_layer(x_t, edge_indices, edge_attrs, s_local)
                layer_outputs.append(conv_out)
            
            conv_output = torch.stack(layer_outputs, dim=1)
            spike_trains, membrane_potential = spike_layer(conv_output, return_membrane=True)
            
            # 收集当前层的脉冲与膜电位统计
            layer_stats = spike_layer.get_neuron_statistics(spike_trains, membrane_potential)
            analysis['spike_statistics'].append(layer_stats)
            analysis['layer_outputs'].append({
                'conv_output': conv_output.detach(),
                'spike_output': spike_trains.detach(),
                'membrane_potential': membrane_potential.detach()
            })
            
            # 估算能耗：脉冲数量 × 单次能耗
            total_spikes = spike_trains.sum().item()
            analysis['energy_consumption'] += total_spikes * 1e-12  # pJ per spike

        # 最终分类输出
        logits = self.readout(spike_trains)
        analysis['logits'] = logits.detach()

        return analysis

    def get_model_statistics(self) -> Dict:
        """获取模型层级与参数统计信息。"""
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
        估算相较传统 ANN 的能耗优势。

        参数：
            num_samples: 样本数量。
            points_per_sample: 每个样本的点数。

        返回：
            energy_stats: 能耗估计结果。
        """
        # 依据文献的经验值进行粗略估计
        ENERGY_PER_SPIKE = 1e-12  # 单个脉冲约 1 皮焦耳
        ENERGY_PER_FLOP = 1e-15   # 单次 FLOP 约 1 飞焦耳（对比用）

        # 估算每层在完整时间窗口内的脉冲数量
        estimated_spikes_per_layer = []
        for i, layer in enumerate(self.spiking_layers):
            # 假设双极神经元约 10% 的放电率
            layer_dim = layer.membrane_dim * 2  # 双极输出通道翻倍
            spikes_per_timestep = num_samples * points_per_sample * layer_dim * 0.1
            total_spikes = spikes_per_timestep * self.num_time_steps
            estimated_spikes_per_layer.append(total_spikes)

        total_spikes = sum(estimated_spikes_per_layer)
        snn_energy = total_spikes * ENERGY_PER_SPIKE

        # 与等价 ANN 进行对比估计：卷积参数量近似为 MAC 次数
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
    面向点云分类任务的 SGWCN 预设配置。
    默认参数针对 ModelNet40 之类的数据集进行了调优。
    """
    
    def __init__(self,
                 num_classes: int = 40,
                 num_points: int = 1024,
                 **kwargs):
        """
        使用合理默认值初始化分类器。

        参数：
            num_classes: 分类类别数（ModelNet40 为 40）。
            num_points: 每个样本的点数上限。
            **kwargs: 传递给基类的其他配置。
        """
        # 针对点云分类的默认超参，可被用户覆盖
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
        
        # 用用户传入的参数覆盖默认值
        defaults.update(kwargs)
        
        super().__init__(
            num_classes=num_classes,
            **defaults
        )
        
        self.num_points = num_points
    
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        针对分类任务的前向过程，兼容不同输入格式。

        参数：
            data: [B, N, 3] 或 [B, N, C] 的点云张量。

        返回：
            logits: [B, num_classes] 分类预测。
        """
        # 兼容额外特征的点云，只取前三个坐标维度
        if data.shape[-1] > 3:
            point_cloud = data[:, :, :3]
        else:
            point_cloud = data

        # 若输入点过多则随机下采样，保证计算成本稳定
        if point_cloud.shape[1] > self.num_points:
            indices = torch.randperm(point_cloud.shape[1])[:self.num_points]
            point_cloud = point_cloud[:, indices, :]

        return super().forward(point_cloud)