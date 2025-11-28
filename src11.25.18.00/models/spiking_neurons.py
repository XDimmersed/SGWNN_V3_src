"""
双极 LIF 神经元实现
--------------------
本文件包含带正负双阈值的脉冲神经元及自适应版本，并提供脉冲读出层。
核心思想是用正/负脉冲同时编码凸、凹特征，提高信息利用率。所有说明均改为中文
并补充了更细的注释，便于理解膜电位更新与重置细节。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from ..utils.spike_utils import SpikeFunction


class BipolarLIFNeuron(nn.Module):
    """
    具有正负双阈值的 LIF 脉冲神经元。

    核心特性：
    1. 双阈值：θ_pos>0 产生正脉冲（兴奋），θ_neg<0 产生负脉冲（抑制）。
    2. 单个神经元输出 2×F 通道，分担凸/凹特征表达，提升参数利用率。
    3. 重置方式遵循正负阈值独立扣除，保证膜电位物理合理。
    """
    
    def __init__(self, 
                 membrane_dim: int,
                 tau_mem: float = 20.0,
                 theta_pos: float = 1.0,
                 theta_neg: float = -1.0,
                 reset_mode: str = 'subtract',
                 dt: float = 1.0):
        """
        初始化双极 LIF 神经元。

        参数：
            membrane_dim: 膜电位维度。
            tau_mem: 膜时间常数，控制电位衰减速度。
            theta_pos: 正阈值（>0）。
            theta_neg: 负阈值（<0）。
            reset_mode: 重置方式（subtract: 减去阈值；zero: 归零）。
            dt: 模拟时间步长。
        """
        super().__init__()
        self.membrane_dim = membrane_dim
        self.tau_mem = tau_mem
        self.theta_pos = theta_pos
        self.theta_neg = theta_neg
        self.reset_mode = reset_mode
        self.dt = dt
        
        # 膜电位衰减系数 α = exp(-dt/τ)
        self.alpha = torch.exp(torch.tensor(-dt / tau_mem))

        # 将膜电位注册为 buffer，以便在 BPTT 中持久化
        self.register_buffer('membrane_potential', torch.zeros(1, 1, 1, membrane_dim))
        
        # 可学习的阈值参数（默认关闭，可按需开启）
        self.learnable_threshold = False
        if self.learnable_threshold:
            self.theta_pos_param = nn.Parameter(torch.tensor(theta_pos))
            self.theta_neg_param = nn.Parameter(torch.tensor(theta_neg))
    
    def reset_state(self, batch_size: int, num_nodes: int, device: torch.device):
        """
        重置膜电位状态，供新批次使用。

        参数：
            batch_size: 当前批大小。
            num_nodes: 图节点数量。
            device: 创建张量的设备。
        """
        self.membrane_potential = torch.zeros(
            batch_size, 1, num_nodes, self.membrane_dim,
            device=device, dtype=torch.float32
        )
    
    def forward(self,
                input_current: torch.Tensor,
                return_membrane: bool = False) -> torch.Tensor:
        """
        前向计算：将连续输入转为正负脉冲。

        参数：
            input_current: [B, T, N, F] 连续电流输入。
            return_membrane: 是否返回膜电位轨迹。

        返回：
            spike_output: [B, T, N, 2*F] 双极脉冲。
            membrane_potential: [B, T, N, F] 膜电位（可选）。
        """
        B, T, N, F = input_current.shape
        device = input_current.device
        
        # 若缓存尺寸不匹配则重置膜电位
        if self.membrane_potential.shape != (B, 1, N, F):
            self.reset_state(B, N, device)
        
        # 获取当前阈值，支持后续扩展为可学习
        if self.learnable_threshold:
            theta_pos = self.theta_pos_param
            theta_neg = self.theta_neg_param
        else:
            theta_pos = self.theta_pos
            theta_neg = self.theta_neg
        
        # 确保阈值符号正确，避免用户输入错误
        if not isinstance(theta_pos, torch.Tensor):
            theta_pos = torch.tensor(theta_pos, device=device, dtype=torch.float32)
        if not isinstance(theta_neg, torch.Tensor):
            theta_neg = torch.tensor(theta_neg, device=device, dtype=torch.float32)
            
        theta_pos = torch.abs(theta_pos)  # 强制为正（原英文注释“Force positive”）
        theta_neg = -torch.abs(theta_neg)  # 强制为负（原英文注释“Force negative”）
        
        spike_output_list = []
        membrane_history = []
        
        for t in range(T):
            # 膜电位动态：V[t] = α*V[t-1] + I[t]
            # 限幅输入，避免极端值导致梯度爆炸
            current_input = torch.clamp(input_current[:, t:t+1, :, :], -5.0, 5.0)
            
            self.membrane_potential = (
                self.alpha.to(device) * self.membrane_potential + 
                current_input  # [B, 1, N, F]
            )
            
            # 膜电位同样限幅，保证数值稳定
            self.membrane_potential = torch.clamp(self.membrane_potential, -20.0, 20.0)
            
            # 生成正、负脉冲：分别比较与正阈值和负阈值
            pos_spikes = SpikeFunction.apply(
                self.membrane_potential, theta_pos, False
            ).squeeze(1)  # [B, N, F]
            
            neg_spikes = SpikeFunction.apply(
                self.membrane_potential, theta_neg, True  
            ).squeeze(1)  # [B, N, F]
            
            # 按配置重置膜电位，subtract 模式精确减去对应阈值
            if self.reset_mode == 'subtract':
                reset_amount = (
                    pos_spikes * theta_pos -
                    neg_spikes * theta_neg
                ).unsqueeze(1)  # [B, 1, N, F]
                self.membrane_potential = self.membrane_potential - reset_amount

            elif self.reset_mode == 'zero':
                spike_mask = (pos_spikes + neg_spikes > 0).unsqueeze(1)  # [B, 1, N, F]
                self.membrane_potential = self.membrane_potential * (~spike_mask).float()

            # 拼接正负脉冲形成双极输出
            bipolar_spikes = torch.cat([pos_spikes, neg_spikes], dim=-1)
            spike_output_list.append(bipolar_spikes)

            if return_membrane:
                membrane_history.append(self.membrane_potential.squeeze(1))  # [B, N, F]

        # 沿时间维堆叠得到最终脉冲序列
        spike_output = torch.stack(spike_output_list, dim=1)
        
        if return_membrane:
            membrane_potential = torch.stack(membrane_history, dim=1)  # [B, T, N, F]
            return spike_output, membrane_potential
        else:
            return spike_output
    
    def compute_firing_rates(self, spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        统计正、负脉冲的放电率。

        参数：
            spikes: [B, T, N, 2*F] 双极脉冲。

        返回：
            pos_rates: 正脉冲放电率。
            neg_rates: 负脉冲放电率。
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
        计算神经元的统计指标，辅助可视化与调参。

        参数：
            spikes: [B, T, N, 2*F] 双极脉冲。
            membrane_potential: [B, T, N, F] 膜电位，可选。

        返回：
            stats: 统计字典。
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
    """带可学习阈值与时间常数的自适应双极 LIF 神经元。"""
    
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
        初始化自适应版本的双极 LIF。

        参数与基类一致，增加了 learnable_tau 与 learnable_threshold 控制可学习性。
        """
        super().__init__(membrane_dim, tau_mem, theta_pos, theta_neg, reset_mode, dt)
        
        # 可学习的时间常数与阈值
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
        """当前的膜时间常数（指数形式存储以确保正值）。"""
        return torch.exp(self.log_tau_mem)
    
    @property
    def alpha_current(self):
        """当前的衰减系数 α。"""
        return torch.exp(-self.dt / self.tau_mem_current)
    
    def forward(self,
                input_current: torch.Tensor,
                return_membrane: bool = False) -> torch.Tensor:
        """
        前向传播，使用可学习的衰减与阈值。

        参数、返回值与基类一致。
        """
        B, T, N, F = input_current.shape
        device = input_current.device
        
        # 与基类相同，确保状态尺寸正确
        if self.membrane_potential.shape != (B, 1, N, F):
            self.reset_state(B, N, device)

        # 读取当前可学习参数
        alpha = self.alpha_current.to(device)
        
        if self.learnable_threshold:
            theta_pos = torch.abs(self.theta_pos_param)  # 强制为正（原英文注释“Force positive”）
            theta_neg = -torch.abs(self.theta_neg_param)  # 强制为负（原英文注释“Force negative”）
        else:
            theta_pos = self.theta_pos
            theta_neg = self.theta_neg
        
        spike_output_list = []
        membrane_history = []
        
        for t in range(T):
            # 使用自适应 τ 进行膜电位更新
            self.membrane_potential = (
                alpha * self.membrane_potential +
                input_current[:, t:t+1, :, :]
            )
            
            # 与基类一致，生成正负脉冲
            pos_spikes = SpikeFunction.apply(
                self.membrane_potential, theta_pos, False
            ).squeeze(1)
            
            neg_spikes = SpikeFunction.apply(
                self.membrane_potential, theta_neg, True
            ).squeeze(1)
            
            # 按选择的模式重置膜电位
            if self.reset_mode == 'subtract':
                reset_amount = (
                    pos_spikes * theta_pos -
                    neg_spikes * theta_neg
                ).unsqueeze(1)
                self.membrane_potential = self.membrane_potential - reset_amount
                
            elif self.reset_mode == 'zero':
                spike_mask = (pos_spikes + neg_spikes > 0).unsqueeze(1)
                self.membrane_potential = self.membrane_potential * (~spike_mask).float()
            
            # 拼接输出
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
    """基于放电率的脉冲网络读出层。"""
    
    def __init__(self, 
                 input_dim: int,
                 output_dim: int,
                 readout_mode: str = 'rate',
                 dropout: float = 0.5):
        """
        初始化读出层。

        参数：
            input_dim: 输入维度（双极输出通常为 2*F）。
            output_dim: 输出维度（类别数）。
            readout_mode: 解码方式（rate/count/last）。
            dropout: 用于正则化的失活比例。
        """
        super().__init__()
        self.readout_mode = readout_mode
        
        # 带 dropout 的线性映射层（原英文注释“Linear projection with dropout”）
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, spike_trains: torch.Tensor) -> torch.Tensor:
        """
        将脉冲序列解码为分类 logits。

        参数：
            spike_trains: [B, T, N, F] 脉冲序列。

        返回：
            logits: [B, output_dim] 分类结果。
        """
        if self.readout_mode == 'rate':
            # 放电率解码：在时间维求平均
            rates = spike_trains.mean(dim=1)  # [B, N, F]

        elif self.readout_mode == 'count':
            # 计数解码：在时间维求和
            rates = spike_trains.sum(dim=1)  # [B, N, F]

        elif self.readout_mode == 'last':
            # 使用最后一个时间步
            rates = spike_trains[:, -1, :, :]  # [B, N, F]
            
        else:
            raise ValueError(f"Unknown readout mode: {self.readout_mode}")
        
        # 在节点维度做全局平均池化（原英文注释“Global pooling over nodes”）
        pooled = rates.mean(dim=1)  # [B, F]
        
        # 使用 dropout 做正则化（原英文注释“Apply dropout for regularization”）
        pooled = self.dropout(pooled)
        
        # 线性分类头输出 logits（原英文注释“Linear classification”）
        logits = self.linear(pooled)  # [B, output_dim]
        
        return logits 