"""脉冲相关的编码、解码与自定义梯度算子。

"""脉冲相关的可微函数、编码/解码与统计工具集。

模块职责：
1. 提供可微分的脉冲函数 ``SpikeFunction``，以替代硬阶跃，使用代理梯度保持可训练性。
2. 给出常见的脉冲编码方式（泊松编码、速率编码），方便将连续特征映射为时序脉冲。
3. 提供脉冲统计与解码工具（计数/放电率/时间编码损失），便于调试与可视化。

所有函数都使用中文注释解释核心思路，并将残留的英文说明翻译到位，避免阅读英文文档的额外成本。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SpikeFunction(torch.autograd.Function):
    """带代理梯度的可微分脉冲函数。

    设计要点：
    - 正/负阈值均支持，适配双极 LIF 神经元的双通道输出。
    - 前向仍是硬脉冲（0/1），反向采用光滑的 Sigmoid 近似以避免梯度消失。
    - 在反向计算中对数值进行裁剪，保证训练稳定性。
    """
    
    @staticmethod
    def forward(ctx, membrane_potential: torch.Tensor,
                threshold: float,
                is_negative: bool = False) -> torch.Tensor:
        """前向计算：根据阈值生成 0/1 脉冲。

        参数：
            membrane_potential: [B, T, N, F] 膜电位张量。
            threshold: 触发阈值（正或负）。
            is_negative: 是否为负阈值分支，用于双极神经元的抑制脉冲。

        返回：
            spikes: 与输入尺寸一致的二值脉冲张量。
        """
        # 记录反向传播所需的变量
        if isinstance(threshold, torch.Tensor):
            threshold_tensor = threshold.clone().detach()
        else:
            threshold_tensor = torch.tensor(threshold, device=membrane_potential.device, dtype=membrane_potential.dtype)
        ctx.save_for_backward(membrane_potential, threshold_tensor)
        ctx.is_negative = is_negative

        # 根据正/负阈值生成脉冲
        if is_negative:
            # 负阈值：膜电位低于阈值触发抑制脉冲
            spikes = (membrane_potential <= threshold).float()
        else:
            # 正阈值：膜电位高于阈值触发兴奋脉冲
            spikes = (membrane_potential >= threshold).float()

        return spikes
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        """反向传播：利用 Sigmoid 代理梯度保持可训练性。

        参数：
            grad_output: 来自上游的梯度。

        返回：
            grad_input: 关于膜电位的梯度；阈值与 is_negative 不需要梯度，返回 None。
        """
        membrane_potential, threshold_tensor = ctx.saved_tensors
        is_negative = ctx.is_negative
        threshold = threshold_tensor.item()

        # 若上游梯度出现 NaN，先行置零，避免继续污染
        if torch.isnan(grad_output).any():
            print("WARNING: NaN detected in grad_output!")
            grad_output = torch.where(torch.isnan(grad_output), torch.zeros_like(grad_output), grad_output)

        # 代理梯度：选用 Sigmoid 的导数，gamma 控制平滑度
        gamma = 0.5

        # 限幅膜电位，避免指数计算溢出
        membrane_potential = torch.clamp(membrane_potential, -10.0, 10.0)

        if is_negative:
            # 负脉冲梯度符号相反
            diff = threshold - membrane_potential
            diff = torch.clamp(diff, -10.0, 10.0)
            surrogate_grad = -gamma * torch.sigmoid(gamma * diff) * \
                            torch.sigmoid(gamma * (-diff))
        else:
            # 正脉冲保留正向梯度
            diff = membrane_potential - threshold
            diff = torch.clamp(diff, -10.0, 10.0)
            surrogate_grad = gamma * torch.sigmoid(gamma * diff) * \
                           torch.sigmoid(gamma * (-diff))

        # 再次限幅，防止梯度异常放大
        surrogate_grad = torch.clamp(surrogate_grad, -1.0, 1.0)

        # 链式法则求得对膜电位的梯度
        grad_input = grad_output * surrogate_grad

        # 最终 NaN 检查，保证输出梯度干净
        if torch.isnan(grad_input).any():
            print("WARNING: NaN detected in grad_input after surrogate gradient!")
            grad_input = torch.where(torch.isnan(grad_input), torch.zeros_like(grad_input), grad_input)
        
        return grad_input, None, None


def poisson_encoding(x: torch.Tensor,
                    T: int,
                    max_rate: float = 1.0,
                    normalize_per_sample: bool = True) -> torch.Tensor:
    """泊松编码：将连续值转为遵循泊松分布的随机脉冲序列。

    参数：
        x: [B, N, F] 连续输入特征。
        T: 时间步数。
        max_rate: 最大放电率（每个时间步允许的最大触发概率）。
        normalize_per_sample: 是否按样本单独归一化，避免尺度差异导致偏置。

    返回：
        spike_trains: [B, T, N, F] 的泊松脉冲序列。
    """
    B, N, F = x.shape
    device = x.device
    
    # 将特征线性归一化到 [0, max_rate]（原英文注释“Normalize features to [0, max_rate]”）
    if normalize_per_sample:
        # 按样本归一化，保证不同样本间范围一致
        x_min = x.amin(dim=(1, 2), keepdim=True)  # [B, 1, 1]
        x_max = x.amax(dim=(1, 2), keepdim=True)  # [B, 1, 1]
    else:
        # 全局归一化，适合数据范围已知的情况
        x_min = x.min()
        x_max = x.max()

    # 避免除零：若范围过小则替换为 1（原英文注释“Avoid division by zero”）
    x_range = x_max - x_min
    x_range = torch.where(x_range > 1e-8, x_range, torch.ones_like(x_range))
    
    # 线性映射到 [0, max_rate]
    rates = (x - x_min) / x_range * max_rate  # [B, N, F]
    
    # 根据放电率在时间维度复制
    rates_expanded = rates.unsqueeze(1).expand(B, T, N, F)  # [B, T, N, F]

    # 与均匀随机数比较得到脉冲
    random_vals = torch.rand_like(rates_expanded)
    spike_trains = (random_vals < rates_expanded).float()
    
    return spike_trains


def rate_encoding(x: torch.Tensor,
                 T: int,
                 normalize_per_sample: bool = True) -> torch.Tensor:
    """速率编码：直接用幅值控制脉冲出现的频率。

    参数：
        x: [B, N, F] 连续输入。
        T: 时间步数。
        normalize_per_sample: 是否按样本归一化。

    返回：
        spike_trains: [B, T, N, F] 速率编码的脉冲序列。
    """
    B, N, F = x.shape
    
    # 将输入缩放到 [0, 1]
    if normalize_per_sample:
        x_min = x.amin(dim=(1, 2), keepdim=True)
        x_max = x.amax(dim=(1, 2), keepdim=True)
    else:
        x_min = x.min()
        x_max = x.max()
    
    x_range = x_max - x_min
    x_range = torch.where(x_range > 1e-8, x_range, torch.ones_like(x_range))
    
    rates = (x - x_min) / x_range  # [B, N, F]
    
    # 时间维复制并使用等距阈值生成脉冲（越大越早触发）
    rates_expanded = rates.unsqueeze(1).expand(B, T, N, F)
    thresholds = torch.linspace(0, 1, T+1, device=x.device)[:-1].view(1, T, 1, 1)
    
    spike_trains = (rates_expanded > thresholds).float()
    
    return spike_trains


def spike_count_decoding(spike_trains: torch.Tensor,
                        dim: int = 1) -> torch.Tensor:
    """脉冲计数解码：统计时间维的脉冲数量。"""
    return spike_trains.sum(dim=dim)


def spike_rate_decoding(spike_trains: torch.Tensor,
                       dim: int = 1) -> torch.Tensor:
    """放电率解码：时间维求平均得到放电率。"""
    return spike_trains.mean(dim=dim)


def temporal_coding_loss(spike_trains: torch.Tensor,
                        targets: torch.Tensor,
                        method: str = 'mse') -> torch.Tensor:
    """针对时间编码脉冲序列的损失函数集合。

    参数：
        spike_trains: [B, T, N, F] 预测的脉冲序列。
        targets: [B, N, F] 目标值或目标放电率。
        method: 损失类型（mse/rate_mse/count_mse）。

    返回：
        loss: 标量损失。
    """
    if method == 'mse':
        # 直接对时序脉冲做 MSE
        targets_expanded = targets.unsqueeze(1).expand_as(spike_trains)
        return F.mse_loss(spike_trains, targets_expanded)

    elif method == 'rate_mse':
        # 对放电率求 MSE
        predicted_rates = spike_rate_decoding(spike_trains, dim=1)
        return F.mse_loss(predicted_rates, targets)

    elif method == 'count_mse':
        # 对脉冲计数求 MSE（除以时间步进行归一化）
        predicted_counts = spike_count_decoding(spike_trains, dim=1) / spike_trains.shape[1]
        return F.mse_loss(predicted_counts, targets)
    
    else:
        raise ValueError(f"Unknown temporal coding loss method: {method}")


def compute_spike_statistics(spike_trains: torch.Tensor,
                           dim: int = 1) -> dict:
    """计算脉冲序列的统计指标，辅助调参与可视化。"""
    # 将时间维移到最后，便于逐样本处理
    spike_trains_transposed = spike_trains.transpose(dim, -1)
    T = spike_trains_transposed.shape[-1]
    
    # 基础统计量
    firing_rates = spike_trains_transposed.mean(dim=-1)  # 平均放电率
    spike_counts = spike_trains_transposed.sum(dim=-1)   # 总脉冲数
    
    # 计算相邻脉冲间隔（仅对有放电的神经元，原英文注释“Inter-spike intervals”）
    isi_means = []
    for batch_idx in range(spike_trains_transposed.shape[0]):
        batch_spikes = spike_trains_transposed[batch_idx]  # [N, F, T]
        batch_isi = []
        for n in range(batch_spikes.shape[0]):
            for f in range(batch_spikes.shape[1]):
                spike_times = torch.nonzero(batch_spikes[n, f], as_tuple=False).squeeze(-1)
                if len(spike_times) > 1:
                    isis = torch.diff(spike_times.float())
                    batch_isi.append(isis.mean().item())
        if batch_isi:
            isi_means.append(torch.tensor(batch_isi).mean().item())
        else:
            isi_means.append(float('inf'))
    
    stats = {
        'mean_firing_rate': firing_rates.mean().item(),
        'max_firing_rate': firing_rates.max().item(),
        'min_firing_rate': firing_rates.min().item(),
        'std_firing_rate': firing_rates.std().item(),
        'total_spikes': spike_counts.sum().item(),
        'sparsity': (spike_trains == 0).float().mean().item(),
        'mean_isi': torch.tensor(isi_means).mean().item() if isi_means else float('inf'),
        'active_neurons_ratio': (firing_rates > 0).float().mean().item(),
    }
    
    return stats 