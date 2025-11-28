"""脉冲图小波卷积网络的核心模型模块聚合。

包含组件：
- :mod:`graph_builder`：密度自适应的稀疏构图器。
- :mod:`wavelet_conv`：节点级自适应的图小波卷积层。
- :mod:`spiking_neurons`：双极 LIF 脉冲神经元与脉冲读出。
- :mod:`sgwcn`：整合上述模块的完整 SGWCN 网络/分类器。

直接 ``from models import *`` 可获得训练脚本需要的所有核心类。"""

from .graph_builder import SparseGraphBuilder
from .wavelet_conv import AdaptiveGraphWaveletConv  
from .spiking_neurons import BipolarLIFNeuron
from .sgwcn import SpikingGraphWaveletNet, SGWCNClassifier

__all__ = [
    'SparseGraphBuilder',
    'AdaptiveGraphWaveletConv', 
    'BipolarLIFNeuron',
    'SpikingGraphWaveletNet',
    'SGWCNClassifier'
] 