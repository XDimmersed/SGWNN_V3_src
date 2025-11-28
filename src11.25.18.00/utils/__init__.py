"""脉冲图小波卷积网络使用的常用工具函数集合。

提供：
- 图构建辅助（kNN 搜索、局部密度估计）。
- 脉冲编码与代理梯度函数。
- 可视化工具（训练曲线、点云渲染）。

在训练或模型脚本中可直接按需导入这些便捷函数。"""

from .graph_utils import knn_search, compute_local_density
from .spike_utils import SpikeFunction, poisson_encoding  
from .visualization import plot_training_curves, visualize_point_cloud

__all__ = [
    'knn_search',
    'compute_local_density', 
    'SpikeFunction',
    'poisson_encoding',
    'plot_training_curves',
    'visualize_point_cloud'
] 