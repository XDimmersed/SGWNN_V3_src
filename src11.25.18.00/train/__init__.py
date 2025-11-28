"""训练流程与指标相关模块。

当前导出：
- :mod:`trainer`：封装 SGWCN 的训练/验证循环、早停与可视化。
- :mod:`metrics`：分类精度、F1/召回、混淆矩阵等评估工具。

便于脚本端 ``from train import ...`` 直接复用训练管线。"""
