"""提供第三天MAPPO入口使用的轻量随机种子工具。"""

import random

import numpy as np
import torch


def set_global_seed(seed, deterministic=True):
    """设置Python、NumPy和PyTorch的全局随机种子。

    本函数必须在创建Actor、Critic和优化器之前调用，才能控制网络的首次
    随机初始化。CPU环境不依赖CUDA；若CUDA可用，则同时设置全部CUDA设备
    的种子，并在 ``deterministic=True`` 时关闭cuDNN自动基准选择。
    """
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("随机种子必须是非负整数")
    if not isinstance(deterministic, bool):
        raise ValueError("deterministic必须是布尔值")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
