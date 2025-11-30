import torch
import numpy as np

from src.utils.visualize_selfspace import SelfSpace
from trace_extractor import TraceExtractor
from src.utils.visualize_selfspace import visualize_self_evolution


# ====== 準備 ======
dim = 64
self_space = SelfSpace(dim=dim)
extractor = TraceExtractor()

# 擬似的Trace: 結果が見やすいようにランダム方向に発生させる
traces = []
for _ in range(8):
    t = torch.randn(dim)
    t = torch.nn.functional.normalize(t, dim=-1)
    traces.append(t)

# ====== 可視化実験 ======
visualize_self_evolution(self_space, traces, title="Self-Space Evolution Demo")
