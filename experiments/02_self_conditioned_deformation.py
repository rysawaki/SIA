import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# ---- SelfSpaceの核のみ簡易定義（最小構成） ----
class SelfSpace:
    def __init__(self, dim, max_axes=6):
        self.dim = dim
        self.max_axes = max_axes
        self.axes = []      # 経験から形成されたSelf軸
        self.strength = []  # 軸の重み（Affect x Shock）

    def update(self, trace, shock=1.0, affect=1.0):
        trace_dir = F.normalize(trace, dim=0)
        if len(self.axes) >= self.max_axes:
            idx = torch.argmin(torch.tensor(self.strength)).item()
            self.axes[idx] = trace_dir
            self.strength[idx] = shock * affect
        else:
            self.axes.append(trace_dir)
            self.strength.append(shock * affect)

    def condition_query(self, query, alpha=0.6):
        if not self.axes:
            return query
        axes = torch.stack(self.axes)              # (k, d)
        weights = torch.tensor(self.strength)
        weights = weights / (weights.sum() + 1e-6)
        sims = torch.matmul(axes, F.normalize(query, dim=0))
        contrib = torch.matmul(sims * weights, axes)
        return F.normalize(query + alpha * contrib, dim=0)


# ---- デモ設定 ----
torch.manual_seed(0)
dim = 16

# ランダムな Query
query = torch.randn(dim)

# SelfSpaceの形成（＝経験によるSelf軸の生成）
self_space = SelfSpace(dim=dim)
for _ in range(4):
    trace = torch.randn(dim)
    shock = torch.rand(()).item()
    affect = torch.rand(()).item()
    self_space.update(trace, shock, affect)

# Self-conditioned Query
query_cond = self_space.condition_query(query)

# ---- 2Dに次元圧縮（PCA） ----
points = [query] + self_space.axes + [query_cond]
labels = ["Query (Before)"] + [f"Self Axis {i+1}" for i in range(len(self_space.axes))] + ["Query (After)"]

pca = PCA(n_components=2)
points_2d = pca.fit_transform(torch.stack(points).detach().numpy())

# ---- プロット ----
plt.figure(figsize=(8, 6))
ax = plt.gca()

# Self軸の点
for i, p in enumerate(points_2d[1:-1], start=1):
    ax.scatter(p[0], p[1], color='red', s=80)
    ax.text(p[0]+0.03, p[1]+0.03, f"Self Axis {i}", color='red')

# Query Before
ax.scatter(points_2d[0][0], points_2d[0][1], color='blue', s=120, label="Query Before")
ax.text(points_2d[0][0]+0.03, points_2d[0][1]+0.03, "Query", color='blue')

# Query After (Self-conditioned)
ax.scatter(points_2d[-1][0], points_2d[-1][1], color='green', s=120, label="Query After")
ax.text(points_2d[-1][0]+0.03, points_2d[-1][1]+0.03, "Query_cond", color='green')

# Query → Query_cond の変形ベクトルを矢印で描く
ax.arrow(points_2d[0][0], points_2d[0][1],
         points_2d[-1][0] - points_2d[0][0],
         points_2d[-1][1] - points_2d[0][1],
         head_width=0.05, head_length=0.1, fc='green', ec='green')

plt.title("Self-conditioned Query Deformation in Self Space (PCA Projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True, alpha=0.3)
plt.show()
