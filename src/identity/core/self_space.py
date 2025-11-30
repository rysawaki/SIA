# ============================
# file: self_space_v3.py
# Core SIA Component: SelfSpace v3
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfSpace(nn.Module):
    """
    Self-Space v3:
    - Trace から自己軸 (axes) を形成・更新する
    - Trace の影響で Self の中心 self_state と Metric を同時に変形させる
    - Attention の Query を「自己の幾何学」を通して歪ませる

    ポイント:
        v1: 軸の蓄積
        v2: Metric の導入（距離構造の変形）
        v3: GrowthKernel 統合（Self と Metric の同時進化）

    Attributes:
        dim:         埋め込み次元 d
        axes:        Self を張る基底ベクトル (k, d)
        strength:    各軸の強度（Shock × Affect の累積）
        num_active:  現在有効な軸の本数
        self_state:  Self の中心（自己の代表ベクトル）
        metric:      Self-Space の距離構造 (d, d)
    """

    def __init__(
        self,
        dim: int,
        max_axes: int = 8,
        init_scale: float = 0.01,
        eta_metric: float = 0.05,
        eta_self: float = 0.1,
        device: str = "cpu",
    ):
        super().__init__()
        self.dim = dim
        self.max_axes = max_axes
        self.device = device

        # ---- Self axes（Trace から形成される）----
        axes = torch.randn(max_axes, dim) * init_scale
        axes = F.normalize(axes, dim=-1)
        self.axes = nn.Parameter(axes)  # 手動更新（SGD ではなく Trace で変化）

        self.strength = nn.Parameter(torch.zeros(max_axes))
        self.register_buffer("num_active", torch.tensor(0, dtype=torch.long))

        # ---- Self の中心（幾何学的な「自分」）----
        # 初期状態はランダム正規化ベクトル
        self.self_state = nn.Parameter(
            F.normalize(torch.randn(dim, device=device), dim=-1)
        )

        # ---- Metric（心理的距離構造）----
        # 初期は等方的（単位行列）
        self.metric = nn.Parameter(torch.eye(dim, device=device))

        # GrowthKernel 的ハイパーパラメータ
        self.eta_metric = eta_metric
        self.eta_self = eta_self

    # ==========================================================
    @torch.no_grad()
    def update(
        self,
        trace: torch.Tensor,
        shock: float,
        affect: float,
        sim_threshold: float = 0.7,
        lr_axes: float = 0.2,
    ):
        """
        Trace を Self に取り込み、Self軸・Self中心・Metric を同時に更新する。

        trace:   (d,)
        shock:   不一致・驚き (0-1)
        affect:  情動の重み (0-1)
        """
        trace = trace.to(self.device)
        influence = float(shock * affect)

        if influence <= 0.0:
            return

        trace_norm = F.normalize(trace, dim=-1)

        # 1) Self axes の更新（v1 相当）
        self._update_axes(trace_norm, influence, sim_threshold, lr_axes)

        # 2) Plasticity（変形のしやすさ）を推定
        plasticity = self._estimate_plasticity(influence)

        if plasticity <= 0.0:
            return

        # 3) GrowthKernel: Self と Metric を同時に変形
        new_self, new_metric = self._growth_step(
            self.self_state.data,
            self.metric.data,
            trace_norm,
            plasticity,
        )

        self.self_state.data = new_self
        self.metric.data = new_metric

    # ==========================================================
    @torch.no_grad()
    def _update_axes(
        self,
        trace_norm: torch.Tensor,
        influence: float,
        sim_threshold: float,
        lr_axes: float,
    ):
        """
        v1 相当の Self 軸更新部分。
        Trace を新しい軸として追加するか、
        既存軸とマージして更新する。
        """
        k = self.num_active.item()

        if k == 0:
            self.axes.data[0] = trace_norm
            self.strength.data[0] = influence
            self.num_active.fill_(1)
            return

        active_axes = self.axes.data[:k]
        sims = F.cosine_similarity(active_axes, trace_norm.unsqueeze(0), dim=-1)
        max_sim, idx = sims.max(dim=0)

        if max_sim > sim_threshold:
            i = idx.item()
            old_axis = active_axes[i]
            new_axis = F.normalize(
                (1 - lr_axes * influence) * old_axis + (lr_axes * influence) * trace_norm,
                dim=-1,
            )
            self.axes.data[i] = new_axis
            self.strength.data[i] += influence
        else:
            if k < self.max_axes:
                self.axes.data[k] = trace_norm
                self.strength.data[k] = influence
                self.num_active.add_(1)
            else:
                weakest_idx = torch.argmin(self.strength.data[:k]).item()
                self.axes.data[weakest_idx] = trace_norm
                self.strength.data[weakest_idx] = influence

    # ==========================================================
    @torch.no_grad()
    def _estimate_plasticity(self, influence: float) -> float:
        """
        Self の「変形しやすさ」を簡易に推定する。

        直感:
            - strength が大きいほど、Self は固くなる（防衛・慣性）
            - 逆に、全体の強度がまだ低いなら、柔らかく変形しやすい
            - shock × affect が大きい経験は、それでも Self を揺らす

        ここでは簡略版:
            rigidity = mean(strength)
            base_plasticity = 1 / (1 + rigidity)
            plasticity = base_plasticity * influence
        """
        k = self.num_active.item()
        if k == 0:
            return float(influence)  # まだ何もない Self は柔らかい

        mean_strength = self.strength.data[:k].clamp(min=0.0).mean().item()
        rigidity = 1.0 + mean_strength  # 防衛・慣性
        base_plasticity = 1.0 / rigidity  # 強度が高いほど硬くなる

        plasticity = base_plasticity * influence
        plasticity = float(max(0.0, min(1.0, plasticity)))  # [0,1] にクリップ
        return plasticity

    # ==========================================================
    @torch.no_grad()
    def _growth_step(
        self,
        self_state: torch.Tensor,
        metric: torch.Tensor,
        trace_norm: torch.Tensor,
        plasticity: float,
    ):
        """
        GrowthKernel 本体:
            Self の中心 self_state
            Metric (距離構造)
        を Trace と Plasticity に応じて同時に変形させる。
        """
        # ---- Self 更新 ----
        new_self = self_state + self.eta_self * plasticity * trace_norm
        new_self = F.normalize(new_self, dim=-1)

        # ---- Metric 更新（rank-1 update）----
        outer = torch.outer(trace_norm, trace_norm)
        new_metric = metric + self.eta_metric * plasticity * outer
        # 安定性のため対角に小さな項を足す
        new_metric = new_metric + 1e-4 * torch.eye(self.dim, device=self.device)

        return new_self, new_metric

    # ==========================================================
    def condition(
        self,
        Q: torch.Tensor,
        alpha_metric: float = 0.4,
        alpha_axes: float = 0.6,
    ) -> torch.Tensor:
        """
        Query ベクトル Q を Self-Space で歪ませる。

        - Metric による全体的な方向歪み（世界の見え方の傾き）
        - axes による自己軸方向のバイアス（自分らしさ・癖）

        Q: (..., d)
        """
        if self.num_active.item() == 0:
            # Self がまだ形成されていない場合は素通し
            return Q

        # ---- Metric による変形 ----
        Q_metric = torch.matmul(Q, self.metric)  # (..., d)

        # ---- Self axes によるバイアス ----
        k = self.num_active.item()
        axes = F.normalize(self.axes[:k], dim=-1)
        strength = F.relu(self.strength[:k]) + 1e-6
        weights = strength / strength.sum()

        Q_norm = F.normalize(Q, dim=-1)
        sims = torch.matmul(Q_norm, axes.t())           # (..., k)
        contrib = torch.matmul(sims * weights.unsqueeze(0), axes)  # (..., d)

        # ---- 統合：Self-Space での幾何学的歪み ----
        Q_new = (
            (1.0 - alpha_metric - alpha_axes) * Q
            + alpha_metric * Q_metric
            + alpha_axes * contrib
        )

        return F.normalize(Q_new, dim=-1)

    # ==========================================================
    @torch.no_grad()
    def metrics(self) -> dict:
        """
        Self-Space の状態をざっくり可視化するためのメトリクス。
        - num_axes:     現在の Self 軸数（経験の多様性）
        - strength_sum: Self 軸全体の強度（安定度・慣性）
        - metric_trace: Metric のトレース（全体感度）
        - metric_norm:  Metric のノルム（歪みの大きさ）
        """
        k = self.num_active.item()
        strength_sum = float(self.strength.data[:k].sum()) if k > 0 else 0.0

        return {
            "num_axes": k,
            "strength_sum": strength_sum,
            "metric_trace": torch.trace(self.metric).item(),
            "metric_norm": torch.norm(self.metric).item(),
        }
