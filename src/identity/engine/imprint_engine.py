# src/identity/geometry/imprint_engine.py
# -*- coding: utf-8 -*-
"""
Imprint-Based Geometry Engine

入力された経験 e_t と、それに対する
    - Attribution（どれだけ「自分事」か）
    - Affect（ヴァレンス・覚醒）
を受け取り、
    - TraceTensor（痕跡ベクトル）
    - Self-space 上の Potental / Metric / Curvature
を更新するモジュール。

芸術（絵・音）、数式、自己体験など異なるモダリティを
共通の「自己幾何の変形」として扱うためのコアエンジン。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImprintEvent:
    """
    1ステップ分の「Imprint イベント」を表現する構造体。

    Attributes:
        u_t: Self-space 上の座標（latent_dim 次元）。
        attribution: [0, 1] のスカラー。「どれだけ自分事か」。
        valence: 負〜正のスカラー。快–不快。
        arousal: [0, 1] などの非負スカラー。情動強度。
        meta: 付随情報（"source": "VanGogh" など）。
    """
    u_t: torch.Tensor
    attribution: float
    valence: float
    arousal: float
    meta: Optional[Dict[str, Any]] = None


class SelfSpace(nn.Module):
    """
    Self-space（自己空間）の幾何構造を管理するクラス。

    - coords:   自己空間上の代表点（格子点でもよい）
    - potential: 各点におけるポテンシャル V(s)
    - metric_scale: 各点における計量スケール（等方的に I_d をスケーリング）
    - curvature: （近似的な）曲率 K(s) を後から載せるためのバッファ
    """

    def __init__(
        self,
        latent_dim: int,
        num_points: int,
        init_coords: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_points = num_points

        if device is None:
            device = torch.device("cpu")

        if init_coords is None:
            coords = torch.randn(num_points, latent_dim, device=device)
        else:
            assert init_coords.shape == (num_points, latent_dim)
            coords = init_coords.to(device)

        # register_buffer を使うことで、学習対象ではないが state_dict には入る
        self.register_buffer("coords", coords)                           # (N, d)
        self.register_buffer("potential", torch.zeros(num_points, device=device))     # (N,)
        self.register_buffer("metric_scale", torch.ones(num_points, device=device))   # (N,)
        self.register_buffer("curvature", torch.zeros(num_points, device=device))     # (N,)

    @property
    def device(self) -> torch.device:
        return self.coords.device

    def update_potential(self, new_potential: torch.Tensor) -> None:
        """
        ポテンシャル V(s) を更新する。

        Args:
            new_potential: shape (num_points,)
        """
        assert new_potential.shape == self.potential.shape
        with torch.no_grad():
            self.potential.copy_(new_potential)

    def update_metric_from_potential(self, alpha: float = 1.0) -> None:
        """
        V(s) から計量スケール（等方的）を計算する。
            g_theta(s) = exp(-alpha * V(s)) * I_d
        """
        with torch.no_grad():
            self.metric_scale.copy_(torch.exp(-alpha * self.potential))

    def update_curvature_from_potential_laplacian(
        self,
        k: int = 8,
        epsilon: float = 1e-5,
    ) -> None:
        """
        V(s) の離散ラプラシアンを簡易的に曲率 proxy として計算し、curvature に格納する。

        非常にラフな近似なので、あとで本格的な幾何に差し替え可能。

        Args:
            k: 近傍点の数（k-NN グラフ）
            epsilon: 数値安定用
        """
        with torch.no_grad():
            # 距離行列（N, N）を計算（N が大きいときは要最適化）
            coords = self.coords  # (N, d)
            N, d = coords.shape
            # (N, 1, d) - (1, N, d) → (N, N, d)
            diff = coords.unsqueeze(1) - coords.unsqueeze(0)
            dist2 = (diff ** 2).sum(dim=-1) + epsilon  # (N, N)

            # k-NN グラフの重み（ガウスカーネルもどき）
            # 自分自身への重みは 0 にしておく
            dist2 = dist2 + torch.eye(N, device=coords.device) * 1e6

            knn_vals, knn_idx = torch.topk(-dist2, k=k, dim=1)  # 最大値=最小距離
            # ガウスカーネル風の重み
            sigma2 = dist2.mean().item()
            weights = torch.exp(knn_vals / (sigma2 + epsilon))  # (N, k)

            # ラプラシアン L f ≈ Σ_j w_ij (f_j - f_i)
            V = self.potential  # (N,)
            V_neighbors = V[knn_idx]  # (N, k)
            V_i = V.unsqueeze(1)      # (N, 1)
            lap = (weights * (V_neighbors - V_i)).sum(dim=1)  # (N,)

            # ラプラシアンの符号等でそのまま曲率 proxy として扱う
            self.curvature.copy_(lap)

    def get_metric(self) -> torch.Tensor:
        """
        現在の計量テンソル g_theta(s_i) を返す。
        ここでは各点で等方的スケーリングされた I_d として扱う。

        Returns:
            metric: (num_points, latent_dim, latent_dim)
        """
        I = torch.eye(self.latent_dim, device=self.device)
        return self.metric_scale.view(-1, 1, 1) * I

    def sample_local_indices(self, center: torch.Tensor, k: int = 8) -> torch.Tensor:
        """
        Self-space 上の点 center の近傍にある代表点インデックスを返す。

        Args:
            center: (latent_dim,) の座標
            k: 取得する近傍数

        Returns:
            idx: (k,) 近傍点インデックス
        """
        center = center.to(self.device)
        diff = self.coords - center.unsqueeze(0)  # (N, d)
        dist2 = (diff ** 2).sum(dim=-1)           # (N,)
        k = min(k, self.num_points)
        _, idx = torch.topk(-dist2, k=k, dim=0)
        return idx


class TraceTensor(nn.Module):
    """
    時間とともに蓄積される Trace（痕跡）を保持する簡易モデル。

    - trace: latent_dim 次元のベクトルとして Self-space 全体の「向き」と強度を表現。
    - decay_rate: 過去の痕跡の減衰率 λ。
    - eta: 新しい Imprint の学習率。
    """

    def __init__(
        self,
        latent_dim: int,
        decay_rate: float = 0.01,
        eta: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.decay_rate = decay_rate
        self.eta = eta

        if device is None:
            device = torch.device("cpu")

        self.register_buffer("trace", torch.zeros(latent_dim, device=device))

    @property
    def device(self) -> torch.device:
        return self.trace.device

    def step(self, imprint_vec: torch.Tensor) -> None:
        """
        痕跡ベクトルを 1 ステップ更新する。

        T_{t+1} = (1 - decay_rate) * T_t + eta * imprint_vec
        """
        imprint_vec = imprint_vec.to(self.device)
        assert imprint_vec.shape == self.trace.shape
        with torch.no_grad():
            self.trace.mul_(1.0 - self.decay_rate).add_(self.eta * imprint_vec)


class ImprintGeometryEngine(nn.Module):
    """
    Imprint-Based Geometry Engine 本体。

    役割:
        - ImprintEvent を受け取り、
            - Imprint ベクトル I_t の生成
            - TraceTensor の更新
            - SelfSpace の Potential / Metric / Curvature の更新
        を行う。
    """

    def __init__(
        self,
        self_space: SelfSpace,
        trace_tensor: TraceTensor,
        alpha_metric: float = 1.0,
    ):
        super().__init__()
        self.self_space = self_space
        self.trace = trace_tensor
        self.alpha_metric = alpha_metric

        # self_center は「現在の自己の重心」として扱う（必要なら外から更新）
        self.register_buffer(
            "self_center",
            torch.zeros(self_space.latent_dim, device=self_space.device),
        )

    @property
    def device(self) -> torch.device:
        return self.self_space.device

    def set_self_center(self, s_t: torch.Tensor) -> None:
        """
        現在の Self の重心を設定する（External Controller から渡される想定）。
        """
        assert s_t.shape == self.self_center.shape
        with torch.no_grad():
            self.self_center.copy_(s_t.to(self.device))

    def compute_imprint_vec(
        self,
        u_t: torch.Tensor,
        attribution: float,
        valence: float,
        arousal: float,
        use_self_center: bool = True,
    ) -> torch.Tensor:
        """
        Imprint ベクトル I_t を計算する。

        直感的には：
            - u_t と self_center の差分方向に
            - Attribution × Arousal × sign(valence)
        を掛けたもの。

        必要に応じて、よりリッチな関数 g(u
        """

