# src/identity/core/trace_tensor.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import Tensor


@dataclass
class TraceTensor:
    """
    Geometric representation of 'Trace' in SIA.

    - T is treated as a Riemannian-ish metric on Self-space.
    - It should be (approximately) symmetric and (ideally) positive semi-definite.
    - This class does NOT decide how T is updated; that is GrowthKernel / IdentityEngine の仕事。
      ここはあくまで「構造」と「計算API」を提供する層。
    """
    T: Tensor  # shape: (d, d)

    def __post_init__(self):
        if self.T.dim() != 2 or self.T.size(0) != self.T.size(1):
            raise ValueError(f"TraceTensor must be square matrix, got {self.T.shape}")
        # 対称性を軽く保証（完全ではなく「投影」）
        self.ensure_symmetric_()

    # ------------------------------------------------------------------
    #  Constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_dim(cls, dim: int, init_scale: float = 0.0, device=None, dtype=None) -> "TraceTensor":
        """
        dim 次元の TraceTensor を生成。
        init_scale=0.0 ならゼロ行列、>0 なら小さなランダムゆらぎを持つ。
        """
        T = torch.zeros(dim, dim, device=device, dtype=dtype or torch.float32)
        if init_scale > 0:
            noise = torch.randn_like(T) * init_scale
            T = T + 0.5 * (noise + noise.T)
        return cls(T)

    @classmethod
    def from_rank1(cls, u: Tensor, scale: float = 1.0) -> "TraceTensor":
        """
        u u^T からなるランク1 TraceTensor を生成。
        """
        if u.dim() != 1:
            raise ValueError("u must be 1D vector")
        T = scale * torch.outer(u, u)
        return cls(T)

    # ------------------------------------------------------------------
    #  Basic properties
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return self.T.size(0)

    def clone(self) -> "TraceTensor":
        return TraceTensor(self.T.clone())

    def to(self, *args, **kwargs) -> "TraceTensor":
        return TraceTensor(self.T.to(*args, **kwargs))

    # ------------------------------------------------------------------
    #  Symmetry / normalization
    # ------------------------------------------------------------------
    def ensure_symmetric_(self, alpha: float = 0.5) -> "TraceTensor":
        """
        T ← (1 - alpha) * T + alpha * (T + T^T)/2
        alpha=1.0 で完全に対称成分だけを残す。
        """
        sym = 0.5 * (self.T + self.T.T)
        self.T = (1 - alpha) * self.T + alpha * sym
        return self

    def normalize_spectral_(self, max_eig: float = 1.0) -> "TraceTensor":
        """
        固有値の最大値を max_eig 以下になるようスケーリング。
        「暴走する曲率」を抑えるための簡易正則化。
        """
        with torch.no_grad():
            # CPUで十分（dはそこまで大きくない前提）
            vals = torch.linalg.eigvalsh(self.T.cpu())
            lam_max = torch.max(torch.abs(vals))
            if lam_max > 0:
                scale = (max_eig / lam_max).item()
                if scale < 1.0:
                    self.T *= scale
        return self

    # ------------------------------------------------------------------
    #  Eigen-decomposition / curvature-like quantities
    # ------------------------------------------------------------------
    def eigendecompose(self) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            eigenvalues: (d,)
            eigenvectors: (d, d)  columns are eigenvectors
        """
        # hermitian=True で対称行列扱い
        vals, vecs = torch.linalg.eigh(self.T)
        return vals, vecs

    def scalar_curvature(self) -> Tensor:
        """
        簡易的な「スカラー曲率」の proxy。
        今は単純に固有値の総和 (trace) を返す。
        必要ならここを本格的な幾何量に置き換える。
        """
        return torch.trace(self.T)

    def anisotropy(self) -> Tensor:
        """
        空間の「ゆがみ具合」の指標。
        ここでは (最大固有値 - 最小固有値) を返す。
        """
        vals, _ = self.eigendecompose()
        return torch.max(vals) - torch.min(vals)

    def sectional_curvature(self, u: Tensor, v: Tensor, eps: float = 1e-8) -> Tensor:
        """
        2つの方向 u, v に対する「断面曲率っぽいもの」を返す。
        本物のRiemann曲率ではなく、SIA用のヒューリスティック指標。

        ここでは以下のような量を返す：
            K(u, v) = (g(u, u) * g(v, v) - g(u, v)^2) / ||u∧v||^2
        where g(x,y) = x^T T y

        u, v: (..., d)
        Returns:
            K: (...,)  same batch shape
        """
        # 正規化済み方向を使う
        u = u / (u.norm(dim=-1, keepdim=True) + eps)
        v = v / (v.norm(dim=-1, keepdim=True) + eps)

        # g(x, y) = x^T T y
        Tu = torch.matmul(u, self.T)        # (..., d)
        Tv = torch.matmul(v, self.T)        # (..., d)

        guu = (u * Tu).sum(dim=-1)
        gvv = (v * Tv).sum(dim=-1)
        guv = (u * Tv).sum(dim=-1)

        wedge_norm2 = torch.clamp(
            (u * u).sum(dim=-1) * (v * v).sum(dim=-1) - (u * v).sum(dim=-1) ** 2,
            min=eps,
        )

        K = (guu * gvv - guv ** 2) / wedge_norm2
        return K

    # ------------------------------------------------------------------
    #  Metric / distance / energy
    # ------------------------------------------------------------------
    def metric(self, v: Tensor) -> Tensor:
        """
        g(v, v) = v^T T v
        v: (..., d)
        Returns:
            (...,)  metric value
        """
        Tv = torch.matmul(v, self.T)       # (..., d)
        return (v * Tv).sum(dim=-1)

    def bilinear(self, u: Tensor, v: Tensor) -> Tensor:
        """
        g(u, v) = u^T T v
        """
        Tv = torch.matmul(v, self.T)
        return (u * Tv).sum(dim=-1)

    def distance(self, q: Tensor, k: Tensor) -> Tensor:
        """
        d_T(q, k)^2 = (q - k)^T T (q - k)
        q, k: (..., d)
        Returns:
            (...,)  squared distance
        """
        diff = q - k
        return self.metric(diff)

    # ------------------------------------------------------------------
    #  Update primitives (GrowthKernel から呼ばれることを想定)
    # ------------------------------------------------------------------
    def add_rank1_(self, u: Tensor, strength: float = 1.0) -> "TraceTensor":
        """
        T ← T + strength * u u^T
        Shock を受けた方向 u に沿って曲率を追加するイメージ。
        """
        if u.dim() != 1 or u.size(0) != self.dim:
            raise ValueError(f"u must be shape ({self.dim},), got {u.shape}")
        self.T = self.T + strength * torch.outer(u, u)
        self.ensure_symmetric_()
        return self

    def decay_(self, lam: float) -> "TraceTensor":
        """
        T ← lam * T
        lam in (0,1] で、Traceエネルギーを徐々に減衰させる。
        """
        self.T = lam * self.T
        return self
