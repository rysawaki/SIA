# src/growth_kernel.py

import torch
from torch import Tensor

EPS = 1e-6


class GrowthKernel:
    """
    SIA Core Mechanism:
    Imprint（気づき）を、方向性と厚みをもって育てるカーネル。

    I_{t+1} = I_t
              + alpha * A_t * g(||delta||)
                * (delta ⊗ delta / ||delta||^2)
                * h(||I_t||)

    - delta: Self と Event の差（足跡の方向）
    - A_t:   Attribution（意味の自己関連度）
    """

    def __init__(
        self,
        alpha: float = 0.05,
        novelty_scale: float = 1.0,
        saturation_center: float = 1.0,
        device: str = "cpu",
    ) -> None:
        self.alpha = alpha
        self.novelty_scale = novelty_scale
        self.saturation_center = saturation_center
        self.device = torch.device(device)

    def novelty_gain(self, delta_norm: Tensor) -> Tensor:
        """
        新規性 g(||delta||):
        Self から遠いほど、成長ゲインを高める。
        """
        return torch.tanh(delta_norm / self.novelty_scale)

    def saturation_factor(self, I: Tensor) -> Tensor:
        """
        飽和 h(||I||_F):
        すでに十分育っている方向では、成長を抑える。
        """
        frob = torch.norm(I, p="fro")
        x = self.saturation_center - frob
        return torch.sigmoid(x)

    def __call__(self, delta: Tensor, I: Tensor, A_t: Tensor) -> Tensor:
        """
        GrowthKernel(delta, I_t, A_t) → ΔI

        delta: (d,)   Self と Event の差
        I:     (d,d)  現在の Imprint（気づきテンソル）
        A_t:   () or (1,)  Attribution（意味の自己関連度）
        """
        delta = delta.to(self.device)
        I = I.to(self.device)
        A_t = A_t.to(self.device)

        delta_norm = torch.norm(delta) + EPS
        dir_tensor = torch.outer(delta, delta) / (delta_norm ** 2)

        g = self.novelty_gain(delta_norm)
        h = self.saturation_factor(I)

        update = self.alpha * A_t * g * h * dir_tensor
        return update
