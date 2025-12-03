# src/core_essence.py

import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Dict, Any, Optional

from design_graveyard.growth_kernel import GrowthKernel, EPS


@dataclass
class SelfState:
    """
    SIA Core Essence の状態表現：
    - s: Self の現在位置（Geometry の1点）
    - I: Imprint（気づきテンソル）
    """
    s: Tensor  # (d,)
    I: Tensor  # (d,d)

    def clone(self) -> "SelfState":
        return SelfState(s=self.s.clone(), I=self.I.clone())

    def to(self, device: str) -> "SelfState":
        dev = torch.device(device)
        return SelfState(s=self.s.to(dev), I=self.I.to(dev))


class SelfCore:
    """
    SIA Core Essence: Trace / Imprint / Geometry を動かす最小クラス。

    - Trace:  delta_t = event - s_t
    - Imprint: I_{t+1} = I_t + GrowthKernel(delta_t, I_t, A_t)
    - Geometry: s_{t+1} = s_t + η * preference_t * direction

    ここでは 2D/少数次元を想定した自己空間の最小モデル。
    """

    def __init__(
        self,
        dim: int = 2,
        device: str = "cpu",
        growth_kernel: Optional[GrowthKernel] = None,
        self_step_size: float = 0.01,
    ) -> None:
        self.dim = dim
        self.device = torch.device(device)
        self.growth_kernel = growth_kernel or GrowthKernel(device=device)
        self.self_step_size = self_step_size

        # 初期状態：原点に立つ Self、ゼロ Imprint
        s0 = torch.zeros(dim, device=self.device)
        I0 = torch.zeros(dim, dim, device=self.device)
        self.state = SelfState(s=s0, I=I0)

    @torch.no_grad()
    def compute_delta(self, event: Tensor) -> Tensor:
        """
        Trace（足跡）: Self と event の「差」として定義。
        """
        event = event.to(self.device)
        return event - self.state.s

    @torch.no_grad()
    def compute_attribution(self, delta: Tensor) -> Tensor:
        """
        Attribution（意味の自己関連度）の暫定版。

        現時点では：
        - Self に近すぎても、遠すぎても弱い
        - 中程度の距離が最も「意味を感じやすい」とみなす

        後で LLM / 意味空間に差し替え可能。
        """
        dist = torch.norm(delta)
        # 距離が 0 付近 / 大きすぎるところで抑制されるガウス的カーブ
        scale = 2.0
        return torch.exp(- (dist / scale) ** 2)

    @torch.no_grad()
    def compute_preference(self, delta: Tensor) -> Tensor:
        """
        Self が「この方向に惹かれている度合い」を測るスカラー。
        preference_t = δ^T I δ
        """
        I = self.state.I
        return torch.dot(delta, I @ delta)

    @torch.no_grad()
    def update_self_position(self, delta: Tensor, preference: Tensor) -> None:
        """
        Self の位置更新：
        s_{t+1} = s_t + η * preference_t * (δ / ||δ||)
        """
        delta_norm = torch.norm(delta) + EPS
        direction = delta / delta_norm
        step_vec = self.self_step_size * preference * direction
        self.state.s = self.state.s + step_vec

    @torch.no_grad()
    def step(self, event: Tensor) -> Dict[str, Any]:
        """
        1ステップの更新：
        1. Trace:    delta_t を計算
        2. A_t:      Attribution（意味の自己関連度）
        3. Imprint:  GrowthKernel により I_t を更新
        4. Geometry: preference_t に応じて Self の位置を更新

        戻り値はログ（可視化や学習用）として使える dict。
        """
        # 1. Trace
        delta = self.compute_delta(event)

        # 2. Attribution（気づきのトリガー強度）
        A_t = self.compute_attribution(delta)

        # 3. Imprint 更新
        dI = self.growth_kernel(delta, self.state.I, A_t)
        self.state.I = self.state.I + dI
        # 対称性を維持
        self.state.I = 0.5 * (self.state.I + self.state.I.T)

        # 4. Self の位置更新（Geometryの変化）
        preference = self.compute_preference(delta)
        self.update_self_position(delta, preference)

        # ログを返す（外側で可視化や記録に使う）
        return {
            "event": event.detach().cpu(),
            "delta": delta.detach().cpu(),
            "attribution": A_t.detach().cpu(),
            "preference": preference.detach().cpu(),
            "self_pos": self.state.s.detach().cpu(),
            "imprint": self.state.I.detach().cpu(),
        }

    def get_state(self) -> SelfState:
        return self.state

    def reset(self) -> None:
        """
        Self と Imprint をリセット。
        """
        s0 = torch.zeros(self.dim, device=self.device)
        I0 = torch.zeros(self.dim, self.dim, device=self.device)
        self.state = SelfState(s=s0, I=I0)
