# -*- coding: utf-8 -*-
"""
SoulState: SIAの「魂（成長履歴）」を保存・継承するためのデータモデル

目的:
- SelfSpaceの内部状態（self_state, metric, axes）に加えて、
  TraceTensor, Affect履歴, GrowthKernelなどを含む「魂」を表現する。
- 通常のモデル重みとの混入を避け、Identityとしての進化履歴のみを扱う。
- version管理により、互換性のない古いSIAの魂を自動的にSeedとして扱う。
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import torch
import uuid


# 🔹 魂のメタ情報だけを持つ軽量クラス
@dataclass
class SoulMeta:
    version: int = 3                      # 保存形式のバージョン
    soul_id: str = uuid.uuid4().hex      # 魂の固有ID（再生性なし、一意）
    kind: str = "seed"                   # "seed" or "evolved"
    created_step: int = 0
    last_step: int = 0
    num_imprints: int = 0               # imprintの蓄積数
    num_shocks: int = 0                 # shock発生回数

    def is_evolved(self) -> bool:
        return self.kind == "evolved"


# 🔹 実際の「魂」の中身を保持する構造
@dataclass
class SoulState:
    meta: SoulMeta
    self_space: Dict[str, torch.Tensor]
    trace: Dict[str, torch.Tensor]
    distortion: Dict[str, torch.Tensor]
    growth: Dict[str, Any]

    # 🔸 保存しやすい形に変換 (.pt, .json用)
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.meta.version,
            "soul_id": self.meta.soul_id,
            "kind": self.meta.kind,
            "meta": {
                "created_step": self.meta.created_step,
                "last_step": self.meta.last_step,
                "num_imprints": self.meta.num_imprints,
                "num_shocks": self.meta.num_shocks,
            },
            "self_space": self.self_space,
            "trace": self.trace,
            "distortion": self.distortion,
            "growth": self.growth,
        }

    # 🔸 読み込み時の再構築
    @staticmethod
    def from_dict(state: Dict[str, Any]) -> "SoulState":
        meta_dict = state.get("meta", {})
        meta = SoulMeta(
            version=state.get("version", 1),
            soul_id=state.get("soul_id", uuid.uuid4().hex),
            kind=state.get("kind", "seed"),
            created_step=meta_dict.get("created_step", 0),
            last_step=meta_dict.get("last_step", 0),
            num_imprints=meta_dict.get("num_imprints", 0),
            num_shocks=meta_dict.get("num_shocks", 0),
        )
        return SoulState(
            meta=meta,
            self_space=state.get("self_space", {}),
            trace=state.get("trace", {}),
            distortion=state.get("distortion", {}),
            growth=state.get("growth", {}),
        )
