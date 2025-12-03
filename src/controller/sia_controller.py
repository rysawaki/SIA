# src/controller/sia_controller.py
# -*- coding: utf-8 -*-

import torch
import uuid
import os
from identity.core.soul_state import SoulState, SoulMeta


class SIAController:
    """
    SIAAgent の中枢コントローラ。
    Body（モデル）と Soul（痕跡・identity成長情報）を切り分けて管理する。
    """

    def __init__(self, body, growth_kernel=None, trace_tensor=None):
        self.body = body  # Transformer, SelfSpace 含む
        self.growth_kernel = growth_kernel
        self.trace_tensor = trace_tensor

        # Optional buffers (必要なら保持)
        self.recent_imprints = None
        self.affect_history = None
        self.distortion_field = None

        # メタ情報
        self.num_imprints = 0
        self.num_shocks = 0
        self.global_step = 0

    # =========================================================
    # 🔹 魂の保存
    # =========================================================
    def save_soul(self, path: str):
        """
        現在のSoulを保存する（Torch形式）。
        path: "experiments/sia_soul_state.pt" など
        """
        soul = self._collect_soul_state()
        torch.save(soul.to_dict(), path)
        print(f"[Soul Saved] → {path} (kind={soul.meta.kind})")

    # =========================================================
    # 🔹 魂の読み込み（Seed/Evolvedを自動判定）
    # =========================================================
    def load_soul(self, path: str):
        """
        魂データを読み込む。
        - version に応じて Seed or Evolved として統合
        - 不完全な古い魂（v1, v2）は seed 扱いで axes のみ復元
        """

        state = torch.load(path, map_location="cpu")
        version = state.get("version", 1)

        if version < 3:
            print("[Legacy Soul Detected] → 認識不能 / seed として取り込む")
            self._load_legacy_seed(state)
            return

        soul_state = SoulState.from_dict(state)
        print(f"[Soul Loaded] kind={soul_state.meta.kind}, step={soul_state.meta.last_step}")

        self._integrate_soul_state(soul_state)

    # =========================================================
    # 🔸 魂の収集（保存用）
    # =========================================================
    def _collect_soul_state(self) -> SoulState:
        """
        現在の Identity（魂）を構造化して SoulState に変換。
        """

        # --- SelfSpace 情報の抽出 ---
        ss = self.body.self_space
        self_space = {
            "self_state": ss.self_state.detach().cpu(),
            "metric": ss.metric.detach().cpu(),
        }
        if hasattr(ss, "axes"):
            self_space["axes"] = ss.axes.detach().cpu()

        # --- Trace / Affect / Growth の収集 ---
        trace = {}
        if self.trace_tensor is not None:
            trace["trace_tensor"] = self.trace_tensor.detach().cpu()
        if self.recent_imprints is not None:
            trace["recent_imprints"] = self.recent_imprints.detach().cpu()
        if self.affect_history is not None:
            trace["affect_history"] = self.affect_history.detach().cpu()

        distortion = {}
        if self.distortion_field is not None:
            distortion["distortion_field"] = self.distortion_field.detach().cpu()

        growth = {}
        if self.growth_kernel is not None:
            growth["growth_kernel_state"] = self.growth_kernel.state_dict()

        # --- メタ情報を構築 ---
        meta = SoulMeta(
            version=3,
            soul_id=uuid.uuid4().hex,
            kind="evolved" if self.num_imprints >= 10 else "seed",
            created_step=0,
            last_step=self.global_step,
            num_imprints=self.num_imprints,
            num_shocks=self.num_shocks,
        )

        return SoulState(
            meta=meta,
            self_space=self_space,
            trace=trace,
            distortion=distortion,
            growth=growth,
        )

    # =========================================================
    # 🔸 魂の統合（本質的な復元処理）
    # =========================================================
    def _integrate_soul_state(self, soul: SoulState):
        """
        魂を SelfSpace / Trace / Growth に統合。
        「seed」と「evolved」で復元範囲を自動で変える。
        """

        ss_state = {}
        if "self_state" in soul.self_space:
            ss_state["self_state"] = soul.self_space["self_state"]
        if "metric" in soul.self_space:
            ss_state["metric"] = soul.self_space["metric"]
        if "axes" in soul.self_space:
            ss_state["axes"] = soul.self_space["axes"]

        self.body.self_space.load_state_dict(ss_state, strict=False)

        # Evolved の場合のみ、成長履歴を完全反映
        if soul.meta.is_evolved():

            if "trace_tensor" in soul.trace and self.trace_tensor is not None:
                self.trace_tensor.copy_(soul.trace["trace_tensor"])

            if "recent_imprints" in soul.trace and self.recent_imprints is not None:
                self.recent_imprints.copy_(soul.trace["recent_imprints"])

            if "affect_history" in soul.trace and self.affect_history is not None:
                self.affect_history.copy_(soul.affect_history)

            if "distortion_field" in soul.distortion and self.distortion_field is not None:
                self.distortion_field.copy_(soul.distortion["distortion_field"])

            if "growth_kernel_state" in soul.growth and self.growth_kernel is not None:
                self.growth_kernel.load_state_dict(soul.growth["growth_kernel_state"])

            print("[Soul Integration] → 完全継承（Evolved）")

        else:
            print("[Soul Integration] → 軸のみ反映（Seed）")

    # =========================================================
    # 🔹 古い魂(v1/v2構造)の読み込み（Seed扱い）
    # =========================================================
    def _load_legacy_seed(self, legacy_data: dict):
        ss = self.body.self_space
        if "self_space" in legacy_data and "axes" in legacy_data["self_space"]:
            ss.load_state_dict({"axes": legacy_data["self_space"]["axes"]}, strict=False)
        print("[Legacy->Seed] → axes のみ継承。Traceなどは破棄。")

