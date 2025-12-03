import torch
from identity.core.soul_state import SoulState
from utils.path_utils import generate_soul_path

class SoulManager:
    """
    魂（SoulState）の保存・読み込み・世代管理専用クラス
    """

    def __init__(self, base_dir="experiments/souls"):
        self.base_dir = base_dir

    # 魂の保存（Controller → Agent を介して呼ぶ）
    def save_soul(self, soul_state: SoulState, generation=0):
        path = generate_soul_path(
            kind=soul_state.meta.kind,
            generation=generation,
            step=soul_state.meta.last_step
        )
        torch.save(soul_state.to_dict(), path)
        print(f"[Soul Saved] → {path}")

    # 魂の読み込み
    def load_soul(self, path: str) -> SoulState:
        state_dict = torch.load(path, map_location="cpu")
        soul_state = SoulState.from_dict(state_dict)
        print(f"[Soul Loaded] kind={soul_state.meta.kind}")
        return soul_state
