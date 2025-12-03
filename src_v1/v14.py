import torch
import torch.nn as nn
import torch.nn.functional as F


# ===============================================================
# 0. Environment (1D Miniworld)
# ===============================================================
class MiniEnv1D:
    """
    1D世界： エージェントは位置 pos を持ち、目標 goal に近づく
    観測: [pos, goal-pos, velocity]
    行動: -1, 0, +1
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.pos = torch.randn(1).item() * 0.5
        self.goal = torch.randn(1).item() * 2.0
        self.vel = 0.0
        return self._obs()

    def _obs(self):
        return torch.tensor([self.pos, self.goal - self.pos, self.vel], dtype=torch.float32)

    def step(self, action):
        # action ∈ {-1, 0, 1}
        self.vel = 0.8 * self.vel + 0.2 * action
        self.pos += self.vel

        obs = self._obs()
        reward = -abs(self.goal - self.pos)
        done = abs(self.goal - self.pos) < 0.05

        return obs, reward, done


# ===============================================================
# 1. Hybrid Trace Module (Oja + Full Hebbian)
# ===============================================================
class HybridTrace(nn.Module):
    """
    mode = 'oja'  : Rank-1 主成分（Oja's Rule）
    mode = 'full' : Full Hebbian 行列
    """

    def __init__(self, d_model, mode='oja', lr=0.05):
        super().__init__()
        self.d_model = d_model
        self.mode = mode  # 'oja' or 'full'
        self.lr = lr

        # Rank-1 (Oja): 主成分ベクトル v
        self.register_buffer('v', F.normalize(torch.randn(d_model), dim=0))

        # Full-Rank (Hebbian): 共分散行列 M
        self.register_buffer('M', torch.zeros(d_model, d_model))

    def get_energy(self, affect: torch.Tensor) -> torch.Tensor:
        """
        共鳴エネルギーを計算
        mode='oja' : E = (v^T x)^2
        mode='full': E = x^T M x
        """
        if self.mode == 'oja':
            proj = torch.dot(self.v, affect)
            return proj * proj  # (v^T x)^2

        elif self.mode == 'full':
            x = affect.unsqueeze(0)        # (1, d)
            energy = (x @ self.M @ x.t()).squeeze()  # scalar
            return energy

        else:
            raise ValueError(f"Unknown trace mode: {self.mode}")

    def update(self, affect: torch.Tensor):
        """
        記憶更新（不可逆）
        affect = Shock後の情動ベクトル
        """
        with torch.no_grad():
            if self.mode == 'oja':
                # Oja's Rule: v ← v + lr * y * (x - y v)
                y = torch.dot(self.v, affect)
                delta = y * (affect - y * self.v)
                self.v += self.lr * delta
                self.v = F.normalize(self.v, dim=0)

            elif self.mode == 'full':
                # Hebbian: M ← decay*M + lr * (x x^T)
                self.M *= 0.95  # 忘却（古い痕跡が少しずつ薄れる）
                delta = torch.ger(affect, affect)
                self.M += self.lr * delta

            else:
                raise ValueError(f"Unknown trace mode: {self.mode}")

    def get_distortion_vector(self, affect: torch.Tensor) -> torch.Tensor:
        """
        トラウマによる歪みベクトル
        mode='oja'  : v 方向に引き寄せ
        mode='full' : M x による複雑な歪み
        """
        if self.mode == 'oja':
            y = torch.dot(self.v, affect)
            return y * self.v

        elif self.mode == 'full':
            return self.M @ affect

        else:
            raise ValueError(f"Unknown trace mode: {self.mode}")


# ===============================================================
# 2. SIA Params / State
# ===============================================================
class SIAParams(nn.Module):
    def __init__(self, d_model=32, obs_dim=3):
        super().__init__()
        self.d_model = d_model

        # obs → embedding (x_t)
        self.obs_proj = nn.Linear(obs_dim, d_model)

        # Shock → Affect
        self.shock_proj = nn.Linear(d_model, d_model)
        self.affect_weight = nn.Parameter(torch.randn(d_model))

        # Self 更新
        self.W_h = nn.Linear(d_model, d_model)

        # Gate パラメータ（調整済み）
        self.flashback_gain = 5.0
        self.gate_sharpness = 7.0
        self.threshold = 0.75  # cos^2 ≈ 0.75 → cos ≈ 0.86 程度

        # 行動決定
        self.action_head = nn.Linear(d_model, 3)

    def embed(self, obs):
        return torch.tanh(self.obs_proj(obs))

    def shock(self, x):
        return torch.tanh(self.shock_proj(x))

    def affect(self, shock):
        return shock * torch.tanh(self.affect_weight)


class SIAState:
    def __init__(self, d_model):
        self.h = torch.zeros(d_model)

    def reset(self):
        self.h.zero_()


# ===============================================================
# 3. SIA Core with Resonance
# ===============================================================
class SIAResonanceCore:
    def __init__(self, params: SIAParams, state: SIAState, trace: HybridTrace):
        self.params = params
        self.state = state
        self.trace = trace

        # 不応期管理
        self.refractory_timer = 0
        self.refractory_limit = 5

    def step(self, obs: torch.Tensor):
        p = self.params
        s = self.state
        t = self.trace

        # 1. Perception → Shock → Affect
        x_t = p.embed(obs)
        shock = p.shock(x_t)
        affect = p.affect(shock)

        # 2. Resonance Check
        energy = t.get_energy(affect)  # scalar

        if t.mode == 'full':
            trace_norm = torch.norm(t.M) + 1e-6
            resonance = energy / trace_norm
        else:  # 'oja'
            # Rank-1: energy = (v^T x)^2 をそのまま使う（強い共鳴だけ反応）
            resonance = energy

        # 3. Gate（不応期付き）
        raw_gate = torch.sigmoid(p.gate_sharpness * (resonance - p.threshold))

        if self.refractory_timer > 0:
            gate = torch.tensor(0.0)
            self.refractory_timer -= 1
        else:
            gate = raw_gate
            if gate > 0.5:
                self.refractory_timer = self.refractory_limit

        # 4. Self 更新（h）
        h_decay = 0.9
        h_rational = torch.tanh(p.W_h(s.h) + x_t)

        distortion = t.get_distortion_vector(affect)
        mixed_input = x_t + gate * p.flashback_gain * distortion

        h_new = torch.tanh(p.W_h(h_decay * s.h) + mixed_input)

        with torch.no_grad():
            s.h = h_new

        # 5. Trace 更新（感情＋ゲート依存）
        # 共鳴が強いほど（gateが大きいほど）深く刻まれる
        t.lr = 0.02 + 0.08 * gate.item()
        t.update(affect)

        # 6. 行動決定
        logits = p.action_head(s.h)
        action = torch.argmax(logits).item() - 1  # {-1,0,+1}

        return action, float(resonance), float(gate), float(torch.norm(s.h))


# ===============================================================
# 4. Demo: 1D Miniworld × Hybrid SIA
# ===============================================================
def demo_miniworld_hybrid():
    torch.manual_seed(0)

    # モード切替: 'oja' (単一トラウマ) or 'full' (多方向トラウマ)
    MODE = 'oja'   # 'full' に変えて比較しても良い

    env = MiniEnv1D()
    obs = env.reset()

    d = 32
    params = SIAParams(d_model=d, obs_dim=3)
    state = SIAState(d_model=d)
    trace = HybridTrace(d_model=d, mode=MODE, lr=0.05)
    sia = SIAResonanceCore(params, state, trace)

    print(f"=== 1D Miniworld × SIA HybridTrace (mode={MODE}) ===")
    print(f"Initial pos={env.pos:.2f}, goal={env.goal:.2f}")

    for t in range(50):
        action, res, gate, h_norm = sia.step(obs)
        obs, reward, done = env.step(action)

        print(
            f"t={t:02d} | pos={env.pos:.2f} | act={action:+d} "
            f"| res={res:.3f} | gate={gate:.2f} | h={h_norm:.2f}"
            + ("  <FLASHBACK>" if gate > 0.6 else "")
        )

        if done:
            print("Goal reached!")
            break


if __name__ == "__main__":
    demo_miniworld_hybrid()
