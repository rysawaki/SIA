import torch
import torch.nn as nn
import torch.nn.functional as F

class IdentityEngine(nn.Module):
    def __init__(self, dim_self=64, alpha_trace=0.1, beta_affect=0.1, gamma_deform=0.05, device="cpu"):
        super().__init__()
        self.dim_self = dim_self
        self.alpha_trace = alpha_trace
        self.beta_affect = beta_affect
        self.gamma_deform = gamma_deform
        self.device = torch.device(device)

        # Self / Trace / Affect を持続的状態として保持
        self.register_buffer("S", torch.zeros(dim_self, device=self.device))
        self.register_buffer("T", torch.zeros(dim_self, device=self.device))
        self.register_buffer("A", torch.zeros(dim_self, device=self.device))

        # Geometryパラメータ（とりあえず線形変換）
        self.geometry = nn.Linear(dim_self, dim_self)

    @torch.no_grad()
    def imprint(self, shock_scalar: torch.Tensor, discrepancy_vec: torch.Tensor):
        """
        ΔT ∝ tanh(|Shock|) · discrepancy
        shock_scalar: ()  or (1,)
        discrepancy_vec: (dim_self,)
        """
        shock_mag = torch.tanh(torch.abs(shock_scalar))  # 0〜1に圧縮
        delta_T = self.alpha_trace * shock_mag * discrepancy_vec
        self.T += delta_T
        return delta_T

    @torch.no_grad()
    def update_affect(self, context_vec: torch.Tensor):
        """
        Affect 更新: A ← (1-β)A + β·context
        context_vec: (dim_self,)
        """
        self.A = (1 - self.beta_affect) * self.A + self.beta_affect * context_vec
        return self.A

    @torch.no_grad()
    def deform_self(self):
        """
        Self変形: S ← S + γ · f(T, A)
        今はシンプルに、(T ⊙ A) を geometry で変換
        """
        interaction = self.T * self.A
        delta_S = self.gamma_deform * self.geometry(interaction)
        self.S += delta_S
        return delta_S

    @torch.no_grad()
    def attribute_self(self, trace_vec: torch.Tensor):
        """
        P(self | Trace) ∝ exp(Trace · A)
        ここではスカラーとして返す（0〜1に正規化してもよい）
        """
        logits = torch.dot(trace_vec, self.A)
        # 温度1のシグモイドで 0〜1
        prob_self = torch.sigmoid(logits)
        return prob_self

    @torch.no_grad()
    def step(self, expected_vec: torch.Tensor, observed_vec: torch.Tensor, context_vec: torch.Tensor):
        """
        1ステップの Identity 更新:
        - discrepancy = observed - expected
        - shock = ||discrepancy||
        - imprint
        - affect更新
        - self変形
        """
        discrepancy = observed_vec - expected_vec
        shock = torch.norm(discrepancy)  # スカラー

        delta_T = self.imprint(shock, discrepancy)
        self.update_affect(context_vec)
        delta_S = self.deform_self()

        return {
            "shock": shock.detach().cpu(),
            "delta_T": delta_T.detach().cpu(),
            "delta_S": delta_S.detach().cpu(),
        }

    @torch.no_grad()
    def get_state(self):
        """LLMに渡すための自己状態"""
        return {
            "S": self.S.detach().clone(),
            "T": self.T.detach().clone(),
            "A": self.A.detach().clone(),
        }
