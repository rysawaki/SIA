import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# ============================================================
# 1. Config
# ============================================================
class SIAConfig:
    h_dim = 16
    d_model = 32
    trace_shape = (3, 10, 32)
    lambda_reg = 0.01
    lr_self = 0.05
    lr_adapter = 0.005
    reality_weight = 0.8


class SIAConfig_Stable(SIAConfig):
    # 生物学的安定性のためのパラメータ
    trace_decay = 0.95  # 忘却係数
    max_h_val = 3.0  # 自己の飽和限界
    input_noise = 0.05  # 感覚ノイズ
    dropout_rate = 0.1  # 脳内ノイズ


# ============================================================
# 2. Components (Architecture)
# ============================================================

class SIAAttention(nn.Module):
    def __init__(self, d_model, h_dim):
        super().__init__()
        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.h_proj_q = nn.Linear(h_dim, d_model, bias=False)
        self.h_proj_k = nn.Linear(h_dim, d_model, bias=False)

    def forward(self, x, h):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        distortion_q = self.h_proj_q(h).unsqueeze(0).unsqueeze(0)
        distortion_k = self.h_proj_k(h).unsqueeze(0).unsqueeze(0)

        Q_distorted = Q + distortion_q
        K_distorted = K + distortion_k

        scores = torch.matmul(Q_distorted, K_distorted.transpose(-2, -1)) / (self.d_model ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights


class TraceTensor_v1_5(nn.Module):
    def __init__(self, depth, world_dim, h_dim, intent_dim, decay=0.9):
        super().__init__()
        self.feature_dim = world_dim + h_dim + intent_dim
        self.register_buffer('T', torch.zeros(depth, self.feature_dim))
        self.decay = decay
        self.split_sizes = [world_dim, h_dim, intent_dim]

    def imprint(self, world_embed, psi, intent):
        if psi.dim() > 1: psi = psi.mean(dim=0)
        if world_embed.dim() > 1: world_embed = world_embed.mean(dim=0)
        if intent.dim() > 1: intent = intent.mean(dim=0)

        new_memory = torch.cat([world_embed, psi, intent], dim=-1)

        shifted = torch.roll(self.T, shifts=1, dims=0)
        shifted[0] = new_memory

        # Note: Decay logic is handled by the caller or configured here.
        # Stable版では外部からdecay制御が入るため、基本実装はシンプルに保つ
        self.T = self.T * self.decay + shifted * (1 - self.decay)

    def read_components(self):
        return torch.split(self.T, self.split_sizes, dim=-1)

    def reconsolidate(self, current_h_intent, intensity=0.1):
        world, pain, intent = self.read_components()
        target_intent = current_h_intent.expand_as(intent)
        new_intent = intent * (1 - intensity) + target_intent * intensity
        self.T = torch.cat([world, pain, new_intent], dim=-1)


class MeaningAdapter_v1_5(nn.Module):
    def __init__(self, trace_depth, total_feature_dim, h_dim):
        super().__init__()
        input_dim = trace_depth * total_feature_dim
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, h_dim)
        )

    def forward(self, trace_tensor):
        return self.net(trace_tensor.unsqueeze(0)).squeeze(0)


# ============================================================
# 3. SIASystem Logic (Base & Stable)
# ============================================================

# 親クラス (v1.5 Base) を先に定義しなければならない
class SIASystem_v1_5(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.h = nn.Parameter(torch.randn(config.h_dim) * 0.1)

        self.sia_attention = SIAAttention(config.d_model, config.h_dim)
        self.output_proj = nn.Linear(config.d_model, config.d_model)

        intent_dim = 8
        self.intent_gen = nn.Sequential(
            nn.Linear(config.h_dim, intent_dim),
            nn.Softmax(dim=-1)
        )

        self.trace = TraceTensor_v1_5(
            depth=10,
            world_dim=config.d_model,
            h_dim=config.h_dim,
            intent_dim=intent_dim
        )

        self.adapter = MeaningAdapter_v1_5(
            trace_depth=10,
            total_feature_dim=config.d_model + config.h_dim + intent_dim,
            h_dim=config.h_dim
        )

        self.opt_adapter = optim.Adam(self.adapter.parameters(), lr=config.lr_adapter)

    def step(self, x, target_x):
        # 1. Forward
        distorted_view, attn_map = self.sia_attention(x, self.h)
        pred_x = self.output_proj(distorted_view)

        # 2. Physics
        diff = pred_x - target_x
        F = 0.5 * diff.pow(2).sum() + 0.5 * self.config.lambda_reg * self.h.pow(2).sum()

        if self.h.grad is not None: self.h.grad.zero_()
        grad_h_true = torch.autograd.grad(F, self.h, create_graph=False)[0]

        # 3. Imprint
        psi = grad_h_true.detach()
        intent = self.intent_gen(self.h)
        world_embed = distorted_view.mean(dim=[0, 1]).detach()
        self.trace.imprint(world_embed, psi, intent)

        # 4. Attribution
        trace_input = self.trace.T.detach()
        grad_h_est = self.adapter(trace_input)

        # 5. Meta-Learning
        loss_adapter = 0.5 * (grad_h_est - psi).pow(2).sum()
        self.opt_adapter.zero_grad()
        loss_adapter.backward()
        self.opt_adapter.step()

        # 6. Self Update
        alpha = self.config.reality_weight
        effective_grad = alpha * grad_h_true + (1 - alpha) * grad_h_est
        with torch.no_grad():
            self.h -= self.config.lr_self * effective_grad

        # 7. Reconsolidation
        recon_rate = 0.05
        with torch.no_grad():
            current_intent = self.intent_gen(self.h)
            self.trace.reconsolidate(current_intent, intensity=recon_rate)

        return {
            "F": F.item(),
            "loss_adapter": loss_adapter.item(),
            "h_norm": self.h.norm().item(),
            "grad_true_norm": grad_h_true.norm().item()
        }


# 子クラス (Stable) は親クラスの後に定義する
class SIASystem_Stable(SIASystem_v1_5):
    def __init__(self, config):
        super().__init__(config)
        self.dropout = nn.Dropout(config.dropout_rate)
        # Configから減衰率を適用
        self.trace.decay = config.trace_decay

    def step(self, x, target_x):
        # 1. Noise Injection
        if self.training or True:
            noise = torch.randn_like(x) * self.config.input_noise
            x_noisy = x + noise
        else:
            x_noisy = x

        # 2. Forward with Dropout
        distorted_view, attn_map = self.sia_attention(x_noisy, self.h)
        distorted_view = self.dropout(distorted_view)
        pred_x = self.output_proj(distorted_view)

        # 3. Physics
        diff = pred_x - target_x
        F = 0.5 * diff.pow(2).sum() + 0.5 * self.config.lambda_reg * self.h.pow(2).sum()

        if self.h.grad is not None: self.h.grad.zero_()
        grad_h_true = torch.autograd.grad(F, self.h, create_graph=False)[0]

        # 4. Imprint (Trinity)
        psi = grad_h_true.detach()
        intent = self.intent_gen(self.h)
        world_embed = distorted_view.mean(dim=[0, 1]).detach()
        self.trace.imprint(world_embed, psi, intent)

        # 5. Attribution
        trace_input = self.trace.T.detach()
        grad_h_est = self.adapter(trace_input)

        # 6. Meta-Learning
        loss_adapter = 0.5 * (grad_h_est - psi).pow(2).sum()
        self.opt_adapter.zero_grad()
        loss_adapter.backward()
        self.opt_adapter.step()

        # 7. Self Update with Saturation (The Stability Patch)
        alpha = self.config.reality_weight
        effective_grad = alpha * grad_h_true + (1 - alpha) * grad_h_est

        with torch.no_grad():
            # Soft Saturation (tanh)
            update = torch.tanh(effective_grad) * self.config.lr_self
            self.h -= update

            # Hard Limit (Clamp)
            self.h.data = torch.clamp(self.h.data, -self.config.max_h_val, self.config.max_h_val)

        # 8. Reconsolidation
        recon_rate = 0.05
        with torch.no_grad():
            current_intent = self.intent_gen(self.h)
            self.trace.reconsolidate(current_intent, intensity=recon_rate)

        return {
            "F": F.item(),
            "loss_adapter": loss_adapter.item(),
            "h_norm": self.h.norm().item(),
            "grad_true_norm": grad_h_true.norm().item()
        }


# ============================================================
# Validation Run
# ============================================================
def test_real_sia():
    # Stable ConfigとStable Systemを使用
    config = SIAConfig_Stable()
    model = SIASystem_Stable(config)

    # Simulate Data
    x = torch.randn(1, 10, 32)
    target = torch.randn(1, 10, 32)

    print(f"{'Step':<5} | {'Free Energy':<12} | {'Attr Error':<12} | {'Self Norm':<10} | {'Grad True':<10}")
    print("-" * 65)

    for t in range(20):
        if t == 8: target = torch.randn(1, 10, 32)  # Trauma Event

        metrics = model.step(x, target)
        print(
            f"{t:<5} | {metrics['F']:<12.4f} | {metrics['loss_adapter']:<12.4f} | {metrics['h_norm']:<10.4f} | {metrics['grad_true_norm']:<10.4f}")


if __name__ == "__main__":
    test_real_sia()