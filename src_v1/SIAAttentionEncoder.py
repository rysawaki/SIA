import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


# ============================================================
# 1. Architecture v4: The Trace-Augmented Self
# ============================================================

class SIAConfig_v4:
    h_dim = 32
    d_model = 64
    trace_decay = 0.95  # 記憶の風化
    imprint_rate = 0.1  # 記憶の刻印率


class TraceMemory(nn.Module):
    """
    Trace: "生き様"の記録
    過去の自己変容(Δh)を、その時の感情価(Affect)と共に蓄積する
    """

    def __init__(self, h_dim, decay=0.95):
        super().__init__()
        # Traceはパラメータではなくバッファ(状態)
        self.register_buffer('memory', torch.zeros(1, h_dim))
        self.decay = decay

    def imprint(self, delta_h, affect):
        # delta_h: 自己の変化量 [B, h_dim]
        # affect: その変化の重要度(Lossの大きさなど) [B, 1]

        # 重要な変化ほど深く刻まれる
        # T_new = T_old * decay + Δh * affect

        # Batch平均をとって単一の「個体の記憶」として更新
        avg_delta = delta_h.mean(dim=0, keepdim=True)
        avg_affect = affect.mean().item()

        # 強い感情(affect)を伴う変化ほど強く残る
        imprint_signal = avg_delta * (1.0 + avg_affect)

        self.memory = self.memory * self.decay + imprint_signal * (1 - self.decay)

    def read(self):
        return self.memory


class CoupledSIAAttention_v4(nn.Module):
    # v3と同じ (World-Self Coupling)
    def __init__(self, channels, h_dim):
        super().__init__()
        self.x_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)
        self.h_proj = nn.Linear(h_dim, channels)
        self.h_gate = nn.Linear(h_dim, 1)

    def forward(self, x, h):
        B, C, H, W = x.shape
        world_map = self.x_conv(x)
        self_bias = self.h_gate(h).view(B, 1, 1, 1)
        mask = torch.sigmoid(world_map + self_bias)
        h_chan = torch.sigmoid(self.h_proj(h)).view(B, C, 1, 1)
        out = x * mask * h_chan
        return out, mask


class SelfRecurrentUnit_v4(nn.Module):
    """
    Self with Trace Bias
    h_new = GRU( World, h_prev + Trace )
    """

    def __init__(self, input_dim, h_dim):
        super().__init__()
        self.rnn = nn.GRUCell(input_dim, h_dim)

    def forward(self, x_summary, h_prev, trace):
        # 過去の経験(Trace)が、現在の思考(h_prev)にバイアスをかける
        # これが「直観」や「トラウマ反応」の源泉
        biased_h = h_prev + trace
        h_new = self.rnn(x_summary, biased_h)
        return h_new


class SIA_MNIST_Net_v4(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Perception
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        # Self & Trace
        self.h0_gen = nn.Parameter(torch.randn(1, config.h_dim) * 0.1)
        self.trace = TraceMemory(config.h_dim, config.trace_decay)
        self.self_rnn = SelfRecurrentUnit_v4(64, config.h_dim)

        # Interaction
        self.sia_attn = CoupledSIAAttention_v4(64, config.h_dim)

        # Output
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.intent_gen = nn.Linear(config.h_dim, 10)

    def forward(self, x, h_prev=None, update_trace=False, affect_val=0.0):
        B = x.shape[0]

        if h_prev is None:
            h_prev = self.h0_gen.expand(B, -1)

        # 1. Perception
        f1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        f2 = self.pool(F.relu(self.bn2(self.conv2(f1))))

        # 2. Trace Reading
        # 記憶を読み出し、現在の自己に重ねる
        current_trace = self.trace.read().expand(B, -1)

        # 3. Interaction
        f2_modulated, mask = self.sia_attn(f2, h_prev)

        # 4. Self Update with Trace
        world_summary = f2_modulated.mean(dim=[2, 3])
        h_new = self.self_rnn(world_summary, h_prev, current_trace)

        # 5. Trace Imprint (Learning from Experience)
        if update_trace:
            # 変化量 Δh
            delta_h = h_new - h_prev
            # 感情価 affect_val (Loss等) を重みとして記憶を刻む
            # detach()しないとTrace自体に勾配が流れてしまう(今回は状態更新として扱う)
            self.trace.imprint(delta_h.detach(), torch.tensor(affect_val))

        # 6. Action
        out_flat = f2_modulated.reshape(B, -1)
        logits = self.fc(out_flat)
        intent_logits = self.intent_gen(h_new)

        return logits, h_new, intent_logits, mask


# ============================================================
# 2. Experiment Runner
# ============================================================

def run_experiment_v4():
    config = SIAConfig_v4()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIA_MNIST_Net_v4(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Dataset (5000 samples)
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    subset_indices = torch.arange(5000)
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"{'Epoch':<5} | {'Phase':<8} | {'Acc':<6} | {'Loss':<8} | {'|h|':<6} | {'|T|':<6} | {'Mask%':<6}")
    print("-" * 75)

    for epoch in range(15):
        # Trauma Phase: Epoch 6-10 (Rotation 90)
        # Recovery Phase: Epoch 11-14 (Back to Normal) -> ここで記憶の効果を見る
        if epoch < 6:
            phase = "Normal"
            angle = 0
        elif epoch < 11:
            phase = "TRAUMA"
            angle = 1  # 90 deg
        else:
            phase = "Recov"  # Back to Normal
            angle = 0

        model.train()
        total = 0
        correct = 0
        epoch_loss = 0

        h_norm_sum = 0
        trace_norm_sum = 0
        mask_mean_sum = 0

        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            if angle > 0:
                imgs = torch.rot90(imgs, angle, [2, 3])

            optimizer.zero_grad()

            # Forward pass needs to handle state sequence if using BPTT,
            # but here we use single step approximation for simplicity in demo.
            # However, for Trace to meaningful, we need h_prev.
            # In this simple training loop, h is re-initialized every batch,
            # BUT Trace is persistent across batches!

            # 1st pass to get loss (Affect)
            logits, h_new, _, mask = model(imgs, update_trace=False)

            loss_cls = nn.CrossEntropyLoss()(logits, labels)

            # Affect calculation: High Loss = High Surprise/Pain
            affect = loss_cls.item()

            # 2nd pass (Imprint) or simply imprint using computed h_new
            # ここでは簡易的に、計算したh_newを使ってTraceを更新
            # h_prevは h0_gen から生成されたものと仮定
            h_prev = model.h0_gen.expand(imgs.shape[0], -1)
            model.trace.imprint((h_new - h_prev).detach(), torch.tensor(affect))

            # Identity Loss (Adaptive)
            if phase == "TRAUMA":
                loss_identity = -0.05 * (h_new - h_prev).norm(dim=1).mean()
            else:
                loss_identity = 0.01 * (h_new - h_prev).norm(dim=1).mean()

            loss = loss_cls + loss_identity

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

            h_norm_sum += h_new.norm(dim=1).mean().item()
            trace_norm_sum += model.trace.read().norm().item()
            mask_mean_sum += mask.mean().item()

        # Stats
        acc = 100 * correct / total
        avg_loss = epoch_loss / (i + 1)
        avg_h = h_norm_sum / (i + 1)
        avg_trace = trace_norm_sum / (i + 1)
        avg_mask = mask_mean_sum / (i + 1)

        print(
            f"{epoch:<5} | {phase:<8} | {acc:<6.2f} | {avg_loss:<8.4f} | {avg_h:<6.3f} | {avg_trace:<6.3f} | {avg_mask:<6.3f}")


if __name__ == "__main__":
    run_experiment_v4()