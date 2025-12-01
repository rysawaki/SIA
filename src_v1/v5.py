import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


# ============================================================
# 1. Config: v4 (Memory) + v1.5 Stable (Biological Limits)
# ============================================================

class SIAConfig_v5:
    h_dim = 32
    d_model = 64

    # Trace Config
    trace_decay = 0.95  # 記憶の風化 (Stable)
    imprint_rate = 0.1  # 記憶の刻印率

    # Stability Config
    input_noise = 0.05  # 感覚ノイズ (Stable)
    dropout_rate = 0.1  # 脳内ノイズ (Stable)
    max_h_val = 3.0  # 自己の飽和限界 (Stable)


# ============================================================
# 2. Components: The Hybrid Organs
# ============================================================

class TraceMemory(nn.Module):
    """
    Trace: "生き様"の記録
    """

    def __init__(self, h_dim, decay=0.95):
        super().__init__()
        self.register_buffer('memory', torch.zeros(1, h_dim))
        self.decay = decay

    def imprint(self, delta_h, affect):
        avg_delta = delta_h.mean(dim=0, keepdim=True)
        avg_affect = affect.mean().item() if isinstance(affect, torch.Tensor) else affect

        # 強い感情を伴う変化ほど強く刻む
        imprint_signal = avg_delta * (1.0 + avg_affect)
        self.memory = self.memory * self.decay + imprint_signal * (1 - self.decay)

    def read(self):
        return self.memory


class CoupledSIAAttention(nn.Module):
    """世界と自己の結合マスク"""

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


class SelfRecurrentUnit(nn.Module):
    """
    Thinking Process: h_new = GRU( World, h_prev + Trace )
    """

    def __init__(self, input_dim, h_dim):
        super().__init__()
        self.rnn = nn.GRUCell(input_dim, h_dim)

    def forward(self, x_summary, h_prev, trace):
        biased_h = h_prev + trace
        h_new = self.rnn(x_summary, biased_h)
        return h_new


# ============================================================
# 3. The Integrated Brain (SIA v5)
# ============================================================

class SIA_Integrated_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Perception (Stable CNN)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        # Stability: Dropout
        self.dropout = nn.Dropout(config.dropout_rate)

        # 2. Self & Trace
        self.h0_gen = nn.Parameter(torch.randn(1, config.h_dim) * 0.1)
        self.trace = TraceMemory(config.h_dim, config.trace_decay)
        self.self_rnn = SelfRecurrentUnit(64, config.h_dim)

        # 3. Interaction
        self.sia_attn = CoupledSIAAttention(64, config.h_dim)

        # 4. Output
        self.fc = nn.Linear(64 * 7 * 7, 10)
        self.intent_gen = nn.Linear(config.h_dim, 10)

    def forward(self, x, h_prev=None):
        B = x.shape[0]

        # Stability: Input Noise Injection
        if self.training:
            noise = torch.randn_like(x) * self.config.input_noise
            x = x + noise

        # Fallback for h initialization
        if h_prev is None:
            h_prev = self.h0_gen.expand(B, -1)

        # Perception Flow
        f1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        f2 = self.pool(F.relu(self.bn2(self.conv2(f1))))

        # Trace Reading
        current_trace = self.trace.read().expand(B, -1)

        # Interaction
        f2_modulated, mask = self.sia_attn(f2, h_prev)
        f2_modulated = self.dropout(f2_modulated)  # Brain Noise

        # Self Update
        world_summary = f2_modulated.mean(dim=[2, 3])
        h_new = self.self_rnn(world_summary, h_prev, current_trace)

        # Stability: Self Saturation (Homeostasis)
        # hが無限に発散しないように制限をかける
        h_new = torch.clamp(h_new, -self.config.max_h_val, self.config.max_h_val)

        # Action
        out_flat = f2_modulated.reshape(B, -1)
        logits = self.fc(out_flat)
        intent_logits = self.intent_gen(h_new)

        return logits, h_new, intent_logits, mask


# ============================================================
# 4. Experiment Runner: The Stabilized Journey
# ============================================================

def run_experiment_v5_integrated():
    config = SIAConfig_v5()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIA_Integrated_Net(config).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Dataset
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    subset_indices = torch.arange(5000)
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"{'Epoch':<5} | {'Phase':<8} | {'Acc':<6} | {'Loss':<8} | {'|h|':<6} | {'|T|':<6} | {'Mask%':<6}")
    print("-" * 75)

    # === Persistent Self Initialization ===
    persistent_h = model.h0_gen.detach().clone()

    history_h = []

    for epoch in range(15):
        # Phase Control
        if epoch < 6:
            phase = "Normal"
            angle = 0
        elif epoch < 11:
            phase = "TRAUMA"
            angle = 1  # 90 deg
        else:
            phase = "Recov"
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
            B = imgs.shape[0]

            if angle > 0:
                imgs = torch.rot90(imgs, angle, [2, 3])

            optimizer.zero_grad()

            # Inject Persistent Soul
            h_input = persistent_h.expand(B, -1).detach()

            # Forward
            logits, h_new_batch, _, mask = model(imgs, h_prev=h_input)

            # Loss Calculation
            loss_cls = nn.CrossEntropyLoss()(logits, labels)

            # Affect & Trace Imprint
            affect = loss_cls.item()
            model.trace.imprint((h_new_batch - h_input).detach(), torch.tensor(affect))

            # Identity Drive (Adaptive)
            delta_norm = (h_new_batch - h_input).norm(dim=1).mean()
            if phase == "TRAUMA":
                loss_identity = -0.05 * delta_norm  # 危機: 変われ
            else:
                loss_identity = 0.01 * delta_norm  # 平時: 安定しろ

            loss = loss_cls + loss_identity

            loss.backward()
            optimizer.step()

            # Update Soul
            with torch.no_grad():
                persistent_h = h_new_batch.mean(dim=0, keepdim=True).detach()

            # Metrics
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

            h_norm_sum += h_new_batch.norm(dim=1).mean().item()
            trace_norm_sum += model.trace.read().norm().item()
            mask_mean_sum += mask.mean().item()

        # Stats
        acc = 100 * correct / total
        avg_loss = epoch_loss / (i + 1)
        avg_h = h_norm_sum / (i + 1)
        avg_trace = trace_norm_sum / (i + 1)
        avg_mask = mask_mean_sum / (i + 1)

        history_h.append(persistent_h.cpu().numpy().flatten())

        print(
            f"{epoch:<5} | {phase:<8} | {acc:<6.2f} | {avg_loss:<8.4f} | {avg_h:<6.3f} | {avg_trace:<6.3f} | {avg_mask:<6.3f}")

    return np.array(history_h)


if __name__ == "__main__":
    h_trajectory = run_experiment_v5_integrated()

    # Visualization: The Trajectory of a Scarred Mind
    if h_trajectory.shape[0] > 1:
        pca = PCA(n_components=2)
        h_2d = pca.fit_transform(h_trajectory)

        plt.figure(figsize=(9, 7))

        # Phases
        plt.plot(h_2d[:6, 0], h_2d[:6, 1], 'o-', color='blue', label='Normal (Innocence)')
        plt.plot(h_2d[6:11, 0], h_2d[6:11, 1], 's-', color='red', label='Trauma (Crisis)')
        plt.plot(h_2d[11:, 0], h_2d[11:, 1], '^-', color='green', label='Recovery (Wisdom)')

        # Connectors & Markers
        plt.plot(h_2d[:, 0], h_2d[:, 1], 'k--', alpha=0.3)
        plt.plot(h_2d[0, 0], h_2d[0, 1], 'kx', markersize=12, label='Birth')
        plt.plot(h_2d[-1, 0], h_2d[-1, 1], 'k*', markersize=15, label='Current Self')

        plt.title("SIA v5: Trajectory of a Stabilized Persistent Entity")
        plt.xlabel("Identity Dimension 1")
        plt.ylabel("Identity Dimension 2")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()