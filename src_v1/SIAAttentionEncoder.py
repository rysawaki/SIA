import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np


# ============================================================
# 1. Config & Components
# ============================================================

class SIAConfig_v3:
    h_dim = 32  # Selfの表現力アップ
    d_model = 64  # CNN特徴マップ深度
    spatial_size = 7  # 最終特徴マップサイズ

    # Loss Weights
    # 自己を殺すペナルティは全廃。
    # 代わりに「変化」を許容する係数。
    plasticity_weight = 0.1


class CoupledSIAAttention(nn.Module):
    """
    修正1 & 2: 世界と自己の結合マスク
    Mask = Sigmoid( Conv(x) + Linear(h) )
    世界の特徴(x)と自己のバイアス(h)の両方を使ってAttentionを決める。
    """

    def __init__(self, channels, h_dim):
        super().__init__()

        # World processing: "世界に何があるか"
        self.x_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        # Self processing: "何を見たいか"
        self.h_proj = nn.Linear(h_dim, channels)  # チャンネルごとの重み
        self.h_gate = nn.Linear(h_dim, 1)  # 空間全体のバイアス

    def forward(self, x, h):
        # x: [B, C, H, W]
        B, C, H, W = x.shape

        # 1. World Context (空間構造の抽出)
        # [B, 1, H, W]
        world_map = self.x_conv(x)

        # 2. Self Bias (自己の意図)
        # h_gate: [B, 1] -> [B, 1, 1, 1] (全体的な感度)
        self_bias = self.h_gate(h).view(B, 1, 1, 1)

        # 3. Coupled Mask Generation
        # 世界の構造 + 自己の感度
        # sigmoidにより 0.0 ~ 1.0 の範囲に収める（二値化暴走を防ぐ）
        mask_logits = world_map + self_bias
        mask = torch.sigmoid(mask_logits)

        # 4. Feature Modulation
        # hによるチャンネル強調 (Channel Attention) も混ぜる
        # h_chan: [B, C] -> [B, C, 1, 1]
        h_chan = torch.sigmoid(self.h_proj(h)).view(B, C, 1, 1)

        # 適用: 空間マスク(どこを見るか) * チャンネル重み(何を見るか)
        # x' = x * mask * channel_weight
        out = x * mask * h_chan

        return out, mask


class SelfRecurrentUnit(nn.Module):
    """
    修正3: Self-seeing-the-world
    h_new = GRU( World_Summary, h_old )
    自己はパラメータではなく、入力によって更新される「状態」である。
    """

    def __init__(self, input_dim, h_dim):
        super().__init__()
        # GRUCell: 入力と過去の隠れ状態から、新しい隠れ状態を作る
        self.rnn = nn.GRUCell(input_dim, h_dim)

    def forward(self, x_summary, h_prev):
        # x_summary: [B, input_dim] (CNNのGlobal Average Pooling)
        # h_prev: [B, h_dim]
        h_new = self.rnn(x_summary, h_prev)
        return h_new


# ============================================================
# 2. SIA Network v3.0 (The Living Agent)
# ============================================================

class SIA_MNIST_Net_v3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Perception (CNN)
        # BatchNormは必須 (勾配消失を防ぐ)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)

        # 2. Dynamic Self (Not Parameter, but State Manager)
        # hの初期状態生成器
        self.h0_gen = nn.Parameter(torch.randn(1, config.h_dim) * 0.1)

        # 自己更新ユニット
        self.self_rnn = SelfRecurrentUnit(64, config.h_dim)

        # 3. Coupled Interaction
        self.sia_attn = CoupledSIAAttention(64, config.h_dim)

        # 4. Classifier
        self.fc = nn.Linear(64 * 7 * 7, 10)

        # 5. Intent (Identity Observer)
        self.intent_gen = nn.Linear(config.h_dim, 10)  # SoftmaxはLoss計算時にかける

    def forward(self, x, h_prev=None):
        B = x.shape[0]

        # Initialize h if not provided (Start of thought process)
        if h_prev is None:
            h_prev = self.h0_gen.expand(B, -1)

        # --- Perception Stream ---
        # CNN Flow
        f1 = self.pool(F.relu(self.bn1(self.conv1(x))))
        f2 = self.pool(F.relu(self.bn2(self.conv2(f1))))  # [B, 64, 7, 7]

        # --- Coupled Interaction ---
        # 世界(f2)と自己(h_prev)が対話して注目領域を決める
        f2_modulated, mask = self.sia_attn(f2, h_prev)

        # --- Self Update (Reflection) ---
        # 歪められた世界を見て、自己を更新する
        # Global Average Pooling: [B, 64, 7, 7] -> [B, 64]
        world_summary = f2_modulated.mean(dim=[2, 3])

        # h_new = GRU( World, h_old )
        h_new = self.self_rnn(world_summary, h_prev)

        # --- Action (Classification) ---
        out_flat = f2_modulated.reshape(B, -1)
        logits = self.fc(out_flat)

        # Intent Observation
        intent_logits = self.intent_gen(h_new)

        return logits, h_new, intent_logits, mask


# ============================================================
# 3. Experiment: The Trauma Test
# ============================================================

def run_experiment_v3():
    config = SIAConfig_v3()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIA_MNIST_Net_v3(config).to(device)

    # Optimizer: 全部まとめて学習させる（共依存関係を作る）
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # Dataset
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # 学習を確実にするためデータ量を確保 (5000枚)
    subset_indices = torch.arange(5000)
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(f"{'Epoch':<5} | {'Phase':<8} | {'Acc':<6} | {'Loss':<8} | {'|h|':<6} | {'Δh':<6} | {'Mask%':<6}")
    print("-" * 75)

    history = {'acc': [], 'h_norm': []}

    for epoch in range(12):
        is_trauma = epoch >= 6
        phase = "TRAUMA" if is_trauma else "Normal"

        model.train()
        total = 0
        correct = 0
        epoch_loss = 0

        h_diff_sum = 0
        h_norm_sum = 0
        mask_mean_sum = 0

        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)

            # === Trauma: 90度回転 ===
            if is_trauma:
                imgs = torch.rot90(imgs, 1, [2, 3])

            optimizer.zero_grad()

            # Forward
            # hは内部で初期化・更新される
            outputs, h_new, intent_logits, mask = model(imgs)

            # --- Loss Calculation (修正4: 構造的損失) ---

            # A. Task Loss (生存必須)
            loss_cls = nn.CrossEntropyLoss()(outputs, labels)

            # B. Identity Distinctiveness (差異の強化)
            # Trauma時は「過去の自分(h0)と違うこと」を推奨する
            # Normal時は「安定」を推奨する
            # h_new と h_init の距離
            h_init = model.h0_gen.expand(h_new.shape[0], -1)
            dist = (h_new - h_init).norm(dim=1).mean()

            if is_trauma:
                # 危機: 変わらなければ死ぬ -> 変化を推奨 (Lossを下げるためにDistを上げる -> -Dist)
                loss_identity = -0.05 * dist
            else:
                # 平時: 安定せよ -> 変化を抑制 (Lossを下げるためにDistを下げる -> +Dist)
                loss_identity = 0.01 * dist

            loss = loss_cls + loss_identity

            loss.backward()
            optimizer.step()

            # Metrics
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

            h_norm_sum += h_new.norm(dim=1).mean().item()
            h_diff_sum += dist.item()
            mask_mean_sum += mask.mean().item()

        # Epoch Stats
        acc = 100 * correct / total
        avg_loss = epoch_loss / (i + 1)
        avg_h = h_norm_sum / (i + 1)
        avg_diff = h_diff_sum / (i + 1)
        avg_mask = mask_mean_sum / (i + 1)

        print(
            f"{epoch:<5} | {phase:<8} | {acc:<6.2f} | {avg_loss:<8.4f} | {avg_h:<6.3f} | {avg_diff:<6.3f} | {avg_mask:<6.3f}")


if __name__ == "__main__":
    run_experiment_v3()