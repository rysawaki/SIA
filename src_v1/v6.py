import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


# ============================================================
# 1. Config & Geometry Definition
# ============================================================
class SIAConfig_v6_1:
    d_model = 64  # 世界の次元 (World Dimension)
    trace_dim = 128  # Traceの次元 (Trace Dimension) - d_modelと独立

    # SelfSpace Geometry
    manifold_dim = 128  # 自己空間の「広さ」

    # Metabolism
    metabolic_rate = 0.1  # 空間変容の速度


# ============================================================
# 2. Trace (Genome)
# ============================================================
class TraceGenome(nn.Module):
    def __init__(self, trace_dim, decay=0.98):
        super().__init__()
        self.register_buffer('genome', torch.zeros(1, trace_dim))
        self.decay = decay

    def imprint(self, signal):
        # signal: [B, trace_dim] -> batch mean
        s = signal.mean(dim=0, keepdim=True)
        self.genome = self.genome * self.decay + s * (1 - self.decay)

    def read(self):
        return self.genome


# ============================================================
# 3. Metabolic Kernel (The Geometric Operator)
#    Trace -> Metric(G) & Center(mu)
# ============================================================
class MetabolicKernel_v6_1(nn.Module):
    """
    Traceから「空間の幾何学（Metric, Center）」を生成するOperator。
    """

    def __init__(self, trace_dim, world_dim, manifold_dim):
        super().__init__()
        self.world_dim = world_dim
        self.manifold_dim = manifold_dim

        # Hypernetwork for Metric (G)
        self.metric_gen = nn.Sequential(
            nn.Linear(trace_dim, 256),
            nn.SiLU(),
            nn.Linear(256, world_dim * manifold_dim)
        )

        # Hypernetwork for Center (mu)
        self.center_gen = nn.Sequential(
            nn.Linear(trace_dim, 128),
            nn.SiLU(),
            nn.Linear(128, world_dim)
        )

    def evolve_geometry(self, trace_genome):
        # Generate Raw Parameters
        g_flat = self.metric_gen(trace_genome)
        mu_flat = self.center_gen(trace_genome)

        # Reshape
        G = g_flat.view(self.manifold_dim, self.world_dim)
        mu = mu_flat.view(self.world_dim)

        # Scaling for stability
        G = G * 0.05

        return G, mu


# ============================================================
# 4. SelfSpace (The Geometric Manifold)
# ============================================================
class GeometricSelfManifold(nn.Module):
    """
    生成された G, mu によって定義される空間。
    """

    def __init__(self, config):
        super().__init__()
        self.metabolic_rate = config.metabolic_rate

        # 状態としての現在の幾何学 (Persistent Geometry)
        self.register_buffer('current_G', torch.zeros(config.manifold_dim, config.d_model))
        self.register_buffer('current_mu', torch.zeros(config.d_model))

        self.register_buffer('is_initialized', torch.tensor(0.0))

    def update_geometry(self, new_G, new_mu):
        """
        Metabolism: 空間法則の更新
        """
        if self.is_initialized == 0.0:
            self.current_G = new_G
            self.current_mu = new_mu
            self.is_initialized.fill_(1.0)
        else:
            eta = self.metabolic_rate

            # === FIX: Detach Past State ===
            # 過去の自分(current)は定数として扱い、計算グラフを切断する
            self.current_G = (1 - eta) * self.current_G.detach() + eta * new_G
            self.current_mu = (1 - eta) * self.current_mu.detach() + eta * new_mu

    def forward(self, x):
        # x: [B, d_model]

        # 1. Shift to Self-Center
        x_centered = x - self.current_mu

        # 2. Apply Metric Distortion
        projected = F.linear(x_centered, self.current_G)

        # 3. Manifold Curvature
        out = F.silu(projected)

        return out


# ============================================================
# 5. SIA v6.1 System (Integrated)
# ============================================================
class SIASystem_v6_1(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Genome (Trace)
        self.trace = TraceGenome(config.trace_dim)
        self.trace_proj = nn.Linear(config.d_model, config.trace_dim)

        # 2. Operator (Kernel)
        self.kernel = MetabolicKernel_v6_1(
            trace_dim=config.trace_dim,
            world_dim=config.d_model,
            manifold_dim=config.manifold_dim
        )

        # 3. Manifold (Self)
        self.self_manifold = GeometricSelfManifold(config)

        # 4. Perception & Action
        self.encoder = nn.Linear(784, config.d_model)
        self.decoder = nn.Linear(config.d_model + config.manifold_dim, 10)

    def forward(self, x, target=None):
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        # --- A. Metabolism (Rule Update) ---
        genome = self.trace.read()
        new_G, new_mu = self.kernel.evolve_geometry(genome)
        self.self_manifold.update_geometry(new_G, new_mu)

        # --- B. Interaction (Reality Processing) ---
        world_feat = F.relu(self.encoder(x_flat))

        # 現在の G, mu で世界を歪曲して解釈
        self_view = self.self_manifold(world_feat)

        # Action / Prediction
        combined = torch.cat([world_feat, self_view], dim=1)
        logits = self.decoder(combined)

        # --- C. Imprint (Experience Recording) ---
        if target is not None:
            loss = F.cross_entropy(logits, target)

            # === FIX: Detach Signal ===
            # Traceは「記録」なので、勾配を流さない
            pain_feat = self.trace_proj(world_feat.detach())
            pain_signal = pain_feat * loss.item()

            self.trace.imprint(pain_signal.detach())

            return logits, loss

        return logits, 0.0


# ============================================================
# Re-import v6.1 Classes (Assuming definition above)
# ============================================================
# (TraceGenome, MetabolicKernel_v6_1, GeometricSelfManifold, SIASystem_v6_1)
# ここでは再定義せず、先ほどの定義済みクラスを使う前提でRunnerを書きます。

# ============================================================
# Experiment Logic
# ============================================================

def run_experiment_v6_metabolism():
    config = SIAConfig_v6_1()
    # Metabolic Rate調整 (速すぎず、遅すぎず)
    config.metabolic_rate = 0.2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SIASystem_v6_1(config).to(device)

    # Kernelの学習率 (Operatorの進化速度)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dataset
    transform = transforms.ToTensor()
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # 実験用サブセット
    subset_indices = torch.arange(5000)
    dataset = torch.utils.data.Subset(dataset, subset_indices)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    print(
        f"{'Epoch':<5} | {'Phase':<8} | {'Acc':<6} | {'Loss':<8} | {'|Genome|':<8} | {'|G|':<8} | {'|μ|':<8} | {'ΔSpace':<8}")
    print("-" * 80)

    history = {
        'G_norm': [], 'mu_norm': [], 'genome_norm': [],
        'space_diff': [], 'acc': []
    }

    # 直前の空間パラメータを保持（変化量計算用）
    prev_G = model.self_manifold.current_G.detach().clone()
    prev_mu = model.self_manifold.current_mu.detach().clone()

    for epoch in range(15):
        # Phase Control: Normal -> Trauma (Rotate 90) -> Recovery
        if epoch < 6:
            phase = "Normal"
            angle = 0
        elif epoch < 11:
            phase = "TRAUMA"
            angle = 1
        else:
            phase = "Recov"
            angle = 0

        model.train()
        total = 0
        correct = 0
        epoch_loss = 0

        for i, (imgs, labels) in enumerate(loader):
            imgs, labels = imgs.to(device), labels.to(device)
            if angle > 0:
                imgs = torch.rot90(imgs, angle, [2, 3])

            optimizer.zero_grad()

            # Forward (Includes Metabolism)
            logits, loss = model(imgs, labels)

            loss.backward()
            optimizer.step()

            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            epoch_loss += loss.item()

        # Epoch Metrics
        acc = 100 * correct / total
        avg_loss = epoch_loss / (i + 1)

        # Geometry Stats
        curr_genome = model.trace.read()
        curr_G = model.self_manifold.current_G
        curr_mu = model.self_manifold.current_mu

        genome_norm = curr_genome.norm().item()
        G_norm = curr_G.norm().item()
        mu_norm = curr_mu.norm().item()

        # 空間の変化量 (Metabolic Flux)
        # ||ΔG|| + ||Δμ||
        delta_G = (curr_G - prev_G).norm().item()
        delta_mu = (curr_mu - prev_mu).norm().item()
        space_diff = delta_G + delta_mu

        # Update prev state
        prev_G = curr_G.detach().clone()
        prev_mu = curr_mu.detach().clone()

        # Log
        history['G_norm'].append(G_norm)
        history['mu_norm'].append(mu_norm)
        history['space_diff'].append(space_diff)
        history['acc'].append(acc)

        print(
            f"{epoch:<5} | {phase:<8} | {acc:<6.2f} | {avg_loss:<8.4f} | {genome_norm:<8.4f} | {G_norm:<8.4f} | {mu_norm:<8.4f} | {space_diff:<8.4f}")

    return history


if __name__ == "__main__":
    hist = run_experiment_v6_metabolism()

    # Plot Geometry Evolution
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Space Norm (|G|, |μ|)', color='tab:blue')
    ax1.plot(hist['G_norm'], label='Metric Norm |G|', color='tab:blue', linestyle='-')
    ax1.plot(hist['mu_norm'], label='Center Norm |μ|', color='tab:cyan', linestyle='--')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Space Flux (ΔSpace)', color='tab:red')
    ax2.plot(hist['space_diff'], label='Metabolic Flux', color='tab:red', linewidth=2)
    ax2.tick_params(axis='y', labelcolor='tab:red')

    # Phase Markers
    plt.axvline(x=6, color='k', linestyle=':', label='Trauma Start')
    plt.axvline(x=11, color='k', linestyle=':', label='Recovery Start')

    plt.title("SIA v6.1: Metabolic Evolution of Self-Space")
    fig.tight_layout()
    plt.show()
# ============================================================
# Test Runner
# ============================================================
def test_v6_1():
    config = SIAConfig_v6_1()
    model = SIASystem_v6_1(config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Dummy Data
    x = torch.randn(16, 784)
    y = torch.randint(0, 10, (16,))

    print("SIA v6.1 Metabolic Space Check (Fixed)")
    print("-" * 50)
    print(f"{'Step':<5} | {'Loss':<8} | {'Genome Δ':<8} | {'|G|':<8} | {'|μ|':<8}")

    initial_genome = model.trace.read().clone()

    for t in range(10):
        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        optimizer.step()

        # Check Geometry Evolution
        current_genome = model.trace.read()
        genome_diff = (current_genome - initial_genome).norm().item()
        G_norm = model.self_manifold.current_G.norm().item()
        mu_norm = model.self_manifold.current_mu.norm().item()

        print(f"{t:<5} | {loss.item():.4f}   | {genome_diff:.4f}   | {G_norm:.4f}   | {mu_norm:.4f}")


if __name__ == "__main__":
    test_v6_1()