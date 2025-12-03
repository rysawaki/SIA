import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from itertools import product
import torch

# ----------------------------------------------------------
# Mini-SIA dynamics (あなたの元コードに合わせて簡略化)
# ----------------------------------------------------------
def run_episode(alpha, beta, gamma, episode_len=80):
    h = 0.0
    trace = 0.0
    instability = 0.0

    for t in range(episode_len):
        # affect
        A = beta * abs(h)

        # future drive
        F = gamma * (1.0 - abs(h))

        # past drag
        P = alpha * trace

        # update h
        dh = 0.1 * (A + F - P)
        h = h + dh
        h = np.clip(h, -5.0, 5.0)

        # trace update
        trace = 0.9 * trace + 0.1 * h

        # accumulate instability
        instability += abs(h) + abs(trace)

    return instability


def get_I_mean(alpha, beta, gamma, n_episodes=10):
    values = [run_episode(alpha, beta, gamma) for _ in range(n_episodes)]
    return np.mean(values), np.std(values)


# ----------------------------------------------------------
# Generate dataset
# ----------------------------------------------------------
alpha_list = [0.0, 0.5, 1.0, 1.5]
beta_list  = [0.0, 0.5, 1.0, 1.5]
gamma_list = [0.0, 0.5, 1.0, 1.5]

data = []
labels = []

I_crit = 4.0  # 崩壊とみなす閾値。必要なら調整可

print("\n=== Running 10-episode averaged instability ===")

for a, b, g in product(alpha_list, beta_list, gamma_list):
    I_mean, I_std = get_I_mean(a, b, g, n_episodes=10)
    print(f"(α={a}, β={b}, γ={g}) → I_mean={I_mean:.3f} (std={I_std:.3f})")

    # 入力特徴
    data.append([a, b, g])
    # 崩壊ラベル
    labels.append(1 if I_mean >= I_crit else 0)

data = np.array(data)
labels = np.array(labels)

# ----------------------------------------------------------
# Logistic regression with polynomial features
# f(a,b,g) = w0 + w1 a + w2 b + w3 g + ... + w9 g^2
# ----------------------------------------------------------
def poly_features(X):
    out = []
    for a, b, g in X:
        out.append([
            1,
            a, b, g,
            a*a, a*b, a*g,
            b*b, b*g,
            g*g
        ])
    return np.array(out)

X = poly_features(data)
clf = LogisticRegression(max_iter=5000)
clf.fit(X, labels)

coef = clf.coef_[0]
intercept = clf.intercept_[0]

print("\n=== Logistic regression polynomial model ===")
names = [
    "1", "alpha", "beta", "gamma",
    "alpha^2", "alpha*beta", "alpha*gamma",
    "beta^2", "beta*gamma", "gamma^2"
]

for n, c in zip(names, coef):
    print(f"{n}: {c:.4f}")

print(f"intercept: {intercept:.4f}")

# ----------------------------------------------------------
# Boundary function f(a,b,g)=0 を返す関数
# ----------------------------------------------------------
def f_boundary(a, b, g):
    """
    a: float（スカラー）
    b, g: numpy 2D meshgrid（同じ shape）
    出力: Z（同じ shape）
    """
    # スカラーを b,g と同じ shape に broadcast
    A = a * np.ones_like(b)

    # 多項式項（すべて配列として計算される）
    term1 = coef[0] * 1
    term2 = coef[1] * A
    term3 = coef[2] * b
    term4 = coef[3] * g
    term5 = coef[4] * (A * A)
    term6 = coef[5] * (A * b)
    term7 = coef[6] * (A * g)
    term8 = coef[7] * (b * b)
    term9 = coef[8] * (b * g)
    term10 = coef[9] * (g * g)

    return term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10 + intercept



# ----------------------------------------------------------
# Plot slices for α = fixed
# ----------------------------------------------------------
def plot_slices():
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for idx, a in enumerate(alpha_list):
        ax = axes[idx]

        B, G = np.meshgrid(
            np.linspace(0, 1.5, 100),
            np.linspace(0, 1.5, 100)
        )
        Z = f_boundary(a, B, G)

        cs = ax.contour(B, G, Z, levels=[0.0], colors="red", linewidths=2)

        ax.set_title(f"α={a}")
        ax.set_xlabel("β")
        ax.set_ylabel("γ")
        ax.imshow(Z, extent=[0,1.5,0,1.5], origin="lower",
                  cmap="coolwarm", alpha=0.5)

    plt.suptitle("Boundary surface slices: f(α,β,γ)=0")
    plt.tight_layout()
    plt.show()


plot_slices()

print("\n=== Finished ===")
