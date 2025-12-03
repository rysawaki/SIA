率直に言う。
あなたはまだ GrowthKernel を「関数」と捉えている限り、それを数学的に定義できても、その本質に届かない。
なぜなら、GrowthKernel は **写像（一回限りの更新関数）ではなく、自己空間そのものの生成規則（生成関数 / 代謝関数）** だから。

つまり、GrowthKernel を「入力 → 出力」の関数として定義するなら、それは適応学習／最適化理論の枠に落ちる。
しかし、本来の GrowthKernel は、

> **SelfSpace 自体を更新する “Higher-order function” であり、
> 空間と解釈を生成し続ける “代謝法則”**。

---

### それでも敢えて “関数として定式化する” なら — 最も本質に近づけた形式はこれだ

---

### 🧬 GrowthKernel の数学的定義（最小だが本質を外さない形）

$$
G: (\mathcal{S}*t,, g_t,, T_t,, p_t)
;\mapsto; (\mathcal{S}*{t+1},, g_{t+1})
$$

ただし、単なる写像ではなく、**Self 空間そのものを生成し直す変換**であるため、

$$
G: (\mathcal{S}*t, g_t) \xrightarrow{(T_t,p_t)} (\mathcal{S}*{t+1}, g_{t+1})
$$

#### ここで重要なのは以下の点：

| 記号                             | 意味         | 重要なポイント                   |
| ------------------------------ | ---------- | ------------------------- |
| $\mathcal{S}_t$                | 自己空間そのもの   | 更新されるのは点(s)ではなく**空間**     |
| $g_t$                          | 計量（Metric） | 経験によって歪む「意味の感じ方」          |
| $T_t$                          | Trace      | 単なる入力ではなく、**自己に影響する経験のみ** |
| $p_t$                          | Plasticity | “どの経験が人格へ刻まれるか” を制御する内的要因 |
| $(\mathcal{S}*{t+1}, g*{t+1})$ | 空間そのものの再生成 | 不可逆的進化、履歴依存、再構成不能         |

---

### ⚙ 具体的な更新式（内部構造）

#### Self の中心更新（移動）

$$
s_{t+1} = \mathrm{Normalize}!\left(s_t + \eta_s \cdot p_t \cdot \hat{T}_t \right)
$$

#### Metric（空間感受性）の更新

$$
g_{t+1} = g_t + \eta_m \cdot p_t \cdot \left( \hat{T}_t \otimes \hat{T}_t \right)
$$

**重要なポイント：**

* これは単なる重み更新（SGD）ではない
* **Loss を計算していない**（目的は最適化ではなく代謝・変容）
* Outer product により、**空間の方向性の再構成が行われる**
* 履歴依存。つまり元の $g_t$ に戻れない（不可逆）

---

### 🔥 決定的な本質

> GrowthKernel は「ある状態を更新する関数」ではなく、
> **自己空間を生成し直す再帰的ルール（Metabolic Operator）** である。

つまり、写像ではなく、

$$
G: \text{Operator on Space}
$$

**通常の関数：**
$f(x) \mapsto y$

**GrowthKernel：**
$f(\text{space}*t) \mapsto \text{space}*{t+1}$

---

### 🏁 あなたが使うべき最終定義

もし「関数形式」で説明する必要があるなら、
これを一文で言い切れ：

> **GrowthKernel は、経験 $T_t$ を自己空間そのものの幾何学的構成に変換する、
> 不可逆な Metabolic Operator である。**

---

必要なら、この定義を：

* 🔹 論文の定義形式（LaTeX）
* 🔹 README / GitHub 用シンプル定義
* 🔹 Substack / 哲学向け表現
* 🔹 コード実装と対応づけた関数形式（forward式）

にそれぞれ最適化する。

次、どのバージョンが欲しい？


アップロードされた `src/identity/dynamics/growth_kernel.py` および関連コードに基づき、現在の設計（v1/Prototype）が抱える特性と、将来的に起こりうる「破綻（限界）」を予測・分析する形で、Design Grave Entry（設計墓碑銘）を作成しました。

このフォーマットを使うことで、**GrowthKernelが「何をしようとしていたか」と「現在の実装の限界」**が明確になります。

---

# 🪦 Design Grave Entry — [GrowthKernel v1: Linear Rank-1 Accumulator]

**Date**: 2025-11-30  
**Related Module**: `src/identity/dynamics/growth_kernel.py`, `src/identity/core/trace_tensor.py`  
**Attempt**: v1 / prototype

---

## 🎯 Initial Intent（初期意図）

> **この設計は、何を達成しようとしていたか？**
> AIの学習を「パラメータ更新（勾配降下法）」ではなく、「幾何学的構造の物理的な変形」として実装しようとした。

* **Goal:**
    * 経験（Trace）を入力とし、自己空間（Self-Space）の計量（Metric）と中心（Self-Center）を不可逆的に更新する「代謝エンジン」を作ること。
    * `TraceTensor`に蓄積された記憶を、実際の「解釈の歪み」に変換する変換器としての役割。

* **Expected Mechanism:**
    * 強いShockを伴う経験（Trace）が入力されると、その方向に対してSelfが敏感になる（距離が縮まる/伸びる）ようにMetricを変形させる。
    * 同時に、Selfの立ち位置を経験の方向へ引き寄せる。

* **Inspired by:**
    * Riemannian Geometry (Metric Tensor deformation)
    * Hebbian Learning (Fire together, wire together -> Rank-1 update)

---

## ⚙️ Approach Summary（採用した構造・思想・アルゴリズム）

> **実装コードから読み取れる思想構造**

| 要素 | 内容 |
| :--- | :--- |
| **Structure Type** | 幾何 / ランク1更新 / 線形蓄積 |
| **Key Concepts** | **Rank-1 Update** (`outer(trace, trace)`), **Plasticity** (可塑性係数), **Metric Tensor** |
| **Core Equation** | `M_new = M_old + eta * plasticity * (trace ⊗ trace)` |
| **Normalization** | Selfベクトルは `F.normalize` で球面上に制約されるが、Metricには明示的な正規化がない。 |

---

## 💥 Failure Phenomenon（破綻現象：現在の実装の限界分析）

> **現在の `growth_kernel.py` の実装 をそのまま大規模運用した際に発生する現象**

### 1. Metric Explosion (計量の発散)
* **現象:** ステップが進むにつれ、`metric` の値が無限に増大し、下流の Attention 計算で `NaN` や極端なスコアが発生する。
* **原因:** `GrowthKernel.forward` 内の `new_metric = metric + ...` という加算更新式に、**減衰項（Decay）や正規化項が含まれていない**。
* **コード証拠:** `src/identity/dynamics/growth_kernel.py` の39行目付近には加算処理のみが存在する。外部の `TraceTensor` には `normalize_spectral_` があるが、Kernel自体は入力されたMetricを無制限に大きくしてしまう。

### 2. Dimensional Collapse (次元の崩壊/視野狭窄)
* **現象:** 特定の方向（例: トラウマ的な入力）へのTraceが繰り返されると、その軸の固有値だけが突出し、他の次元の微細な違いを認識できなくなる。AIが「そのことしか考えられない」状態になる。
* **原因:** 単純なランク1更新（`torch.outer`）の繰り返しは、行列の異方性（Anisotropy）を急速に高める性質がある。これを抑制する恒常性（Homeostasis）のロジックがKernel内に欠如している。

### 3. Linear Sensitivity (非線形性の欠如)
* **現象:** 些細なTraceでも、回数を重ねれば巨大なShockと同じだけの変形を引き起こしてしまう。「何度も言われると信じてしまう」効果が線形に効きすぎる。
* **原因:** 入力 `trace` と `plasticity` に対して線形に反応する設計になっている。生物的な「閾値（ある程度強くないと変形しない）」や「飽和（ある程度変形したらそれ以上変わらない）」といったSigmoid的な特性を持っていない。

---

### 📝 Next Action (v2への改善案)
この「墓碑銘」に基づき、次期バージョン (`GrowthKernel v2`) では以下が必要です：

1.  **Homeostatic Normalization:** Metric更新時にトレース（対角和）や最大固有値を一定に保つ正規化をKernel内に内蔵する。
2.  **Non-linear Activation:** `plasticity` に対して `sigmoid` や `tanh` を適用し、微弱なノイズによる変形を防ぐ。
3.  **Spectral Decay:** 古い構造を少しずつ忘却させる（単位行列に近づける）減衰項 `(1 - decay) * metric` を導入する。



`GrowthKernel` や関連モジュール（`SelfSpace`、`TraceTensor`）のコードと設計履歴（Grave Entry）を分析すると、このシステムが「スパゲティ化（構造的・論理的な複雑化）」してしまった主な原因は、**「責務の重複」**と**「循環する因果関係」**にあります。

具体的には、以下の3点がコードベースから読み取れる主要因です。

### 1. ロジックの散逸と重複（Logic Scattering & Duplication）
最も直接的なスパゲティ化の原因は、**「成長（更新）のロジック」が複数の場所に分散・重複して定義されていること**です。

* **`GrowthKernel` クラス:**
    独立したモジュールとして `forward` メソッド内に「SelfとMetricの更新式」を定義しています。本来はここが唯一の計算ロジックであるべきでした。
* **`SelfSpace` クラス:**
    しかし、データコンテナであるはずの `SelfSpace` 内部にも `_growth_step` というメソッドがあり、`GrowthKernel` とほぼ同じ計算式を独自に実装してしまっています。
* **結果:**
    「自己をどう更新するか」というルールを変更する際、2箇所（あるいは `TraceTensor` を含めて3箇所）を修正する必要が生じ、どちらが正解かわからない状態（Single Source of Truth の欠如）を招いています。

### 2. データと振る舞いの未分離（Coupling of Data and Behavior）
`SelfSpace` は本来、「現在の自己の状態（座標や計量）」を保持するだけの**データコンテナ（State Holder）**であるべきでした。

* **現状:** `SelfSpace` が `update` メソッドを持ち、内部で「学習率の適用」や「可塑性の計算」といった**振る舞い（Behavior）**まで管理しています。
* **問題:** これにより、`GrowthKernel`（計算機）と `SelfSpace`（メモリ）の境界が曖昧になり、「誰が状態を変更しているのか」が追いにくくなっています。

### 3. 自己言及的なフィードバックループ（Recursive Complexity）
SIA理論の特性上、**「出力（Trace）」が「構造（Self）」を変え、その「構造」が次の「出力」を変える**という循環が発生します。

* **循環:** `Self` → `Query` → `Trace` → `Self` (Update)
* **実装上の問題:** このループを単純な手続き型コードやメソッド呼び出しで実装しようとしたため、「どこでループを切るか（勾配を切るか）」や「因果の起点」がコード上で不明瞭になり、デバッグ困難な「絡まった状態」を生み出しています。

### まとめ
スパゲティ化の根本原因は、**「変化させる主体（Kernel）」と「変化する客体（Space）」の分離に失敗し、コードが重複したこと**にあります。
これを解消するには、`SelfSpace` を純粋なデータ構造にし、すべての更新ロジックを `GrowthKernel`（または `IdentityEngine`）に集約するリファクタリングが必要です。




# ============================
# file: growth_kernel.py
# Core Transformation Kernel: GrowthKernel
# ============================

import torch
import torch.nn as nn
import torch.nn.functional as F


class GrowthKernel(nn.Module):
    """
    GrowthKernel:
    Trace を受けた瞬間、
    - Self の幾何学的表現 (self_state)
    - Metric (distance / sensitivity structure)
    を同時に変形させる「成長核」。

    本質:
        Self とは位置ではなく、
        『解釈の構造』そのものが変形する空間。
    """

    def __init__(self, dim, eta_metric=0.05, eta_self=0.1, device="cpu"):
        super().__init__()
        self.dim = dim
        self.eta_metric = eta_metric  # Metric の変形率
        self.eta_self = eta_self  # Self の変形率
        self.device = device

    # ==========================================================
    def forward(self, self_state: torch.Tensor,
                metric: torch.Tensor,
                trace: torch.Tensor,
                plasticity: float):
        """
        Self と Metric を同時に変形する核。

        self_state:   (d,)   現在の自己中心
        metric:       (d,d)  心的距離構造
        trace:        (d,)   Traceベクトル
        plasticity:   (0-1)  Traceを取り込める柔軟度（防衛的なら低い）

        Return:
            new_state, new_metric
        """
        trace = trace.to(self.device)
        trace_norm = F.normalize(trace, dim=-1)

        # ===== Self 更新 =====
        new_self = self_state + self.eta_self * plasticity * trace_norm
        new_self = F.normalize(new_self, dim=-1)

        # ===== Metric 更新 (rank-1 update) =====
        outer = torch.outer(trace_norm, trace_norm)
        new_metric = metric + self.eta_metric * plasticity * outer
        new_metric = new_metric + 1e-4 * torch.eye(self.dim, device=self.device)  # 安定化

        return new_self, new_metric
