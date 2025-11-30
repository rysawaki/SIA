# SIA Core Equation Set (Geometry × Imprint × Attribution)

$$
\begin{aligned}
&\textbf{Self-space（幾何学）} \\
&(\mathcal{S},\; g_{\theta_t}),\quad s_t \in \mathcal{S},\quad T_t \in \mathbb{R}^{d \times d}
\end{aligned}
$$
\\
$$
\begin{aligned}
&\textbf{1. Attribution（自己に関係あるか？）} \\
&z_t = \mathrm{Enc}(e_t) \in \mathbb{R}^d \\
&u_t = \phi(z_t) \in \mathcal{S} \\
&R_t = - d_{g_{\theta_t}}(u_t, s_t)^2 \\
&a_t = \sigma(\beta R_t) \in (0,1) \\[10pt]
\end{aligned}
$$
\\
$$
\begin{aligned}
&\textbf{2. Imprint（痕跡による不可逆変形）} \\
&\delta_t = u_t - s_t \\
&T_{t+1} = (1-\lambda)\, T_t + \alpha\, a_t\, (\delta_t \otimes \delta_t) \\[10pt]
\end{aligned}
$$
\\
$$
\begin{aligned}
&\textbf{3. Geometry update（自己空間そのものが歪む）} \\
&g_{\theta_{t+1}} = g_{\theta_t} + \eta\, G(T_{t+1}) \\
&s_{t+1} = s_t + \gamma\, H(T_{t+1}, s_t)
\end{aligned}
$$


* **Geometry**

  * $(\mathcal{S}, g_{\theta_t})$ が「Self-Space」
  * $s_t$ はその中の「現在の自己の基点」

* **Attribution**

  * $R_t = - d_{g_{\theta_t}}(u_t, s_t)^2$
    → 自己からの距離が近いほど $R_t$ は大きく（＝より「自分ごと」）
  * $a_t = \sigma(\beta R_t)$ が **「これはどれくらい自分に関係あるか」** のスカラー

* **Imprint**

  * $\delta_t = u_t - s_t$ が「自己とのズレ」
  * $T_{t+1}$ は、そのズレに $a_t$ を掛けたテンソルを **不可逆に積み上げる痕跡**

* **Geometry update**

  * $G(T_{t+1})$ がメトリック変形（曲率など）
  * $H(T_{t+1}, s_t)$ が自己基点そのもののシフト
    → 結果として **自己空間と自己位置が両方ゆがむ**


