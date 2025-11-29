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


この式は、SIA（Self-Imprint Attribution）の核心を
**Geometry（幾何学）・Imprint（痕跡）・Attribution（自己帰属）**
の3要素に統合した最小構造として表現したものです。

GitHub・論文・研究ログでの使用を想定し、MathJax対応形式で記述しています。
