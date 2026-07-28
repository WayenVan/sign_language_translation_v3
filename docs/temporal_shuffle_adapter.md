# TemporalShuffleAdapter

`TemporalShuffleAdapter` 将每帧一个视觉 token 的可变长度视频批次，按固定的局部时间窗口融合并下采样。实现位于 `src/csi_slt/modeling_slt/visual_adapters/patch_shuffle.py`。

设输入为按视频顺序拼接的帧特征：

\[
X \in \mathbb{R}^{N \times D}, \qquad
\boldsymbol{T} = [T_1, \ldots, T_B], \qquad
N = \sum_{i=1}^{B} T_i.
\]

其中每个 token 对应一帧，`D` 是视觉特征维度。设时间下采样因子为 `s`（即 `scale_factor`）。输入仍是按视频顺序 concat 的 packed tensor，不需要构造 batch padding；每个 \(T_i\) 必须能被 \(s\) 整除。adapter 会在 reshape 前验证此条件，因此窗口绝不会跨越视频边界，且 token 数具有固定缩放比例。

## Window shuffle

每 \(s\) 个相邻帧构成一个窗口：

\[
W_j = [x_{js}, x_{js+1}, \ldots, x_{js+s-1}]
    \in \mathbb{R}^{s \times D}.
\]

因此 token 总数从 \(N\) 变为 \(N / s\)，每个视频的新长度为：

\[
T_i' = T_i / s.
\]

## Motion-aware gated fusion

每个窗口的输出由稳定内容的 base path 和运动信息的 gated residual path 相加：

\[
y_j = b_j + \sigma(g) \cdot m_j.
\]

### Base path

先对窗口内帧取平均，以保留共享的外观/手形等稳定信息：

\[
b_j = W_b\, \operatorname{LN}\!\left(
        \frac{1}{s}\sum_{k=0}^{s-1} W_{j,k}
      \right).
\]

### Motion path

拼接有序帧特征与相邻帧差分：

\[
z_j = [W_{j,0}; \ldots; W_{j,s-1};
       W_{j,1}-W_{j,0}; \ldots;
       W_{j,s-1}-W_{j,s-2}]
    \in \mathbb{R}^{(2s-1)D}.
\]

经过 LayerNorm 和 SwiGLU：

\[
[v_j, q_j] = W_{in}\, \operatorname{LN}(z_j),
\]

\[
m_j = W_{out}\left(v_j \odot \operatorname{SiLU}(q_j)\right).
\]

其中 \(v_j\) 是候选运动特征，\(\operatorname{SiLU}(q_j)\) 是由当前窗口内容决定的逐通道门控。门控可抑制冗余/静态帧带来的运动特征，也可在手形变化或手部位移时放大相关通道。

全局可学习标量 \(g\) 控制运动残差的初始强度。代码中 \(g=-2\) 初始化，故 \(\sigma(g)\approx0.12\)：模型训练开始时以稳定 base path 为主，再逐步学习使用 motion path。

当 `scale_factor=2` 时，运动分支的输入就是：

\[
z_j = [x_t; x_{t+1}; x_{t+1}-x_t].
\]

顺序拼接使 \([x_t; x_{t+1}]\) 与 \([x_{t+1}; x_t]\) 不同；差分项则显式提供运动方向和幅度。

输出维度为 `output_hidden_size`。构造函数仍接受 `mlp_depth` 以兼容已有配置，但该参数不参与新的 SwiGLU 融合结构。
