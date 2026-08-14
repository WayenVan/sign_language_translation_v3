# Short-Window Static–Motion Query Adapter

## 目标

在保持与当前 `DINOFrameAdapterCrossV2` 相近视觉 token 数量的前提下，同时保留：

- DINO patch 的静态外观和手形信息；
- 相邻帧匹配后得到的显式运动变化；
- 短时窗口内的全局 CLS 语义；
- 多个互补的局部时空表示。

建议初始配置：

```yaml
window_size: 3
window_stride: 2
num_queries: 4
num_heads: 4
diversity_loss_weight: 0.001
```

窗口必须基于真实 `visual_length` 在每个视频内部构造，不能跨视频边界，也不能把为了整除而复制的尾帧当成真实帧。

## 总体结构

```text
每帧 DINO 输出
  ├─ CLS c_t
  └─ patch x_t,p
          │
          ├─ 与下一帧局部相似 patch 对齐
          │
          └─ delta_t,p = aligned(x_t+1)_p - x_t,p

三帧短时窗口 [t-1, t, t+1]
  ├─ 3 个 CLS                  → 直接时序融合 → z_cls
  ├─ 静态 patch x              → static cross-attention
  └─ 动态 patch delta          → motion cross-attention
                                  ↑
                          共享的 4 个 learned queries
                                  │
                                  ▼
                       query-wise gated fusion
                                  │
                                  ▼
                         每个窗口输出 4 tokens
```

## 1. 显式 patch motion

沿用 CrossV2 的基本思路。对当前帧 patch `x_t` 和下一帧 patch `x_{t+1}` 做相似度匹配：

\[
\hat{x}_{t+1,p}=\operatorname{Align}(x_{t,p},X_{t+1})
\]

\[
\Delta_{t,p}=\hat{x}_{t+1,p}-x_{t,p}
\]

建议最终加入局部空间约束和匹配置信度，避免左手匹配右手、手匹配衣服或遮挡区域被强制匹配。最后一帧的 delta 必须 mask 为零。

静态 patch 与 motion delta 不应直接相加。二者使用独立的 LayerNorm、输入投影、key/value 投影。

## 2. 短时窗口

默认 `window_size=3, stride=2`：

```text
输入:       x0  x1  x2  x3  x4  x5
窗口中心:   ↑       ↑       ↑
窗口 0:  [x0, x0, x1]
窗口 1:          [x1, x2, x3]
窗口 2:                  [x3, x4, x5]
```

边界采用 replicate padding，但保留 `valid_frame_mask`。复制位置可以提供 attention 上下文，但不参与 diversity loss 统计。

窗口中的 patch 加入：

- 二维空间位置 embedding；
- 相对时间 embedding `{-1, 0, +1}`；
- static/motion 类型 embedding。

## 3. CLS 直接融合

每个窗口只有三个 CLS，不使用 query attention。推荐中心帧残差形式：

\[
z^{cls}_w=c_t+operatorname{MLP}_{cls}
\left(\operatorname{LN}[c_{t-1},c_t,c_{t+1}]\right)
\]

三个 CLS 可以先分别归一化和映射，再拼接。`z_cls` 不单独输出为 LLM token，而是作为全局条件注入四个最终 query token。

## 4. 共享 query、独立 attention 分支

定义四个共享的可学习 query：

\[
Q=[q_1,q_2,q_3,q_4]\in\mathbb{R}^{4\times H}
\]

相同的 query 参数分别读取静态和运动特征，但使用两套独立的 cross-attention 投影：

\[
Z^s=\operatorname{CrossAttn}_{static}(Q,X_w)
\]

\[
Z^m=\operatorname{CrossAttn}_{motion}(Q,\Delta_w)
\]

共享 query 的目的，是让 `static query k` 和 `motion query k` 更容易形成同一语义槽位的外观—运动对应；独立投影则允许两种输入保持不同的特征分布。

## 5. Query-wise gated fusion

对每个 query 独立计算 motion gate：

\[
g_k=\sigma\left(
\operatorname{MLP}_g[Z^s_k,Z^m_k,z^{cls}_w]
\right)
\]

\[
Z_k=Z^s_k+g_k\odot Z^m_k+W_cz^{cls}_w
\]

如果提供 patch 匹配置信度，应将 motion attention 的汇总置信度也输入 gate。错误或不确定的匹配不应产生强 motion residual。

融合后经过 LayerNorm、FFN 和输出投影，得到四个 LLM visual tokens。

## 6. 输出与时域压缩

每个 stride-2 窗口直接输出四个 token，不再接额外的 `TemporalShuffleAdapter`：

```text
窗口 0: [Q1_0, Q2_0, Q3_0, Q4_0]
窗口 1: [Q1_1, Q2_1, Q3_1, Q4_1]
...
```

同一窗口的四个 token 共享同一个 `position_id`：

```python
position_ids = torch.arange(num_windows).repeat_interleave(4)
```

当输入帧数为 `T` 且 stride 为 2 时，输出 token 数约为：

\[
4\times T/2=2T
\]

与当前 CrossV2 每帧输出 `[CLS, PATCH]` 的 token budget 基本相同。

## 7. Diversity loss

分别约束 static queries 和 motion queries 内部的 attention 多样性；不要强迫 static attention 与 motion attention 互斥，因为同一个部位的外观和运动本来就应该对应。

多头 attention 先在 head 维求均值并做 L2 归一化：

\[
\tilde A_{w,k}=\frac{A_{w,k}}{\lVert A_{w,k}\rVert_2}
\]

分支内 diversity loss：

\[
L_{div}(A)=\frac{1}{WK(K-1)}
\sum_w\sum_{i\ne j}\tilde A_{w,i}^{\top}\tilde A_{w,j}
\]

总约束：

\[
L_{div}=L_{div}(A^{static})+L_{div}(A^{motion})
\]

总训练损失：

\[
L=L_{SLT}+\lambda_{div}L_{div}
\]

第一版使用 `lambda_div=0.001`，并消融 `0 / 0.001 / 0.01`。

Diversity 建议只在中心真实帧的 patch attention 上计算，避免 CLS、replicate padding 和相邻帧重复内容干扰约束。

## 8. 推荐监控项

训练和验证时记录：

- `static_attention_overlap`；
- `motion_attention_overlap`；
- static/motion attention entropy；
- query 输出两两 cosine similarity；
- 每个 query 的 motion gate；
- patch matching confidence；
- query attention 热图。

如果 attention overlap 下降但翻译性能也下降，通常意味着 diversity 太强，部分 query 被迫关注背景。

## 9. 最小消融实验

| 实验 | Queries | Motion branch | Diversity |
|---|---:|---:|---:|
| Baseline | 当前 CrossV2 | 是 | 无 |
| W4-S | 4 | 否 | 无 |
| W4-SM | 4 | 是 | 无 |
| W4-SM-D1 | 4 | 是 | 0.001 |
| W4-SM-D2 | 4 | 是 | 0.01 |

重点比较：

- `W4-S` vs `W4-SM`：显式 motion 分支是否有效；
- `W4-SM` vs `W4-SM-D1`：diversity 是否缓解 query collapse；
- `W4-SM-D1` vs Baseline：在相近 token budget 下是否提升翻译；
- `W4-SM-D1` vs `W4-SM-D2`：过强 diversity 是否损伤表现。

## 当前设计决策

1. 使用 boundary-aware 三帧短时窗口和 stride 2。
2. patch motion 沿用“当前 patch 与下一帧相似 patch 的差分”。
3. CLS 不做 query，三个 CLS 直接融合为窗口全局条件。
4. 静态和动态特征使用独立 attention 分支。
5. 两个分支共享四个 learned query 参数，以建立对应槽位。
6. static/motion query 输出通过内容相关 motion gate 融合。
7. 每个窗口最终只输出四个 token。
8. diversity loss 在两个分支内部独立计算，不跨分支互斥。
