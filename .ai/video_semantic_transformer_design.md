# 视频全局—局部语义 Transformer 设计

## 目标

在保留每帧 CLS 和 PATCH 局部信息的同时，学习一个视频级全局语义表示，并将三类 token 全部输入 LLM：

```text
GLOBAL token
CLS tokens
PATCH tokens
```

概念验证阶段只引入一个轻量双向 Transformer 和全局对比损失，暂不加入 concept tokens、局部 OT 对齐、局部对比损失或多样性损失。

## 总体结构

Adapter 输出交错的帧级 token：

```text
CLS₀, PATCH₀, CLS₁, PATCH₁, ..., CLSₜ, PATCHₜ
```

在每个视频前添加一个可学习的 GLOBAL token：

```text
GLOBAL, CLS₀, PATCH₀, CLS₁, PATCH₁, ..., CLSₜ, PATCHₜ
```

所有 token 经过一个小型双向 Transformer：

```text
                    ┌─ GLOBAL' ──── 全局对比学习
Adapter tokens ─── Transformer
                    └─ CLS'/PATCH' ─ 局部时序特征

GLOBAL' + CLS'/PATCH' ───────────── 输入 LLM
```

最终 LLM 接收的视频 token 序列为：

```text
<video_start>
GLOBAL'
CLS₀'
PATCH₀'
CLS₁'
PATCH₁'
...
<video_end>
```

GLOBAL token 通过双向 self-attention 聚合整个视频。CLS/PATCH token 同时保留自己的局部内容，并获得全局和跨时间上下文。

## 轻量化设计

LLM hidden size 可能为 2048，直接使用 2048 维 Transformer 成本较高。建议在语义 Transformer 内部使用 512 维表示：

```python
local_hidden = input_projection(local_features)  # 2048 -> 512
tokens = concat(global_token, local_hidden)
tokens = semantic_transformer(tokens)

global_hidden = tokens[:, 0]
local_hidden = tokens[:, 1:]
```

再将输出映射回 LLM 空间：

```python
global_llm = output_projection(global_hidden)  # 512 -> 2048

local_llm = original_local + torch.tanh(local_residual_gate) * output_projection(
    local_hidden
)
```

`local_residual_gate` 初始化为 0：

```python
self.local_residual_gate = nn.Parameter(torch.tensor(0.0))
```

因此训练开始时：

```text
local_llm ≈ original_local
```

这可以避免随机初始化的 Transformer 在训练初期破坏 Adapter 已经产生的局部表示。GLOBAL token 没有对应的原始输入残差，直接使用映射后的 `global_llm`。

## Token 类型与时间位置

使用三种可学习的类型表示：

```text
global_type_embedding
cls_type_embedding
patch_type_embedding
```

Transformer 输入构造如下：

```python
CLS_t = down(CLS_t) + temporal_position[t] + cls_type_embedding
PATCH_t = down(PATCH_t) + temporal_position[t] + patch_type_embedding
GLOBAL = global_token + global_type_embedding
```

同一帧的 CLS 和 PATCH 必须共享时间位置：

```text
CLS₀   -> temporal position 0
PATCH₀ -> temporal position 0
CLS₁   -> temporal position 1
PATCH₁ -> temporal position 1
```

类型 embedding 用于区分外观/全局帧信息与局部运动信息，时间 embedding 用于表达帧顺序。

## 全局对比学习

只使用 Transformer 输出的 GLOBAL token 计算视频—文本全局对比损失：

```python
global_visual = global_projection(global_hidden)
global_visual = F.normalize(global_visual, dim=-1)
```

文本侧使用当前的上下文化全局表示：

```python
global_text = 0.5 * contextual_mean + 0.5 * last_valid_state
global_text = F.normalize(global_text, dim=-1)
```

两侧使用双向 CLIP/InfoNCE 损失：

```python
contrastive_loss = clip_loss(global_visual, global_text)
loss = lm_loss + contrastive_loss_weight * contrastive_loss
```

GLOBAL token 同时输入 LLM，因此它受到两种监督：

1. 全局对比损失要求它表达与翻译文本一致的视频级语义。
2. 语言模型损失要求它对最终翻译有实际帮助。

CLS/PATCH token 由语言模型损失直接训练，同时也通过 GLOBAL token 的 attention 路径接收全局对比损失的梯度。

## 推荐的概念验证配置

```yaml
semantic_hidden_dim: 512
semantic_num_layers: 2
semantic_num_heads: 8
semantic_ffn_dim: 1024
semantic_dropout: 0.1
local_residual_gate_init: 0.0
contrastive_dim: 512
contrastive_loss_weight: 0.1
```

如果需要更小的第一轮实验，可将 `semantic_num_layers` 设为 1。

对比损失权重建议从 `0.1` 开始，而不是直接设为 `1.0`。训练时应分别记录 LM loss、contrastive loss、可学习温度和 residual gate。

## 长度和 Prompt 适配

每个视频会额外产生一个 GLOBAL token。假设 Adapter 原本产生 `L` 个 token：

```text
旧视觉长度：L
新视觉长度：L + 1
```

对于每帧两个 token 的 Adapter：

```text
旧视觉长度：2T
新视觉长度：1 + 2T
```

Prompt 中的视频 placeholder 数量必须同步增加。包含视频边界 token 后，应为：

```text
Adapter 输出 token 数 + 1 个 GLOBAL token + 2 个视频边界 token
```

现有仅依赖固定 `video_token_scale` 推导长度的逻辑需要适配这个额外 token，否则 processor 生成的 placeholder 数量会与模型实际视觉 token 数不一致。

## 概念验证范围

第一阶段只验证以下假设：

1. 小型双向 Transformer 能否改善跨帧上下文建模。
2. GLOBAL token 能否学习稳定的视频级语义并降低全局对比损失。
3. GLOBAL token 输入 LLM 后能否改善或至少不损害翻译性能。
4. 零初始化 residual gate 能否在保留局部特征的同时逐渐引入语义 Transformer 输出。

第一阶段不加入：

- 多个 concept tokens；
- Sinkhorn/Optimal Transport；
- token 级视频—文本对齐；
- concept diversity loss；
- 视频相似度软标签；
- 复杂的分阶段训练。

只有当基础结构验证有效后，再考虑增加局部跨模态对齐。

## 建议监控项

训练过程中至少记录：

```text
main_loss
contrastive_loss
contrastive logit scale / temperature
local_residual_gate
GLOBAL token norm
CLS token norm
PATCH token norm
正样本与负样本的平均余弦相似度
```

需要特别观察：

- GLOBAL token 是否对所有视频趋于相同；
- CLS/PATCH token 是否因 Transformer 而同质化；
- residual gate 是否长期停留在 0；
- contrastive loss 下降时 LM loss 是否显著恶化；
- 增加 GLOBAL token 后生成长度与 placeholder 是否完全对齐。
