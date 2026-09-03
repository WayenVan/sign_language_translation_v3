# Gloss-free SLT 设计方案:CTC 离散码本 + Qwen3 端到端

> 交接文档。目标读者是负责实现的工程 agent。
> 文中标注 `[待确认]` 的地方需要先向项目负责人索取,不要自行假设。

---

## 1. 任务与现状

| 项 | 值 |
|---|---|
| 数据集 | RWTH-PHOENIX-Weather 2014T(德语手语 DGS) |
| 规模 | train 7096 / dev 519 / test 642,约 11 小时视频 |
| 文本词表 | 2887 个德语词 |
| 设定 | **gloss-free**,不使用逐样本真 gloss 标注 |
| 当前成绩 | BLEU-4 约 12 `[待确认:什么配置跑出来的]` |
| 目标 | BLEU-4 20+ |

### 参照系(PHOENIX14T,gloss-free)

| 方法 | 语言侧 | 论文报告 | 统一复现 |
|---|---|---|---|
| C²RL | 12 层 mBART | 26.75 | — |
| SCL-SLT | mBART | 26.00 | — |
| LLaVA-SLT | LLM | 23.43 | — |
| FLa-LLM | 12 层 mBART | 23.09 | **19.84 ±0.28** |
| Sign2GPT | XGLM 1.7B(decoder-only) | 22.52 | — |
| SignCL | 3 层 mBART | 22.74 | 22.35 ±0.28 |
| GFSLT-VLP | 3 层 mBART | 21.44 | ~21.4 |

统一复现来自 Surrey 的 *Gloss-Free SLT: An Unbiased Evaluation*(代码库 `github.com/ozgemercanoglu/sltbaselines`,三个种子 0/42/100)。

**注意两点:**

1. 榜上前排几乎全是 encoder-decoder。唯一的 decoder-only(Sign2GPT)排中游,SCL-SLT 论文明确指出自己比 LLM-centric 的 LLaVA-SLT 高 2.57。**本方案走 decoder-only,在这个基准上是逆风局**,合理预期是 20–22,不是 26。
2. 12 → 22 这 10 分的差距不是架构选择造成的,是实现问题。先定位再优化。

---

## 2. 伪 gloss

### 生成方式

从德语口语句子**删除功能词**(冠词、系动词、部分介词)得到,保留全部实词。

**硬约束:不做语序重排。**

CTC 要求标签序列与输入单调对齐。视频是按时间顺序的,伪 gloss 若按 DGS 语法重排成 SOV,单调性被破坏,CTC 不可解。文献里(如 SignBind-LLM 的 prompt)会做重排,那是因为它们的下游不依赖 CTC 单调性。**本方案必须只删不换序。**

语序的调整交给 Qwen3——它的职责就是从残缺序列还原流畅德语。

### 数据增强

同一德语句子用**不同删除强度**生成多份伪 gloss(保留 60% / 70% / 80% 实词各一份),7096 句可扩到 2–3 万对。

目的:让模型见过不同程度的残缺,推理时 CTC 输出质量波动也能扛。成本几乎为零,对训练外挂码本尤其有用。

### 词表

`[待确认:实际词表大小]`。预期落在 1500–2200(原始 2887 减去功能词)。

若超过 2000,做词频截断:出现次数 < 5 的词映射到 `<UNK>`。7096 个样本训不出低频词的可靠 embedding。

---

## 3. 架构

```
视频帧
  │
  ▼
视觉编码器                      [待确认:现有结构]
  │  输出 [T, d_v],T = CTC 槽位数 [待确认]
  │
  ├──────────────────────────────┐
  ▼                              │
CTC head W_ctc [d_v, |V|+1]      │
  │  logits [T, |V|+1]           │
  │                              ▼
  ├──► L_ctc(伪 gloss 监督)   (仅前向路径,不共享权重)
  │
  ▼
Gumbel-Softmax / Straight-Through
  │  y [T, |V|+1]
  │
  ▼
外挂码本 E [|V|+1, d_e]          ← 独立参数,与 W_ctc 不绑定
  │  out = y @ E   →  [T, d_e]
  │
  ▼
软 collapse:门控 (1 - y[:, blank])
  │
  ▼
Projection
  │  LayerNorm → Linear(d_e→d_llm) → GELU → Linear(d_llm→d_llm) → LayerNorm
  │  输出 scale 对齐 Qwen3 embed_tokens 的平均 norm
  │
  ▼
prefix [T, d_llm]  →  Qwen3 (LoRA)  →  L_lm
```

**总损失:** `L = L_lm + λ · L_ctc`,λ 初始 1.0。

### 3.1 CTC head 与外挂码本必须分离

技术上可以做权重绑定(`E = W_ctc.T`),**但不要这么做**。

两个目标对同一张表的要求冲突:

- CTC 要**可分性**:不同 gloss 的向量越远越好
- Qwen3 要**语义结构**:语义相近的 gloss 应该相近

绑定会导致两头不讨好。分开的代价只是 `|V| × d_e` 约 100 万参数,可忽略。

`d_e` 建议 512,不必等于 `d_llm`。projection 负责升维。

### 3.2 外挂码本初始化

用 Qwen3 tokenizer 编码每个伪 gloss,取子词 embedding 的平均值作为该行的初值。

多子词取平均在**终态**是错的(落在流形外、norm 按 1/√k 系统性收缩、且收缩程度与词频相关),但作为**初始化**是合理的——它只提供一个语义合理的起点,后续会被训练更新。

`E` 必须是可训练参数,不能冻结。

### 3.3 软 collapse

**不做 argmax,不合并重复槽位,不物理删除 blank。**

blank 通过门控实现:每个槽位输出 `(1 - y[blank]) × (y @ E)`,blank 概率高的槽位自然趋近零向量。

理由:

- argmax + collapse 是硬判决,梯度到此断裂,视觉编码器收不到语言损失的信号,整个端到端设计失去意义
- 重复 token 对 LLM 不构成障碍,它能从上下文消化
- T 约 75(`[待确认]`),不合并的 prefix 长度完全可接受

**位置信息**:软 collapse 后时序结构会被削弱,建议在 projection 前叠加一层位置编码,保留槽位的原始时间索引。手语的时序对翻译有信息量。

### 3.4 梯度路径

```
L_lm
 └─► prefix ─► projection ─► (y @ E)
                              ├─► E                    (码本更新)
                              └─► y ─[Gumbel 重参数化]─► logits ─► W_ctc ─► 视觉编码器

L_ctc ──────────────────────────────────────────────► logits ─► W_ctc ─► 视觉编码器
```

视觉编码器收到**两股梯度**。二者的平衡(λ)是本方案的主要调参对象。

### 3.5 Straight-Through 实现

```python
y_soft = F.softmax((logits + gumbel_noise) / tau, dim=-1)
y_hard = F.one_hot(y_soft.argmax(-1), num_classes=V+1).float()
y = y_hard - y_soft.detach() + y_soft      # ST trick
out = y @ E
```

第三行:前向 `y == y_hard`(真 one-hot,`y @ E` 等价于精确索引);反向梯度只走 `y_soft`,可导。

推理时关闭 Gumbel 噪声,直接 `argmax(logits)`。

**已知局限:**

- ST 梯度是有偏估计。τ 过小时 softmax 雅可比趋零,梯度消失 → τ 下界取 0.1,不要更低
- Gumbel 噪声使梯度方差偏大,loss 曲线会抖,属正常

---

## 4. 训练计划

### Phase A — CTC 预热(必做)

冻结 Qwen3(仅 LoRA 参数存在但不参与),**只用 `L_ctc`** 训视觉编码器 + CTC head,跑若干 epoch。

理由:logits 随机时,Gumbel 采样输出的是噪声混合向量,会把 projection 带偏。

**通过标准:** CTC WER 明显下降并趋稳。若 WER 始终很高,问题在视觉侧,停止推进,先修视觉编码器。

### Phase B — 联合训练(纯软)

加入 `L_lm`,τ **固定在 1.0**,使用纯软路径(不用 ST)。

目的是验证管线接通:梯度能流、两个 loss 都在降。这是最稳的配置,用来排除接线错误。

### Phase C — ST + τ 退火

切换到 Straight-Through,τ 从 1.0 退火到 0.1。参考 SignBind-LLM 是 50K 步内退完;本数据集样本少,建议退慢一些。

若切换后 loss 反而不降,可确定是 ST 引入的问题(因为 Phase B 已建立参照系),而非其他环节。

### 并行任务 — 纯文本上界(优先级高,不阻塞主线)

**独立脚本,不需要视频,不依赖主线代码。**

伪 gloss 序列 → 外挂码本 → projection → Qwen3 → 德语句子。用第 2 节的增强数据训练。

这个实验回答三个问题:

1. Qwen3 的德语生成能力上界是多少
2. 删词策略丢掉的信息(冠词、系动词、语序)能还原多少
3. 外挂码本 + projection 能否训得动

**这是整条 pipeline 的天花板。** 端到端结果必须对照它来解读——否则拿到 15 分时无法区分是视觉侧、对齐、还是语言侧的问题。

若上界只有 20 出头,应立即回头修改伪 gloss 生成策略(减少删除强度、保留更多功能词),而不是继续优化视觉侧。

---

## 5. 超参

### LoRA(Qwen3)

参考 SignBind-LLM 的配置:

```
r = 16, alpha = 32, dropout = 0.05
target_modules = [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
lr = 2e-4, optimizer = AdamW, warmup_ratio = 0.03
```

**不要全冻 Qwen3。** 全冻的话 projection 需独自跨越模态鸿沟,7K 样本量级训不出来。

### Prompt 构造

prefix 部分的 label 全部设为 `-100`,**loss 只在德语目标句上计算**。这一点容易漏。

建议加分隔特殊 token(参考 SignBind-LLM 的 `<S2S>` / `<GLOSS>` / `</GLOSS>` / `<TEXT>`),随机初始化后训练。

### 其他

- `d_e = 512`
- projection 用 **2 层 MLP,不要单层线性**
- 末尾 LayerNorm 后需 rescale:先统计 `Qwen3.embed_tokens.weight` 的平均 L2 norm,把 projection 输出对齐到同一量级。**不做这步,Qwen3 前几层会直接失效**
- 若 Qwen3 输入输出 embedding 存在 tying,注意生成时的 id 范围(本方案不改 Qwen3 词表,风险较低,但需确认)
- `d_llm` = `[待确认:取决于选用的 Qwen3 规模,不要凭记忆填]`

---

## 6. 监控指标

必须在训练过程中持续记录:

| 指标 | 危险信号 | 含义与处理 |
|---|---|---|
| **CTC WER** vs **L_lm** | L_lm 在降但 WER 不降或上升 | **语言先验压制视觉学习**(FLa-LLM 分析过的失败模式)。Qwen3 在靠语言先验硬编天气预报。立即加大 λ |
| **有效 gloss 数** | 塌到词表的 10–15% 以下 | 模型只用少数高频 gloss 蒙混。检查 τ 和 CTC 权重 |
| **dev BLEU-4 曲线** | 快速上升后迅速平台 | 早期过拟合。7096 样本 + 大解码器是已知高危组合(统一复现中 680M mBART 就已过拐点,跌到 19.84) |
| **prefix 向量 norm** | 与 Qwen3 embed norm 差一个量级 | projection 的 rescale 没生效 |

---

## 7. 评测协议

必须与文献对齐,否则数字不可比:

- **sacreBLEU**,签名 `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp`
- **句尾追加 `" ."`** 到预测和参考两侧。PH14T 原始句子无标点,但社区惯例会追加,这会抬高 BLEU;不追加则无法与已发表数字对比
- 打分前**剥离语言标记 token**(如使用 mBART 系时的 `de_DE`)。只剥一侧会导致每句固定错一个 token,4-gram 全线受损
- 报 **test** 集,dev(519 句)通常偏高
- 建议跑 **3 个种子**(0/42/100)并报均值 ± 标准差,与统一复现的口径一致

关于 BLEU 库:nlgeval 与 sacreBLEU 在 PH14T 上的 BLEU-4 差异仅零点几分(21.44 vs 21.38),因为该数据集小写无标点,13a 近似空操作。**不是主要风险,但 BLEU-1 差异明显(43.71 vs 46.04),报低阶指标时注意说明。**

---

## 8. 已知风险

1. **数据量**。7096 句、2887 词、单一天气预报领域。统一复现显示 680M 解码器已过拟合拐点,3B 级只会更严重。若 dev 曲线早平台,优先考虑更强正则(label smoothing 0.2 是这条线的标配)而非更大模型。
2. **德语**。mBART/XGLM 被选用的明文理由是多语种预训练。Qwen3 的德语能力弱于专门做翻译预训练的模型,**这是纯文本上界实验要测的核心变量**。
3. **PH14T 基准本身**。训练/测试集重叠会虚高 BLEU 并掩盖过拟合。最终结论建议在 CSL-Daily 上交叉验证。
4. **decoder-only 在小数据集上的劣势**。mBART decoder 每层可 cross-attention 回视觉侧重新取信息,decoder-only 只读一次 prefix。数据少时前者的归纳偏置是优势。本方案的收益预期应放在**可扩展到更大数据集**上,而非 PH14T 刷分。

---

## 9. 开工前必须索取的三个数字

1. **伪 gloss 词表大小** —— 决定 CTC 头输出维度和码本行数
2. **CTC 槽位数 T** —— 必须**明显大于**伪 gloss 序列的最大长度,否则 CTC 不可解。若不满足,先改下采样率
3. **当前 12 BLEU-4 的完整配置** —— 是否端到端联合训练、有无视觉预训练、优化器与学习率。这决定新方案接在哪一层

第 2 项是硬约束,开工前必须验算。
