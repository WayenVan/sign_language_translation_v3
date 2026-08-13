# Round 2 Refinement

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Anchor Check
- Original bottleneck: Qwen 没有充分利用保细节的视觉条件；retrieval-style auxiliary 与生成目标错位。
- Why revised method still addresses it: 用 source-less counterfactual 隔离 gloss 对生成分布的增量影响，再只把这部分信息传给视觉接口。
- Reviewer suggestions rejected as drift: 不再把 input-statistic closeness 当必要成功标准；它与功能等价的生成条件不是一一对应，只保留作诊断。

## Simplicity Check
- Dominant contribution: Source-Influence Distillation（SID）：用 `qg` 与 `q0` 的反事实差异筛选/加权跨模态 response KD。
- Components removed: calibrator、OT、early-state loss、confidence gate、top-k cache 均从核心方法删除。
- Rejected complexity: 不同时蒸馏 logit residual；先采用 source-influence weighted KL。
- Smallest adequate route: 两个冻结 teacher 条件 forward + 一个 weighted KL；推理零开销，零新增参数。

## Changes Made

### 1. 加入 source-less counterfactual
- Reviewer said: qg 的大部分信息可能来自 target prefix。
- Action: 构造 q0，并以 `JS(qg,q0)` 形成每位置权重。
- Reasoning: 只蒸馏 gloss 实际改变分布的位置。
- Impact: 从标准 privileged KD 变成 source-influence-aware KD。

### 2. 严格梯度路由
- Reviewer said: LoRA 可绕开视觉接口逼近 teacher。
- Action: SID loss 只更新 adapter；CE 按基线更新 adapter+LoRA。
- Reasoning: 强制 privileged source information 进入视频路径。
- Impact: 提高机制可归因性。

## Revised Proposal

# 研究方案：Source-Influence Distillation for Sign-to-LLM

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Technical Gap
全局 contrastive alignment 回答“视频与文本是否在检索空间接近”，而 SLT 需要“哪些 target decisions 必须由 source evidence 改变”。直接 qg→video KD 仍会复制 target-prefix language prior。缺失机制是一个反事实 source baseline：比较同一冻结 Qwen 在有/无 gloss source 时的生成分布，只迁移由 source 引起的增量影响。

## Method Thesis
- One-sentence thesis: 用 source-less Qwen 作为反事实基线定位 pseudo-gloss 真正影响下一词预测的位置，并将 gloss-conditioned teacher 的分布只在这些位置蒸馏到 video adapter。
- Smallest adequate intervention: 无新模型参数；复用冻结 Qwen 计算 `qg/q0`，新增 source-influence weighted KL；推理路径不变。
- Frontier relevance: foundation model 同时充当 conditional teacher 与 counterfactual critic，直接定义 source evidence 对生成行为的边际贡献。

## Contribution Focus
- Dominant contribution: SID，一个反事实、source-influence-aware 的跨模态 privileged distillation objective。
- Supporting contribution: 无。
- Non-contributions: adapter、backbone、pseudo-gloss generation、embedding model、OT、RL。

## Proposed Method
### Complexity Budget
- Reused: 当前 backbone、adapter、Qwen、prompt/data pipeline。
- New trainable components: 0。
- Training-only extra compute: frozen Qwen teacher 对 `gloss source` 与 `empty source` 各一次 forward。
- Excluded: calibrator、queue、negative mining、token alignment、probe tokens。

### Exact Paths
令 gold translation 为 `Y=(y1...yN)`，pseudo-gloss 为 `G`。

- Student: `p_v^t = Q_θ(y_t | P_V, Z(V), y_<t)`。
- Gloss teacher: `q_g^t = Q_θ0(y_t | P_G, E(G), y_<t)`。
- Source-less teacher: `q_0^t = Q_θ0(y_t | P_0, y_<t)`。

`θ0` 是同一 base Qwen 的永久 frozen/eval/LoRA-off 逻辑副本；teacher 输入绝不包含 future target。三个路径使用相同 target delimiter、tokenization 和 target span mask。`P0` 与 `PG` 的 instruction 完全相同，只将 source 内容替换为专门的 `<no-source>` 文本，避免 prompt wording confound。

### Source Influence
首轮使用 full-vocabulary float32 概率，温度 T=1：

`d_t = JS(q_g^t || q_0^t)`。

在训练集 teacher statistics 上预先固定尺度 `τ = percentile_75({d_t})`，不按验证指标调参：

`w_t = stopgrad(min(d_t / τ, 1))`。

若 `mean(d_t)` 近零或 gloss-conditioned NLL 不优于 q0，路线在训练 student 前终止。

### Loss and Gradient Routing
`L_SID = [Σ_t mask_t w_t KL(q_g^t || p_v^t)] / [Σ_t mask_t w_t + ε]`。

`L = L_CE + λ(s)L_SID`，λ warm-up 后在末 30% decay 到 0。实现时分别求梯度：

- `L_CE` → adapter + 基线允许的 student LoRA；
- `L_SID` → adapter only；
- `∇LoRA L_SID = 0`，`∇θ0=0`，backbone 保持冻结。

首个机制实验完全冻结 student Qwen，仅训练 adapter，以排除绕过；确认后再测试与 LoRA 共存。SID 对同一 video-conditioned logits 反传，因此会通过 Qwen 的固定 Jacobian 塑造 adapter，而不更新 Qwen。

### Inference
删除 `qg/q0` 两条 teacher 路径，保持当前 video→adapter→Qwen；零新增参数与推理开销。

### Why not Qwen Embedding / Contrastive
embedding 模型强化句级检索不变性，InfoNCE 依赖可能错误的 negatives，二者都不标识哪些生成决策需要 source。SID 不压缩视频序列、不构造 negatives，只在 gloss 相对无 source 明确改变 Qwen 输出的 target positions 提供监督。

### Failure Modes and Diagnostics
- qg≈q0: gloss teacher 没提供 source evidence，直接停止。
- qg 错误: 同时报告 qg/q0 gold NLL；可在最终阶段把 `w_t` 乘 `1[qg gold log-prob > q0]` 作为预注册安全变体，但不与主方法并列。
- Student ignores video: video-zero/masked 与 temporal shuffle；SID 必须对 zero-source 显著退化且不比 CE-only 更不敏感。
- Target-prefix dominance: 报告 JS 随 target position 的曲线，以及高权重 token 类型。
- Gradient conflict: adapter 上 CE/SID gradient cosine；λ 初始设为使 SID/CE adapter gradient norm ratio≈0.1。
- Compute: 核心实验在线 full-vocab；缓存/Top-k 不属于方法结论。

### Novelty and Elegance Argument
LUPI 与跨模态 KD 已有大量工作，近期 VLM KD 也直接对齐 teacher/student logits，因此“text teacher→video student”本身不是贡献。SID 的边界是：用同一生成器的 source-less counterfactual 显式扣除 target-prefix language prior，再通过严格 adapter-only gradient routing 把 source-induced distribution change 注入视觉接口。相较 SLT 的 SignCL/GFSLT-VLP/SignLLM/SCL-SLT，它不做 embedding density、random negatives 或 source-token OT。论文主张严格限定为：在相同 pseudo-gloss privileged supervision 下，source-influence selection 比无差别 KD 和 embedding alignment 更有效且更少冲突。

## Claim-Driven Validation Sketch
### Claim 1: source-influence weighting 隔离了 gloss 的有效增量
- Experiment: q0/qg 的 NLL、JS position curve；unweighted KD vs instruction-only KD vs SID。
- Metric: BLEU-4/ROUGE-L/NLL；高权重 token 分布。
- Decisive evidence: SID > unweighted KD > instruction-only，且 qg NLL < q0。

### Claim 2: SID 优于同监督的 embedding alignment 且不绕过视频
- Experiment: CE-only、当前 InfoNCE、unweighted GPCD、SID；student Qwen frozen，统一训练预算。
- Metric: 3-seed BLEU-4/ROUGE-L、adapter CE/SID gradient cosine、video-zero/shuffle ΔNLL/ΔBLEU。
- Decisive evidence: SID 稳定提升且 zero/masked video 显著破坏输出。

## Experiment Handoff Inputs
- Must-prove: qg 提供非零 source influence；SID 优于普通 KD/InfoNCE；视频不可被忽略。
- Must-run: teacher sanity、四臂对照、zero/masked test。
- Critical metrics: PHOENIX14T BLEU-4/ROUGE-L/NLL，最终 3 seeds。
- Highest risks: pseudo-gloss 噪声；SID 本质上是已有 conditional KD；额外 teacher compute。

## Compute & Timeline Estimate
- GPU-hours: teacher sanity <0.2 次训练；四臂约 4 次 1.7B；最终两个候选补 seeds，总约 8 次训练。在线双 teacher 约增加训练 FLOPs，但 teacher 无反向。
- Data: 无新增标注，完整披露 pseudo-gloss 来源。
- Timeline: 1–2 天实现 q0/qg 与 gradient routing，2–4 天筛选，再补 seeds。
