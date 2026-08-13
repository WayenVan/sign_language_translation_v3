# Round 1 Refinement

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Anchor Check
- Original bottleneck: 视觉条件未被 Qwen 有效消费，而 retrieval-style auxiliary 与生成 CE 可能冲突。
- Why revised method still addresses it: 直接蒸馏 Qwen 对目标 translation 的条件分布，不再要求视觉 token 接近静态词向量。
- Reviewer suggestions rejected as drift: 不采用纯 moment matching 作为论文主路线；它只能修尺度统计，无法提供手语内容的条件监督。它保留为诊断基线。

## Simplicity Check
- Dominant contribution after revision: training-only 的 gloss-privileged conditional distribution distillation（GPCD）。
- Components removed or merged: 删除 early-state matching、student-dependent OT、NULL、token correspondence、external embedding encoder；校准器降级为普通 baseline component。
- Reviewer suggestions rejected as unnecessary complexity: 不加入 probe token，因为 next-token logits 已提供直接 readout。
- Why smallest adequate: 复用同一个 Qwen 运行 teacher/student 两条条件路径，只新增一个 KL loss，推理零新增分支。

## Changes Made

### 1. 从 hidden-state calibration 改为 conditional distribution distillation
- Reviewer said: early-state matching 不能证明 distribution calibration，且 student-dependent OT 有弱循环。
- Action: teacher 使用 pseudo-gloss 条件，student 使用 video 条件；二者在同一 gold translation prefix 上输出 next-token logits，做 KL；完全删除 OT。
- Reasoning: logits 是最终生成行为的直接对象，且 translation positions 天然一一对应。
- Impact: 机制从混合式对齐变成单一、可归因的生成蒸馏。

### 2. 明确 privileged supervision 与公平对照
- Reviewer said: pseudo-gloss 可能是 label-derived privileged supervision。
- Action: 明确披露来源；所有 static/contrastive/contextual 对照使用相同 pseudo-gloss；另设 CE-only 衡量额外监督的总收益。
- Reasoning: 避免把监督预算差异误写成方法优势。
- Impact: 主张限定为“相同 pseudo-gloss 下，生成分布蒸馏优于 embedding alignment”。

## Revised Proposal

# 研究方案：Gloss-Privileged Conditional Distillation for Sign-to-LLM

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Technical Gap
当前 InfoNCE 优化“整段视频与 pseudo-gloss 在检索空间可分”，但 SLT 需要的是“给定视频，Qwen 在每个生成位置形成正确的条件分布”。全局 pooling 丢失时序；随机/queue negatives 可能包含重复或相似语义；raw word-embedding mean 不是 Qwen 的 contextual semantics。独立 Qwen embedding model 会进一步把序列压向 retrieval invariant representation，接口目标仍与 autoregressive generation 不一致。

## Method Thesis
- One-sentence thesis: 在训练时以 pseudo-gloss 作为 privileged condition，让冻结 Qwen teacher 的逐步翻译分布监督 video-conditioned student，从而把跨模态监督从 embedding similarity 改成 generation behavior matching。
- Smallest adequate intervention: 只增加一次 teacher forward 和 token-level KL；无新 encoder、无 OT、无 negatives，推理路径完全不变。
- Foundation-model-era relevance: 将 LLM 本身作为 task-specific conditional teacher，而不是用外部 embedding space 代理其需求。

## Contribution Focus
- Dominant contribution: GPCD，一种生成目标一致、无需负样本与跨长度 token 对齐的 SLT auxiliary。
- Optional supporting contribution: 无。
- Explicit non-contributions: adapter architecture、视觉 backbone、pseudo-gloss 生成器、RL。

## Proposed Method
### Complexity Budget
- Frozen/reused: backbone；teacher Qwen `θ0` 永久 eval/frozen/LoRA-disabled；student Qwen 沿用当前 frozen 或 LoRA 配置；现有 adapter。
- New trainable components: 零个必需新模块。可选 identity-gated residual calibrator 仅作为共同架构，所有对照共享。
- Excluded: Qwen embedding model、contrastive head/queue、OT、Q-Former、VQ。

### System Overview
Student: `[P_video; Z_video; P_target; y_<t] → Qwen_θ → p_v(y_t)`

Teacher: `[P_gloss; E(G); P_target; y_<t] → Qwen_θ0 → p_g(y_t)`

两条路径只在 gold target token positions 对齐，因此无需对齐 video/gloss 的源长度。

### Core Mechanism
- Inputs: 同一样本的 video、pseudo-gloss `G`、target translation `Y`。teacher 输入不包含未移位的 future target；标准 teacher forcing 的 `y_<t` 与 CE 完全相同，不构成额外泄漏。
- Prompts: `P_video` 与 `P_gloss` 使用语义等价的 instruction，仅 source span 类型不同；target delimiter 及 target position IDs 规则一致。两路径独立构造 causal attention mask。
- Teacher: 从 base Qwen checkpoint 复制逻辑权重 `θ0`，永久 `eval()`、无 LoRA、stop-gradient。teacher logits 可离线缓存为 top-k log-prob + residual mass，降低显存/算力；首版可在线验证正确性。
- Loss: `L_GPCD = mean_t w_t * T² KL(q_g^T(.|G,y_<t) || p_v^T(.|V,y_<t))`。仅在非 padding target positions 计算；T∈{1,2}。为避免错误 teacher 过度约束，`w_t = 1[teacher top-1 == y_t]` 或平滑置信权重 `q_g(y_t)^γ`，两者预先选一种，默认平滑权重避免大量 hard drop。
- Total: `L = L_CE + λ(s)L_GPCD`。λ warm-up 后保持，后 30% cosine decay 到 0；以 adapter 参数上 auxiliary/CE gradient norm ratio 约 0.1 为初始化准则。
- Inference: 完全删除 gloss teacher，只运行原 video→adapter→student Qwen。

### Why not Qwen Embedding
Qwen embedding 模型的输出空间由检索/排序目标塑造，通常强调句级不变性与 pooling；SLT 的错误来自细粒度 source evidence 未能改变逐步生成分布。GPCD 直接回答“若 Qwen 已读懂 pseudo-gloss，它下一 token 会怎样分布”，再让视频条件复现该行为。

### Training Plan
1. Teacher sanity：在冻结 Qwen 下比较 instruction-only 与 pseudo-gloss-conditioned 的 gold translation NLL；若 gloss 不显著降低 NLL，则停止 GPCD 路线。
2. 固定共同 student 架构、数据与 steps，关闭现有 contrastive/OT；训练 CE-only、static embedding alignment、当前 InfoNCE、GPCD。
3. 仅当 GPCD 优于同监督的 static/InfoNCE 且至少不劣于 CE-only，才补 3 seeds；否则不继续叠模块。

### Failure Modes and Diagnostics
- Weak/wrong teacher: gloss-conditioned NLL 不优于 instruction-only；停止或按 teacher confidence 降权。
- Label-derived pseudo-gloss: 明确披露生成流程；同监督对照全部使用相同 gloss，不把它与无额外监督 baseline 直接归因于方法。
- Student copies language prior but ignores video: 做 temporal shuffle/frame masking；若 GPCD 模型降幅小于 CE-only，说明条件被绕过，应减 λ/加强视频扰动一致性诊断，而非宣称成功。
- Gradient conflict: 记录 adapter 上 CE 与 KL gradient cosine；若长期为负，缩短蒸馏阶段或降低 λ。
- Compute: 用 top-k cached teacher logits；验证 cached KL 与 full-logit KL 误差。

### Novelty and Elegance Argument
与 SignCL/GFSLT-VLP 的 embedding contrast、SignLLM 的 VQ+OT 不同，GPCD 不对齐 source representations，而对齐相同 target prefix 下的 conditional generation behavior。它与一般 knowledge distillation 的区别在于 teacher/student 共享生成器先验但接收不同模态、teacher 仅在训练期获得 privileged pseudo-gloss。最强可证伪主张是：在完全相同的 pseudo-gloss 监督预算下，conditional distribution distillation 比 retrieval/static embedding alignment 更少与 CE 冲突并带来更高 SLT 指标。

## Claim-Driven Validation Sketch
### Claim 1: pseudo-gloss 是有信息的 privileged teacher condition
- Minimal experiment: frozen Qwen 的 instruction-only vs gloss-conditioned target NLL。
- Metric: token NLL/perplexity、teacher token accuracy、coverage/confidence。
- Decision: 若无显著改善，终止路线。

### Claim 2: generation behavior matching 优于 embedding alignment
- Minimal experiment: CE-only、同 pseudo-gloss static alignment、现有 InfoNCE、GPCD，统一 adapter/steps。
- Metric: BLEU-4、ROUGE-L、validation CE、aux–CE gradient cosine；最终候选 3 seeds。
- Expected evidence: GPCD 稳定优于同监督 baselines，且 gradient conflict 更低。

### Claim 3: 增益仍依赖视频证据
- Minimal experiment: temporal shuffle/frame masking sensitivity。
- Metric: clean-to-corrupted ΔBLEU/ΔNLL。
- Expected evidence: GPCD 不比 CE-only 更不敏感。

## Experiment Handoff Inputs
- Must-prove: gloss teacher 有效；GPCD 胜过同监督 embedding alignment；模型未忽略视频。
- Must-run: teacher sanity、四臂对照、shuffle sensitivity。
- Critical metrics: PHOENIX14T BLEU-4/ROUGE-L/NLL；最终 3 seeds。
- Highest risks: teacher 太弱；pseudo-gloss 噪声；cached top-k KL 偏差；KL 与 CE 冲突。

## Compute & Timeline Estimate
- GPU-hours: teacher sanity <0.1 次训练；四臂筛选约 4 次 1.7B 训练；2 个候选补 3 seeds 后总约 8 次训练。
- Annotation: 无新增标注；复用现有 pseudo-gloss。
- Timeline: 1 天实现/验证 loss，2–4 天筛选，随后补 seeds。
