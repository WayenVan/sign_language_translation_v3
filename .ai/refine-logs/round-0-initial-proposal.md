# 研究方案：面向生成的 LLM-Native Visual Token Calibration

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Technical Gap
当前代码把 adapter 输出投到 2048 维、加视觉位置编码并乘一个全局标量后，直接替换 `<video_soft_token>` 的 text embedding。维度兼容不等于分布和功能兼容：Qwen 的输入 embedding 是离散词元训练形成的流形，而视频 token 需要表达尚未词汇化的连续、局部、多对多证据。

把 adapter 输出再输入通用 Qwen embedding 模型通常不合适：句向量 embedding 模型主要为检索训练，常含深层双向编码和 pooling；它既不提供“连续向量作为输入后逐 token 输出”的天然接口，也会将序列压成全局语义，损失 SLT 生成所需的局部运动证据。若只是取 Qwen 的 token embedding matrix，当前代码已经在用同一个空间作为文本侧 anchor。

现有对比学习掉点并不反常。当前 global branch 对视频作 learned attention pooling、对 pseudo-gloss 作 raw token-embedding mean，然后以 batch/queue negatives 做对称 InfoNCE。它可能同时遭遇：(1) 相似句/重复 gloss 被当作负例；(2) word embedding 的均值不是 Qwen 的上下文化句义；(3) global invariance 与 token-level autoregressive generation 目标冲突；(4) queue 中旧文本特征与快速变化的视觉投影形成监督失配；(5) auxiliary 权重总和相对 CE 过强。2026 年 SCL-SLT 也专门指出随机 in-batch negatives 经常语义无效或错误。

## Method Thesis
- One-sentence thesis: 不引入外部 Qwen embedding 模型，而用 Qwen 自身早期层作为冻结的“可读性教师”，通过一个残差校准器和生成一致性蒸馏，让视觉 token 在不丢失时序细节的前提下变成 LLM-native 条件。
- Why this is the smallest adequate intervention: 只在现有 adapter 与 Qwen `inputs_embeds` 之间加入一个近似恒等的残差校准器；训练信号来自同一个生成模型，无需新的语义编码器或负样本。
- Why this route is timely in the foundation-model era: 将冻结 LLM 作为接口 teacher/critic，而不是用独立检索 embedding 空间规定视觉表示，直接优化 foundation model 的可消费性。

## Contribution Focus
- Dominant contribution: 生成目标一致的 LLM-native visual token calibration，替代全局对比对齐。
- Optional supporting contribution: 无负样本的 pseudo-gloss teacher-state distillation，仅作为校准器训练信号。
- Explicit non-contributions: 新视觉 backbone、新 tokenizer、新大规模预训练、新 RL 算法。

## Proposed Method
### Complexity Budget
- Frozen / reused backbone: DINOv2/C-RADIO、现有 temporal adapter、Qwen 主体；可保持已有 LoRA 设置。
- New trainable components: 一个共享的 gated residual calibrator（RMSNorm → bottleneck MLP → scalar gate）；无第二个 encoder。
- Tempting additions intentionally not used: Qwen embedding model、Q-Former、VQ codebook、额外 contrastive head、memory queue。

### System Overview
`video → frozen visual backbone → existing temporal adapter → position embedding → gated residual calibrator → Qwen inputs_embeds → translation CE`

训练时额外教师路径：
`pseudo-gloss text → frozen Qwen embedding + first K layers → contextual teacher states`

学生路径：
`calibrated visual tokens → same frozen first K Qwen layers → visual states`

只在训练时用软对齐得到 teacher target；推理时删除教师路径，主路径不增加独立模型。

### Core Mechanism
- Input / output: 输入 packed visual tokens `[ΣM, D]`，输出同形状 token；按视频 padding 后计算辅助损失。
- Architecture or policy: `z = x + sigmoid(g) * W2(SiLU(W1(RMSNorm(x))))`，`g` 初始化为负值使系统从近似 identity 开始；随后使用 Qwen 原生 input RMS/首 K 层处理 `z`。
- Training signal / loss: 主损失始终为 translation CE。辅助项不直接匹配 raw word embedding，而匹配冻结 Qwen 在包含 pseudo-gloss prompt 时的第 K 层 contextual states。用现有 semi-unbalanced OT/monotonic soft alignment 产生 stop-gradient transport plan，仅最小化 transported cosine/Huber state loss；不使用 batch negatives。总损失 `L = L_CE + λ(t)L_native`，λ 先 warm-up、峰值不超过使辅助梯度范数约为 CE adapter 梯度的 10–20%，后半程衰减到 0，让最终解由生成目标收口。
- Why this is the main novelty: 对齐目标从“视频与文本在检索空间接近”改为“视觉条件经 Qwen 早期计算后产生与语言条件相容的内部状态”，且保留逐 token 序列，不要求视觉 token 伪装成某个静态词向量。

### Optional Supporting Component
- Only include if truly necessary: 复用现有 NULL-aware OT，仅作为训练期 stop-gradient correspondence estimator；若消融无增益则删除，改用分段 mean/temporal interpolation。
- Input / output: Qwen early states 的视觉序列与 pseudo-gloss 序列，输出软 transport plan。
- Training signal / loss: cosine cost + NULL；不让 OT loss 本身反向优化视觉表示，只让 transport 后的 teacher target 监督 calibrator。
- Why it does not create contribution sprawl: 它不是第二贡献，只解决两序列长度不同的问题，并复用已有实现。

### Modern Primitive Usage
- Which primitive is used: frozen LLM internal states as a teacher/critic。
- Exact role: 定义“Qwen-readable”接口，而非生成标签或执行检索。
- Why more natural: 下游消费者就是 Qwen；用同一模型的 contextual computation作 teacher 比引入另一个 Qwen embedding 模型减少 domain/objective mismatch。

### Integration into Base Generator / Downstream Pipeline
校准器放在 `visual_position_embedding_forward` 之后、`visual_scale` 与 `inputs_embeds` 合并之前。第一阶段冻结 Qwen 和 backbone，仅训练 adapter+calibrator；第二阶段若当前最佳基线需要则启用相同 LoRA。推理只保留 adapter+calibrator+Qwen，序列长度不变。

### Training Plan
1. 先复现实证基线：CE-only、当前 InfoNCE、当前 InfoNCE+OT，固定 seed/data/decode，并记录每项 loss 和 adapter 梯度 cosine。
2. Native calibration：关闭 global contrastive queue；以 identity-gated calibrator 起步，λ 线性 warm-up 10%，保持 30–40%，随后 cosine decay 到 0。
3. 仅在验证集 BLEU/ROUGE 与 CE 同向时保留辅助项；若 native loss 下降而 BLEU 不升，优先减 λ/K，而非增加模块。

### Failure Modes and Diagnostics
- Teacher forcing leakage / pseudo-gloss 过强: 观察训练与推理差距；教师路径严格仅训练使用，并报告无 pseudo-gloss 推理。
- 状态匹配抹平视觉细节: 比较 temporal shuffle 前后输出与局部 token 方差；若 shuffle 敏感性下降，减小 λ 或只匹配少量 anchor positions。
- 辅助梯度与 CE 冲突: 记录 adapter 参数上两者 gradient cosine；持续为负时使用 λ 衰减或只在预热阶段蒸馏。
- 早期层选择不当: 只测 K∈{0,2,4}，不扩展为大规模层搜索；K=0 等价于输入空间校准。

### Novelty and Elegance Argument
SignLLM 使用离散 VQ token 与 OT，SAGE 强调 segment-aware tokenization/token alignment，GFSLT-VLP 与 SignCL 使用对比式预训练；这些路线证明语言先验重要，但不等于“越贴近句向量越好”。本方案的区别是让同一个 downstream Qwen 定义接口，并以 generation-compatible early-state distillation 替代 retrieval-compatible global contrast。论文主张限定为：对 SLT，LLM-native conditional-state calibration 比全局 embedding alignment 更符合生成目标。

## Claim-Driven Validation Sketch
### Claim 1: Qwen embedding 模型/全局对比不是当前瓶颈的合适修复，LLM-native calibration 更匹配生成
- Minimal experiment: 在相同 checkpoint/预算下比较 CE-only、现有 InfoNCE(+queue)、external embedding projection（仅作诊断）、native calibration。
- Baselines / ablations: calibrator-only（无蒸馏）、native loss-only、当前完整 auxiliary。
- Metric: PHOENIX14T BLEU-4/ROUGE-L；三 seed 均值与方差；验证 CE。
- Expected evidence: native calibration 相对 CE-only 稳定正增益，且优于或至少不出现 contrastive 的掉点。

### Claim 2: 收益来自“可读且保细节”的接口，而非额外参数
- Minimal experiment: 统计输入范数/协方差、Qwen 第 1/2/4 层激活稳定性、temporal shuffle 性能降幅、aux-vs-CE gradient cosine。
- Baselines / ablations: 参数量匹配的两层 MLP；去掉 gate；λ 不衰减。
- Metric: BLEU-4、shuffle ΔBLEU、梯度冲突比例、token effective rank。
- Expected evidence: 校准改善早期层稳定性但不降低顺序敏感性；恒定强辅助监督更易掉点。

## Experiment Handoff Inputs
- Must-prove claims: generation-aligned teacher 比 retrieval embedding/InfoNCE 更适合当前接口。
- Must-run ablations: CE-only；calibrator-only；native calibration；当前 contrastive；λ decay vs constant。
- Critical datasets / metrics: PHOENIX-2014T BLEU-4、ROUGE-L、validation CE，至少 3 seeds 用于最终结论。
- Highest-risk assumptions: pseudo-gloss contextual states确实是有用 teacher；已有 OT correspondence 足够稳定；当前掉点不是数据/解码配置差异。

## Compute & Timeline Estimate
- Estimated GPU-hours: 先用 1.7B 做 4 个单 seed 筛选，约等于 4 次当前训练成本；最终只对前 2 个配置补至 3 seeds，总预算约 8 次基线训练。
- Data / annotation cost: 无新增人工标注，复用 pseudo-gloss。
- Timeline: 1 天完成诊断与最小实现，2–4 天筛选，随后按单次训练时长补 seeds。
