# Round 3 Refinement

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Anchor Check
- Original bottleneck: 生成器未充分消费视觉 source evidence。
- Preserved: SID 直接把由 gloss source 改善的生成决策蒸馏给 video adapter。
- Drift rejected: internal-statistic closeness 降为解释性诊断；必要成功条件是翻译提升和 source sensitivity。

## Simplicity Check
- Dominant contribution: direction-aware SID objective。
- Removed: 所有新模块、OT、calibrator、multiple gates。
- Smallest adequate: 仅修改 loss 的权重与梯度路由。

## Changes Made
- 固定分母为 batch 内有效 target token 数，低 influence 样本产生绝对更弱梯度。
- 主权重固定为 `min(JS/τ,1) * 1[log qg(gold)>log q0(gold)]`；纯 JS 只作消融。
- `q0` 使用与 gloss prompt 完全相同的 source delimiters，中间 source span 长度为 0，不创建 special token。
- novelty 初查未发现相同的 conditional-vs-empty divergence token weighting 用于 SLT；但 counterfactual/causal KD 和跨模态 LUPI 广泛存在，最终只主张在 sign-to-LLM 接口中的特定机制。

## Revised Proposal

# 研究方案：Direction-aware Source-Influence Distillation（D-SID）

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Technical Gap and Thesis
Embedding/InfoNCE 对齐优化 retrieval similarity，而普通 gloss→video KD 又混入 target-prefix language prior。D-SID 用同一冻结 Qwen 的 empty-source 反事实，找出 gloss 不仅改变而且正确提高 gold-token 概率的位置，只在这些位置把 gloss-conditioned distribution 蒸馏给 video adapter。核心成功标准是 BLEU/ROUGE、conditional-logit matching 与 source sensitivity；内部统计仅作诊断。

## Contribution Focus and Complexity
- 唯一贡献：direction-aware source-influence weighted cross-modal response distillation。
- 冻结/复用：backbone、Qwen teacher、现有 adapter；首轮 student Qwen 也冻结。
- 新参数：0；推理新增开销：0。
- 明确排除：embedding encoder、contrastive queue、OT、Q-Former、calibrator、RL。

## Exact Method
对每个 target position t：

- `p_v^t = Q_θ(y_t | P, Z(V), y_<t)`；
- `q_g^t = Q_θ0(y_t | P, E(G), y_<t)`；
- `q_0^t = Q_θ0(y_t | P, empty_span, y_<t)`。

三路共享 source delimiters、target delimiter、target tokenization 与 causal target prefix；q0 的 delimiters 之间没有 token。`θ0` 永久 frozen/eval/LoRA-off，teacher 不含 future target。

训练前在 training teacher statistics 上固定 `τ=P75(JS(qg,q0))`。唯一主权重为：

`w_t = stopgrad[min(JS(q_g^t,q_0^t)/τ,1) · 1(log q_g^t(y_t)>log q_0^t(y_t))]`。

主损失：

`L_DSID = (1/N_valid) Σ_t mask_t w_t KL(q_g^t || p_v^t)`，其中 `N_valid=Σ_t mask_t` 是 batch 内固定有效 target token 数，不用 `Σw` 归一化。

`L=L_CE+λ(s)L_DSID`；λ warm-up，末 30% decay 到 0。`L_DSID` 只更新 video adapter：`∇LoRA L_DSID=0`；`L_CE` 按基线更新 adapter（后续兼容实验可含 LoRA）。主机制实验冻结 student Qwen，使用 online full-vocabulary logits。

若 direction gate 保留率过低、`NLL(qg)≥NLL(q0)` 或平均 JS 近零，则在训练 student 前判定 pseudo-gloss teacher 无效并停止路线。

## Why This Rather Than Qwen Embedding
Qwen embedding 模型输出为检索不变性服务，常压缩为句级表示；D-SID 不接触/压缩 source token 序列，直接监督“哪些下一词决策应因 source 改变”，与 SLT 的 autoregressive objective 同构。

## Failure Modes and Diagnostics
- Noisy gloss：direction gate 排除降低 gold probability 的位置；报告 gate coverage。
- Language-prior copying：instruction-only KD 与 unweighted GPCD 对照。
- Ignoring video：video-zero/masked 和 temporal-shuffle 测试；若 D-SID 比 CE-only 更不敏感则否定机制。
- Gradient conflict：报告 adapter 上 CE/D-SID gradient cosine 和 norm ratio；λ 初值令 ratio≈0.1。
- Functional vs manifold：输入范数/协方差只作解释，不作为成败条件。

## Novelty Boundary
普通 LUPI、跨模态 KD、counterfactual KD 与 causal KD 均不是新点。初步检索未发现把同一生成器的 conditional-vs-empty source divergence 与 gold-direction gate 结合、以 adapter-only gradients 训练 sign-to-LLM 的相同方法。最终主张只限定在 SLT/多模态生成接口；若后续查到同 objective，则将其作为强工程路线而非通用新范式。

## Claim-Driven Validation Sketch
### Claim 1: D-SID 迁移的是正确 source influence
- Teacher sanity：qg/q0 NLL、JS curve、direction-gate coverage。
- 对照：instruction-only KD、unweighted GPCD、pure-JS SID、D-SID。
- 判据：D-SID > unweighted/pure-JS；qg NLL < q0。

### Claim 2: D-SID 比 embedding alignment 更匹配生成且依赖视频
- 对照：CE-only、当前 InfoNCE、unweighted GPCD、D-SID，统一冻结 Qwen/steps/data。
- 指标：3-seed BLEU-4/ROUGE-L/NLL、gradient cosine、zero/masked/shuffle ΔBLEU。
- 判据：D-SID 稳定提升且破坏视频显著恶化。

## Experiment Handoff Inputs
- 必证：有效 qg teacher、direction/source weighting 必要、视频不可绕过。
- 核心实验：teacher sanity；四臂生成对照；source corruption。
- 风险：pseudo-gloss gate coverage 太低；增益仅来自 KD；conditional-null weighting 已有先例。

## Compute & Timeline
- 筛选配置：CE-only、InfoNCE、unweighted GPCD、D-SID，共 4 个；pure-JS 在同一 D-SID run/短程消融中验证。
- 预算：约 4 次单-seed 1.7B 筛选；最优 2 个补至 3 seeds，总约 8 次训练等价；双 teacher 无反向但增加 forward FLOPs。
- 时间：1–2 天实现，2–4 天筛选，再补 seeds。
