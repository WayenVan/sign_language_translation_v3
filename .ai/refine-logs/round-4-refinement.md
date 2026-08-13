# Round 4 Refinement

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Anchor Check
- 原问题、生成目标与 source sensitivity 均保留；内部统计仍仅作诊断。
- 无 reviewer 建议造成 drift。

## Simplicity Check
- 唯一贡献仍为 D-SID objective；没有新增组件。
- 本轮只补复现规范和 related-work 边界。

## Changes Made
- Tau：仅训练集非 padding target positions、float32、student 训练前一次计算；P75<1e-8 则 teacher-invalid；跨 seeds/同 teacher checkpoint 固定，不按 BLEU 调整。
- Prompt：单元测试断言 qg/q0 token IDs 只在 source-content span 不同；empty span 不含空格/换行/default text。
- pure-JS：计入一个独立单-seed 消融，总筛选从 4 改为 5 runs。
- Closest work：完成类别级逐项定位，结论限定为“未发现相同组合”，不声称首个 counterfactual KD。

## Revised Proposal

# 研究方案：Direction-aware Source-Influence Distillation（D-SID）

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；优先只新增一个小型可训练接口；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 组以内。
- Success condition: 相同数据、decode 和训练预算下，主翻译指标稳定优于当前 CE-only 与当前 contrastive/OT 配置；同时视觉 token 的尺度/协方差和 Qwen 早期层响应更接近真实 text embedding 条件，但时序打乱敏感性不下降。

## Thesis and Method
D-SID 不将 adapter token 送入 Qwen embedding 模型。它让冻结 Qwen 分别读取 pseudo-gloss 与空 source，在相同 gold target prefix 下得到 `qg` 与 `q0`；只对 gloss 引起显著且提高 gold-token probability 的位置，把完整 `qg` 分布蒸馏给 video-conditioned student。

`p_v^t=Qθ(y_t|P,Z(V),y_<t)`；`q_g^t=Qθ0(y_t|P,E(G),y_<t)`；`q_0^t=Qθ0(y_t|P,empty_span,y_<t)`。

`w_t=stopgrad[min(JS(qg^t,q0^t)/τ,1)·1(log qg^t(y_t)>log q0^t(y_t))]`。

`L_DSID=(1/N_valid)Σ_t mask_t w_t KL(qg^t||pv^t)`；`L=L_CE+λ(s)L_DSID`，λ 后 30% 衰减。D-SID 只更新现有 video adapter；主机制实验冻结 student Qwen。teacher 永久 frozen/eval/LoRA-off。推理删除 teacher，零新增参数和开销。

### Reproduction Contract
- qg/q0 共享全部 prompt token IDs，唯一差别是 source-content span；q0 span 长度严格为 0。测试断言无空格、换行、默认 placeholder 或 role 差异。
- `τ=P75(JS(qg,q0))`，仅训练集有效 target positions，以 float32、固定 teacher checkpoint 在训练前一次计算。`τ<1e-8` 或 `NLL(qg)≥NLL(q0)` 则停止。τ 不按 seed/config/validation 重估。
- Gate 使用相同精度、相同 gold token index 的严格 `>`；不设 margin/soft gate。
- 首轮在线 full vocabulary；任何 top-k/cache 是后续工程优化，不能替代核心验证。

## Why Contrastive Can Hurt Here
当前实现对视频 attention-pool、对 pseudo-gloss raw embedding mean，再用 batch/queue InfoNCE。它把多对多、可能重复的翻译样本变成实例判别，且长期与 CE 共同更新 adapter。SCL-SLT 已指出随机 in-batch negatives 会把相似/相同语义当负例且大量 negatives 无效；这与用户观察的掉点一致。D-SID 无 negatives、无 source pooling、无跨长度匹配，并通过 adapter-only routing 防止绕过。

## Closest-work Positioning

| 类别 | 已有机制 | 与 D-SID 的关键差别 |
|---|---|---|
| LUPI / cross-modal KD | 训练期 privileged modality 教师，推理移除 | 通常直接蒸馏 embedding/logits，不用同一生成器的 empty-source 反事实扣除 target-prefix prior |
| Causal Distillation for LMs | 用 interchange interventions 蒸馏 teacher 的 causal computation | 蒸馏模型内部干预一致性，不是 conditional-vs-empty source divergence 的 target-position selection |
| Switch-KD / VLM logits KD | teacher/student logits 与视觉路径切换 | teacher/student 主要共享视觉输入或模型压缩目标；没有 gloss-vs-empty source influence 与 gold-direction gate |
| Counterfactual Distillation (EMNLP 2024) | 生成 label-changing counterfactual examples/CoT | counterfactual 位于数据生成，不是 source ablation 后的同位置 teacher divergence weighting |
| SignCL / GFSLT-VLP / SCL-SLT | representation density、VLP contrast、negative selection | 仍优化表示/negative geometry，不直接迁移 source-induced autoregressive decisions |
| SignLLM / SAGE | VQ+OT 或 segment/token alignment | 需要 source tokenization/alignment；D-SID 不改变 source 表示与推理结构 |

可主张的最窄 novelty：在 sign-to-LLM 中，把同一 frozen generator 的 gloss-conditioned 与 empty-source 分布差用于 target-position source attribution，再以 gold-direction gate 和 adapter-only gradients 蒸馏 source-induced generation behavior。现阶段不主张 D-SID 是一般意义上第一个 counterfactual 或 privileged KD。

## Failure Handling
- Direction gate coverage 过低、qg NLL 不优于 q0、τ 近零：停止路线。
- video-zero/masked 或 temporal shuffle 不造成显著退化：判定 student 绕过视频，不接受 BLEU 增益为机制证据。
- CE/D-SID adapter gradient 长期冲突：降低 λ/缩短蒸馏期，不添加模块。
- pseudo-gloss 来自 gold translation：明确作为 label-derived privileged supervision，所有核心 KD/contrastive 对照获得相同 gloss。

## Minimal Validation
1. Teacher/source attribution sanity：qg/q0 NLL、JS curve、gate coverage。
2. 相同训练预算的 CE-only、当前 InfoNCE、unweighted GPCD、pure-JS SID、D-SID；关键候选补 3 seeds，BLEU-4/ROUGE-L/NLL。
3. video-zero/masked + temporal shuffle，报告 ΔBLEU/ΔNLL 与 adapter CE/D-SID gradient cosine。

## Compute & Timeline
- 5 个单-seed 筛选（含 pure-JS 独立 run）；最佳 2 个补到 3 seeds，总约 9 次训练等价。
- 1–2 天实现/单测，2–5 天筛选，随后补 seeds；无新增标注。
