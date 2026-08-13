# 研究方案：Direction-aware Source-Influence Distillation（D-SID）

## Problem Anchor
- Bottom-line problem: 在当前 DINO/C-RADIO → video adapter → Qwen3 LLM 的手语翻译系统中，提高 PHOENIX-2014T 等数据集上的最终翻译质量；核心不是让全局视频和文本向量更相似，而是让保留细粒度手形、运动和时序信息的视觉 token 成为 Qwen 可有效消费的条件。
- Must-solve bottleneck: 当前 adapter token 虽然维度等于 Qwen hidden size，却未必落在 Qwen 首层可稳定处理的输入分布和条件接口中；现有全局 video–pseudo-gloss InfoNCE 与 token-level OT 会把不同粒度、非单调且多对多的手语内容强行拉向静态 token embedding，已经观察到最终生成指标下降。
- Non-goals: 不把系统改造成检索模型；不以 embedding-space retrieval 指标替代翻译质量；不堆叠独立 Qwen embedding encoder、Q-Former、VQ tokenizer 和 RL；不声称已获得尚未运行的增益。
- Constraints: 复用冻结的视觉 backbone 与 Qwen3-1.7B/8B；沿用现有数据、pseudo-gloss 和训练框架；核心验证控制在 3 个 blocks。
- Success condition: 相同数据、decode 和训练预算下，BLEU-4/ROUGE-L 稳定优于 CE-only 与当前 contrastive/OT；模型对 video-zero/masked 与 temporal shuffle 保持显著敏感。

## Technical Gap
当前 InfoNCE 优化句级 retrieval similarity：视频被 attention-pool，pseudo-gloss 是 raw token-embedding mean，batch/queue 中相似或重复语义仍可能成为负例。该目标既丢时序又可能和 autoregressive CE 冲突。独立 Qwen embedding 模型同样由检索/排序目标塑造，通常更强调句级不变性与 pooling，不能直接解决“哪些下一词决策必须由视频证据改变”。

## Method Thesis
训练时让冻结 Qwen 分别读取 pseudo-gloss 与空 source，在相同 gold target prefix 下得到 $q_g$ 与 $q_0$；只在 gloss 引起显著变化且提高 gold-token probability 的位置，把完整 $q_g$ 条件分布蒸馏给 video-conditioned student。推理保持原 pipeline。

## Exact Method
对 target position $t$：

$$
p_v^{(t)} = Q_{\theta}\!\left(y_t \mid P, Z(V), y_{<t}\right)
$$

$$
q_g^{(t)} = Q_{\theta_0}\!\left(y_t \mid P, E(G), y_{<t}\right)
$$

$$
q_0^{(t)} = Q_{\theta_0}\!\left(y_t \mid P, \varnothing, y_{<t}\right)
$$

$\theta_0$ 永久 frozen/eval/LoRA-off。三路共享 delimiters、target tokenization 和 causal prefix；teacher 不含 future target。训练前固定：

$$
\tau = \operatorname{P}_{75}\!\left(
\left\{\operatorname{JS}\!\left(q_g^{(t)},q_0^{(t)}\right)
: t \in \mathcal{T}_{\mathrm{train}}^{\mathrm{valid}}\right\}
\right)
$$

$$
w_t = \operatorname{stopgrad}\!\left[
\min\!\left(
\frac{\operatorname{JS}\!\left(q_g^{(t)},q_0^{(t)}\right)}{\tau},
1
\right)
\cdot
\mathbf{1}\!\left\{
\log q_g^{(t)}(y_t) > \log q_0^{(t)}(y_t)
\right\}
\right]
$$

$$
\mathcal{L}_{\mathrm{D\text{-}SID}}
=
\frac{1}{N_{\mathrm{valid}}}
\sum_t m_t w_t\,
\operatorname{KL}\!\left(q_g^{(t)}\,\|\,p_v^{(t)}\right),
\qquad
N_{\mathrm{valid}} = \sum_t m_t
$$

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda(s)\,\mathcal{L}_{\mathrm{D\text{-}SID}}
$$

$\lambda(s)$ 先 warm-up，并在训练的最后 30% decay。$\mathcal{L}_{\mathrm{D\text{-}SID}}$ 只更新现有 video adapter；首个机制实验冻结 student Qwen。无新参数、无 OT、无 negatives、无 source pooling、推理零开销。

## Reproduction Contract
- $q_g/q_0$ token IDs 唯一差别必须是 source-content span；$q_0$ span 长度严格为 0，无空格、换行、placeholder 或 role 差异。
- $\tau$ 仅用训练集有效 target positions，以 float32 在 student 训练前一次计算；同 teacher checkpoint 跨 seed 固定，不按 BLEU 调参。
- $\tau < 10^{-8}$、$\operatorname{NLL}(q_g) \geq \operatorname{NLL}(q_0)$ 或 direction-gate coverage 过低时停止路线。
- 主验证用 online full-vocabulary logits；Top-k/cache 仅是后续工程优化。
- pseudo-gloss 若由 gold translation 生成，明确披露为 label-derived privileged supervision；核心对照使用相同 gloss 预算。

## Novelty Boundary
一般 LUPI、跨模态 KD、counterfactual/causal KD 都不是新点。D-SID 的窄贡献是在 sign-to-LLM 中结合：(1) 同一 frozen generator 的 gloss-conditioned vs empty-source divergence；(2) gold-direction target-position selection；(3) adapter-only gradient routing，以迁移 source-induced generation behavior。若正式查重发现相同 objective，则将其定位为 SLT 工程方法而非通用蒸馏新范式。

## Claim-Driven Validation
1. Teacher sanity：qg/q0 NLL、JS-position curve、gate coverage；不满足停止条件则不训练 student。
2. 同预算比较 CE-only、当前 InfoNCE、unweighted gloss KD、pure-JS SID、D-SID；候选补 3 seeds，指标 BLEU-4/ROUGE-L/NLL。
3. video-zero/masked 与 temporal shuffle；报告 ΔBLEU/ΔNLL、adapter 上 CE/D-SID gradient cosine。若 D-SID 更不依赖视频，则否定机制。

## Compute & Timeline
5 个单-seed 筛选，最佳 2 个补至 3 seeds，总约 9 次训练等价。预计 1–2 天实现与单测，2–5 天筛选，再补 seeds；无新增人工标注。
