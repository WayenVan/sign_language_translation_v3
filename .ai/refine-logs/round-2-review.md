# Round 2 Review

<details>
<summary>GPT-5.6-Sol 原始审查</summary>

CALIBRATION: none

## Round 2 总评

方案已从混合 hidden-state/OT calibration 收敛为单一 gloss-privileged response distillation；teacher/student 在相同 target positions 对齐，贡献更聚焦。主要缺口是完整 next-token KL 没有分离 pseudo-gloss source information 与 target-prefix language information，置信权重还可能偏向最容易由 target prefix 预测的 token。

| 维度 | 分数 |
|---|---:|
| Problem Fidelity | 8/10 |
| Method Specificity | 7/10 |
| Contribution Quality | 7/10 |
| Frontier Leverage | 8/10 |
| Feasibility | 8/10 |
| Validation Focus | 7/10 |
| Venue Readiness | 6/10 |

加权分数 7.35，**OVERALL SCORE: 7.4/10**。

**GAP:** 需要 source-less teacher 对照并限制 KL 梯度只能塑造视觉接口，否则 conditional distillation 不可唯一归因。

## 核心剩余问题

1. **CRITICAL — source-conditioned information 未分离。** 定义 `q0=Q(y_t|empty,y_<t)` 与 `qg=Q(y_t|G,y_<t)`；用 `JS(qg,q0)` 或 `ReLU[log qg(y_t)-log q0(y_t)]` 形成 source-influence weight，并加入 instruction-only KD 对照。
2. **CRITICAL — KL 可能绕过 adapter。** 明确 `∇LoRA L_GPCD=0`，KL 只更新 video adapter；首个机制实验可冻结 student Qwen。
3. **IMPORTANT — anchor tension。** 核心成功标准应改为 translation quality/source sensitivity/conditional-logit matching；内部统计只作诊断。
4. **MINOR — top-k tail 未精确定义。** 首轮在线 full-vocab KL；有效后才工程优化。

## Simplification Opportunities

移除 optional calibrator；首轮冻结 student Qwen；删除多种 confidence gate，只保留 source-influence weight。

## Modernization Opportunities

升级为 source-influence-aware conditional distillation；若 weighting 失败才考虑直接蒸馏 logit residual，不能并列堆叠。

## Validation Focus

Teacher sanity（q0/qg NLL与JS）；CE-only、instruction-only KD、InfoNCE、GPCD；clean、temporal shuffle、video-zero/masked。

## Novelty / Venue 风险

需对比 cross-modal KD、LUPI、text-teacher/video-student response distillation、MLLM projector distillation。普通 gloss→video KD 可能只是领域迁移。

## Drift Warning

PARTIAL, non-blocking at task level：底层任务保留，但方法已从 input manifold calibration 转为 privileged generation distillation。

## Verdict

**REVISE**

</details>
