# Round 1 Review

<details>
<summary>GPT-5.6-Sol 原始审查</summary>

CALIBRATION: none

## 总体评价

方向是合理的：拒绝额外 Qwen embedding encoder、把下游 Qwen 本身作为接口参照，比继续强化全局 InfoNCE 更贴合当前问题。但方案尚未证明“冻结 Qwen 早期层状态蒸馏”是最小且正确的干预。

当前核心因果链条存在断点：诊断是 input distribution/interface mismatch，实际监督却是 pseudo-gloss semantic distillation + student-dependent OT。因此，即使实验提升，也无法区分收益来自输入分布校准、额外 pseudo-gloss 监督、contextual teacher、OT 或 gated MLP。

## 评分

| 维度 | 分数 |
|---|---:|
| Problem Fidelity | 8/10 |
| Method Specificity | 6/10 |
| Contribution Quality | 5/10 |
| Frontier Leverage | 7/10 |
| Feasibility | 7/10 |
| Validation Focus | 6/10 |
| Venue Readiness | 5/10 |

加权分数为 6.25，**OVERALL SCORE: 6.3/10**。

**GAP:** 提案已有不错的问题诊断和克制的模块预算，但缺少能被唯一归因的机制。必须先确定主张是“分布校准”还是“consumer-conditioned contextual distillation”。

## 主要方法问题与修复要求

- Method Specificity（CRITICAL）：teacher/student 序列、position、mask、LoRA 冻结方式未定义；student-dependent OT 有自适应逃逸风险。应永久冻结 teacher、相同 prefix、不含目标翻译，并首版删除 OT。
- Contribution Quality（CRITICAL）：early-state matching 不等于输入分布校准。只能选择真正的 moment calibration 或承认核心是 contextual privileged distillation。
- Validation Focus（IMPORTANT）：需要参数量/训练步数相同的 decisive controls，并先验证 gloss 条件确实降低目标 translation NLL。
- Venue Readiness（CRITICAL）：主张应压缩为“相同 pseudo-gloss 监督下，下游生成器定义的 contextual teacher 比静态/检索式对齐更少梯度冲突且翻译更好”。

## Target leakage / circularity

- 人工 gloss 不是测试泄漏，但属于额外监督。
- gold translation 生成的 pseudo-gloss 是 label-derived privileged supervision，必须披露并保证对照同监督预算。
- teacher prompt 包含 gold translation 是严重泄漏，应禁止。
- 固定 Qwen teacher states 不是严格循环；student-dependent OT 有弱循环与逃逸风险。

## Simplification Opportunities

1. 删除 OT，改固定 bins。
2. 先测 calibrator-only/moment calibration。
3. teacher 关闭 LoRA并缓存 states。

## Modernization Opportunities

1. 用 gloss-conditioned frozen Qwen 在相同 gold translation prefix 上产生 next-token logits，再做 KL；比 early-state 坐标更直接对应生成，也无需 OT。
2. 若 full logits 太贵，可使用固定 probe tokens。

## Drift Warning

NONE on the bottom-line task；但机制叙事从 distribution calibration 漂移到了 contextual semantic distillation。

## Verdict

**REVISE**

</details>
