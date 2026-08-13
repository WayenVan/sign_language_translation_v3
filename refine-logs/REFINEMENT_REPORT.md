# Refinement Report

**Problem**: Qwen embedding 是否能改善 video-adapter-to-LLM SLT，以及 contrastive loss 掉点原因。
**Initial Approach**: adapter → external Qwen embedding → Qwen LLM。
**Date**: 2026-08-14
**Rounds**: 5 / 5
**Final Score**: 9.0 / 10
**Final Verdict**: READY

## Outputs
- Final proposal: `refine-logs/FINAL_PROPOSAL.md`
- Review summary: `refine-logs/REVIEW_SUMMARY.md`
- Score history: `refine-logs/score-history.md`
- Raw/round records: `refine-logs/round-*-review.md`, `round-*-refinement.md`

## Final Thesis
- 不使用独立 Qwen embedding encoder。
- 用同一冻结 Qwen 的 gloss-vs-empty 分布差定位 source-dependent target decisions。
- 只蒸馏对 gold token 有益的 source influence，且梯度只进入 adapter。
- 若 teacher sanity、gate coverage 或 video-dependence 失败，停止而非堆模块。

## Score Evolution
| Round | Fidelity | Specificity | Contribution | Frontier | Feasibility | Validation | Venue | Overall | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 8 | 6 | 5 | 7 | 7 | 6 | 5 | 6.3 | REVISE |
| 2 | 8 | 7 | 7 | 8 | 8 | 7 | 6 | 7.4 | REVISE |
| 3 | 8 | 8 | 8 | 9 | 8 | 8 | 7 | 8.1 | REVISE |
| 4 | 9 | 9 | 8 | 9 | 9 | 9 | 8 | 8.7 | REVISE |
| 5 | 9 | 9.5 | 9 | 9 | 8.5 | 9 | 8 | 9.0 | READY |

## Remaining Weaknesses
双 teacher forward 有训练开销；qg/q0 仍含 source length 差；pseudo-gloss 可能低 coverage；顶会竞争力依赖三 seed 实际收益与 closest-work 复核。

## Next Steps
进入 claim-driven experiment planning：先实现 teacher sanity 与 prompt/gradient 单测，再决定是否值得完整训练。
