# Review Summary

**Problem**: 视觉 adapter 后是否应接 Qwen embedding，以及为何 contrastive learning 掉点。
**Initial Approach**: 外部 embedding model 后再输入 Qwen LLM。
**Date**: 2026-08-14
**Rounds**: 5 / 5
**Final Score**: 9.0 / 10
**Final Verdict**: READY

## Problem Anchor
目标始终是提升生成式 SLT，让 Qwen 消费保留细粒度与时序的视频条件，而非提升 retrieval embedding 相似度。

## Round-by-Round Resolution Log
| Round | Main Reviewer Concerns | Simplified / Modernized | Result |
|---|---|---|---|
| 1 | input calibration 与 gloss distillation 混杂 | 删除 early-state/OT，改 target-logit KD | partial |
| 2 | KD 主要复制 target-prefix prior | 加 empty-source counterfactual、adapter-only gradients | partial |
| 3 | JS 无方向、动态归一化抵消低 influence | gold-direction gate + fixed valid-token denominator | solved |
| 4 | closest-work 与复现边界不足 | 正式定位类别差异，固定 tau/prompt contract | solved |
| 5 | 最终审查 | 无新增模块 | READY |

## Final Status
- Anchor: preserved
- Focus: tight
- Modernity: appropriately frontier-aware
- Strongest part: 直接蒸馏 source-induced generation behavior，无 negatives、无跨长度对齐、推理零开销。
- Remaining risk: pseudo-gloss teacher 可能无效；实际 novelty 与收益需实验和正式检索确认。
