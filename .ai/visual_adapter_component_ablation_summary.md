# 视觉 adapter 部件消融：表 B 总结（12 行 / 12 假设）

日期：2026-09-05

## 有效的（🟢）

| 改动 | 增益 | 备注 |
|---|---|---|
| **NextFrame Fusion**（行 3）| eval +2.95 | 运动残差，最强单模块 |
| **MotionTemporal Fusion**（行 1）| eval +2.3 | 另一种运动建模，与上者**二选一**（行 4 证明叠加冗余）|
| **激进 gate 初始化**（行 7）| +0.55 | 模型确实要运动信息，提前给权重 |
| **project dropout 0.5**（行 9）| gap 37.6→15.1，eval +0.8 | **唯一同时降过拟合 + 涨 eval 的操作** |
| **hand-ROI + NextFrame 结合**（行 10）| eval 12.3（全表最高）| 功能互补，用门控残差和注入 |
| CLS token（行 2）| +0.8 | 几乎免费，略优于 pooled patch |

## 无效的（🔴 / 🟡）

- 行 4：NextFrame + MotionTemporal 叠加 → 冗余掉点
- 行 5/6：hand-ROI 单独 → 只 +0.7（弱）
- 行 8：displacement Kaiming init → 无明显收益
- 行 11：spatial patch dropout → gap 不动（过拟合在 projection 层，不在 patch）
- 行 12：mean → conv → eval −0.3，过拟合最重（均匀 mean 本身是有用正则）

## 推荐最终结构

```
视觉特征：CLS token
  + NextFrame Fusion（fusion_gate 初值 +1.0）
  + hand-ROI 分支（门控残差和注入）
  + projection dropout 0.5
去掉：MotionTemporal Fusion / spatial dropout / mean→conv / displacement Kaiming
```

**下一个该跑的**：行 10 + project dropout（= 行 9 的 dropout 套到行 10 上）。行 10 eval 12.3
但 gap 49.7，加上 project dropout 大概率能把 gap 压下来同时保住 eval——这基本就是最终结构的
验证。
