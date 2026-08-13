# Grouped Shuffle 视觉适配器效果诊断

日期：2026-08-13

## 结论先行

`dinoframe_adapter_cross_v2_grouped_shuffle.py` 相比旧的
`dinoframe_adapter_cross_v2_shuffle.py`，增加的主要是时序模块容量和固定局部窗口偏置，
但没有增加新的监督信息、没有保留更多空间细节，也没有减少送入 LLM 的 token 数量。
目前更像是**表示瓶颈、时序归纳偏置和多目标冲突**，而不是模型容量不足。

现有 checkpoint 在相近 epoch 下确实表现更差：

| 实验 | checkpoint / epoch | BLEU-4 | ROUGE-L |
| --- | ---: | ---: | ---: |
| 旧 Shuffle | step 32000 / 5.60 | 0.1496 | 0.3551 |
| 新 Grouped Shuffle | step 30000 / 5.64 | 0.1200 | 0.3142 |

但这不是严格的单变量比较：两次实验的 backbone 输出层、辅助 loss、代码版本和训练配置均有变化，
所以只能说“当前整套新方案在相近训练阶段落后”，暂时不能把差距单独归因于 grouped shuffle。
而且新实验曲线仍在上升，以上结果也不能证明它最终一定无法追平。

## 1. 模型究竟增大了多少

从两个 `model.safetensors` 统计：

| 组成 | 旧 Shuffle | 新 Grouped Shuffle |
| --- | ---: | ---: |
| LLM | 1,720,574,976 | 1,720,574,976 |
| Visual backbone | 431,237,240 | 431,237,240 |
| CrossV2 frame adapter | 13,951,617 | 13,951,617 |
| CLS temporal adapter | 33,579,009 | 50,364,417 |
| PATCH temporal adapter | 33,579,009 | 50,364,417 |
| 总参数量 | 2,233,974,525 | 2,268,598,014 |
| 估算可训练参数量 | 约 82.2M | 约 116.8M |

总参数只增加约 1.5%，但可训练参数增加约 42%，两个时序适配器自身增加约 50%。
数据中实际只有约 7,096 条独立视频，三种目标语言只是重复使用同一批视觉样本；
因此新增容量很容易用于拟合 signer、背景和常见句式，而不是学到更好的动作表征。

更关键的是，新旧模块每个视频最终都输出约 `T` 个视觉 token。
Grouped Shuffle 没有换来更短的 LLM 上下文或更低的生成成本。

## 2. 新旧模块的结构差异

相关实现：

- `src/csi_slt/modeling_slt/visual_adapters/dinoframe_adapter_cross_v2.py`
- `src/csi_slt/modeling_slt/visual_adapters/dinoframe_adapter_cross_v2_shuffle.py`
- `src/csi_slt/modeling_slt/visual_adapters/dinoframe_adapter_cross_v2_grouped_shuffle.py`
- `src/csi_slt/modeling_slt/visual_adapters/patch_shuffle.py`

旧模块以不重叠的两帧为一组，分别处理 CLS 和 pooled patch token。新模块默认使用
`window=3, stride=2` 的重叠窗口，并拼接三帧特征和两组帧差：

```text
[x0, x0, x1], [x1, x2, x3], [x3, x4, x4], ...
```

然后执行近似如下计算：

```text
base = Linear(LayerNorm(mean(window)))
motion = MLP(concat(frames, frame_deltas))
output = base + sigmoid(gate) * motion
```

### 2.1 时序建模发生了重复

底层 `DINOFrameAdapterCrossV2` 已经做过：

1. 当前帧和下一帧全部 patch 的余弦相似度匹配；
2. 计算对齐后的 next-current 差分；
3. 用带门控的 temporal MLP 写回特征；
4. 把每帧全部 patch attention-pool 成一个 token。

之后 Grouped Shuffle 又用三帧拼接和帧差做一轮运动建模。
两层模块在优化近似相同的信号，新增参数不一定带来互补信息，反而可能重复放大背景、衣服纹理、
摄像机噪声或错误 patch 匹配。

建议把“CrossV2 motion”和“后级 temporal grouping”看成两个竞争方案，先验证各自独立价值，
而不是默认二者叠加一定更强。

### 2.2 固定窗口与手语的可变时长不匹配

固定 `window=3, stride=2` 强制所有手势使用同样的局部时间尺度，而且窗口相位固定在
`0, 2, 4, ...`。一个符号可能很短，也可能包含准备、保持、转移等不同长度阶段；随机速度增强还会改变
语义边界相对固定窗口的位置。

首尾的复制 padding 还让边界帧被不均匀地重复使用。相比旧的两帧 shuffle，新模块虽然覆盖更宽，
但没有根据动作边界或手部运动自适应分段。

SAGE 的实验指出，uniform downsampling 和固定 sliding window 容易忽略不同 sign duration 与 signing
speed，并使用手部/姿态驱动的 segment-aware tokenizer 获得更好的压缩与翻译结果。

### 2.3 新模块并不是近似恒等初始化

Grouped Shuffle 的 motion gate 初始化为 `-2`，只会把 motion residual 初始缩小到约 0.12；
但 `base` 本身仍经过一个随机初始化的 `D -> D` 线性层。因此整个模块并非“先保留旧表示，再逐步学习修正”，
而是一开始就随机旋转已经训练良好的视觉特征。

更稳妥的形式是：

```text
output = center_or_mean(window) + sigmoid(gate) * low_rank_motion_residual
```

其中 residual 使用 `D -> 256/512 -> D`，或把 base projection 做成 identity/zero-residual 初始化。

### 2.4 在时序模块之前，空间信息已经被压缩

CrossV2 把一帧所有 patch attention-pool 成一个 token，随后 Grouped Shuffle 只能在全局 patch summary
上做时序混合。手型、手-脸关系、双手相对位置和面部非手部信息可能在此之前已经丢失。

SpaMo 的做法更接近“空间与运动解耦”：图像编码器负责 spatial cue，预训练视频编码器负责 motion cue，
随后再用轻量 connector 融合。其消融中不同图像/视频预训练组合差异很大，说明动作信息的质量和预训练来源
通常比在冻结图像特征后堆一个大 MLP 更重要。

## 3. 为什么当前对比 loss 可能没有帮助

当前的 global contrastive loss 大致是：

- 视觉端：对最终视觉 token 做 learned attention pooling；
- 文本端：对冻结 Qwen 的原始 input embeddings 直接 mean pooling；
- 训练 symmetric InfoNCE，并使用大小 1024 的文本 queue；
- 文本特征 detach，只推动视觉端追随文本空间。

### 3.1 对齐目标过于粗糙

整句平均向量会丢失词序和局部动作对应关系。对德语、英语、中文三语而言，原始 Qwen token embedding
的平均值也未必是稳定的跨语言句义空间。视觉表示可能被迫接近一个并不适合视觉对齐的代理目标。

SAGE 的结果显示 token-level cross-modal contrastive learning 优于普通的 global CLIP-style alignment；
但其 loss 权重也呈非单调关系，过强时 BLEU 会下降。这说明“对比学习无用”并不是最可靠的结论，
更可能是当前粒度、text representation、负样本定义或权重不合适。

### 3.2 小 batch 和重复语义会产生不稳定或错误负样本

当前每卡 batch size 为 2，两卡全局 batch 约为 4。虽然 queue 增加了文本负样本，PHOENIX14T 中天气句式
高度重复，语义相近但来源视频不同的句子仍会被当作负样本。semantic ID 只把同一个源视频的多语翻译识别为
positive，无法消除“不同视频、相同或近似含义”的 false negatives。

### 3.3 该 loss 并非没有作用，而可能是在强力优化错误代理任务

Grouped checkpoint 后期大致为：

```text
main loss          ~= 1.886
contrastive raw    ~= 2.521, weight 0.25
alignment loss     ~= 0.733, weight 1.0
pooling distill    ~= 0.058, weight 0.5
weighted auxiliary ~= 1.392
total              ~= 3.278
```

辅助目标约占总 loss 的 42%，不能视为“没有起作用”。更可能的情况是，它们显著改变了视觉表示，
但改变方向未必有利于生成 BLEU。训练后期还观察到约 49--54 的 gradient norm spike，值得进一步确认
是否来自辅助目标冲突。

## 4. OT alignment 也存在表示空间错配

当前 `minimal_null_ot_alignment.py` 在 `video_dim == text_dim == 2048` 时使用 Identity projector。
但维度相等并不意味着 C-RADIO/adapter 特征和 Qwen token embedding 位于同一个语义坐标系。

当前 OT、global contrastive、pooling distillation 和生成 CE 同时直接作用于同一批视觉 prompt token：

```text
视觉 token
  ├─ 适配 Qwen 生成所需的 soft-prompt 空间（CE）
  ├─ 接近原始 Qwen token embedding（OT）
  ├─ 匹配整句平均 embedding（InfoNCE）
  └─ 模仿 OT relevance（pooling distillation）
```

这些目标不必共享最优解。特别是严格 pseudo gloss 来自口语文本，可能遗漏真实 sign、添加未被表达的词，
顺序也不一定等于 sign order。TV regularization、重叠窗口和平滑 temporal adapter 又都倾向于平滑时间表示，
可能进一步抹掉符号边界。

SignLLM 中的 OT 是与学习得到的离散视觉 token/codebook 一起设计的，并非简单把最终连续 prompt
直接拉向 LLM embedding。当前实现缺少这样的中间对齐空间。

## 5. 当前实验还有几个重要混杂变量

### 5.1 新旧配置并不等价

旧 checkpoint 配置主要使用：

```text
backbone output_layer = -1
contrastive weight    = 1.0
无当前 OT alignment / pooling distillation
```

新 grouped 配置主要使用：

```text
backbone output_layers = [-1, -2, -3, -4]，归一化后等权平均
contrastive weight     = 0.25
alignment weight       = 1.0
pooling distill weight = 0.5
TV beta                = 2.0
```

两个 checkpoint 对应提交之间还有大量代码变化。因此当前差异至少同时包含 adapter、backbone 层融合、
loss 组合和训练代码变化。最后四层等权平均本身也可能冲淡不同层所保留的细节或语义信息，必须单独消融。

### 5.2 当前用 test set 做训练期评估和 best-model 选择

`src/csi_slt/commands/train.py` 当前把 `datamodule.test_dataset` 传给 Trainer 的 `eval_dataset`，
同时配置启用了 `load_best_model_at_end=True` 并以 BLEU 选择模型。

这意味着 test set 被反复查看并参与 checkpoint 选择，最终 test 分数不再是独立估计。
在判断几个百分点的模块差异之前，应先改为：

```text
训练期 / early stopping / best checkpoint：validation set
实验方案冻结后：test set 只评估一次
```

这是当前最高优先级的方法学问题。

### 5.3 仅增大模型与论文中的 scaling 不是同一件事

《Scaling Sign Language Translation》中的收益来自同时扩大预训练数据、模型和翻译方向；
SignCLIP 也使用了约 50 万 dictionary clips 做多语言预训练。当前只有约 7,096 条独立视觉序列，
因此不能期待单独扩大 adapter 就复现 data-model co-scaling 的收益。

## 6. 建议的最小实验路线

### 第一步：先建立可信基线

所有实验固定以下条件：

- 同一 commit；
- 同一数据划分和增强；
- 同一 backbone output layer，建议先固定为 `-1`；
- 同一学习率、batch、训练步数和评估频率；
- 只用 validation 选 checkpoint；
- 至少 3 个随机种子，报告均值和标准差。

先运行一个纯生成基线：

```text
CE only
contrastive = off
OT alignment = off
pooling distillation = off
```

否则无法知道辅助目标是在帮助，还是在掩盖 adapter 本身的效果。

### 第二步：做一个 2 x 2 的时序消融

| 实验 | CrossV2 patch motion | 后级 temporal module |
| --- | --- | --- |
| A | 开 | 简单 pair mean / pooling |
| B | 关 | Grouped Shuffle 或小型 TCN |
| C | 开 | Grouped Shuffle |
| D | 关 | 简单 pooling |

最关键的问题是：运动建模应该放在哪一层，而不是继续把两个更大的运动模块叠起来。
如果 A 与 B 均不弱于 C，就基本证明当前重复建模没有带来互补收益。

### 第三步：把 Grouped Shuffle 缩小并稳定初始化

优先尝试：

1. `base = center frame` 或 `mean(window)`，不使用随机 `D -> D` base projection；
2. motion residual 改成 `D -> 256/512 -> D`；
3. CLS/PATCH 共享 temporal module，通过 type embedding 区分；
4. residual 最后一层 zero-init，或门控从接近 0 开始；
5. 输出 token 数如果仍为 `T`，明确验证其收益，否则直接做真正的 temporal compression。

这会让 temporal 模块从约 100.7M 参数降到更合理的数量，并减少小数据过拟合风险。

### 第四步：逐个恢复辅助目标

推荐顺序：

1. 最佳 CE-only adapter；
2. 加一个独立 projection head 的 global contrastive；
3. 单独测试 token-level alignment；
4. 最后才测试两者同时启用。

不要让辅助 loss 直接约束生成 token 的全部维度。可以让 generation path 保持自由，只在单独的低维
projection head 上计算 alignment，然后比较：

- 原始 Qwen token embedding；
- 1--3 层浅层 contextualized text representation；
- global alignment 与 token-level alignment；
- 2k--4k step alignment warmup 后关闭/降低权重；
- 冻结 LLM 与小型 LoRA。

SpaMo 的消融中，加入 LLM LoRA 的 BLEU-4 从 19.67 提升到 24.32；这不能直接保证当前项目也获得
相同幅度，但说明完全冻结 LLM 可能限制其适应视觉 soft prompt 的新分布。建议在视觉表示稳定以后再测试小 LoRA，
不要与 adapter 架构变更同时进行。

## 7. 最有信息量的诊断指标

在继续训练大模型前，建议先补以下检查：

1. **真实视频对照**：用正常、帧打乱、全黑/均值视频分别推理；若 BLEU 下降很小，模型主要依赖语言先验。
2. **梯度余弦**：分别计算 CE 与 contrastive、OT 对 adapter 参数的 gradient cosine；长期为负表示目标冲突。
3. **门控幅度**：记录 CrossV2 gate 与 Grouped motion gate 的分布；若长期接近 0，新增分支没有被使用。
4. **注意力熵**：检查每帧 patch pooling 是否长期只看背景或躯干，而不是手、脸和双手关系。
5. **过拟合差距**：同时画 train CE、validation CE/BLEU；参数增加只改善 train 时就是典型容量过剩。
6. **边界与速度敏感性**：同一视频改变起始偏移一帧或轻微变速，检查 grouped window 输出及翻译是否剧烈变化。
7. **分布外划分**：补充 signer-disjoint / sentence-disjoint 诊断；PHOENIX14T 标准划分可能高估对 signer 和模板的泛化。

## 8. 建议的决策标准

短期内不建议继续升级更大的 adapter、LLM 或 hidden dimension。先满足以下标准再扩大规模：

- 单变量、三种子实验确认某个 temporal module 稳定提升 validation BLEU；
- 真实视频相对 shuffled/blank video 有显著优势；
- 辅助 loss 与 CE 的梯度大部分时间不冲突；
- 新模块要么明显提升质量，要么实际减少视觉 token / 显存 / 延迟；
- 最终 test set 未参与调参和 checkpoint 选择。

如果只能优先做一件事，建议先完成：

```text
修正 validation/test 流程
  -> 固定 output_layer=-1
  -> 跑 CE-only 的旧 Shuffle 与 Grouped Shuffle 三种子对照
  -> 再做 CrossV2 motion × 后级 temporal module 的 2x2 消融
```

这组实验会比继续添加 loss 或扩大模块更快地回答“性能瓶颈究竟在哪里”。

## 参考研究

- [SAGE: Segment-Aware Gloss-Free Sign Language Translation](https://arxiv.org/html/2507.09266)
- [SpaMo: Spatiotemporal Motioning for Sign Language Translation](https://aclanthology.org/2025.naacl-long.197/)
- [FLa-LLM: Fine-tuning Large Language Models for Gloss-free Sign Language Translation](https://aclanthology.org/2024.lrec-main.620/)
- [SignCL: Sign Language Translation with Contrastive Learning](https://papers.nips.cc/paper_files/paper/2024/hash/c225136cfe52a8fd66658bbcf9d894ab-Abstract-Conference.html)
- [SignCLIP: Connecting Text and Sign Language by Contrastive Learning](https://aclanthology.org/2024.emnlp-main.518/)
- [Scaling Sign Language Translation](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ced76a666704e381c3039871ffe558ee-Abstract-Conference.html)
- [SignLLM: Sign Languages Production and Translation with Large Language Models](https://openaccess.thecvf.com/content/CVPR2024/html/Gong_LLMs_are_Good_Sign_Language_Translators_CVPR_2024_paper.html)
- [Text CTC Alignment in Sign Language Translation](https://aclanthology.org/2025.coling-main.219/)
- [Rethinking Sign Language Translation: The Impact of Signer Dependence on Model Evaluation](https://aclanthology.org/2025.findings-emnlp.997/)

