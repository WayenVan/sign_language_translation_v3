# D-SID 讨论说明

## 1. 当前训练设定

训练中有三条 teacher-forcing 路径。三条路径使用相同的目标翻译 `label`，区别只在于 source。

### 视频学生路径

```text
<prompt> + <video> + <label prefix> → 预测下一个 label token
```

记该路径在目标位置 $t$ 的完整词表概率分布为：

$$
p_v^{(t)}
$$

这是最终部署时使用的路径。

### 伪标签教师路径

```text
<prompt> + <pseudo label> + <label prefix> → 预测下一个 label token
```

记输出分布为：

$$
q_g^{(t)}
$$

当前 pseudo label 是从 gold label 中提取的名词和动词。更准确的名称是：

> label-derived privileged keywords（从目标标签派生的训练期特权关键词）

它不是真正的手语 gloss。

### Empty-source 教师路径

```text
<prompt> + <empty source> + <label prefix> → 预测下一个 label token
```

记输出分布为：

$$
q_0^{(t)}
$$

Empty source 的构造方式是：保留 prompt、role、分隔符、assistant prefix 和 label prefix，只把原来放 video 或 pseudo label 的 source span 设为空。

```text
视频：prompt_before + VIDEO          + prompt_after + label
教师：prompt_before + PSEUDO_LABEL   + prompt_after + label
空源：prompt_before + EMPTY_STRING   + prompt_after + label
```

Empty source 不是：

- 删除整个 prompt；
- 使用全零 video features；
- 放入未经训练的 `<no-source>` 特殊 token；
- 改变 user/assistant role 或 instruction。

## 2. 三个输出如何比较

### 第一步：$q_g$ 与 $q_0$ 决定“教不教”

比较伪标签教师和空源教师，判断伪标签在目标位置 $t$ 是否提供了新的、正确的 source 信息。

如果伪标签使完整预测分布发生明显变化，并提高正确 token $y_t$ 的概率，则该位置值得蒸馏：

$$
q_g^{(t)}(y_t) > q_0^{(t)}(y_t)
$$

如果二者几乎相同，说明 Qwen 仅凭 label prefix 就能预测该词，伪标签没有提供多少额外信息。

如果伪标签降低正确词概率，说明它在该位置有误导作用，该位置不蒸馏。

因此：

```text
qg vs q0 → 判断伪标签是否在该位置提供有效 source information
         → 计算蒸馏权重 wt
```

### 第二步：$q_g$ 与 $p_v$ 决定“教什么”

在伪标签确实有帮助的位置，让视频学生的完整词表分布接近伪标签教师：

$$
\operatorname{KL}\!\left(q_g^{(t)}\,\|\,p_v^{(t)}\right)
$$

这里比较的是两条路径对下一个 label token 的完整概率分布，不是：

- video embedding 与 pseudo-label embedding；
- 两个序列的 source tokens；
- 两条路径的总 loss；
- 两条序列的绝对位置。

虽然 video、pseudo label 和 empty source 的长度不同，但都可以根据 label span 内部的第 $t$ 个位置对齐。

### 第三步：$p_v$ 与 gold label 保证“最终答对”

视频路径继续使用正常 teacher-forcing cross-entropy：

$$
\mathcal{L}_{\mathrm{CE}}
=
-\frac{1}{N_{\mathrm{valid}}}
\sum_t m_t \log p_v^{(t)}(y_t)
$$

它保证最终训练目标仍然是正确翻译，而不是无条件模仿教师。

最短总结：

```text
qg vs q0：决定“教不教”
qg vs pv：决定“教什么”
pv vs label：保证“最终答对”
```

## 3. D-SID 权重与总损失

用 $q_g$ 和 $q_0$ 的 Jensen–Shannon divergence 衡量伪标签造成的分布变化幅度，并用正确 token 的概率变化判断方向：

$$
w_t
=
\operatorname{stopgrad}\!\left[
\min\!\left(
\frac{\operatorname{JS}\!\left(q_g^{(t)},q_0^{(t)}\right)}{\tau},
1
\right)
\mathbf{1}\!\left\{
\log q_g^{(t)}(y_t) > \log q_0^{(t)}(y_t)
\right\}
\right]
$$

D-SID loss 为：

$$
\mathcal{L}_{\mathrm{D\text{-}SID}}
=
\frac{1}{N_{\mathrm{valid}}}
\sum_t m_t w_t
\operatorname{KL}\!\left(q_g^{(t)}\,\|\,p_v^{(t)}\right)
$$

最终损失为：

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{CE}}
+
\lambda\mathcal{L}_{\mathrm{D\text{-}SID}}
$$

D-SID 梯度只更新 video adapter；核心机制实验中冻结 student Qwen，避免模型通过修改语言参数、不看视频也能模仿教师。

## 4. D-SID 实际在判断什么

D-SID 的直观目标是：

> 通过伪标签判断 label 中哪些生成决策应获得 source 支持，再要求视频 adapter 为这些决策提供相似的信息。

更严谨地说，它不能证明某个词在语言学意义上一定来自视频。它测量的是：

> 对当前 Qwen 而言，加入伪标签后，目标 token 的预测是否得到改善。

因此，其逻辑链条为：

```text
从 label 提取视频相关关键词
          ↓
比较 qg 与 q0
          ↓
定位伪标签真正改善预测的 label positions
          ↓
把这些 positions 视为需要 source evidence 的生成决策
          ↓
让视频输出 pv 在这些位置模仿 qg
          ↓
迫使 video adapter 学会提供相关信息
```

## 5. 当前伪标签能否使用

从 label 中提取名词和动词可以作为第一版实验，但必须明确：

- 它属于训练期 privileged supervision；
- 它来源于 gold translation；
- 测试和实际推理时不能使用；
- 对照方法必须获得相同的关键词监督预算；
- 不能声称训练阶段完全不使用文本派生中间监督。

推理仍然是 gloss-free：

```text
video → adapter → Qwen → translation
```

伪标签教师和 empty-source 教师仅在训练时存在。

## 6. 伪标签需要满足的条件

### 正确

关键词必须与 label 表达的内容一致，最好也能从当前视频片段中观察到。错误实体、动作或语义会污染教师分布。

### 有额外信息

关键词需要帮助 Qwen 做出仅靠 label prefix 难以完成的预测。若 $q_g\approx q_0$，该伪标签对 D-SID 没有价值。

### 可从视频恢复

如果 label 中的信息来自新闻上下文或语言补全，而视频没有明确表达，就不适合拿来监督 video adapter。

### 不能丢失关键语义算子

只抽名词和动词可能导致严重问题，例如：

```text
label：明天不会下雨
pseudo label：明天 下雨
```

否定被删除后语义反转。因此建议至少保留：

- 名词和专有名词；
- 实义动词；
- 否定词；
- 时间和地点；
- 数字和数量；
- 方向词；
- 情态词，例如“可能、必须、应该”；
- 疑问词。

更合适的定义是：

> 内容词 + 关键语义算子

### 不应近似复制完整答案

如果关键词覆盖大部分 label 并保留原始顺序，教师可能直接获得答案骨架。建议区分：

- 固定无序关键词：主实验，主要提供内容；
- 保留原顺序关键词：上界或消融，同时包含部分词序信息。

若使用随机顺序，应对每个样本固定随机种子，不能每个 epoch 改变。

## 7. 训练前的必要检查

### Teacher sanity

比较伪标签教师与空源教师的目标 NLL：

$$
\Delta\operatorname{NLL}
=
\operatorname{NLL}(q_0)
-
\operatorname{NLL}(q_g)
$$

- $\Delta\operatorname{NLL}>0$：伪标签整体有帮助；
- 接近 0：信息量很低；
- $<0$：伪标签整体有误导作用。

### Direction-gate coverage

$$
\operatorname{coverage}
=
\frac{
\#\left\{t:q_g^{(t)}(y_t)>q_0^{(t)}(y_t)\right\}
}{N_{\mathrm{valid}}}
$$

需要检查高权重位置是否主要落在实体、动作、否定、时间和数量等合理 token 上。

### 数据质量统计

- 空伪标签比例；
- 每条样本的关键词数量；
- 关键词占 label 的比例；
- 名词、动词、否定、数字、时间的覆盖率；
- 高频词和低信息泛化词分布；
- tokenizer 切分情况；
- 随机人工抽查 100–200 条。

## 8. 最小实验对照

1. CE-only：不使用伪标签。
2. 当前 InfoNCE：使用相同伪标签监督预算。
3. Unweighted KD：所有 label positions 都蒸馏 $q_g\rightarrow p_v$。
4. Pure-JS SID：只按 $q_g/q_0$ 的变化幅度加权，不做正确方向过滤。
5. D-SID：JS 幅度 + gold-direction gate。
6. 打乱伪标签诊断：在 batch 内把关键词换给其他视频；如果仍然提升，可能主要是语言正则，而不是 source-information transfer。

还可以分别比较：

- only nouns；
- only verbs；
- nouns + verbs；
- content words + semantic operators。

## 9. 最终一句话

> D-SID 先用“有伪标签”和“无 source”的 Qwen 输出差异，找出伪标签真正帮助预测的目标词，再让视频分支只在这些位置学习伪标签教师的生成判断。
