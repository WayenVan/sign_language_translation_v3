# GRPO 首版 run 把模型跑差的诊断

日期：2026-09-08

背景：更早以前用 `scripts/run_tubbs_grpo.sh` + `configs/train/grpo.yaml` 跑过一次
GRPO，起点是 SFT 好的 SLT checkpoint，结果模型在真实指标上明显**下降**。当时的疑问是
「是不是 `do_sample` 对小数据集（PHOENIX14T，~7k）本身有问题」。本文档结论：**采样是放大器，
不是根因**；真正的原因是四个配置，其中第一个在当前这版 GRPO 代码里是硬写死的。

## 结论先行

1. **`beta: 0.0`——没有 KL 约束，而且 `trainer.py:169-170` 强制必须为 0。** 这是模型退化的
   首要原因。没有 KL 项把 policy 拽回 SFT 分布，配一个有噪声的 sentence-BLEU reward，
   被训的 visual adapter 就朝噪声 advantage 自由漂移，小数据集上几乎必然坍缩。
2. **`num_generations: 2`。** GRPO advantage = `(r − group_mean) / (group_std + 1e-4)`
   （`trainer.py:471-475`）。组内只有 2 个样本时，advantage 退化成一个 **±1 的符号位**，
   跟「好多少」无关，方差极大；2 个点算 std 不是有意义的归一化。正常要 8–64。
3. **sentence BLEU 当 reward（`reward.py`）。** 8–15 词的德语天气句，句级 BLEU-4 由少数
   4-gram 命中主导，稀疏、高方差，且可往通用高重合句上 hack。同一 prompt 的 2 个样本常拿到
   接近的低分，advantage 就是零附近的噪声。
4. **`metric_for_best_model: eval_reward`。** 按 reward 最高挑 checkpoint。reward 只要略可
   hack，选出来的就是真实指标上漂得最狠的那个。应按 held-out 上 **greedy 解码的 BLEU** 选。

次要项：起点是老的 v3.0 qwen3-1.7b checkpoint；`temperature: 0.9` 偏高。

---

## 1. 复现该 run 的配置

| 项 | 值 | 出处 |
| --- | --- | --- |
| 启动脚本 | `scripts/run_tubbs_grpo.sh` | — |
| 配置 | `configs/train/grpo.yaml` | — |
| 起点 checkpoint | `outputs/v3.0-qwen3-1.7b-cradio-l-dinoframecrossv3-0815-224x224/checkpoint-24000` | `grpo.yaml` `model.checkpoint_dir` |
| 被训参数 | **只有 visual adapter**（peft=none，不加 LoRA） | `train_grpo.py:58`, `trainer.py` docstring |
| `num_generations` | 2（train 和 eval 都是 2） | `grpo.yaml` |
| `beta`（KL 系数） | 0.0，且硬性要求 | `grpo.yaml`；`trainer.py:169-170` `raise ValueError("The first SLT GRPO version requires beta=0.0")` |
| advantage 归一化 | `(r − grouped_mean) / (grouped_std + 1e-4)`，`std(dim=1, unbiased=False)` | `trainer.py:471-475` |
| reward | 平滑句级 BLEU，`sacrebleu` `sentence_score`，`smooth_method="exp"`，`effective_order=True`，德语 `13a` tokenizer，大小写敏感，分数 `/100` 裁剪到 `[0,1]` | `src/csi_slt/engine/grpo/reward.py` |
| 采样 | `do_sample: true, temperature: 0.9, top_p: 0.95, top_k: 50` | `grpo.yaml` `generation_kwargs` |
| `max_completion_length` | 64 | `grpo.yaml` |
| `num_iterations` / `steps_per_generation` | 1 / 1（纯 on-policy，无 PPO 式多轮更新） | `grpo.yaml` |
| lr / scheduler | `5.0e-6` / cosine，warmup 30，`max_steps: 4500` | `grpo.yaml` |
| checkpoint 选择 | `load_best_model_at_end: true`，`metric_for_best_model: eval_reward`，`greater_is_better: true` | `grpo.yaml` |
| `disable_dropout` | true（RL 常规做法，无问题） | `grpo.yaml` |

---

## 2. 四个点详解

### 2.1 `beta = 0`：没有缰绳（首要）

`src/csi_slt/engine/grpo/trainer.py:169-170`：

```python
if args is not None and getattr(args, "beta", 0.0) != 0.0:
    raise ValueError("The first SLT GRPO version requires beta=0.0")
```

标准 GRPO / RLHF 的 loss 里带一项 `β · KL(π_policy ‖ π_ref)`，`π_ref` 是冻结的 SFT 模型。
这一项的作用是**不让 policy 离 SFT 分布太远**。`β = 0` 时这项完全消失：

- reward 只要有偏差或可 hack，visual adapter 就会一路优化到 reward 的漏洞里；
- 小数据 + 句级 BLEU 的噪声让 advantage 频繁给错方向，没有 KL 把它拉回来，误差累积成坍缩；
- 表现就是「一次 GRPO 把 SFT 模型跑差了」。

**当前代码开不了 KL。要重新做 RL，第一步是给 `SltGRPOConfig` / `SltGRPOTrainer` 接回 KL 项**
（冻结 ref 模型、每步算 policy 对 ref 的 per-token KL、按 `β` 加进 loss），`β` 起手 0.02–0.05。

### 2.2 `num_generations = 2`：advantage 退化成符号位

`trainer.py:471-475`：

```python
grouped_std = grouped_rewards.std(dim=1, unbiased=False)
...
return (rewards - grouped_mean) / (grouped_std + 1e-4)
```

组内 2 个样本、reward 为 `r1, r2`：

- `mean = (r1+r2)/2`，`std = |r1−r2|/2`；
- 两个样本的 advantage = `±(r1−r2)/(|r1−r2|)` ≈ `±1`，**只保留了「哪个好」，丢掉了「好多少」**；
- `std` 是 2 个点估的，不是有意义的归一化量，`+1e-4` 在 `r1≈r2` 时还会让 advantage 爆。

结果是高方差的「1 样本 baseline REINFORCE」。GRPO 的组内 baseline 要能压方差，一般需要
**8–64** 个 generation。n=2 时这套机制基本不工作。

修：`num_generations` 提到 ≥ 8（受 batch / 显存约束，可配合减小 `per_device_train_batch_size`
或上梯度累积；注意 generation batch 必须能被 `num_generations` 整除）。

### 2.3 sentence BLEU 当 reward：信号太稀疏

`reward.py` 的 `SentenceBLEUReward` 在 8–15 词的德语天气句上：

- BLEU-4 由少数 4-gram 是否命中主导，句级方差很大；
- 对「通用高重合句」友好（`und nun die wettervorhersage` 这类模板句能白拿分）；
- 同一 prompt 的 2 个采样经常都是低分且接近，advantage ≈ 噪声。

修：换更平滑的 reward——`chrF++`（无额外模型依赖，字符级、稠密）或 `BLEURT` / `COMET` /
`BERTScore`（需模型，但梯度平滑），也可以 `chrF++ + BERTScore` 混合。保留 BLEU 只作为
**监控指标**，不作 reward。

### 2.4 `metric_for_best_model: eval_reward`：挑出最坏的那个

`load_best_model_at_end` 会回滚到 eval reward 最高的 checkpoint。若 reward 可 hack，
「reward 最高」往往正是在真实翻译质量上漂得最狠的一个——**选择机制本身在帮倒忙**。

修：`metric_for_best_model` 换成 held-out val 上 **greedy 解码**算的 `bleu`（或 chrF），
`eval_strategy=steps` 时确保 eval 走确定性解码而不是 `do_sample`。

---

## 3. 关于 `do_sample`（回答最初的疑问）

「`do_sample` 对小数据集有问题」——方向对，但归因偏了：

- **GRPO 去不掉采样。** on-policy RL 需要同一 prompt 的多样 completion 才能估 advantage；
  greedy 会给 N 个完全相同的样本，学习信号严格为 0。
- 真正的问题：视觉 grounding 弱时，`temp=0.9` 的采样波动主要落在**语言先验方向**，跟
  「手语有没有读对」无关。于是 2 个样本的 advantage 量的是 LM 噪声，不是视觉层面的进步。
- 修法不是改 greedy，而是：**样本多**（2.2）+ **加 KL**（2.1）+ **换平滑 reward**（2.3），
  外加 `temperature` 降到 ~0.7。这些到位后，采样是必要且没问题的。

---

## 4. 大方向

一次「把模型跑差」的 GRPO 是**配错了 RL** 的结果，不是 RL 在这里没用的证据。但要现实看待
上限：在 ~7k 样本、greedy BLEU-4 ≈ 0.14 的模型上，即使 RL 配对了也是收尾手段，量级约
**+1–3 BLEU**，不会质变，且小数据上 reward hacking 风险高。

**优先级**：先把 supervised 模型（尤其视觉 grounding，见
`.ai/cradio_pooling_handshape_diagnosis.md`）拉到能拉的上限，再考虑 RL。碰 RL 之前，
代码必须先补回 KL 项。

## 5. 重启 RL 前的最小改动清单

- [ ] `SltGRPOTrainer` / `SltGRPOConfig` 接回 KL 项，解除 `beta != 0` 的硬性拦截，`β` 起手 0.02–0.05
- [ ] `num_generations` ≥ 8
- [ ] reward 换 `chrF++` 或加 `BERTScore` 混合，BLEU 降级为监控指标
- [ ] `metric_for_best_model` 换成 held-out greedy BLEU/chrF；eval 走确定性解码
- [ ] `temperature` 0.9 → 0.7
- [ ] 起点换成当前最好的 v5.0 4b SFT checkpoint，不用老的 v3.0 1.7b
