# Target-language 训练配置结构

入口：`scripts/ph14t/run_cognition_target_language.sh`，通过 `accelerate launch -m csi_slt.commands.train` 启动，默认目标语言为 `en`，支持 `de/en/zh`、`debug`、`share`。

## 组合顺序

1. `configs/train/pretrain_adapter/base.yaml`：定义 model/data/datamodule/prompt/peft 配置组，以及 engine 的训练、优化和生成设置。
2. `configs/train/pretrain_adapter/baseline_ablation.yaml`：继承 base，将数据改为单语言、prompt 改为 fixed，设置时间压缩和日志频率。
3. 脚本的 `model=...`：选择 `qwen3-4b-cradio-l-spatiotemporal-next-frame-handroi-cls-20m` 模型组，替换 baseline 默认的 pooled-linear 模型。
4. 脚本的点路径参数：最终覆盖目标语言、视觉层、scorer、dropout、位置编码、训练轮数和输出目录等。

Hydra defaults 中的 `_self_` 表示当前文件内容在其 defaults 后合并；命令行点路径覆盖最后生效。只读模型 YAML 不能代表一次运行的最终配置。

## 原脚本的最终设置

| 层次 | 配置与作用 |
| --- | --- |
| model | Qwen3-4B（hidden size 2560）+ C-RADIOv4-SO400M（patch 1152、CLS 2304） |
| visual backbone | output_layer=-8，与 `outputs/hand_patch_scorer_L8` 配对 |
| visual adapter | next-frame patch fusion + global / hand-ROI / CLS 三路门控投影；每两帧一个 token |
| regularization | 三路 projection dropout=0.5；learned visual position embedding |
| objectives | LM loss + CTC（weight=1，vocab=2417，blank=2） |
| data | `ph14t_*x224x224_qwen_single_language`；224×224；processor token scale=0.5，额外边界 token=2 |
| prompt | `fixed_prompt`，各 split 根据目标语言选 canonical prompt |
| datamodule | `standard`，train_with_val=false |
| peft | `none`，没有 LoRA |
| trainability | LLM / 视觉骨干冻结；adapter、CTC head、visual position embedding、边界 embedding、visual scale 训练；scorer 冻结 |
| optimization | 默认 LR=1e-4、weight decay=0.05；adapter gates LR=1e-3；cosine，warmup=0.03 |
| schedule | 80 epochs；每卡 train batch=2、eval batch=1；eval_steps=6000、logging_steps=15 |
| runtime | 2 processes，LLM bfloat16，bf16 mixed precision；原脚本申请两张 L40S |
| checkpoint | 按 eval_overall_weighted_bleu4 保存最佳模型，训练末加载最佳模型，再 predict |

注意：当前 `commands/train.py` 把 `datamodule.test_dataset` 传给 Trainer 的 `eval_dataset`，训练中评估的 split 应以这段实际接线为准，不能仅凭 `val` 配置键判断。

## Qwen3-32B dense 扩展约束

官方配置：https://huggingface.co/Qwen/Qwen3-32B/blob/main/config.json 。模型 ID 为 `Qwen/Qwen3-32B`，架构 `Qwen3ForCausalLM`，hidden size=5120；模型的 hidden_size 与 adapter output_dim 必须同时改为 5120。

仓库已有的 `qwen3-32b-cradio-l-dinoframecrossv28shuffle.yaml` 使用不同 adapter、token scale 和注意力设置，不适合作为本次实验的直接替代。

原三路 rank=2349/1265/620。保持这些 rank，将输出从 2560 加宽至 5120，adapter 参数量从 20,000,781 增至 30,847,501（不含冻结 scorer），与现有 14B 同结构配置一致。这里的 adapter 预算不包含 CTC head 和位置 embedding 等其他训练组件。已确认保持与 14B 一致的 rank=2349/1265/620，包含 CLS bottleneck，不进一步扩容。

32B 的 bf16 权重本身约 64GB，原两卡 L40S 普通数据并行不能直接沿用。仓库已有 `configs/accelerate/fsdp2.yaml`，支持 Qwen3DecoderLayer 分片及 activation checkpointing；新脚本可复用。实际训练显存、吞吐和 checkpoint 保存仍须 GPU 实跑验证。

## 新增的 32B 运行入口

- 模型：`configs/model/qwen3-32b-cradio-l-spatiotemporal-next-frame-handroi-cls-31m.yaml`。与 14B 配置的有效字段仅模型 ID 不同。
- 脚本：`scripts/ph14t/run_cognition_scale_max.sh`。默认通过 best_adapter_multilang 使用 de/en/zh 联合训练及 diverse_train prompt（验证/测试使用 canonical prompt）；多语言默认 25 epochs、每 12000 steps 评估；单语言默认 80 epochs、每 6000 steps 评估；可传正整数覆盖轮数，保留原优化、正则设置。
- 资源：两张 H100、256GB CPU RAM、60 小时，显式使用 `configs/accelerate/fsdp2.yaml`；FSDP2 负责 decoder 分片；当前关闭 activation checkpointing 和 forward 后 reshard，以更多显存换取速度。
- 启动：`sbatch scripts/ph14t/run_cognition_scale_max.sh share`；不传参数默认多语言 + diverse。可用 `de fixed` 切换单语言固定 prompt，或 `multi fixed` 切换多语言固定 prompt。`debug` 仅关闭 WandB 并更换输出目录，仍会完整训练。
- 新输出目录和 WandB 标签包含 32B / scale-max 标识。
