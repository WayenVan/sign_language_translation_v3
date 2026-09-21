# 基于 checkpoint 的 prompt-suite 评测脚本

日期：2026-09-17

> 存档：这套脚本的设计决策、用法、验证范围。原来这个文件里是实现前的方案讨论，
> 已被下面的实现记录取代（决策条目保留在「落定的决策」一节）。

## 结论先行

评测被拆成 7 个 suite（2026-09-20 加了 `cf_first` / `cf_last` 两个反事实 suite），
每个 suite = 一个输出目录 = n 个 prompt 变体各一次完整
predict，同一次 run 里三种语言用同一个 instruction 变体（这样
`mean ± std across prompt variants` 才是关于 prompt 措辞的量，而不是措辞和语言混在一起）。
所有脚本本地跑，没有 slurm。

## 文件清单

| 文件 | 作用 |
|---|---|
| `configs/prompt/eval_fixed.yaml` | 只含 `test` 的 `FixedPromptResolver`，bank 和三个 prompt id 由 launcher 覆盖 |
| `src/csi_slt/commands/list_prompt_variants.py` | 枚举 bank 里的变体，输出 `变体名<TAB>de_id<TAB>en_id<TAB>zh_id`，三语不对齐直接报错 |
| `src/csi_slt/commands/summarize_prompt_suite.py` | suite 汇总 → `summary.json` + `summary.md`（mean±std、min–max、missing、diagnostic 分区） |
| `src/csi_slt/commands/evaluate.py` | predict 后把 resolved cfg 写成 `eval_config.yaml`（`SaveHydraConfigCallback` 只在 `on_save` 触发，eval run 之前没有任何 provenance）；`engine.model_dtype: checkpoint` 分支 |
| `src/csi_slt/utils/checkpoint_dtypes.py` | 读 safetensors 头，把模型每个张量铸回 checkpoint 里存的 dtype（`tests/test_checkpoint_dtypes.py` 7 个用例） |
| `src/csi_slt/modeling_slt/misc.py` | `mark_adapter_modules_as_initialized()` / `mark_module_tree_as_initialized()` / `validate_rope_buffers()`——见下面的 RoPE bug（`tests/test_module_initialization_marking.py` 7 个用例） |
| `src/csi_slt/commands/train.py:38` | `cast_module_dtype` 不再铸非持久 buffer |
| `scripts/ph14t/eval/run_eval.sh` | worker：本地 `accelerate launch`，回放 ckpt 的 processor 设置，跳过已完成的 run |
| `scripts/ph14t/eval/lib_prompt_suite.sh` | 共用 suite 驱动：枚举变体 → 逐个跑 → 写 `variants.tsv` → 汇总 |
| `scripts/ph14t/eval/run_eval_{fixed,diverse,unseen,wrong_task,unrelated}_prompt.sh` | 5 个 suite 入口 |
| `scripts/ph14t/eval/sweep14b/run_multilang_diverse_sweep.sh` | 两个 14B 多语言 stage-2 run 各取 `eval_overall_macro_bleu4` 最高的 checkpoint 跑 diverse suite，输出到 `outputs/eval/14b-multilang-diverse-eval/` |
| `prompts/generic/counterfactual_{first,last}.jsonl` | 反事实 bank，各 6 条（family `cf_first` / `cf_last`），held-out，禁止进训练池 |
| `scripts/ph14t/eval/run_eval_cf_{first,last}_prompt.sh` | 两个反事实 suite 入口，各 2 个变体 |
| `src/csi_slt/commands/summarize_counterfactual.py` | 把两个反事实 suite 合成一份报告：逐条件 LAcc、BSA、目标语言混淆矩阵、主表那一行（`tests/test_summarize_counterfactual.py` 19 个用例） |
| `scripts/ph14t/eval/sweep{4b,14b}/run_multilang_counterfactual_sweep.sh` | 对 diverse sweep 已完成的每个 checkpoint 跑两个反事实 suite 并汇总 |

## 用法

```bash
bash scripts/ph14t/eval/run_eval_fixed_prompt.sh     <CKPT>        # multi，canonical_001
bash scripts/ph14t/eval/run_eval_fixed_prompt.sh     <CKPT> de     # 单语言（只有这个 suite 支持）
bash scripts/ph14t/eval/run_eval_diverse_prompt.sh   <CKPT>        # 8 轮：canonical_001 + diverse_001..007
bash scripts/ph14t/eval/run_eval_unseen_prompt.sh    <CKPT>        # 8 轮：heldout_001..008
bash scripts/ph14t/eval/run_eval_wrong_task_prompt.sh <CKPT>       # 1 轮，diagnostic
bash scripts/ph14t/eval/run_eval_unrelated_prompt.sh  <CKPT>       # 1 轮，diagnostic
bash scripts/ph14t/eval/run_eval_cf_first_prompt.sh   <CKPT>       # 2 轮：cf_first_001..002
bash scripts/ph14t/eval/run_eval_cf_last_prompt.sh    <CKPT>       # 2 轮：cf_last_001..002
# 任何一个加 dry-run 只打印命令；share 用仓库内数据集
# PROMPT_VARIANTS="diverse_003 diverse_005" 跑子集；FORCE=1 重跑；EVAL_NUM_PROCESSES 改 GPU 数
```

14B 多语言 sweep（每个 run 取验证集 `eval_overall_macro_bleu4` 最高的 checkpoint，
当前是 fixed run 的 `checkpoint-38064`（0.2627）和 diverse run 的 `checkpoint-49776`（0.2534），
2 个 checkpoint × 8 变体 = 16 次评测）：

```bash
bash scripts/ph14t/eval/sweep14b/run_multilang_diverse_sweep.sh                       # 自动选最好的
CHECKPOINT_STEPS="29280 46848" bash scripts/ph14t/eval/sweep14b/run_multilang_diverse_sweep.sh  # 手动指定
ALL_CHECKPOINTS=1 bash scripts/ph14t/eval/sweep14b/run_multilang_diverse_sweep.sh     # 全部 checkpoint
```

分数从最新 checkpoint 的 `trainer_state.json` → `log_history` 里读。注意训练时
`metric_for_best_model` 是 `eval_overall_weighted_bleu4`，这里按 macro 选；三种语言样本数
都是 642，所以两者目前一致，不一致时脚本会告警。

单独补跑某个 prompt 条件（不走 suite）用 worker 自己：

```bash
EVAL_OUTPUT_DIR=outputs/eval/<run>/<step>/unseen/heldout_002 \
EVAL_PROMPT_BANK=prompts/generic/heldout.jsonl \
EVAL_PROMPT_IDS="de=heldout_en_de_002,en=heldout_en_en_002,zh=heldout_en_zh_002" \
  bash scripts/ph14t/eval/run_eval.sh <CKPT> multi
```

汇总器可以单独重跑（纯派生文件，删掉再生成即可）：

```bash
python -m csi_slt.commands.summarize_prompt_suite outputs/eval/<run>/<step>/diverse
python -m csi_slt.commands.summarize_prompt_suite outputs/eval/<run>/<step>/wrong_task \
    --diagnostic --language-distribution
```

## 落盘布局

文件夹名直接用 prompt 变体名，不另存 index 映射表：

```
outputs/eval/<ckpt_run_name>/<checkpoint-step>/
  fixed/<multi|de|en|zh>/canonical_001/
  diverse/{canonical_001, diverse_001 … diverse_007}/
  unseen/{heldout_001 … heldout_008}/
  wrong_task/wrong_task_001/
  unrelated/unrelated_001/
  cf_first/{cf_first_001, cf_first_002}/
  cf_last/{cf_last_001, cf_last_002}/
  counterfactual_summary.{json,md}        # 跨两个 cf suite 的汇总，见下
```

每个 suite 目录里还有 `variants.tsv`（这个 suite 由哪些变体构成 + 用的哪个 bank，
汇总器据此把没跑出来的变体报成 missing）和 `summary.{json,md}`；
每个变体目录里是 `predictions.jsonl` / `prompts.jsonl` / `predictions_metrics.json` / `eval_config.yaml`。

## 落定的决策

- **diverse suite 跑 8 轮**：`canonical_001` 也算一轮，不复用 fixed suite 的结果，
  这样 8 个点来自同一次同配置 sweep。
- **weather bank 完全不碰**：bank 固定 `prompts/generic/*`，worker 不从 ckpt 反推 bank family。
- **无 slurm**：没有 `--sbatch`，只有本地顺序执行 + `dry-run`；保留 tubbs/集群的 host
  allowlist 和 `prepare_dataset`（tubbs 直接用仓库内数据集），GPU 数走 `EVAL_NUM_PROCESSES`（默认 2）。
- **diverse / unseen / adversarial 只接受 multi**，传单语言直接报错（理由同
  `run_llm_lora_inject.sh` 拒绝单语言 diverse 训练：单语言的 paraphrase 没有对照物）。
- **worker 回放 ckpt 的预处理**：从 `<CKPT>/hydra_config.yaml` 读出全部
  `data.processor.*` 用 `++` 重新应用。那个 14B multilang ckpt 是
  `video_token_scale=0.5` + `do_normalize=false`，而 `eval/base` 默认 `1.0`——
  `run_cognition_eval.sh` 里那句 `do_normalize=False # NOTE: 很重要！！！` 是手抄的，
  抄错就是静默掉分。data **group** 仍按本次请求选（多语言 ckpt 可以合法地只评一种语言），
  只继承 processor。
- **dtype 按 checkpoint 存储的来**（2026-09-17 加）：`configs/eval/base.yaml` 的
  `engine.model_dtype` 默认值从 `auto` 改成 `checkpoint`。`auto` 读的是 `config.json`
  的 `dtype` 字段，而那个字段记的是保存时 config 对象上的值——`train.py:155` 是**载入后**
  才 `cast_module_dtype(llm, bfloat16)`，只改 module 不回写 config，所以 config 一直写着
  `float32`，磁盘上却是 bf16。14B stage-2 checkpoint 的真实存储是
  **llm bf16 29.4 GiB + 视觉侧 fp32 1.8 GiB = 31.2 GiB**，`auto` 会把它全升成 fp32
  ≈ 63 GiB/卡（而且 eval 是普通 DDP，每卡一份完整模型），显存翻倍、生成变慢、精度一点没多。
  `checkpoint` 的做法：先按 torch 默认 fp32 载入，再逐张量铸回 safetensors 头里记录的
  dtype。bf16 → fp32 → bf16 是无损往返，所以 GPU 上的权重和 checkpoint **逐 bit 相同**，
  fp32 的张量完全不碰，bool/int64 buffer 也不碰。`auto` / `float32` / `bfloat16` 仍可用，
  worker 里 `EVAL_MODEL_DTYPE` 可覆盖。没走这条路的 `configs/train/grpo.yaml` 保持 `auto`。
  **边界**：它只处理 checkpoint 里存在的张量。非持久 buffer（RoPE 的 `inv_freq`）不在 checkpoint 里，
  由 transformers 重算成 fp32，这个 guard 一次都没碰过它们——2026-09-18 排查时我曾据此误判过原因。
- **语言守卫**：单语言 ckpt 跑多语言 suite 在 suite 入口就 exit 5（`FORCE_LANG=1` 放行），
  不会 8 个变体各报一遍。
- **断点续跑**：已有 `predictions_metrics.json` 的变体默认跳过（`FORCE=1` 重跑），
  单个变体失败不中断整个 suite。
- **`wrong_task` / `unrelated` 标成 diagnostic**：它们的指令不是「翻译」，BLEU 对参考译文
  没有意义，汇总里 BLEU/ROUGE 进 appendix 区，主位是 `lacc` + 输出语言分布
  （`--language-distribution` 用的是 metric 自己那个检测器）+ 几条样例输出。
  口径同 `refine-logs/PROMPT_PROTOCOL.md`。
- `PromptIdFromRowResolver`（`configs/prompt/heldout_eval.yaml` 那条路）是死的：
  它要求数据集行带 `prompt_id` 列，数据集没有这一列，`prompts/assignments/` 也不存在。
  「每轮固定一个 id、跑 n 轮」的做法绕开了它，只用 `FixedPromptResolver`。

## 反事实指令切换实验（2026-09-20 加）

方案在 `.ai/experiment_plan.md`。问的是：模型选目标语言，靠的是指令表达的语义关系，
还是仅仅检测到了一个语言名 / 按它在 prompt 里的位置来选。

**12 个条件怎么塞进 4 个变体**。suite 机制的硬约束是「一次 run = 一个变体 = 三种语言各
一个 prompt id」，所以 2 模板 x 3 语言对 x 2 方向 = 12 个条件 = **4 个变体 x 3 个目标语言**，
干扰项按循环相邻取：

| 变体 | 模板 | de 的干扰项 | en 的 | zh 的 |
|---|---|---|---|---|
| `cf_first_001` | `into {目标}, not {干扰项}` | en | zh | de |
| `cf_first_002` | 同上 | zh | de | en |
| `cf_last_001` | `not into {干扰项}; use {目标}` | en | zh | de |
| `cf_last_002` | 同上 | zh | de | en |

每个模板的 6 个格子正好是 3 个语言对的全部 6 个有序方向。一个方向和它的反向落在**不同
变体**里，所以 BSA 只能跨变体算，不可能从单个 `summary.json` 里读出来——这就是
`summarize_counterfactual.py` 存在的理由。

**尾行必须中性**。canonical 模板第二行是 `only the German translation`，照搬会让目标语言
在 prompt 里出现两次、而且是最后一次出现的语言名——「数出现次数」和「取最后一个」两种
作弊策略都能刷满，两个模板的位置差异也被抹平。所以 cf bank 的尾行统一改成
`only the translation`，每条 prompt 里两个语言名**各出现恰好一次**，两个模板的唯一差别
就是先后顺序。于是作弊策略有了尖锐且相反的预测：只认第一个语言名 → cf_first ≈ 1.0 /
cf_last ≈ 0.0，只认最后一个 → 反过来。代价是 prompt 格式相对训练时有漂移，
所以 canonical → cf 的下降混了「不懂否定」和「格式漂移」两个原因；12 个条件之间的比较
（A vs B、两个方向、BSA）共享同一个骨架，不受影响。曾考虑加一个「canonical 指令 + 中性尾行」
的对照 bank 把漂移单独测出来，评估后不做。

**一个 suite 一个 bank**。跟 `heldout` / `wrong_task` / `unrelated` 同构。`train.jsonl` 靠
`--families` 切分是因为它同时是训练 bank（`PromptSampler.random()` 要从整池抽），cf 没有这个
约束。拆开的代价是没法再轻松加一个「4 个变体一起 mean±std」的合并 suite
（`EVAL_PROMPT_BANK` 是单个字符串，且 launcher 有存在性检查），但主表要的就是两列分开，
跨 12 个条件的总体数由汇总器自己算，所以这个损失是零。family 名仍然分 `cf_first`/`cf_last`，
让汇总器光看 prompt id 就能解出模板和目标语言。

**`--language-distribution` 从 `--diagnostic` 解耦**（`lib_prompt_suite.sh`）。cf suite 的
BLEU 是对目标语言参考译文算的、有意义，所以 `diagnostic=false`；但混淆矩阵又必须要。
改成默认值跟随 `$diagnostic`、`EVAL_LANGUAGE_DISTRIBUTION` 可双向覆盖，非法值报错。
现有五个 suite 行为不变，两个 cf launcher 用 `: "${EVAL_LANGUAGE_DISTRIBUTION:=1}"` 打开。

**BSA 的配对为什么成立**。BSA 要把同一个视频的 de 行和 en 行配起来，这是**不同的数据集行**，
`index` 只能对齐同一行，`reference` 又是两种语言的文本，都做不了键。成立的理由是
**gather 后的行序就是数据集行序**：`per_device_eval_batch_size` 行一个 rank，
`gather_for_metrics` 先接 rank 0 的片再接 rank 1 的，每个 batch 是一段连续升序，置换是恒等。
实测在 `diverse/canonical_001` 的 1926 行上 language 列零失配确认了这一点（reference 有 1256
处不同，那只是标点被归一化掉，不是错位）。于是行 `i` 就是数据集行 `i`，视频名直接从数据集拿。
数据集本身是按语言分块的（en / zh / de 各 642），**三个块里的视频顺序完全一致**——BSA 整个
建立在这条性质上，所以汇总器把它和 language 列一起**硬断言**，不符就报错，不猜。

**其它**。输出语言判定用的是 `SLTMetric` 同一个检测器（口径一致），结果缓存成
`<变体>/predictions_langid.jsonl`，predictions 变了就自动重算（`--force-langid` 强制）；
CPU 上 1926 条约 5 分钟，四个变体一次约 20 分钟。主表的 canonical 那一列不重跑，直接读
`diverse/canonical_001/predictions_metrics.json` 已经算好的 lacc。设计不完整（缺任何一个
方向）时汇总器直接报错而不是少算——BSA 缺一个方向不是「少一点数据」，是错的；sweep 脚本
里某个 suite 失败时也会跳过该 checkpoint 的汇总。

成本：每个 checkpoint 4 次 predict x 1926 行 = 7704 次生成，四个 checkpoint 共 30816 次。

## 一个严重 bug：LoRA checkpoint 载回来输出崩坏（2026-09-18 修复）

**症状**：任何 stage-2（`llm_lora=True`）checkpoint 用 `SltModel.from_pretrained()` 载入后，
自由生成变成词汇沙拉 + 复读、给英语目标吐德语、平均生成 112 token 撞满上限；
macro BLEU-4 **0.016**（训练时同一份权重、同一批样本是 **0.263**）。
teacher forcing 同样坏（loss 7.7、next-token acc 0.10），所以和生成机制无关。

**根因**：`SltModel._inject_llm_lora` 结尾调 `mark_module_tree_as_initialized(self.llm)`，
把整棵 LLM 子树（含 `llm.model.rotary_emb`）标记成 `_is_hf_initialized = True`。
而 transformers 5.15 的加载流程是：

1. `modeling_utils.py:4769-4772`——非持久 buffer 不在任何 checkpoint 里，用 `torch.empty_like` 重新分配，**内容是未初始化内存**
2. `_initialize_missing_keys` → `_initialize_weights` 本该由 `modeling_utils.py:2428-2436` 的 RotaryEmbedding 分支把 `inv_freq` 重算回来
3. 但 `modeling_utils.py:2442` 开头 `if getattr(module, "_is_hf_initialized", False): return` 短路了

于是 `inv_freq` 留着 1.1e17 量级的垃圾，RoPE 变噪声。`config.llm_lora=True` 时注入发生在
`__init__` 内部（`slt.py:303-304`），即**权重还没加载**就打了"已初始化"的断言——对参数来说这句话
马上会被加载变成真，对非持久 buffer 来说永远不会。

**为什么 stage-1 正常**：`llm_lora=False`，注入分支不执行，LLM 子树从未被标记，rope 正常重算。
**为什么训练时正常**：`train.py` 从 stage-1 checkpoint 构造（那次加载 rope 是对的），
**之后**才调 `inject_llm_lora()`；`load_best_model_at_end` 走的是 `load_state_dict`，不触发重新初始化。

**修复**（两道防线 + 一道闸）：

- `slt.py:527,541` 两处注入改用 `mark_adapter_modules_as_initialized()`——只标 PEFT 自己造的
  `adapter_layer_names` 子模块（`lora_A`/`lora_B`…），不碰 `base_layer`、不碰整棵树
- `mark_module_tree_as_initialized()` 跳过「只持有非持久 buffer」的模块（无参数、无持久 buffer）。
  刻意收窄：同时持有参数的模块（`SltModel` 根节点、`visual_adapter.next_frame_patch_fusion`）
  保持标记，否则它们的参数会暴露给重新初始化
- `validate_rope_buffers()`：`inv_freq` 必须有限且落在 (0,1]，`evaluate.py` 加载后立刻调用。
  这次是静默掉到 BLEU 0.01，有这道闸会当场报错

**定位方法（下次可复用）**：怀疑"checkpoint 载回来行为不对"时，不要逐个猜配置，直接
**对撞两条构造路径并逐张量对比**——`from_pretrained(stage2)` vs
`from_pretrained(stage1) + inject_llm_lora() + load_state_dict(stage2)`，导出每个张量的
(dtype, shape, 求和, absmax) 摘要再 diff。1138 个张量里只有那 2 个 rotary buffer 不同，
一步就定位了。诊断脚本当时在 scratchpad（`diag_construct.py` / `diag_diff.py`），未入库。

**作废的数据**：2026-09-17 那次 sweep 产出的 11 个变体结果（`outputs/eval/14b-multilang-diverse-eval/`，
BLEU 0.01~0.03）全部无效，已删除。checkpoint 本身完好，**不需要重训**。

**顺带修掉的源头**：`cast_module_dtype` 原来用 `Module.to(dtype=...)`，会把非持久 buffer 一起铸成
bf16——训练时的 RoPE 因此是 bf16。现在它只铸参数和持久 buffer，所以训练和 eval 的 rope
都是 fp32。实测 fp32 vs bf16 对质量没有影响（next-token acc 完全相同，两条样本里差一个同义词），
选 fp32 是因为它才是 Qwen3 预训练时的状态，也少一层为复刻意外而存在的机制。
代价：`canonical_001` 不会精确等于训练日志里的 0.2627，只会接近——**不要把 trainer 日志的数
和 suite 的数混进同一张表**，统一只报 suite 自己的数。

## 验证范围

已验证：四个 bank 的变体枚举；worker 生成的完整参数表（18 条继承 override + prompt id）
能被 hydra 正常 compose；`FixedPromptResolver` 用 heldout id 实例化后三语模板正确；
语言守卫的拒绝/通过；skip 已完成变体、`PROMPT_VARIANTS` 子集、`variants.tsv` 写入、
`summary.json`/`summary.md` 生成（普通与 diagnostic 两种模式，含 missing 变体告警）。

2026-09-18 补充：上面那个 RoPE bug 的修复已经端到端验证——两条构造路径 1138 张量 0 差异，
`validate_rope_buffers` 双双通过，两条样本的输出与训练时留下的 `predictions.jsonl` 逐句一致
（tf loss 3.57 / 0.80，acc 0.61 / 0.875）。完整 suite 尚未重跑。

## 一个事故（2026-09-17）

做非 dry-run 集成测试时 4 个变体里只造了 3 个假结果，第 4 个走了真实分支调用
`prepare_dataset`；该函数在没有 `.extract_complete` 标记时会先 `rm -rf "$DATASET_PATH"`
再重新解压，所以 `~/localscratch/ph14t` 被清掉并重新解压了约 2 分钟后被中止。
当时状态：解压到 42G 里的约 25G，`.extract_complete` / `.data_complete` 都不存在。
因为标记文件没写，下一次不带 `share` 的运行会自动 `rm -rf` 重新解压 + 预处理；
想省掉这次重做就带 `share` 用仓库内数据集。仓库文件和 GPU 任务未受影响。
