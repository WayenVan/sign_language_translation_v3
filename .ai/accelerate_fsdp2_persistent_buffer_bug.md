# accelerate FSDP2：持久化 buffer 导致 `fsdp2_load_full_state_dict` 崩溃

日期：2026-08-22
状态：已确认，未修复（upstream 最新版 1.14.0 仍有）
目标：给 huggingface/accelerate 提 issue + PR

## 结论先行

`accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict` 假设 `model.state_dict()`
的每一项都是 `DTensor`，但 `fully_shard` 只把**参数**转成 DTensor，**buffer 保持普通
Tensor**。因此只要模型带任何**持久化 buffer**，FSDP2 + `cpu_ram_efficient_loading`
就会在 `accelerator.prepare()` 阶段崩：

```
AttributeError: 'Tensor' object has no attribute 'device_mesh'
```

修复方向：在两个广播循环里识别非 DTensor 条目，走「整份广播 + 直接赋值」而不是
`distribute_tensor`。约 12 行。

**注意：有一个看起来很自然但会造成静默权重破坏的错误修法，见第 5 节，PR 描述里应该
主动说明为什么不那么做。**

## 1. 环境

> 行号约定：`fsdp_utils.py` 一律对应 **1.14.0**（PR 的目标版本）；其它文件如未标注，
> 对应本地安装的 **1.12.0**。


| 组件 | 版本 |
| --- | --- |
| accelerate | 1.12.0（本地）；1.13.0 / 1.14.0 已下载 wheel 核对，**同样有 bug** |
| torch | 2.8.0+cu128 |
| transformers | 5.15.0 |

触发条件（三者同时满足）：

1. `distributed_type: FSDP` + `fsdp_version: 2`
2. `fsdp_cpu_ram_efficient_loading: true`
3. 模型含至少一个**持久化** buffer（即不在 `_non_persistent_buffers_set` 里的 buffer）

条件 3 是大多数人碰不到它的原因：HF 的纯文本模型把 `rotary_emb.inv_freq` 之类都注册成
`persistent=False`，不进 `state_dict`。本项目因为挂了 NVIDIA C-RADIO 视觉塔才撞上——
它的 `summary_idxs`、`input_conditioner.norm_mean`、`input_conditioner.norm_std`
都是持久化的。

## 2. 复现

最小复现（单进程 gloo 即可暴露，不需要多卡）：

```python
import os, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29543")
dist.init_process_group("gloo", rank=0, world_size=1)

class M(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 4)
        self.head = nn.Linear(4, 8, bias=False)
        self.head.weight = self.emb.weight                                  # tied
        self.register_buffer("norm_mean", torch.zeros(3, 1, 1))             # persistent
        self.register_buffer("inv_freq", torch.zeros(4), persistent=False)  # non-persistent

m = M()
fully_shard(m)
for k, v in m.state_dict().items():
    print(k, "DTensor" if isinstance(v, DTensor) else type(v).__name__,
          "has device_mesh =", hasattr(v, "device_mesh"))
```

实测输出：

```
norm_mean    Tensor   has device_mesh = False     ← 持久化 buffer
emb.weight   DTensor  has device_mesh = True
head.weight  DTensor  has device_mesh = True      ← 绑定权重是好的
```

三点值得记录：

- **持久化 buffer 是普通 Tensor** → 就是崩溃来源。
- **非持久化 buffer 根本不在 `state_dict` 里** → 不会触发，走的是另一条路（第 4 节）。
- **绑定权重（tied weight）是 DTensor**，FSDP2 通过 `ParamModuleInfo.shared_modules`
  正确传播了（`torch/distributed/fsdp/_fully_shard/_fsdp_param.py:596-598`）。
  一开始怀疑过 `lm_head.weight`，可以排除。

真实 traceback（本项目 Qwen3 + C-RADIO，行号对应本地安装的 **1.12.0**）：

```
File ".../accelerate/accelerator.py", line 1711, in _prepare_fsdp2
    model = fsdp2_prepare_model(self, model)
File ".../accelerate/utils/fsdp_utils.py", line 683, in fsdp2_prepare_model
    fsdp2_load_full_state_dict(accelerator, model, original_sd)
File ".../accelerate/utils/fsdp_utils.py", line 526, in fsdp2_load_full_state_dict
    device_mesh = sharded_param.device_mesh
AttributeError: 'Tensor' object has no attribute 'device_mesh'
```

## 3. 根因

`fsdp2_load_full_state_dict`（1.14.0 里在 `fsdp_utils.py:467`）第 484 行取
`meta_sharded_sd = model.state_dict()`，然后两个分支都无条件读 `.device_mesh`：

```python
# fsdp_utils.py:513-538  rank 0 分支
if accelerator.is_main_process:
    for param_name, sharded_param in meta_sharded_sd.items():
        if param_name not in full_sd:
            raise KeyError(...)
        full_param = full_sd[param_name]
        device_mesh = sharded_param.device_mesh          # ← :521 崩在这
        full_param = full_param.detach().to(device_mesh.device_type)
        ...
        dist.broadcast(full_param, src=0, group=dist.group.WORLD)
        sharded_tensor = distribute_tensor(full_param, device_mesh, sharded_param.placements)
        sharded_sd[param_name] = sharded_tensor

# fsdp_utils.py:541-557  其它 rank 分支
else:
    for param_name, sharded_param in meta_sharded_sd.items():
        device_mesh = sharded_param.device_mesh          # ← :543 同样的问题
        full_tensor = torch.empty(sharded_param.size(), device=device_mesh.device_type,
                                  dtype=sharded_param.dtype)
        dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
        sharded_tensor = distribute_tensor(full_tensor, device_mesh, sharded_param.placements)
        sharded_sd[param_name] = sharded_tensor
```

`device_mesh` 和 `placements` 都是 DTensor 的属性。buffer 没有分片，自然两者都没有。

注：1.12.0 的 rank0 分支写的是 `zip(full_sd.items(), meta_sharded_sd.values())`，
即**按位置**配对两个 dict，键一旦不对齐就会静默错配。1.14.0 已经改成遍历
`meta_sharded_sd` 并显式 `raise KeyError`。**PR 请基于 1.14.0/main**，那个错配问题
不必重复处理。

## 4. 为什么 accelerate 会分成两条路

`fsdp2_prepare_model` 对 buffer 有两条互斥的处理路径，理解这个是写对补丁的前提：

| | 持久化 buffer | 非持久化 buffer |
| --- | --- | --- |
| 在 `state_dict` 里 | 是 | 否 |
| 恢复方式 | 随 `full_sd` 从 rank 0 **广播**（:782 处调用） | 转 meta 前 **deepcopy**（:733-735），之后本地 re-register（:787 起） |
| 需要通信 | 需要 | 不需要 |

设计意图是清楚的：`state_dict` 是「需要从 rank 0 同步过来的东西」的清单；清单外的漏网之鱼
（非持久化 buffer）会在 `.to(meta)` 后永远停在 meta 上，所以单独备份再放回。accelerate
自己的注释（`fsdp_utils.py:730`）：

> We need to keep the original non-persistent buffers, as those MAY not be in the
> state_dict, resulting in them staying on meta device

deepcopy 那条路**不做通信**，前提是「清单外的值在每个 rank 上本来就相同」。transformers
保证了这一点：非 rank0 上 `_initialize_missing_keys`（`modeling_utils.py:4783`）把
`state_dict()` 里的东西全标成 `_is_hf_initialized=True`（反正等广播），而不在 `state_dict`
里的非持久化 buffer 没被标记，于是各自本地重建。

**所以两条路的分界不是「buffer vs 参数」，而是「在不在 state_dict 里」。**
持久化 buffer 本来就该走广播路，只是那条路的实现漏了非 DTensor 的情况。

## 5. ⚠️ 一个会造成静默权重破坏的错误修法

最自然的想法是：把这些 buffer 改成 `persistent=False`，让它们走 deepcopy 路绕开崩溃。
**这是错的，而且不会报错。**

`transformers/modeling_utils.py:4744-4750`：

```python
if is_fsdp_enabled() and not is_local_dist_rank_0() and not is_quantized:
    for key, param in self.named_parameters():
        _load_parameter_into_model(self, key, torch.zeros_like(param, device="cpu"))
    for key, buffer in self.named_buffers():
        _load_parameter_into_model(self, key, torch.zeros_like(buffer, device="cpu"))
    return
```

非 local-rank-0 上**所有参数和 buffer 都被置零**，等广播来填。如果在加载完成之后才把某个
持久化 buffer 降级成非持久化，它就同时失去了两条路：广播路看不见它了，deepcopy 路又只是
把本地那份零复制一遍。结果是 **rank 0 正确、其余 rank 全零**，静默传播。

本项目实测到的后果（C-RADIO 的 input conditioner 会做 `(x - norm_mean) / norm_std`）：
`norm_std = 0` → 除零 → 除 rank 0 外所有卡的视觉特征变 NaN。

同理，「在 `__init__` 里就把它们注册成非持久化」也不行：`from_pretrained` 是在空权重上构造
再从 checkpoint 填数，buffer 不在 checkpoint 里就没有任何一步会给它赋值。本项目实测拿到
`norm_std = [2.46e+14, 4.59e-41, 3.52e+14]`（未初始化内存）。

**正确的方向是让持久化 buffer 留在广播路上，把那条路修好。**

## 6. 建议的修复

在两个循环开头各加一个非 DTensor 分支：整份广播、直接赋值，不做 `distribute_tensor`。
设备用 `accelerator.device`（没有 `device_mesh` 可问）。

```python
    # rank 0 分支，插在 `full_param = full_sd[param_name]` 之后
            if not isinstance(sharded_param, DTensor):
                # Buffers are not sharded by `fully_shard`, so they have no mesh or
                # placements. Broadcast them whole and assign them as-is.
                full_param = full_param.detach().to(accelerator.device)
                dist.broadcast(full_param, src=0, group=dist.group.WORLD)
                sharded_sd[param_name] = full_param
                continue

    # 其它 rank 分支，插在循环开头
            if not isinstance(sharded_param, DTensor):
                full_tensor = torch.empty(
                    sharded_param.size(),
                    device=accelerator.device,
                    dtype=sharded_param.dtype,
                )
                dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
                sharded_sd[param_name] = full_tensor
                continue
```

要点：

- **两个分支必须发出数量和顺序完全一致的 `broadcast`**，否则死锁。1.14.0 里两个分支都遍历
  `meta_sharded_sd`，顺序天然一致；`continue` 前后各恰好一次 broadcast，保持这个不变量。
- 结尾的 `model.load_state_dict(sharded_sd, assign=True)`（:559）对普通 Tensor 的 buffer
  条目可以直接工作，不用改。
- `_infer_parameter_dtype` / `_cast_and_contiguous` 对 buffer 可以跳过：广播出来的张量 dtype
  就是 `full_sd` 里的原 dtype，不需要转换。
- **边界情况：bool buffer。** NCCL 对 `torch.bool` 的支持不完整，某些模型有持久化的 bool
  mask buffer。可能需要 `.to(torch.uint8)` 往返，或者至少在 PR 里说明这个限制。
- **边界情况：`cpu_offload`。** 1.14.0 的签名多了 `cpu_offload` 参数，参数路径下会
  `sharded_tensor.to("cpu")`。buffer 要不要跟着走 CPU 需要确认——倾向于不要，buffer 不参与
  分片，留在 device 上更合理。

## 7. 测试方案

accelerate 仓库里加一个多进程测试（参考 `tests/fsdp/` 下现有用例）：

1. 构造一个小模型，含：一个持久化 buffer（非零值，且各 rank 初始值**不同**，用于证明确实
   发生了广播而不是碰巧相等）、一个非持久化 buffer、一对绑定权重。
2. `Accelerator` 用 FSDP2 + `cpu_ram_efficient_loading=true`，2 进程。
3. `prepare` 之后断言：
   - 不抛 `AttributeError`；
   - **所有 rank 上的持久化 buffer 都等于 rank 0 的原值**（这是关键断言，第 5 节那个错误
     修法在这条上会失败）；
   - 非持久化 buffer 仍然正确（不受回归影响）；
   - 参数值正确（不受回归影响）。

单进程 gloo 复现可以进 CI 做冒烟测试，但**必须有 ≥2 进程的用例**，否则 rank 0 永远是对的，
静默破坏那一类问题测不出来。本项目就是因为一开始只在单进程验证，差点把第 5 节那个错误修法
提交上去。

## 8. 相关但不在本次范围内

**FSDP2 的 `MixedPrecisionPolicy` 没有 `buffer_dtype`。** FSDP1 的 `MixedPrecision` 有，
FSDP2 删掉了。后果是 fp32 buffer 和 bf16 激活做算术时会把结果提升回 fp32，下一个 matmul 就
报 `expected mat1 and mat2 to have the same dtype`。这是 **PyTorch** 的功能缺失，不是
accelerate 的 bug，应该单独提到 pytorch/pytorch。

**FSDP2 路径下 accelerate 不套 autocast。** `_prepare_fsdp2`（1.14.0 `accelerator.py:1673`）
不经过 `prepare_model`，而 autocast 的包裹写在 `prepare_model`（1.14.0 `accelerator.py:1818` 起）；
同时 transformers 的 `autocast_smart_context_manager`（`trainer.py:2066`）故意返回
`nullcontext`，注释说「We rely on accelerate for autocast」。两边互相指望，FSDP2 下一层
autocast 都没有。但 `native_amp` 在 FSDP 下确实是 `True`（1.12.0 `accelerator.py:585-596` 只排除了
DeepSpeed 和 Megatron），说明 accelerate 认为 AMP 是开着的——这个不自洽也值得单独提一个
issue，不过和本次 buffer 修复是两件事。

## 9. 本仓库的现状

在上游修好之前，本项目的规避手段是**关掉** `fsdp_cpu_ram_efficient_loading`
（`configs/accelerate/fsdp2.yaml`，那里有完整注释说明原因）。关掉之后
`fsdp2_load_full_state_dict` 根本不被调用，整条出问题的路径都不走；代价是每个 rank 都要在
CPU 上完整加载一份模型（Qwen3-32B 约 70GB/进程）。

上游修复合并并发版之后，可以把那个开关重新打开、把 `run_cognition_qwen.sh` 的 `--mem`
调回去，并删掉 config 里那段解释。
