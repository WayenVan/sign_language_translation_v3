# Dataset 编排层独立 Python 库设计

## 一句话目标

```text
Mapping[str, Dataset] -> Strategy Pipeline -> Mapping[str, Dataset]
```

不把整个当前 `DataModule` 抽成库，只抽出“输入若干命名 Dataset，通过 Strategy 编排，产生若干命名 Dataset”的纯编排层。

## 推荐边界

```text
项目内部                              独立 Python 库
────────                              ─────────────
Hydra 配置
Dataset 实例化       ──datasets──▶    DatasetPlanner
tokenizer.prepare()                    └─ Strategy Pipeline
processor / collator                  输出任意命名的 datasets
训练框架的 stage
```

当前 `src/csi_slt/data/datamodule.py` 混合了四类职责：

1. 从 Hydra 配置创建 Dataset
2. 调用 `dataset.prepare(tokenizer)`
3. 用 Strategy 编排 Dataset
4. 创建 processor 和 collator

只有第 3 项值得成为通用库。

## 核心数据模型

不要把 split 限制成：

```python
Split = Literal["train", "val", "test"]
```

改成任意字符串：

```python
from collections.abc import Mapping
from torch.utils.data import Dataset

DatasetMap = Mapping[str, Dataset]
```

输入和输出都不限于三个：

```python
inputs = {
    "train_original": train_dataset,
    "validation_original": validation_dataset,
    "external": external_dataset,
}

outputs = {
    "train": ...,
    "validation": ...,
    "test": ...,
    "calibration": ...,
    "debug": ...,
}
```

通常输出三个数据集，但库本身不应该知道“三个”这个概念。

## Strategy 最小接口

Strategy 是纯 Dataset 变换：

```python
from collections.abc import Mapping
from typing import Protocol

from torch.utils.data import Dataset


class Strategy(Protocol):
    def apply(
        self,
        datasets: Mapping[str, Dataset],
    ) -> dict[str, Dataset]:
        ...
```

使用方式：

```python
result = strategy.apply({
    "train": train_dataset,
    "val": val_dataset,
    "test": test_dataset,
})
```

Strategy 不负责：

- Hydra 实例化
- tokenizer
- collator
- processor
- DataLoader
- Lightning/Hugging Face 的 `fit`、`test`、`predict` stage

这些都是上层应用的职责。

## Strategy 应该可组合

当前 `datamodule_strategies.py` 中三个类混合了不同操作：

- `StandardSplitStrategy`：保留、合并
- `SplitSubsetStrategy`：采样
- `SharedSubsetStrategy`：采样、复制、重命名

应把它们拆成小操作，再组成 Pipeline。

共享一份子集：

```python
strategy = Pipeline([
    Subset("train", count=100, seed=42),
    Alias("train", as_names=["val", "test"]),
])
```

正常训练的分 split 采样：

```python
strategy = Pipeline([
    Subset("train", fraction=0.4, seed=42),
    Subset("val", fraction=0.5, seed=42),
    Subset("test", fraction=0.5, seed=42),
])
```

训练集追加验证集：

```python
strategy = Pipeline([
    Concat(inputs=["train", "val"], output="train"),
    Select(["train", "val", "test"]),
])
```

第一版优先提供五个基础 Strategy：

1. `Select`：选择、重命名输出
2. `Subset`：按数量或比例采样
3. `Concat`：合并多个 Dataset
4. `Alias`：同一 Dataset 暴露为多个名字
5. `Pipeline`：顺序组合其他 Strategy

这五个已经能覆盖当前全部需求。

## Alias 和独立切分是两种语义

“一个 Dataset 组成 train、val、test”存在两种不同需求。

### 共享同一批样本

适合 overfit sanity check：

```python
Pipeline([
    Subset("source", count=100, seed=42),
    Alias("source", ["train", "val", "test"]),
])
```

三个 key 指向同一个对象、同一批样本。

### 切成互不重叠的三份

适合真正的数据划分：

```python
RandomSplit(
    source="all",
    outputs={
        "train": 0.8,
        "val": 0.1,
        "test": 0.1,
    },
    seed=42,
)
```

`RandomSplit` 应作为独立 Strategy，保证：

- 划分比例完整
- index 不重叠
- seed 可复现
- 所有样本如何处理清楚

不要让 `Subset + Alias` 同时承担这两种语义。

## Stage 放在项目内部

当前 Strategy 有：

```python
required_splits(stage)
arrange(datasets, stage)
```

这使通用策略与训练框架生命周期耦合，而且当前 `fit` 仍要求加载 `test`，边界比较含混。

建议改成：

```python
# 通用库
datasets = strategy.apply(source_datasets)

# 项目内部
if stage == "fit":
    required_outputs = {"train", "val"}
elif stage in {"test", "predict"}:
    required_outputs = {"test"}
```

如果以后需要避免创建未使用的输入 Dataset，可以增加静态声明：

```python
strategy.required_inputs
```

第一版不用急着做惰性构建，先让纯编排层稳定。

## 推荐包结构

```text
dataset-compose/
├── pyproject.toml
├── src/dataset_compose/
│   ├── __init__.py
│   ├── typing.py
│   ├── pipeline.py
│   └── strategies/
│       ├── select.py
│       ├── subset.py
│       ├── concat.py
│       ├── alias.py
│       └── random_split.py
└── tests/
```

候选库名：

- `dataset-compose`：推荐；不仅做 split，也做采样、合并、别名和组合
- `dataset-strategies`
- `splitcraft`

## 第一版 API 承诺

```python
outputs = compose(inputs, strategy)
```

其中：

- `inputs`：任意名称、任意数量的 Dataset
- `strategy`：一个 Strategy 或 Pipeline
- `outputs`：任意名称、任意数量的 Dataset
- 库依赖最好只有 `torch`
- 所有随机行为必须接收显式 `seed`
- 不包含 Hydra、tokenizer、collator 和训练 stage

核心接口：

```python
compose(
    inputs: Mapping[str, Dataset],
    strategy: Strategy,
) -> dict[str, Dataset]
```

## 工作量估计

- 拆出最小可用库：约 2–3 小时
- 补齐类型、边界测试、README 和发布配置：约半天
