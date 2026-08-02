from omegaconf import DictConfig
from hydra.utils import instantiate
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset, ConcatDataset
from typing import Literal
import os
from functools import cached_property


class DataModule:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
        accelerator=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.accelerator = accelerator

    @property
    def chat_template(self) -> str | None:
        """Read chat template from the data-level config."""
        if hasattr(self.cfg, "chat_template_jinjia"):
            path = self.cfg.chat_template_jinjia
            # convert relative path to absolute if needed
            if not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            if os.path.exists(path):
                with open(path, "r") as f:
                    return f.read()
            else:
                raise FileNotFoundError(f"Chat template file not found: {path}")
        return None

    @property
    def _prompt_paths(self) -> dict[str, str]:
        """Resolve prompt_templates config paths into prompt_paths_per_language dict."""
        paths: dict[str, str] = {}
        if not hasattr(self.cfg, "prompt_templates"):
            return paths
        for lang, rel_path in self.cfg.prompt_templates.items():
            abs_path = (
                rel_path
                if os.path.isabs(rel_path)
                else os.path.join(os.getcwd(), rel_path)
            )
            paths[lang] = abs_path
        return paths

    @staticmethod
    def get_fraction_subset_dataset(
        dataset, labels=None, fraction=0.3, stratify=True, random_state=None
    ):
        """
        从原始 dataset 中抽取 fraction 比例的子集，仅返回 Subset 对象。

        参数：
        - dataset: PyTorch map-style Dataset 实例；
        - labels: 可选，长度与 dataset 相同，用于 stratify；
        - fraction: float，抽样比例 (0 < fraction < 1)；
        - stratify: 是否分层抽样，如果 labels 为 None，则强制 False；
        - random_state: 随机种子，用于复现。

        返回：
        - subset_dataset: torch.utils.data.Subset，仅包含选定索引。
        """

        total = len(dataset)
        n = int(total * fraction)
        if n <= 0 or n >= total:
            raise ValueError("fraction must be between 0 and 1 (exclusive)")

        indices = list(range(total))
        if stratify and labels is not None:
            chosen_idx, _ = train_test_split(
                indices,
                train_size=n,
                random_state=random_state,
                shuffle=True,
                stratify=labels,
            )
        else:
            generator = (
                torch.Generator().manual_seed(random_state)
                if random_state is not None
                else None
            )
            perm = torch.randperm(total, generator=generator).tolist()
            chosen_idx = perm[:n]

        subset_dataset = Subset(dataset, chosen_idx)
        return subset_dataset

    def setup(self, stage: Literal["train", "test", None] = None):
        # Set up the dataset for training, validation, and testing
        if stage == "train" or stage is None:
            self.train_dataset = instantiate(self.cfg.train.dataset)
            self.val_dataset = instantiate(self.cfg.val.dataset)

            self.train_dataset.prepare(self.tokenizer)
            self.val_dataset.prepare(self.tokenizer)

            if self.cfg.fraction_dataset:
                # If fraction_dataset is True, create a subset of the training dataset
                #
                if self.cfg.train_fraction < 1.0:
                    self.train_dataset = self.get_fraction_subset_dataset(
                        self.train_dataset,
                        fraction=self.cfg.train_fraction,
                    )

                if self.cfg.val_fraction < 1.0:
                    self.val_dataset = self.get_fraction_subset_dataset(
                        self.val_dataset,
                        fraction=self.cfg.val_fraction,
                    )

            if self.cfg.assemble:
                self.train_dataset = ConcatDataset(
                    [self.train_dataset, self.val_dataset]
                )

        if stage == "test" or stage is None:
            if self.cfg.test is not None:
                self.test_dataset = instantiate(self.cfg.test.dataset)
                self.test_dataset.prepare(self.tokenizer)
                if self.cfg.fraction_dataset:
                    # If fraction_dataset is True, create a subset of the test dataset
                    if self.cfg.test_fraction < 1.0:
                        self.test_dataset = self.get_fraction_subset_dataset(
                            self.test_dataset,
                            fraction=self.cfg.test_fraction,
                        )

    @cached_property
    def processor(self):
        return instantiate(
            self.cfg.processor,
            tokenizer=self.tokenizer,
            chat_template=self.chat_template,
            prompt_paths_per_language=self._prompt_paths,
            _convert_="all",
        )

    @property
    def train_collator(self):
        return instantiate(
            self.cfg.train.collator,
            processor=self.processor,
        )

    @property
    def val_collator(self):
        return instantiate(self.cfg.val.collator, processor=self.processor)

    @property
    def test_collator(self):
        return instantiate(self.cfg.test.collator, processor=self.processor)

    def print_batch(self, batch_size, num_workers=1, random=False):
        dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            collate_fn=self.train_collator,
            num_workers=num_workers,
            shuffle=random,
        )
        for batch in dataloader:
            input_text = self.tokenizer.batch_decode(
                batch["input_ids"], skip_special_tokens=False
            )
            if "labels" in batch:
                labels = batch["labels"]
                mask = labels == -100
                labels = labels.masked_fill(mask, self.tokenizer.pad_token_id)
                label_text = self.tokenizer.batch_decode(
                    labels, skip_special_tokens=False
                )
                for i in range(len(input_text)):
                    print(f"INPUT: \n {input_text[i]}")
                    print(f"LABEL: \n {label_text[i]}")
                    print("-" * 50)

            break
