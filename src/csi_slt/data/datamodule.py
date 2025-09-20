from torch.utils.data import DataLoader
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import List
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from typing import Literal


class DataModule:
    def __init__(
        self,
        cfg: DictConfig,
        tokenizer,
    ):
        super().__init__()
        self.cfg = cfg

        self.pipline_train = instantiate(getattr(cfg.train, "pipline", None))
        self.pipline_val = instantiate(getattr(cfg.val, "pipline", None))
        self.pipline_test = instantiate(
            getattr(cfg.test, "pipline", cfg.val.pipline),
        )
        self.tokenizer = tokenizer

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
            self.train_dataset = instantiate(
                self.cfg.train.dataset, pipline=self.pipline_train
            )
            self.val_dataset = instantiate(
                self.cfg.val.dataset, pipline=self.pipline_val
            )
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

        if stage == "test" or stage is None:
            if self.cfg.test is not None:
                self.test_dataset = instantiate(
                    self.cfg.test.dataset, pipline=self.pipline_test
                )
                if self.cfg.fraction_dataset:
                    # If fraction_dataset is True, create a subset of the test dataset
                    if self.cfg.test_fraction < 1.0:
                        self.test_dataset = self.get_fraction_subset_dataset(
                            self.test_dataset,
                            fraction=self.cfg.test_fraction,
                        )

    def train_dataloader(self):
        # Return the training dataloader
        return DataLoader(
            self.train_dataset,
            collate_fn=instantiate(
                self.cfg.train.collator,
                tokenizer=self.tokenizer,
            ),
            **self.cfg.train.loader_kwargs,
        )

    def val_dataloader(self):
        # Return the validation dataloader
        return DataLoader(
            self.val_dataset,
            collate_fn=instantiate(self.cfg.val.collator, tokenizer=self.tokenizer),
            **self.cfg.val.loader_kwargs,
        )

    def test_dataloader(self):
        kwargs = getattr(
            self.cfg, "test_dataloader_kwargs", self.cfg.val_dataloader_kwargs
        )
        collator = getattr(self.cfg.test, "collator", self.cfg.val_collator)
        return DataLoader(
            self.test_dataset,
            collate_fn=instantiate(collator, tokenizer=self.tokenizer),
            **kwargs,
        )
