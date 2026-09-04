from __future__ import annotations

import os

import hydra
from datasets import Dataset as HFDataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoTokenizer, set_seed

from csi_slt.data.datamodule import DataModule
from csi_slt.commands.config import instantiate_prompt_resolvers
from csi_slt.engine.grpo.trainer import SltGRPOTrainer
from csi_slt.engine.grpo.training_args import SltGRPOConfig
from csi_slt.modeling_slt.slt import SltModel


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))


class _TorchDatasetAsHFDataset(HFDataset):
    """Expose a lazy PyTorch map dataset through TRL's required HF type."""

    def __init__(self, dataset: TorchDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

    def __getitems__(self, indices: list[int]):
        return [self.dataset[index] for index in indices]

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.dataset!r})"


def _as_trl_dataset(dataset: TorchDataset | HFDataset) -> HFDataset:
    if isinstance(dataset, HFDataset):
        return dataset
    return _TorchDatasetAsHFDataset(dataset)


def _set_grpo_trainable_parameters(model: SltModel) -> None:
    """Freeze the SLT base model and expose only GRPO policy adapters."""

    # -------------------------------------------------------------------------
    # 1. Checkpoint loading materializes new Parameters, so restore the intended
    # frozen base-model state explicitly after ``from_pretrained`` returns.
    # -------------------------------------------------------------------------
    for parameter in model.parameters():
        parameter.requires_grad = False

    # -------------------------------------------------------------------------
    # 2. The native SLT GRPO policy trains only the visual adapter.
    # -------------------------------------------------------------------------
    for parameter in model.visual_adapter.parameters():
        parameter.requires_grad = True

    # -------------------------------------------------------------------------
    # 3. When requested, keep the newly added LLM LoRA policy trainable too.
    # -------------------------------------------------------------------------
    if model.config.llm_lora:
        for name, parameter in model.llm.named_parameters():
            if "lora_" in name:
                parameter.requires_grad = True


def _load_policy(cfg: DictConfig) -> SltModel:
    """Load an SFT checkpoint, optionally adding a fresh LoRA policy adapter."""
    checkpoint_dir = str(cfg.model.checkpoint_dir)
    model = SltModel.from_pretrained(
        checkpoint_dir,
        dtype=cfg.engine.model_dtype,
        trust_remote_code=True,
    )
    if cfg.peft.type == "lora":
        lora_args = OmegaConf.to_container(cfg.peft.lora_config, resolve=True)
        peft_config = LoraConfig(
            **lora_args,
            task_type=TaskType.CAUSAL_LM,
        )
        model.inject_llm_lora(peft_config)
    elif cfg.peft.type != "none":
        raise ValueError(f"Unknown peft type: {cfg.peft.type}")

    _set_grpo_trainable_parameters(model)
    return model


@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name="train/grpo",
)
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))

    slt_model = _load_policy(cfg)
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.checkpoint_dir,
        config=slt_model.config.llm_config,
    )

    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(
            cfg.prompt, ("train", "val")
        ),
    )
    datamodule.setup("fit")

    training_args = SltGRPOConfig(
        **OmegaConf.to_container(cfg.engine.training_args, resolve=True)
    )
    trainer = SltGRPOTrainer(
        model=slt_model,
        args=training_args,
        processing_class=datamodule.processor,
        train_prompt_resolver=datamodule.prompt_resolvers["train"],
        eval_prompt_resolver=datamodule.prompt_resolvers["val"],
        train_dataset=_as_trl_dataset(datamodule.train_dataset),
        eval_dataset=_as_trl_dataset(datamodule.val_dataset),
    )

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=cfg.engine.resume_from_checkpoint)


if __name__ == "__main__":
    main()
