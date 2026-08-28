"""Unified SLT training entrypoint for full, connector, and PEFT training."""

import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, set_seed

from csi_slt.commands.prompt_setup import instantiate_prompt_resolvers
from csi_slt.data.datamodule import DataModule
from csi_slt.engine.sft.metrics import SLTMetric
from csi_slt.engine.sft.trainer import SltTrainer
from csi_slt.engine.sft.training_args import SltTrainingArguments
from csi_slt.engine.trainability import (
    SltTrainabilityPlan,
    apply_trainability_plan,
)
from csi_slt.modeling_slt.slt import SltConfig, SltModel
from csi_slt.utils.generation_config import merge_generation_config


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


def cast_module_dtype(module: torch.nn.Module, dtype: str | torch.dtype) -> None:
    """Cast one loaded checkpoint component, treating ``auto`` as no-op."""
    if dtype == "auto":
        return
    if isinstance(dtype, str):
        resolved_dtype = getattr(torch, dtype, None)
        if not isinstance(resolved_dtype, torch.dtype):
            raise ValueError(f"Unsupported dtype: {dtype!r}")
        dtype = resolved_dtype
    if not isinstance(dtype, torch.dtype):
        raise TypeError("dtype must be a torch.dtype or dtype name")
    if not (dtype.is_floating_point or dtype.is_complex):
        raise ValueError(f"dtype must be floating point or complex, got {dtype}")
    module.to(dtype=dtype)


def initialize_model(
    model_cfg: DictConfig,
    *,
    llm_lora_config: LoraConfig | None,
    visual_lora_config: LoraConfig | None,
    llm_dtype: str | torch.dtype,
    visual_backbone_dtype: str | torch.dtype,
) -> tuple[SltModel, str]:
    """Create an SLT model from a checkpoint or pretrained components."""
    load_from_checkpoint = model_cfg.get("load_from_checkpoint", False)
    if not isinstance(load_from_checkpoint, bool):
        raise TypeError("model.load_from_checkpoint must be a boolean")

    if load_from_checkpoint:
        checkpoint_dir = model_cfg.get("checkpoint_dir")
        if not checkpoint_dir:
            raise ValueError(
                "model.checkpoint_dir is required when model.load_from_checkpoint=true"
            )
        model = SltModel.from_pretrained(checkpoint_dir)
        tokenizer_source = str(checkpoint_dir)
    else:
        config_node = model_cfg.get("config")
        if config_node is None:
            raise ValueError(
                "model.config is required when model.load_from_checkpoint=false"
            )
        slt_config = SltConfig(**OmegaConf.to_container(config_node, resolve=True))
        model = SltModel.from_pretrained_components(
            config=slt_config,
            llm_dtype=llm_dtype,
            visual_backbone_dtype=visual_backbone_dtype,
        )
        tokenizer_source = str(slt_config.llm_model_name_or_path)

    if llm_lora_config is not None:
        model.inject_llm_lora(llm_lora_config)
    if visual_lora_config is not None:
        model.inject_visual_lora(visual_lora_config)

    return model, tokenizer_source


def build_lora_configs(
    peft_cfg: DictConfig,
) -> tuple[LoraConfig | None, LoraConfig | None]:
    """Create optional language and visual LoRA configs."""
    llm_lora_config = None
    if (node := peft_cfg.get("llm_lora_config")) is not None:
        llm_lora_config = LoraConfig(
            **OmegaConf.to_container(node, resolve=True),
            task_type=TaskType.CAUSAL_LM,
        )

    visual_lora_config = None
    if (node := peft_cfg.get("visual_lora_config")) is not None:
        visual_lora_config = LoraConfig(**OmegaConf.to_container(node, resolve=True))
    return llm_lora_config, visual_lora_config


@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name="train/base",
)
def main(cfg: DictConfig) -> None:
    llm_lora_config, visual_lora_config = build_lora_configs(cfg.peft)
    slt_model, tokenizer_source = initialize_model(
        cfg.model,
        llm_lora_config=llm_lora_config,
        visual_lora_config=visual_lora_config,
        llm_dtype=cfg.engine.llm_dtype,
        visual_backbone_dtype=cfg.engine.visual_backbone_dtype,
    )
    cast_module_dtype(slt_model.llm, cfg.engine.llm_dtype)
    cast_module_dtype(slt_model.visual_backbone, cfg.engine.visual_backbone_dtype)
    apply_trainability_plan(
        slt_model,
        SltTrainabilityPlan.from_mapping(cfg.engine.trainability),
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(
            cfg.prompt, ("train", "val", "test")
        ),
    )
    datamodule.setup()

    generation_config_args = OmegaConf.to_container(
        cfg.engine.generation_config, resolve=True
    )
    training_args = SltTrainingArguments(
        generation_config=merge_generation_config(
            slt_model.generation_config,
            generation_config_args,
        ),
        **cfg.engine.training_args,
    )
    metrics = SLTMetric(processor=datamodule.processor)
    train_probe_metrics = None
    train_probe_interval = OmegaConf.select(
        cfg, "engine.train_probe.every_n_evaluations", default=-1
    )
    if train_probe_interval != -1:
        train_probe_metrics = SLTMetric(processor=datamodule.processor)

    trainer = SltTrainer(
        model=slt_model,
        args=training_args,
        hydra_config=cfg,
        processing_class=datamodule.processor,
        train_dataset=datamodule.train_dataset,
        eval_dataset=datamodule.test_dataset,
        train_data_collator=datamodule.train_collator,
        eval_data_collator=datamodule.test_collator,
        compute_metrics=metrics,
        train_probe_compute_metrics=train_probe_metrics,
    )

    if OmegaConf.select(cfg, "engine.evaluate_before_training", default=False):
        trainer.evaluate()
    if training_args.do_train:
        trainer.train()
    if training_args.do_predict:
        predictions = trainer.predict(
            test_dataset=datamodule.test_dataset,
            test_collator=datamodule.test_collator,
        )
        trainer.save_predictions(predictions)


if __name__ == "__main__":
    main()
