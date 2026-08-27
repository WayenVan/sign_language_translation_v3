import hydra

from omegaconf import DictConfig, OmegaConf
import os
from ..engine.sft.trainer import SltTrainer
from ..engine.sft.training_args import SltTrainingArguments
from ..data.datamodule import DataModule
from transformers import set_seed
from transformers import AutoTokenizer
import torch
from ..modeling_slt.slt import SltModel
from ..utils.generation_config import merge_generation_config
from peft import (
    LoraConfig,
    TaskType,
)
from csi_slt.engine.sft.metrics import SLTMetric
from csi_slt.commands.prompt_setup import instantiate_prompt_resolvers


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


def apply_peft_trainability(
    model: SltModel,
    *,
    train_llm_lora: bool,
    train_visual_lora: bool,
    train_visual_adapter: bool,
) -> int:
    """Freeze the model, then enable the explicitly selected PEFT components."""
    flags = {
        "train_llm_lora": train_llm_lora,
        "train_visual_lora": train_visual_lora,
        "train_visual_adapter": train_visual_adapter,
    }
    for name, enabled in flags.items():
        if not isinstance(enabled, bool):
            raise TypeError(f"{name} must be a boolean")

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_parameter_count = 0
    for component_name, module, enabled in (
        ("LLM", model.llm, train_llm_lora),
        ("visual", model.visual_backbone, train_visual_lora),
    ):
        lora_parameters = [
            parameter
            for name, parameter in module.named_parameters()
            if "lora_" in name
        ]
        if enabled and not lora_parameters:
            raise ValueError(
                f"{component_name} LoRA training was requested, but the model "
                "contains no matching LoRA parameters"
            )
        if enabled:
            for parameter in lora_parameters:
                parameter.requires_grad_(True)
                trainable_parameter_count += parameter.numel()

    if train_visual_adapter:
        if model.visual_adapter is None:
            raise ValueError(
                "visual adapter training was requested, but the model has no "
                "visual adapter"
            )
        for parameter in model.visual_adapter.parameters():
            parameter.requires_grad_(True)
            trainable_parameter_count += parameter.numel()

    if trainable_parameter_count == 0:
        raise ValueError("The trainability policy selected no trainable parameters")

    return trainable_parameter_count


@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name="train/ft_peft",
)
def main(cfg: DictConfig):
    # create model
    # peft confg
    llm_lora_config = None
    llm_lora_config_node = cfg.peft.get("llm_lora_config")
    if llm_lora_config_node is not None:
        lora_args = OmegaConf.to_container(llm_lora_config_node, resolve=True)
        llm_lora_config = LoraConfig(
            **lora_args,
            task_type=TaskType.CAUSAL_LM,
        )
    visual_lora_config = None
    visual_lora_config_node = cfg.peft.get("visual_lora_config")
    if visual_lora_config_node is not None:
        visual_lora_args = OmegaConf.to_container(visual_lora_config_node, resolve=True)
        visual_lora_config = LoraConfig(**visual_lora_args)

    if llm_lora_config is None and visual_lora_config is None:
        slt_model = SltModel.from_pretrained(cfg.model.checkpoint_dir)
    else:
        slt_model = SltModel.from_pretrained_with_new_lora(
            checkpoint_dir=cfg.model.checkpoint_dir,
            llm_lora_config=llm_lora_config,
            visual_lora_config=visual_lora_config,
        )
    cast_module_dtype(slt_model.llm, cfg.engine.llm_dtype)
    cast_module_dtype(slt_model.visual_backbone, cfg.engine.visual_backbone_dtype)
    trainability = cfg.peft.trainability
    apply_peft_trainability(
        slt_model,
        train_llm_lora=trainability.llm_lora,
        train_visual_lora=trainability.visual_lora,
        train_visual_adapter=trainability.visual_adapter,
    )

    # create datamodule
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint_dir)

    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(
            cfg.prompt, ("train", "val", "test")
        ),
    )
    datamodule.setup()

    # generation config
    generation_config_args = OmegaConf.to_container(
        cfg.engine.generation_config, resolve=True
    )
    # create trainer
    training_args = SltTrainingArguments(
        generation_config=merge_generation_config(
            slt_model.generation_config,
            generation_config_args,
        ),
        **cfg.engine.training_args,
    )

    metrics = SLTMetric(processor=datamodule.processor)
    train_probe_metrics = None
    if cfg.engine.train_probe.every_n_evaluations != -1:
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

    # trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
