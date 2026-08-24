import hydra

from omegaconf import DictConfig, OmegaConf
import os
from ..engine.sft.trainer import SltTrainer
from ..engine.sft.training_args import SltTrainingArguments
from ..data.datamodule import DataModule
from transformers import set_seed
from transformers import AutoTokenizer
from ..modeling_slt.slt import SltConfig, SltModel
from ..utils.generation_config import merge_generation_config
from peft import (
    LoraConfig,
    TaskType,
)
from csi_slt.engine.sft.metrics import SLTMetric
from csi_slt.commands.prompt_setup import instantiate_prompt_resolvers


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


def freeze_except_lora_adapters(
    model: SltModel,
    *,
    unfreeze_visual_adapter: bool = False,
) -> int:
    """Freeze the SLT model except LoRA and, optionally, the visual adapter."""
    if not isinstance(unfreeze_visual_adapter, bool):
        raise TypeError("unfreeze_visual_adapter must be a boolean")

    trainable_parameter_count = 0
    for name, parameter in model.named_parameters():
        is_lora_parameter = "lora_" in name
        parameter.requires_grad_(is_lora_parameter)
        if is_lora_parameter:
            trainable_parameter_count += parameter.numel()

    if trainable_parameter_count == 0:
        raise ValueError("No LoRA adapter parameters were found in the model")

    if unfreeze_visual_adapter:
        visual_adapter = getattr(model, "visual_adapter", None)
        if visual_adapter is None:
            raise ValueError(
                "unfreeze_visual_adapter=True requires model.visual_adapter"
            )
        for parameter in visual_adapter.parameters():
            parameter.requires_grad_(True)
            trainable_parameter_count += parameter.numel()

    return trainable_parameter_count


def get_transform_layers_from_strategy(llm_num_layers, strategy_config):
    strategy_name = strategy_config.name
    if strategy_name == "none":
        raise ValueError("No transform layers specified in strategy config.")
    elif strategy_name == "all_layers":
        return list(range(llm_num_layers))
    elif strategy_name == "last_n_layers":
        n = strategy_config.n_layers
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("n_layers must be an integer")
        if not 1 <= n <= llm_num_layers:
            raise ValueError(
                f"n_layers must be between 1 and {llm_num_layers}, got {n}"
            )
        return list(range(llm_num_layers - n, llm_num_layers))
    else:
        raise ValueError(f"Unknown strategy name: {strategy_name}")


@hydra.main(
    version_base=None,
    config_path=DEFAULT_CONFIG_PATH,
    config_name="train/ft_peft",
)
def main(cfg: DictConfig):
    # create model
    # peft confg
    slt_config = SltConfig.from_pretrained(cfg.model.checkpoint_dir)
    lora_args = OmegaConf.to_container(cfg.peft.llm_lora_config, resolve=True)
    peft_config = LoraConfig(
        **lora_args,
        layers_to_transform=get_transform_layers_from_strategy(
            slt_config.llm_config.num_hidden_layers, cfg.peft.llm_lora_strategy
        ),
        task_type=TaskType.CAUSAL_LM,
    )
    slt_model = SltModel.from_pretrained_with_new_lora(
        peft_config, cfg.model.checkpoint_dir
    )
    freeze_except_lora_adapters(
        slt_model,
        unfreeze_visual_adapter=cfg.peft.unfreeze_adapter,
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
        eval_dataset=datamodule.val_dataset,
        train_data_collator=datamodule.train_collator,
        eval_data_collator=datamodule.val_collator,
        compute_metrics=metrics,
        train_probe_compute_metrics=train_probe_metrics,
    )

    # trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
