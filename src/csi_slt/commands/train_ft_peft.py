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


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


def get_transform_layers_from_strategy(llm_num_layers, strategy_config):
    strategy_name = strategy_config.name
    if strategy_name == "none":
        raise ValueError("No transform layers specified in strategy config.")
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

    # create datamodule
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint_dir)

    datamodule = DataModule(cfg.data, cfg.datamodule, tokenizer=tokenizer)
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
