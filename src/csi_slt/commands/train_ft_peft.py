import hydra

from omegaconf import DictConfig, OmegaConf
import os
from ..engine.trainer import SltTrainer
from ..engine.training_args import SltTrainingArguments
from ..data.datamodule import DataModule
from transformers import set_seed
from transformers import AutoTokenizer
from ..modeling_slt.slt import SltConfig, SltModel
from ..utils.generation_config import merge_generation_config
from peft import (
    LoraConfig,
    TaskType,
)
from csi_slt.engine.metrics import SLTMetric


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="peft/base")
def main(cfg: DictConfig):
    # create model
    # peft confg
    lora_args = OmegaConf.to_container(cfg.model.peft_config, resolve=True)
    peft_config = LoraConfig(
        **lora_args,
        task_type=TaskType.CAUSAL_LM,
    )
    slt_model = SltModel.from_pretrained_with_new_lora(
        peft_config, cfg.model.checkpoint_dir
    )

    # create datamodule
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.checkpoint_dir)

    datamodule = DataModule(cfg.data, tokenizer=tokenizer)
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
    )

    # trainer.evaluate()
    trainer.train()


if __name__ == "__main__":
    main()
