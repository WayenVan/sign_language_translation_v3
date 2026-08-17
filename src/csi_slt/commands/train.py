from accelerate import Accelerator
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
from csi_slt.engine.sft.metrics import SLTMetric
from csi_slt.engine.sft.priodic_metrics import XCometLiteMetric
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


def get_transform_layers_from_strategy(llm_num_layers, strategy_config):
    if strategy_config.type == "none":
        raise ValueError("No transform layers specified in strategy config.")
    elif strategy_config.type == "last_n_layers":
        n = strategy_config.n_layers
        return list(range(llm_num_layers - n, llm_num_layers))
    else:
        raise ValueError(f"Unknown strategy type: {strategy_config.type}")


@hydra.main(
    version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="train/base"
)
def main(cfg: DictConfig):
    # accelerate initialize
    acc = Accelerator()

    # create model
    slt_config = SltConfig(**OmegaConf.to_container(cfg.model.config, resolve=True))

    if cfg.peft.type == "lora":
        from peft import LoraConfig, TaskType

        lora_args = OmegaConf.to_container(cfg.peft.llm_lora_config, resolve=True)
        peft_config = LoraConfig(
            **lora_args,
            task_type=TaskType.CAUSAL_LM,
        )
        slt_model = SltModel.from_pretrained_components_with_lora(
            slt_config,
            peft_config,
            cfg.engine.llm_dtype,
            cfg.engine.visual_backbone_dtype,
        )
    elif cfg.peft.type == "none":
        slt_model = SltModel.from_pretrained_components(
            slt_config, cfg.engine.llm_dtype, cfg.engine.visual_backbone_dtype
        )
    else:
        raise ValueError(f"Unknown peft type: {cfg.peft.type}")

    # fix parameters
    # for param in slt_model.llm.parameters():
    #     param.requires_grad = False

    # for param in slt_model.visual_backbone.backbone.parameters():
    #     param.requires_grad = False

    # create datamodule
    llm_name = cfg.model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    datamodule = DataModule(cfg.data, cfg.datamodule, tokenizer=tokenizer)
    datamodule.setup()

    # generation config
    #
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
    metrics = SLTMetric(
        processor=datamodule.processor,
        priodic_metrics=[
            XCometLiteMetric(
                accelerator=acc,
                every_n_evaluations=cfg.engine.metrics.xcomet_every_n_evaluations,
            )
        ],
    )
    train_probe_metrics = None
    train_probe_interval = OmegaConf.select(
        cfg, "engine.train_probe.every_n_evaluations", default=-1
    )
    if train_probe_interval != -1:
        # Probe calls must not advance the state of the normal evaluation metric.
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
