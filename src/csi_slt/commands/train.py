from accelerate.test_utils.testing import AccelerateTestCase
from accelerate import Accelerator
import hydra

from omegaconf import DictConfig, OmegaConf
import os
from ..engine.trainer import SltTrainer
from ..engine.training_args import SltTrainingArguments
from ..data.datamodule import DataModule
from transformers import set_seed
from transformers import AutoTokenizer
from ..modeling_slt.slt import SltConfig, SltModel
from ..misc.utils import deep_merge
from transformers.generation.configuration_utils import GenerationConfig
from csi_slt.engine.metrics import SLTMetric
from csi_slt.engine.priodic_metrics import XCometLiteMetric
from huggingface_hub import login

login(token=os.getenv("HF_TOKEN"))


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(
    version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="train/base"
)
def main(cfg: DictConfig):
    # accelerate initialize
    acc = Accelerator()

    # create model
    slt_config = SltConfig(**OmegaConf.to_container(cfg.model.config, resolve=True))

    if cfg.model.type == "qwenvl":
        from ..modeling_slt.slt_qwen_vl import SltQwenVLModel

        slt_model = SltQwenVLModel(slt_config)
    else:
        slt_model = SltModel.from_pretrained_components(
            slt_config, cfg.engine.llm_dtype, cfg.engine.visual_backbone_dtype
        )

    # fix parameters
    # for param in slt_model.llm.parameters():
    #     param.requires_grad = False

    # for param in slt_model.visual_backbone.backbone.parameters():
    #     param.requires_grad = False

    # create datamodule
    llm_name = cfg.model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(llm_name)

    datamodule = DataModule(cfg.data, tokenizer=tokenizer)
    datamodule.setup()

    # generation config
    #
    generation_config_args = OmegaConf.to_container(
        cfg.engine.generation_config, resolve=True
    )
    model_generation_config = slt_model.generation_config.to_dict()

    # create trainer
    training_args = SltTrainingArguments(
        generation_config=GenerationConfig(
            **deep_merge(model_generation_config, generation_config_args)
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
