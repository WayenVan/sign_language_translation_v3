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
from transformers.trainer_utils import PredictionOutput
from csi_slt.data.processors.slt_processor import SignTranslationProcessor
from csi_slt.engine.sft.metrics import SLTMetric

from accelerate import Accelerator


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))

set_seed(42)


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="eval/base")
def main(cfg: DictConfig):
    # accelerate initialize
    acc = Accelerator()

    # create model
    slt_model = SltModel.from_pretrained(
        cfg.model.checkpoint_dir,
    )

    # create datamodule
    llm_name = slt_model.config.llm_model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(llm_name, map_location="cpu")

    datamodule = DataModule(cfg.data, tokenizer=tokenizer)
    datamodule.setup()

    # generation config
    generation_config_args = OmegaConf.to_container(
        cfg.engine.generation_config, resolve=True
    )
    metrics = SLTMetric(
        processor=datamodule.processor,
    )

    # create trainer
    training_args = SltTrainingArguments(
        generation_config=merge_generation_config(
            slt_model.generation_config,
            generation_config_args,
        ),
        **cfg.engine.training_args,
    )
    trainer = SltTrainer(
        model=slt_model,
        args=training_args,
        hydra_config=cfg,
        processing_class=datamodule.processor,
        compute_metrics=metrics,
    )

    gen_kwargs = {}
    if cfg.experiment.permutation is True:
        gen_kwargs["permute_video_tokens"] = True

    predictions = trainer.predict(
        test_dataset=datamodule.test_dataset,
        test_collator=datamodule.test_collator,
        **gen_kwargs,
    )

    trainer.save_predictions(predictions)


if __name__ == "__main__":
    main()
