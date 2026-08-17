from __future__ import annotations

import os
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, set_seed

from csi_slt.data.datamodule import DataModule
from csi_slt.engine.sft.metrics import SLTMetric
from csi_slt.engine.sft.trainer import SltTrainer
from csi_slt.engine.sft.training_args import SltTrainingArguments
from csi_slt.modeling_slt.slt import SltModel
from csi_slt.utils.generation_config import merge_generation_config


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))


def save_prediction_arrays(trainer: SltTrainer, prediction_output) -> None:
    """Save gathered prediction and reference arrays on the main process."""
    if not trainer.is_world_process_zero():
        return

    prediction_ids, sequence_lengths, prompt_lengths = prediction_output.predictions
    label_ids, language_ids = prediction_output.label_ids
    output_path = Path(trainer.args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path / "predictions.npz",
        prediction_ids=prediction_ids,
        sequence_lengths=sequence_lengths,
        prompt_lengths=prompt_lengths,
        label_ids=label_ids,
        language_ids=language_ids,
    )


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="eval/base")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))

    slt_model = SltModel.from_pretrained(
        cfg.model.checkpoint_dir,
        dtype=cfg.engine.model_dtype,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        slt_model.config.llm_model_name_or_path,
        config=slt_model.config.llm_config,
    )

    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
    )
    datamodule.setup("predict")

    generation_config_args = OmegaConf.to_container(
        cfg.engine.generation_config,
        resolve=True,
    )
    training_args = SltTrainingArguments(
        generation_config=merge_generation_config(
            slt_model.generation_config,
            generation_config_args,
        ),
        **OmegaConf.to_container(cfg.engine.training_args, resolve=True),
    )
    metrics = SLTMetric(processor=datamodule.processor)
    trainer = SltTrainer(
        model=slt_model,
        args=training_args,
        hydra_config=cfg,
        processing_class=datamodule.processor,
        test_data_collator=datamodule.test_collator,
        compute_metrics=metrics,
    )

    generation_kwargs = {}
    if cfg.experiment.permutation:
        generation_kwargs["permute_video_tokens"] = True

    predictions = trainer.predict(
        test_dataset=datamodule.test_dataset,
        metric_key_prefix="test",
        **generation_kwargs,
    )
    trainer.log_metrics("test", predictions.metrics)
    trainer.save_metrics("test", predictions.metrics)
    save_prediction_arrays(trainer, predictions)


if __name__ == "__main__":
    main()
