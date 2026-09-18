from __future__ import annotations

import os
from pathlib import Path

import hydra
from accelerate import Accelerator
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, set_seed

from csi_slt.data.datamodule import DataModule
from csi_slt.commands.config import (
    build_slt_trainer_kwargs,
    instantiate_prompt_resolvers,
)
from csi_slt.engine.sft.metrics import SLTMetric
from csi_slt.engine.sft.trainer import SltTrainer, apply_fsdp2_autocast
from csi_slt.engine.sft.training_args import SltTrainingArguments
from csi_slt.modeling_slt.misc import validate_rope_buffers
from csi_slt.modeling_slt.slt import SltModel
from csi_slt.utils.checkpoint_dtypes import restore_checkpoint_dtypes
from csi_slt.utils.generation_config import merge_generation_config


DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.getcwd(), "configs"))


def prepare_model_for_fsdp2(trainer: SltTrainer) -> None:
    """Shard the model up-front when predicting without a preceding training run.

    ``Trainer.evaluation_loop`` prepares the model itself when eval is called
    without ``train()``, but that path is incompatible with FSDP2 in two ways:
    it calls ``accelerator.prepare(model)`` with no optimizer, which Accelerate
    rejects outright, and it never runs ``register_fsdp_forward_method``, which
    Transformers only does inside ``_prepare_for_training`` -- without it
    ``generate`` mixes plain tensors with the sharded ``DTensor`` parameters.
    Preparing here also populates ``accelerator._models``, so the evaluation
    loop skips its own preparation.

    The optimizer is never stepped; it only exists to satisfy Accelerate's
    "model and optimizer must be prepared together" invariant for FSDP2.
    """
    accelerator = trainer.accelerator
    if not getattr(accelerator, "is_fsdp2", False):
        return

    from torch.distributed.fsdp import register_fsdp_forward_method

    trainer.create_optimizer()
    model, trainer.optimizer = accelerator.prepare(trainer.model, trainer.optimizer)
    trainer.model = trainer.model_wrapped = model
    register_fsdp_forward_method(model, "generate")
    apply_fsdp2_autocast(accelerator, model)


@hydra.main(version_base=None, config_path=DEFAULT_CONFIG_PATH, config_name="eval/base")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.seed))

    # Initializing the process group before the checkpoint is read lets
    # Transformers' ``is_fsdp_enabled()`` take effect, so only local rank 0
    # materializes the weights instead of every rank holding a full copy.
    Accelerator()

    # ``checkpoint`` loads at torch's default dtype and then puts every tensor
    # back to the dtype the checkpoint stores, which is also the dtype layout
    # training ran with. ``csi_slt.utils.checkpoint_dtypes`` explains why
    # ``auto`` cannot be trusted for that; any explicit dtype still casts the
    # whole model.
    if cfg.engine.model_dtype == "checkpoint":
        slt_model = SltModel.from_pretrained(cfg.model.checkpoint_dir)
        restore_checkpoint_dtypes(slt_model, cfg.model.checkpoint_dir)
    else:
        slt_model = SltModel.from_pretrained(
            cfg.model.checkpoint_dir,
            dtype=cfg.engine.model_dtype,
        )

    # A rotary buffer that was never recomputed costs no error and all of the
    # quality, so it is checked before anything expensive runs.
    validate_rope_buffers(slt_model)

    tokenizer = AutoTokenizer.from_pretrained(
        slt_model.config.llm_model_name_or_path,
        config=slt_model.config.llm_config,
    )

    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(cfg.prompt, ("test",)),
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
        **build_slt_trainer_kwargs(cfg),
        model=slt_model,
        args=training_args,
        hydra_config=cfg,
        processing_class=datamodule.processor,
        test_data_collator=datamodule.test_collator,
        compute_metrics=metrics,
    )

    prepare_model_for_fsdp2(trainer)

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
    trainer.save_predictions(predictions)

    # Record what was actually evaluated. ``SaveHydraConfigCallback`` only
    # fires ``on_save``, so a predict-only run would otherwise leave no trace
    # of its checkpoint, prompt IDs, or generation settings next to its
    # predictions -- and a prompt suite is exactly one output directory per
    # prompt variant. ``csi_slt.commands.summarize_prompt_suite`` reads this.
    if trainer.is_world_process_zero():
        output_dir = Path(training_args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "eval_config.yaml")


if __name__ == "__main__":
    main()
