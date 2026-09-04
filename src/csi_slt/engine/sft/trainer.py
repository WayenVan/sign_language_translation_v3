from types import MethodType

from accelerate.utils import get_mixed_precision_context_manager
from transformers.trainer_seq2seq import Seq2SeqTrainer
from .callbacks import (
    SltTrainerCallbackHandler,
    ModelInfoCallback,
    LogHydraConfigCallback,
    SaveGitInfoCallback,
    SaveHydraConfigCallback,
    ETACallback,
    EvalInformationVisualizationCallback,
    TrainSubsetMetricsCallback,
)
from torch import nn
import torch
from torch.distributed.fsdp import FSDPModule, FullyShardedDataParallel
from typing import Any, Optional, Union
import contextlib
from omegaconf import OmegaConf

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.trainer_utils import PredictionOutput, seed_worker
from transformers.utils import is_datasets_available


import datasets
from datasets import Dataset
from torch.utils.data import DataLoader, Subset
from transformers.utils import logging
from typing import Callable, Literal, Tuple
from functools import partial
import inspect
import json
import random
import re
import numpy as np
from pathlib import Path

from csi_slt.data.sampler import (
    GlobalLengthBucketSampler,
    get_dataset_lengths,
)
from csi_slt.modeling_slt.info_utils import InformationRequest
from csi_slt.engine.optimization import (
    OptimizationPlan,
    build_optimizer_parameter_groups,
)

logger = logging.get_logger(__name__)


def apply_fsdp2_autocast(accelerator, model: nn.Module) -> None:
    """Re-apply the autocast wrapper that Accelerate skips under FSDP2.

    On every other path ``Accelerator.prepare`` routes the model through
    ``prepare_model``, which wraps ``model.forward`` in ``torch.autocast``
    (``accelerator.py:1774``). ``_prepare_fsdp2`` takes its own branch and never
    calls it, while Transformers' ``autocast_smart_context_manager`` returns a
    null context on purpose because it delegates AMP to Accelerate -- so under
    FSDP2 nothing applies autocast at all.

    FSDP2's ``MixedPrecisionPolicy`` does not cover the gap. It casts parameters
    on all-gather and module inputs on entry, but it has no ``buffer_dtype``
    (FSDP1 did), so arithmetic between a bf16 activation and an fp32 buffer
    promotes the result back to fp32 and the next matmul dies with "expected
    mat1 and mat2 to have the same dtype". The C-RADIO input conditioner does
    exactly that with its ``norm_mean``/``norm_std`` buffers. Autocast resolves
    dtypes per operation, so it covers that whole class rather than one site,
    and it restores the fp32 treatment of softmax/cross-entropy that the earlier
    DDP runs were trained with.

    Accelerate additionally wraps the forward in ``convert_outputs_to_fp32``.
    That is deliberately skipped here: the policy sets ``output_dtype`` to bf16,
    and FSDP's own post-forward cast runs after this wrapper, so converting back
    to fp32 inside it would be undone immediately.
    """
    if not getattr(accelerator, "is_fsdp2", False):
        # Accelerate already wrapped `forward` on the DDP/FSDP1/single-device paths.
        return
    if getattr(model, "_original_forward", None) is not None:
        return

    autocast_context = get_mixed_precision_context_manager(
        accelerator.native_amp, accelerator.autocast_handler
    )
    model._original_forward = model.forward
    if hasattr(model.forward, "__func__"):
        model.forward = MethodType(autocast_context(model.forward.__func__), model)
    else:
        model.forward = autocast_context(model.forward)


class SltTrainer(Seq2SeqTrainer):
    # NOTE: Parameters whose zero point is not "this path is switched off".
    # Transformers only filters biases and normalization layers by name, which
    # misses SLT's marker tokens, learned positions, and adapter token-type
    # embeddings.  These all carry 2 or more dimensions, so the ndim rule below
    # cannot catch them and they must be named explicitly.  Keep the terms
    # specific: a broad pattern such as "gate" would wrongly exempt Qwen3's
    # mlp.gate_proj.weight, which is an ordinary weight matrix.
    _NO_DECAY_NAME_PATTERN = re.compile(
        r"type_embedding|position_embedding|start_video_embds|end_video_embeds"
    )

    def __init__(
        self,
        hydra_config=None,
        eval_data_collator=None,
        train_data_collator=None,
        test_data_collator=None,
        train_probe_compute_metrics=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # ``SltModel.forward`` accepts arbitrary LLM kwargs, which makes
        # Transformers infer that it consumes ``num_items_in_batch``.  Its
        # language-model CE and D-SID losses are already mean-reduced and do
        # not use that value. Keep the standard DDP averaging
        # path; otherwise Trainer multiplies the complete loss by world size
        # when ``average_tokens_across_devices=True``.
        self.model_accepts_loss_kwargs = False

        self.eval_data_collator = (
            eval_data_collator if eval_data_collator is not None else self.data_collator
        )
        self.train_data_collator = (
            train_data_collator
            if train_data_collator is not None
            else self.data_collator
        )
        self.test_data_collator = (
            test_data_collator if test_data_collator is not None else self.data_collator
        )
        self.train_probe_compute_metrics = train_probe_compute_metrics
        self.forward_mode = (
            OmegaConf.select(hydra_config, "engine.forward_mode", default="joint")
            if hydra_config is not None
            else "joint"
        )
        if self.forward_mode not in ("ctc_only", "joint"):
            raise ValueError(
                "engine.forward_mode must be 'ctc_only' or 'joint', got "
                f"{self.forward_mode!r}"
            )
        optimization_config = (
            OmegaConf.select(hydra_config, "engine.optimization", default={})
            if hydra_config is not None
            else {}
        )
        if OmegaConf.is_config(optimization_config):
            optimization_config = OmegaConf.to_container(
                optimization_config, resolve=True
            )
        self.optimization_plan = OptimizationPlan.from_mapping(optimization_config)

        # NOTE: add custom callbacks
        # self.add_callback(
        #     SaveBestMetricCallback(metric_name="test_overall_sentence_bleu_4")
        # )
        self.add_callback(ModelInfoCallback())
        self.add_callback(LogHydraConfigCallback(hydra_config))
        self.add_callback(SaveHydraConfigCallback(hydra_config))
        self.add_callback(SaveGitInfoCallback())
        self.add_callback(ETACallback())
        eval_information_kwargs = {}
        if hydra_config is not None:
            eval_information_config = OmegaConf.select(
                hydra_config,
                "engine.eval_information",
                default=None,
            )
            if eval_information_config is not None:
                eval_information_kwargs = OmegaConf.to_container(
                    eval_information_config,
                    resolve=True,
                )
        self.add_callback(
            EvalInformationVisualizationCallback(**eval_information_kwargs)
        )
        train_probe_kwargs = {}
        if hydra_config is not None:
            train_probe_config = OmegaConf.select(
                hydra_config, "engine.train_probe", default=None
            )
            if train_probe_config is not None:
                train_probe_kwargs = OmegaConf.to_container(
                    train_probe_config, resolve=True
                )
        self.add_callback(TrainSubsetMetricsCallback(**train_probe_kwargs))

        # if _is_peft_model(unwrap_model(self.model)):
        #     self.add_callback(SaveBaseModelInPEFT())

        self.callback_handler = SltTrainerCallbackHandler(
            self,
            self.callback_handler.callbacks,  # WARN: replaceing the original callback handler
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )

        # Accumulate detached per-micro-batch values.  They are reduced in
        # ``log`` so their cadence exactly matches Trainer's ``logging_steps``.
        self._logging_scalar_totals: dict[str, torch.Tensor] = {}
        self._logging_scalar_counts: dict[str, int] = {}

        # adjust arguments for seq2seq training
        if self.args.predict_with_generate is False:
            logger.warning(
                "Overriding predict_with_generate to True for Customized Prediction Step"
            )
        self.args.predict_with_generate = True
        self.hydra_config = hydra_config
        self._is_predicting = False
        self._is_train_probe = False

    def get_decay_parameter_names(self, model: nn.Module) -> list[str]:
        """Narrow the decayed set to genuine weight matrices.

        Weight decay only regularizes a parameter when shrinking it toward zero
        simplifies the function it parameterizes.  On top of the base class's
        bias/normalization filter this drops:

        * every parameter with fewer than two dimensions -- residual gates such
          as ``motion_gate`` and ``visual_scale`` live here.  Gates store a
          pre-sigmoid logit, so decaying them does not close the branch, it
          pulls it toward ``sigmoid(0) = 0.5`` and *opens* it.
        * the explicitly named coordinate-like tensors in
          ``_NO_DECAY_NAME_PATTERN``.

        The rule is applied here rather than in ``create_optimizer`` so that it
        also covers the base-class optimizer path taken when no per-component
        learning rate or weight decay is configured.
        """
        decay_parameters = super().get_decay_parameter_names(model)
        dimensions = {
            name: parameter.ndim for name, parameter in model.named_parameters()
        }
        return [
            name
            for name in decay_parameters
            # Default to 2 so an unexpectedly absent name keeps the base
            # class's decision instead of being silently exempted.
            if dimensions.get(name, 2) >= 2
            and not self._NO_DECAY_NAME_PATTERN.search(name)
        ]

    def create_optimizer(self, model=None) -> torch.optim.Optimizer:
        """Create component groups, resolving overrides against global defaults."""
        if self.optimizer is not None:
            return self.optimizer

        opt_model = self.model if model is None else model
        unwrapped_model = self.accelerator.unwrap_model(opt_model)
        optimizer_grouped_parameters = build_optimizer_parameter_groups(
            model=opt_model,
            ownership_model=unwrapped_model,
            plan=self.optimization_plan,
            default_learning_rate=self.args.learning_rate,
            default_weight_decay=self.args.weight_decay,
            decay_parameter_names=self.get_decay_parameter_names(opt_model),
        )

        optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
            self.args, opt_model
        )
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

    def _prepare_for_training(self, *args, **kwargs):
        """Prepare as usual, then restore autocast on the FSDP2 path."""
        result = super()._prepare_for_training(*args, **kwargs)
        apply_fsdp2_autocast(self.accelerator, self.model)
        return result

    @torch.no_grad()
    def collect_eval_information(self, num_samples: int) -> list[dict[str, Any]]:
        """Collect last-layer LLM attention for the first evaluation samples."""
        if self.eval_dataset is None:
            raise ValueError("evaluation information requires an eval_dataset")
        if isinstance(self.eval_dataset, dict):
            raise TypeError(
                "evaluation information does not yet support a dictionary of datasets"
            )

        sample_count = min(num_samples, len(self.eval_dataset))
        samples = [self.eval_dataset[index] for index in range(sample_count)]
        batch_size = max(1, min(self.args.eval_batch_size, sample_count))

        unwrapped_model = self.accelerator.unwrap_model(self.model)
        forward_parameters = set(inspect.signature(unwrapped_model.forward).parameters)
        was_training = self.model.training
        self.model.eval()
        records = []
        try:
            for offset in range(0, sample_count, batch_size):
                raw_batch = self.eval_data_collator(
                    samples[offset : offset + batch_size]
                )
                model_inputs = {
                    name: value
                    for name, value in raw_batch.items()
                    if name in forward_parameters and name != "information_request"
                }
                model_inputs = self._prepare_inputs(model_inputs)
                current_batch_size = len(samples[offset : offset + batch_size])
                model_inputs["information_request"] = InformationRequest(
                    llm_attentions=True,
                    sample_indices=tuple(range(current_batch_size)),
                    llm_layers=(-1,),
                    reduce_heads=True,
                )
                outputs = self.model(**model_inputs)
                information = outputs.information.detach_to_cpu()
                attention_mask = model_inputs["attention_mask"].detach().cpu()

                for batch_index in range(current_batch_size):
                    records.append(
                        {
                            "sample_index": offset + batch_index,
                            "attention_mask": attention_mask[batch_index],
                            "information": type(information)(
                                llm_attentions=tuple(
                                    layer[batch_index : batch_index + 1]
                                    for layer in information.llm_attentions
                                ),
                                llm_visual_mask=information.llm_visual_mask[
                                    batch_index : batch_index + 1
                                ],
                                visual_lengths=information.visual_lengths[
                                    batch_index : batch_index + 1
                                ],
                                visual_position_ids=(
                                    information.visual_position_ids[
                                        batch_index : batch_index + 1
                                    ]
                                    if information.visual_position_ids is not None
                                    else None
                                ),
                            ),
                        }
                    )
        finally:
            self.model.train(was_training)

        return records

    def evaluate_train_subset(
        self,
        num_samples: int = 200,
        seed: int = 42,
        metric_key_prefix: str = "train_probe",
    ) -> dict[str, float]:
        """Predict a reproducible train subset with an isolated metric object."""
        if self.train_dataset is None:
            raise ValueError("train subset evaluation requires a train_dataset")
        if isinstance(self.train_dataset, dict):
            raise TypeError(
                "train subset evaluation does not support dataset dictionaries"
            )
        if self.train_probe_compute_metrics is None:
            raise ValueError(
                "train subset evaluation requires train_probe_compute_metrics"
            )
        if num_samples <= 0:
            raise ValueError("num_samples must be a positive integer")

        sample_count = min(num_samples, len(self.train_dataset))
        indices = random.Random(seed).sample(
            range(len(self.train_dataset)), sample_count
        )
        subset = Subset(self.train_dataset, indices)

        previous_collator = self.test_data_collator
        previous_compute_metrics = self.compute_metrics
        was_training = self.model.training
        self.test_data_collator = self.eval_data_collator
        self.compute_metrics = self.train_probe_compute_metrics
        self._is_train_probe = True
        try:
            output = self.predict(
                test_dataset=subset,
                metric_key_prefix=metric_key_prefix,
            )
        finally:
            self._is_train_probe = False
            self.test_data_collator = previous_collator
            self.compute_metrics = previous_compute_metrics
            self.model.train(was_training)
        return dict(output.metrics)

    def save_predictions(self, prediction_output: PredictionOutput) -> None:
        """Write decoded predictions, their prompts, and metrics to ``output_dir``.

        Three files are written on the main process, each overwritten on every
        call:

        * ``predictions.jsonl`` -- one decoded sample per line.
        * ``prompts.jsonl`` -- the generation prompt of that same sample.
        * ``predictions_metrics.json`` -- ``prediction_output.metrics``.

        Line ``i`` of the two JSONL files always describes the same sample.
        Both are written in a single pass over the gathered arrays and both
        carry a matching ``index`` field, so the correspondence survives an
        editor that reorders or filters one of the files.

        Decoding is delegated to ``compute_metrics`` rather than repeated here,
        so the saved text is exactly what the reported BLEU was computed from.
        Prompts are the one exception: they are not part of the metric, and are
        decoded here from ``prediction_ids[:prompt_length]`` -- the prompt that
        ``prediction_step`` handed to ``generate``.
        """
        # ``predict`` gathers across ranks, so rank zero already holds every
        # sample and the other ranks would only overwrite it with the same data.
        if not self.is_world_process_zero():
            return

        decoder = self.compute_metrics
        if decoder is None or not hasattr(decoder, "decode_batch"):
            raise TypeError(
                "save_predictions needs compute_metrics to expose decode_batch(), "
                f"got {type(decoder).__name__}"
            )
        batch = decoder.decode_batch(prediction_output)

        prediction_ids, _, prompt_lengths = prediction_output.predictions
        prediction_ids = np.asarray(prediction_ids)
        prompt_lengths = np.asarray(prompt_lengths).reshape(-1)
        if prediction_ids.shape[0] != len(batch):
            raise RuntimeError(
                f"decoded {len(batch)} samples but received "
                f"{prediction_ids.shape[0]} prediction rows"
            )

        # WARN: skip_special_tokens stays False on purpose. The prompt is worth
        # saving precisely because of its structure -- chat template markers,
        # the repeated video placeholder token, and the left padding that
        # batched decoder-only generation inserts are all part of what the
        # model actually saw.
        prompts = self.processing_class.tokenizer.batch_decode(
            [
                prediction_ids[index, : int(prompt_lengths[index])].tolist()
                for index in range(len(batch))
            ],
            skip_special_tokens=False,
        )

        output_path = Path(self.args.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        predictions_path = output_path / "predictions.jsonl"
        prompts_path = output_path / "prompts.jsonl"
        metrics_path = output_path / "predictions_metrics.json"

        with (
            predictions_path.open("w", encoding="utf-8") as predictions_file,
            prompts_path.open("w", encoding="utf-8") as prompts_file,
        ):
            for index in range(len(batch)):
                # ensure_ascii=False keeps Chinese and German references
                # readable instead of escaping them into \uXXXX.
                predictions_file.write(
                    json.dumps(
                        {
                            "index": index,
                            "language": batch.languages[index],
                            "prediction": batch.predictions[index],
                            "reference": batch.references[index],
                            "n_tokens": batch.total_token_counts[index],
                            "n_tokens_generated": batch.generated_token_counts[index],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                prompts_file.write(
                    json.dumps(
                        {
                            "index": index,
                            "prompt_length": int(prompt_lengths[index]),
                            "prompt": prompts[index],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        metrics_path.write_text(
            json.dumps(prediction_output.metrics or {}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(
            "Saved %d predictions to %s, prompts to %s, metrics to %s",
            len(batch),
            predictions_path,
            prompts_path,
            metrics_path,
        )

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[Union[torch.Tensor, int]] = None,
    ):
        """Compute loss and retain model-provided scalar values for logging."""
        model_inputs = {
            name: value
            for name, value in inputs.items()
            if not name.startswith("generation_")
        }
        model_inputs["forward_mode"] = self.forward_mode
        loss, outputs = super().compute_loss(
            model,
            model_inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        logging_scalars = getattr(outputs, "logging_scalars", None)
        if logging_scalars is None and isinstance(outputs, dict):
            logging_scalars = outputs.get("logging_scalars")
        for name, value in (logging_scalars or {}).items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(
                    f"logging_scalars[{name!r}] must be a scalar tensor, got "
                    f"{type(value).__name__}"
                )
            value = value.detach()
            self._logging_scalar_totals[name] = (
                self._logging_scalar_totals.get(name, torch.zeros_like(value)) + value
            )
            self._logging_scalar_counts[name] = (
                self._logging_scalar_counts.get(name, 0) + 1
            )

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Add component LRs and averaged model scalars to training logs."""
        if self.optimizer is not None and "learning_rate" in logs:
            for group in self.optimizer.param_groups:
                component = group.get("slt_component")
                if component is not None:
                    logs[f"learning_rate/{component}"] = float(group["lr"])
        if "loss" in logs and self._logging_scalar_totals:
            for name, total in self._logging_scalar_totals.items():
                count = torch.tensor(
                    self._logging_scalar_counts[name],
                    device=self.args.device,
                    dtype=torch.float,
                )
                global_count = self.accelerator.gather_for_metrics(count).sum().item()
                global_total = self.accelerator.gather_for_metrics(total).sum().item()
                logs[name] = global_total / global_count
            self._logging_scalar_totals.clear()
            self._logging_scalar_counts.clear()

        super().log(logs, start_time=start_time)

    def _prepare_generation_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve evaluation-loop generation kwargs and distributed defaults."""
        if not gen_kwargs and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        else:
            gen_kwargs = gen_kwargs.copy()

        for name in ("num_beams", "max_length", "max_new_tokens"):
            if gen_kwargs.get(name, ...) is None:
                gen_kwargs.pop(name)

        # ``is_fsdp_managed_module`` only recognizes FSDP2 through the
        # ``_is_fsdp_managed_module`` flag that Transformers' own ``apply_fsdp2``
        # sets; Accelerate's ``fully_shard`` path never sets it. Check for the
        # FSDP2 module type directly so generation stays synchronized when the
        # model is sharded by ``accelerate launch --config_file .../fsdp2.yaml``.
        default_synced_gpus = (
            is_deepspeed_zero3_enabled()
            or is_fsdp_managed_module(self.model)
            or isinstance(self.model, FSDPModule)
        )
        gen_kwargs.setdefault("synced_gpus", default_synced_gpus)
        return gen_kwargs

    @staticmethod
    def _build_generation_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        """Select prompt-only and shared video fields from an evaluation batch."""
        generation_field_map = {
            "generation_input_ids": "input_ids",
            "generation_attention_mask": "attention_mask",
            "generation_token_type_ids": "token_type_ids",
        }
        required_fields = (*generation_field_map, "pixel_values", "pixel_values_length")
        missing_fields = [name for name in required_fields if name not in inputs]
        if missing_fields:
            raise ValueError(
                "Evaluation batches are missing required generation fields: "
                + ", ".join(missing_fields)
            )

        generation_inputs = {
            model_name: inputs[batch_name]
            for batch_name, model_name in generation_field_map.items()
        }
        generation_inputs.update(
            pixel_values=inputs["pixel_values"],
            pixel_values_length=inputs["pixel_values_length"],
        )
        return generation_inputs

    @staticmethod
    def _build_teacher_forcing_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
        """Select the complete reference sequence used to compute prediction loss."""
        required_fields = (
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "position_ids",
            "pixel_values",
            "pixel_values_length",
            "labels",
        )
        missing_fields = [name for name in required_fields if name not in inputs]
        if missing_fields:
            raise ValueError(
                "Evaluation batches are missing required teacher-forcing fields: "
                + ", ".join(missing_fields)
            )
        teacher_forcing_inputs = {name: inputs[name] for name in required_fields}
        for name in ("pseudo_gloss_ids", "pseudo_gloss_length"):
            if name in inputs:
                teacher_forcing_inputs[name] = inputs[name]
        return teacher_forcing_inputs

    @staticmethod
    def _padded_ctc_greedy_decode(
        logits: torch.Tensor,
        lengths: torch.Tensor,
        blank_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse packed CTC argmax paths and return padded token sequences."""
        if logits.ndim != 2:
            raise ValueError("ctc_logits must have shape [sum(T_i), vocab_size].")
        lengths = lengths.to(device=logits.device, dtype=torch.long)
        if lengths.ndim != 1 or int(lengths.sum()) != logits.shape[0]:
            raise ValueError("ctc_lengths must describe every row of packed ctc_logits.")

        paths = logits.argmax(dim=-1).split(lengths.tolist())
        sequences = []
        for path in paths:
            collapsed = torch.unique_consecutive(path)
            sequences.append(collapsed[collapsed.ne(blank_id)])
        sequence_lengths = lengths.new_tensor([sequence.numel() for sequence in sequences])
        max_length = max((sequence.numel() for sequence in sequences), default=0)
        padded = torch.full(
            (len(sequences), max_length),
            blank_id,
            dtype=torch.long,
            device=logits.device,
        )
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.numel()] = sequence
        return padded, sequence_lengths

    @staticmethod
    def _pad_packed_sequences(
        token_ids: torch.Tensor,
        lengths: torch.Tensor,
        padding_value: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Turn packed references into gather-safe padded sequences."""
        lengths = lengths.to(device=token_ids.device, dtype=torch.long)
        if token_ids.ndim != 1 or lengths.ndim != 1:
            raise ValueError("Packed CTC references require 1-D token_ids and lengths.")
        if int(lengths.sum()) != token_ids.numel():
            raise ValueError("pseudo_gloss_length does not match pseudo_gloss_ids.")
        sequences = token_ids.split(lengths.tolist())
        max_length = int(lengths.max()) if lengths.numel() else 0
        padded = torch.full(
            (lengths.numel(), max_length),
            padding_value,
            dtype=token_ids.dtype,
            device=token_ids.device,
        )
        for index, sequence in enumerate(sequences):
            padded[index, : sequence.numel()] = sequence
        return padded, lengths

    def _prediction_step_ctc_only(
        self,
        model: nn.Module,
        inputs: dict[str, Any],
        prediction_loss_only: bool,
    ):
        """Evaluate Phase A with one CTC forward and no language generation."""
        required_fields = (
            "pixel_values",
            "pixel_values_length",
            "pseudo_gloss_ids",
            "pseudo_gloss_length",
        )
        missing_fields = [name for name in required_fields if name not in inputs]
        if missing_fields:
            raise ValueError(
                "CTC-only evaluation batches are missing required fields: "
                + ", ".join(missing_fields)
            )
        model_inputs = {name: inputs[name] for name in required_fields}
        model_inputs["forward_mode"] = "ctc_only"
        with torch.no_grad():
            with self.compute_loss_context_manager():
                outputs = model(**model_inputs)

        def output_value(name: str):
            return (
                outputs.get(name)
                if isinstance(outputs, dict)
                else getattr(outputs, name, None)
            )

        loss = output_value("loss")
        if loss is not None:
            loss = loss.detach().mean()
        if prediction_loss_only:
            return loss, None, None

        ctc_logits = output_value("logits")
        ctc_lengths = output_value("lengths")
        if ctc_logits is None or ctc_lengths is None:
            raise RuntimeError(
                "ctc_only forward must return packed logits and sequence lengths."
            )
        blank_id = int(model.config.ctc_blank_id)
        predictions = self._padded_ctc_greedy_decode(
            ctc_logits.detach(), ctc_lengths.detach(), blank_id
        )
        references = self._pad_packed_sequences(
            inputs["pseudo_gloss_ids"],
            inputs["pseudo_gloss_length"],
            blank_id,
        )
        return loss, predictions, references

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[
        Optional[float],
        Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
        Optional[torch.Tensor],
    ]:
        # NOTE: this method is modified to support text generation during evaluation
        # I REMOVE the label prediction part
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate and self.forward_mode != "ctc_only":
            raise NotImplementedError(
                "Only `predict_with_generate=True` is implemented in SltTrainer."
            )

        inputs = self._prepare_inputs(inputs)
        if self.forward_mode == "ctc_only":
            return self._prediction_step_ctc_only(
                model,
                inputs,
                prediction_loss_only,
            )
        if prediction_loss_only:
            raise NotImplementedError(
                "Joint prediction_loss_only without generation is not implemented."
            )
        gen_kwargs = self._prepare_generation_kwargs(gen_kwargs)
        generation_inputs = self._build_generation_inputs(inputs)
        lang_ids = inputs.get("lang_ids")

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )
        with summon_full_params_context:
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        loss = None
        ctc_predictions = None
        ctc_references = None
        needs_ctc_metrics = bool(
            getattr(self.compute_metrics, "requires_ctc_outputs", False)
        )
        needs_teacher_forcing = needs_ctc_metrics or (
            self._is_predicting
            and getattr(self.args, "predict_with_teacher_forcing", False)
        )
        if needs_teacher_forcing:
            teacher_forcing_inputs = self._build_teacher_forcing_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    outputs = model(**teacher_forcing_inputs)
            if self.label_smoother is not None:
                loss = (
                    self.label_smoother(outputs, teacher_forcing_inputs["labels"])
                    .detach()
                    .mean()
                )
            else:
                loss_value = (
                    outputs.get("loss")
                    if isinstance(outputs, dict)
                    else getattr(outputs, "loss", None)
                )
                if loss_value is None:
                    raise RuntimeError(
                        "The teacher-forcing forward pass did not return a loss"
                    )
                loss = loss_value.detach().mean()

            if needs_ctc_metrics:
                ctc_logits = getattr(outputs, "ctc_logits", None)
                ctc_lengths = getattr(outputs, "ctc_lengths", None)
                pseudo_gloss_ids = inputs.get("pseudo_gloss_ids")
                pseudo_gloss_length = inputs.get("pseudo_gloss_length")
                if ctc_logits is None or ctc_lengths is None:
                    raise RuntimeError(
                        "CTC metrics require forward() to return ctc_logits and ctc_lengths."
                    )
                if pseudo_gloss_ids is None or pseudo_gloss_length is None:
                    raise ValueError(
                        "CTC metrics require pseudo_gloss_ids and pseudo_gloss_length."
                    )
                blank_id = int(model.config.ctc_blank_id)
                ctc_predictions = self._padded_ctc_greedy_decode(
                    ctc_logits.detach(), ctc_lengths.detach(), blank_id
                )
                ctc_references = self._pad_packed_sequences(
                    pseudo_gloss_ids, pseudo_gloss_length, blank_id
                )

        # Avoid rebuilding GenerationConfig from the model config for every batch.
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        labels = inputs.get("labels")
        batch_size = generated_tokens.shape[0]
        prompt_length_value = generation_inputs["input_ids"].shape[1]
        generated_sequence_lengths = torch.full(
            (batch_size,),
            generated_tokens.shape[1],
            dtype=torch.long,
            device=generated_tokens.device,
        )
        prompt_lengths = torch.full(
            (batch_size,),
            prompt_length_value,
            dtype=torch.long,
            device=generated_tokens.device,
        )

        predictions = (generated_tokens, generated_sequence_lengths, prompt_lengths)
        label_output = (labels, lang_ids if lang_ids is not None else None)
        if needs_ctc_metrics:
            predictions = (*predictions, *ctc_predictions)
            label_output = (*label_output, *ctc_references)

        return loss, predictions, label_output

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
        mode: Literal["train", "eval", "test"] = "train",
        persistent_workers: Optional[bool] = None,
    ) -> DataLoader:
        """Create a [`~torch.utils.data.DataLoader`] from the given dataset."""

        if mode == "train":
            data_collator = self.train_data_collator
        elif mode == "eval":
            data_collator = self.eval_data_collator
        elif mode == "test":
            data_collator = self.test_data_collator

        if is_datasets_available() and isinstance(dataset, datasets.Dataset):
            dataset = self._remove_unused_columns(dataset, description=description)
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description=description
            )

        if persistent_workers is None:
            persistent_workers = self.args.dataloader_persistent_workers

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            # PyTorch only accepts prefetch_factor when multiprocessing workers
            # are enabled. Keeping the single-process path valid is useful for
            # diagnosing worker deadlocks in distributed jobs.
            if self.args.dataloader_num_workers > 0:
                dataloader_params["prefetch_factor"] = (
                    self.args.dataloader_prefetch_factor
                )
            if is_training:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )

        dataloader = self.accelerator.prepare(DataLoader(dataset, **dataloader_params))

        # Store the prepared dataloader for subsequent evaluations if using persistent workers.
        if dataloader_key is not None and self.args.dataloader_persistent_workers:
            if hasattr(self, "_eval_dataloaders"):
                self._eval_dataloaders[dataloader_key] = dataloader
            else:
                self._eval_dataloaders = {dataloader_key: dataloader}

        return dataloader

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset

        video_lengths, label_ids_lengths = get_dataset_lengths(dataset)

        # calculate total lenghts
        lengths = (
            video_lengths * self.model.config.video_token_scale + label_ids_lengths
        )
        seed = (
            self.args.data_seed if self.args.data_seed is not None else self.args.seed
        )

        sampler = GlobalLengthBucketSampler(
            lengths=lengths,
            per_device_batch_size=self._train_batch_size,
            num_processes=self.accelerator.num_processes,
            seed=seed,
            drop_last=self.args.dataloader_drop_last,
            balance_batches=True,
        )

        logger.warning(
            "⚠️ using global length bucketing: "
            f"dataset_size={len(dataset)}, "
            f"per_device_batch_size={self._train_batch_size}, "
            f"num_processes={self.accelerator.num_processes}, "
            f"global_batch_size={sampler.global_batch_size}"
        )

        return sampler

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return self._get_dataloader(
            dataset=self.train_dataset,
            description="Training",
            batch_size=self._train_batch_size,
            sampler_fn=self._get_train_sampler,
            is_training=True,
            mode="train",
        )

    def get_eval_dataloader(
        self, eval_dataset: Optional[Union[str, Dataset]] = None
    ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`str` or `torch.utils.data.Dataset`, *optional*):
                If a `str`, will use `self.eval_dataset[eval_dataset]` as the evaluation dataset. If a `Dataset`, will override `self.eval_dataset` and must implement `__len__`. If it is a [`~datasets.Dataset`], columns not accepted by the `model.forward()` method are automatically removed.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        # If we have persistent workers, don't do a fork bomb especially as eval datasets
        # don't change during training
        dataloader_key = eval_dataset if isinstance(eval_dataset, str) else "eval"
        if (
            hasattr(self, "_eval_dataloaders")
            and dataloader_key in self._eval_dataloaders
            and self.args.dataloader_persistent_workers
        ):
            return self._eval_dataloaders[dataloader_key]

        eval_dataset = (
            self.eval_dataset[eval_dataset]
            if isinstance(eval_dataset, str)
            else eval_dataset
            if eval_dataset is not None
            else self.eval_dataset
        )

        return self._get_dataloader(
            dataset=eval_dataset,
            description="Evaluation",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            dataloader_key=dataloader_key,
            mode="eval",
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        """
        Returns the test [`~torch.utils.data.DataLoader`].

        Prediction arrays are positionally aligned with ``test_dataset`` when
        the evaluation sampler remains sequential (no length grouping or
        custom shuffled sampler), ``test_dataset`` is not iterable, and
        ``dataloader_drop_last`` is false. Under those conditions Accelerate
        preserves/restores dataset order when gathering distributed results,
        so prediction item ``i`` corresponds to ``test_dataset[i]``.

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            test_dataset (`torch.utils.data.Dataset`, *optional*):
                The test dataset to use. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        return self._get_dataloader(
            dataset=test_dataset,
            description="test",
            batch_size=self.args.eval_batch_size,
            sampler_fn=self._get_eval_sampler,
            mode="test",
            # A train probe creates a short-lived test DataLoader on every
            # evaluation. Its workers must exit after the probe rather than
            # inheriting the long-lived train/eval setting.
            persistent_workers=False if self._is_train_probe else None,
        )

    def predict(self, test_dataset: Dataset, test_collator=None, **gen_kwargs):
        if test_collator is not None:
            self.test_data_collator = test_collator
        self._is_predicting = True
        try:
            return super().predict(test_dataset=test_dataset, **gen_kwargs)
        finally:
            self._is_predicting = False
