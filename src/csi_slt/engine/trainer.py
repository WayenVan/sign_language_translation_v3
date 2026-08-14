from transformers.trainer_seq2seq import Seq2SeqTrainer
from .callbacks import (
    SltTrainerCallbackHandler,
    ModelInfoCallback,
    LogHydraConfigCallback,
    SaveGitInfoCallback,
    SaveHydraConfigCallback,
    ETACallback,
    DSIDWeightSchedulerCallback,
)
from torch import nn
import torch
from torch.distributed.fsdp import FullyShardedDataParallel
from typing import Any, Optional, Union
import contextlib
from omegaconf import OmegaConf

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available


import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.utils import logging
from typing import Callable, Literal, Tuple
from functools import partial

from csi_slt.data.sampler import (
    GlobalLengthBucketSampler,
    get_dataset_lengths,
)

logger = logging.get_logger(__name__)


class SltTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        hydra_config=None,
        eval_data_collator=None,
        train_data_collator=None,
        test_data_collator=None,
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

        # NOTE: add custom callbacks
        # self.add_callback(
        #     SaveBestMetricCallback(metric_name="test_overall_sentence_bleu_4")
        # )
        self.add_callback(ModelInfoCallback())
        self.add_callback(LogHydraConfigCallback(hydra_config))
        self.add_callback(SaveHydraConfigCallback(hydra_config))
        self.add_callback(SaveGitInfoCallback())
        self.add_callback(ETACallback())
        dsid_scheduler_kwargs = {}
        if hydra_config is not None:
            dsid_scheduler_config = OmegaConf.select(
                hydra_config,
                "engine.dsid_scheduler",
                default=None,
            )
            if dsid_scheduler_config is not None:
                dsid_scheduler_kwargs = OmegaConf.to_container(
                    dsid_scheduler_config,
                    resolve=True,
                )
        self.add_callback(DSIDWeightSchedulerCallback(**dsid_scheduler_kwargs))

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
        self._loss_component_totals: dict[str, torch.Tensor] = {}
        self._loss_component_count = 0

        # adjust arguments for seq2seq training
        if self.args.predict_with_generate is False:
            logger.warning(
                "Overriding predict_with_generate to True for Customized Prediction Step"
            )
        self.args.predict_with_generate = True
        self.hydra_config = hydra_config

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch: Optional[Union[torch.Tensor, int]] = None,
    ):
        """Compute the optimization loss and retain individual loss terms for logging."""
        model_inputs = {
            name: value
            for name, value in inputs.items()
            if not name.startswith("generation_")
        }
        loss, outputs = super().compute_loss(
            model,
            model_inputs,
            return_outputs=True,
            num_items_in_batch=num_items_in_batch,
        )

        loss_info = getattr(outputs, "loss_info", None)
        if loss_info is None and isinstance(outputs, dict):
            loss_info = outputs.get("loss_info")
        for name, value in (loss_info or {}).items():
            if not isinstance(value, torch.Tensor) or value.numel() != 1:
                raise TypeError(
                    f"loss_info[{name!r}] must be a scalar tensor, got "
                    f"{type(value).__name__}"
                )
            value = value.detach()
            self._loss_component_totals[name] = (
                self._loss_component_totals.get(name, torch.zeros_like(value)) + value
            )
        self._loss_component_count += 1

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        """Add detached loss components to training logs."""
        if "loss" in logs and self._loss_component_count:
            count = torch.tensor(
                self._loss_component_count, device=self.args.device, dtype=torch.float
            )
            global_count = self.accelerator.gather_for_metrics(count).sum().item()
            for name, total in self._loss_component_totals.items():
                global_total = self.accelerator.gather_for_metrics(total).sum().item()
                logs[name] = global_total / global_count
            self._loss_component_totals.clear()
            self._loss_component_count = 0

        super().log(logs, start_time=start_time)

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

        try:
            if not self.args.predict_with_generate or prediction_loss_only:
                raise NotImplementedError(
                    "Only `predict_with_generate=True` is implemented in `SltTrainer` for now."
                )
                # WARN: this should not happen during evaluation
                return super().prediction_step(
                    model,
                    inputs,
                    prediction_loss_only=prediction_loss_only,
                    ignore_keys=ignore_keys,
                )

            has_labels = "labels" in inputs
            inputs = self._prepare_inputs(inputs)

            # Priority (handled in generate):
            # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
            if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
                gen_kwargs = self._gen_kwargs.copy()
            if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
                gen_kwargs.pop("num_beams")
            if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
                gen_kwargs.pop("max_length")

            default_synced_gpus = (
                is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self.model)
            )
            gen_kwargs["synced_gpus"] = gen_kwargs.get(
                "synced_gpus", default_synced_gpus
            )

            generation_field_map = {
                "generation_input_ids": "input_ids",
                "generation_attention_mask": "attention_mask",
                "generation_token_type_ids": "token_type_ids",
            }
            missing_generation_fields = [
                name for name in generation_field_map if name not in inputs
            ]
            if missing_generation_fields:
                raise ValueError(
                    "Evaluation batches must provide prompt-only generation "
                    "fields; missing: " + ", ".join(missing_generation_fields)
                )

            generation_inputs = {
                model_name: inputs[batch_name]
                for batch_name, model_name in generation_field_map.items()
            }
            for shared_name in ("pixel_values", "pixel_values_length"):
                if shared_name not in inputs:
                    raise ValueError(
                        f"Evaluation batch is missing required {shared_name!r}"
                    )
                generation_inputs[shared_name] = inputs[shared_name]

            summon_full_params_context = (
                FullyShardedDataParallel.summon_full_params(self.model)
                if isinstance(self.model, FullyShardedDataParallel)
                else contextlib.nullcontext()
            )

            # NOTE: language_ids will be save per batch
            lang_ids = inputs.get("lang_ids")

            with summon_full_params_context:
                generated_tokens = self.model.generate(
                    **generation_inputs, **gen_kwargs
                )

            # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
            # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
            # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
            if self.model.generation_config._from_model_config:
                self.model.generation_config._from_model_config = False

            # Retrieves GenerationConfig from model.generation_config
            gen_config = self.model.generation_config
            default_gen_config = gen_config._get_default_generation_params()
            gen_config.update(**default_gen_config, defaults_only=True)

            prompt_length_value = generation_inputs["input_ids"].shape[1]
            max_new_tokens = gen_kwargs.get("max_new_tokens")
            if max_new_tokens is None:
                max_new_tokens = gen_config.max_new_tokens

            max_length = gen_kwargs.get("max_length")
            if max_length is None:
                max_length = gen_config.max_length

            target_length = (
                prompt_length_value + max_new_tokens
                if max_new_tokens is not None
                else max_length
            )

            # Pad decoder-only outputs to prompt length plus the generation budget.
            if target_length is not None and generated_tokens.shape[-1] < target_length:
                generated_tokens = self._pad_tensors_to_max_len(
                    generated_tokens, target_length
                )

                # with torch.no_grad():
                #     if has_labels:
                #         with self.compute_loss_context_manager():
                #             outputs = model(**inputs)
                #         if self.label_smoother is not None:
                #             loss = (
                #                 self.label_smoother(outputs, inputs["labels"]).detach().mean()
                #             )
                #         else:
                #             loss = (
                #                 (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                #                 .detach()
                #                 .mean()
                #             )
                #     else:
            loss = None  # WARN: we do not compute loss during evaluation, so it always be None

            if self.args.prediction_loss_only:
                return loss, None, None

            if has_labels:
                labels = inputs["labels"]
                if target_length is not None and labels.shape[-1] < target_length:
                    labels = self._pad_tensors_to_max_len(labels, target_length)
            else:
                labels = None

            B = generated_tokens.shape[0]
            generated_batch_size = torch.full(
                (B,),
                generated_tokens.shape[1],
                dtype=torch.long,
                device=generated_tokens.device,
            )
            prompt_length = torch.full(
                (B,),
                prompt_length_value,
                dtype=torch.long,
                device=generated_tokens.device,
            )
        except Exception as e:
            import traceback
            import sys

            traceback.print_exc(file=sys.stderr)
            raise e

        return (
            loss,
            (generated_tokens, generated_batch_size, prompt_length),
            (labels, lang_ids if lang_ids is not None else None),
        )

    def _get_dataloader(
        self,
        dataset: Dataset,
        description: str,
        batch_size: int,
        sampler_fn: Optional[Callable[[Dataset], torch.utils.data.Sampler]] = None,
        is_training: bool = False,
        dataloader_key: Optional[str] = None,
        mode: Literal["train", "eval", "test"] = "train",
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

        dataloader_params = {
            "batch_size": batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
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
        )

    def predict(self, test_dataset: Dataset, test_collator=None, **gen_kwargs):
        if test_collator is not None:
            self.test_data_collator = test_collator
        return super().predict(test_dataset=test_dataset, **gen_kwargs)
