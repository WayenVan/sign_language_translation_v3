from transformers.trainer_seq2seq import Seq2SeqTrainer
from .callbacks import (
    SltTrainerCallbackHandler,
    ModelInfoCallback,
    LogHydraConfigCallback,
    SaveGitInfoCallback,
    SaveHydraConfigCallback,
    ETACallback,
    DSIDWeightSchedulerCallback,
    EvalInformationVisualizationCallback,
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
import inspect

from csi_slt.data.sampler import (
    GlobalLengthBucketSampler,
    get_dataset_lengths,
)
from csi_slt.modeling_slt.info_utils import InformationRequest

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
        self._is_predicting = False

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

    def _prepare_generation_kwargs(self, gen_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Resolve evaluation-loop generation kwargs and distributed defaults."""
        if not gen_kwargs and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        else:
            gen_kwargs = gen_kwargs.copy()

        for name in ("num_beams", "max_length", "max_new_tokens"):
            if gen_kwargs.get(name, ...) is None:
                gen_kwargs.pop(name)

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(
            self.model
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
        return {name: inputs[name] for name in required_fields}

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

        if not self.args.predict_with_generate or prediction_loss_only:
            raise NotImplementedError(
                "Only `predict_with_generate=True` is implemented in SltTrainer."
            )

        inputs = self._prepare_inputs(inputs)
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
        if self._is_predicting and getattr(
            self.args, "predict_with_teacher_forcing", False
        ):
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

        return (
            loss,
            (generated_tokens, generated_sequence_lengths, prompt_lengths),
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
        self._is_predicting = True
        try:
            return super().predict(test_dataset=test_dataset, **gen_kwargs)
        finally:
            self._is_predicting = False
