from transformers.trainer_seq2seq import Seq2SeqTrainer
from .callbacks import SltTrainerCallbackHandler, SaveBestMetricCallback
from transformers.trainer_utils import EvalLoopOutput
from torch import nn
import torch
from torch.distributed.fsdp import FullyShardedDataParallel
from typing import Any, Optional, Union
import contextlib

from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.integrations.fsdp import is_fsdp_managed_module
from transformers.trainer_utils import seed_worker
from transformers.utils import is_datasets_available


import datasets
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers.utils import logging

from typing import Callable, Literal
from functools import partial

logger = logging.get_logger(__name__)


class SltTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        eval_data_collator=None,
        train_data_collator=None,
        test_data_collator=None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

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

        self.add_callback(SaveBestMetricCallback(metric_name="bleu4"))
        self.callback_handler = SltTrainerCallbackHandler(
            self,
            self.callback_handler.callbacks,  # WARN: replaceing the original callback handler
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )

        self.compute_metrics = self._compute_metrics

        # adjust arguments for seq2seq training
        if self.args.predict_with_generate is False:
            logger.warn(
                "Overriding predict_with_generate to True for Customized Prediction Step"
            )
        self.args.predict_with_generate = True

    @staticmethod
    def _compute_metrics(pred: EvalLoopOutput):
        preds = pred.predictions
        labels = pred.label_ids
        return {"bleu": 0.1, "bleu4": 0.1}

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list[str]] = None,
        **gen_kwargs,
    ) -> tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
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

        default_synced_gpus = is_deepspeed_zero3_enabled() or is_fsdp_managed_module(
            self.model
        )
        gen_kwargs["synced_gpus"] = gen_kwargs.get("synced_gpus", default_synced_gpus)

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape
            == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v
                for k, v in inputs.items()
                if k not in ("decoder_input_ids", "decoder_attention_mask")
            }

        summon_full_params_context = (
            FullyShardedDataParallel.summon_full_params(self.model)
            if isinstance(self.model, FullyShardedDataParallel)
            else contextlib.nullcontext()
        )

        # NOTE: pop possible unused keys for generation
        unsed_keys = ["names", "target_text", "prompts", "input_text", "lang"]
        for key in unsed_keys:
            if key in generation_inputs:
                generation_inputs.pop(key)

        with summon_full_params_context:
            generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_length
            )
        elif (
            gen_config.max_new_tokens is not None
            and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1
        ):
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_config.max_new_tokens + 1
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
        loss = (
            None  # WARN: we do not compute loss during evaluation, so it always be None
        )

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif (
                gen_config.max_new_tokens is not None
                and labels.shape[-1] < gen_config.max_new_tokens + 1
            ):
                labels = self._pad_tensors_to_max_len(
                    labels, gen_config.max_new_tokens + 1
                )
        else:
            labels = None

        return loss, generated_tokens, labels

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
            "multiprocessing_context": "spawn"
            if self.args.dataloader_num_workers > 0
            else None,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if sampler_fn is not None:
                dataloader_params["sampler"] = sampler_fn(dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
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
