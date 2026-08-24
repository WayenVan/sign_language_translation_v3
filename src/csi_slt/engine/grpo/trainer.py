"""Educational TRL/GRPO adapter for :class:`~csi_slt.modeling_slt.slt.SltModel`.

The module is intentionally isolated from the existing supervised trainer.  It
contains the project-specific mechanics needed by an eventual GRPO integration:

* convert raw video examples with ``SignTranslationProcessor``;
* repeat packed, variable-length videos for multiple completions;
* call SLT generation and policy/reference forward passes consistently.

It is a scaffold, not yet a drop-in training entry point. In particular,
``_generate_and_score_completions`` remains the TRL integration seam because its
exact signature and returned dictionary vary between TRL releases.

Distributed reward handling gathers only the small BLEU tensors, computes
advantages from globally complete ``num_generations`` groups, and slices the
advantages back to each process before local packed-video shuffling. Rollouts,
``pixel_values``, and micro-batch construction always remain GPU-local.

TODO (evaluation): Validate checkpoints over the complete test dataset in a
separate generation/evaluation pass, using ``csi_slt.engine.sft.metrics`` to
compute corpus-level BLEU. Do not treat the mean rollout sentence-BLEU reward
as the test-set BLEU metric.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from transformers import Trainer, logging

try:
    from trl import GRPOTrainer
except ImportError as error:  # pragma: no cover - depends on an optional package
    raise ImportError(
        "SltGRPOTrainer requires TRL. Install a TRL version compatible with the "
        "project's Transformers version before importing this module."
    ) from error

from csi_slt.constants import LANGUAGE_MAP
from csi_slt.data.processors.slt_processor import SignTranslationProcessor
from csi_slt.engine.prompt_resolver import PromptResolver
from csi_slt.engine.sft.callbacks import (
    ModelInfoCallback,
    SltTrainerCallbackHandler,
)
from csi_slt.modeling_slt.slt import SltModel

from .reward import SentenceBLEUReward
from .training_args import SltGRPOConfig

logger = logging.get_logger(__name__)


def _unused_slt_reward_placeholder(*args: Any, **kwargs: Any) -> list[float]:
    """Satisfy TRL's reward-source check; SLT rewards are computed internally."""
    del args, kwargs
    raise RuntimeError(
        "The TRL reward placeholder must not be called; SltGRPOTrainer computes "
        "rewards in _calculate_slt_rewards()."
    )


class SltGRPOTrainer(GRPOTrainer):
    """Base framework for applying TRL GRPO to packed sign-language videos.

    This class does not alter ``SltModel`` or the existing ``SltTrainer``.  The
    helper methods are intentionally small so they can be read independently
    and then connected to the installed TRL version's rollout hook.
    """

    def __init__(
        self,
        model: SltModel,
        *args: Any,
        processing_class: SignTranslationProcessor,
        train_prompt_resolver: PromptResolver,
        eval_prompt_resolver: PromptResolver,
        **kwargs: Any,
    ) -> None:
        grpo_args = kwargs.get("args")
        if grpo_args is None and args:
            # GRPOTrainer accepts ``args`` positionally, although keyword use is
            # clearer. Validation below is best-effort for that less common form.
            grpo_args = args[1] if len(args) > 1 else None
        peft_config = kwargs.get("peft_config")
        # In GRPOTrainer's positional signature, peft_config is the tenth
        # argument after model. Keyword usage is strongly preferred, but guard
        # the positional form as well.
        if peft_config is None and len(args) > 9:
            peft_config = args[9]
        if model.config.llm_lora and peft_config is not None:
            raise ValueError(
                "SltModel already contains LoRA inside model.llm; do not pass "
                "peft_config to SltGRPOTrainer because TRL would wrap the outer "
                "model in a second PEFT layer."
            )
        self._validate_slt_setup(model, processing_class, grpo_args)
        self.train_prompt_resolver = train_prompt_resolver
        self.eval_prompt_resolver = eval_prompt_resolver

        # This adapter computes processor-aligned sentence BLEU internally, but
        # upstream TRL requires at least one reward source during construction.
        # Install an unreachable placeholder unless the caller supplied one.
        positional_args = list(args)
        if kwargs.get("reward_funcs") is None:
            if positional_args:
                if positional_args[0] is None:
                    positional_args[0] = _unused_slt_reward_placeholder
            else:
                kwargs["reward_funcs"] = _unused_slt_reward_placeholder
        super().__init__(
            model,
            *positional_args,
            processing_class=processing_class,
            **kwargs,
        )

        # ---------------------------------------------------------------------
        # Reuse the SFT model-information callback before the first rollout.
        # Its handler passes the trainer instance expected by ModelInfoCallback.
        # ---------------------------------------------------------------------
        self.add_callback(ModelInfoCallback())
        self.callback_handler = SltTrainerCallbackHandler(
            self,
            self.callback_handler.callbacks,
            self.model,
            self.processing_class,
            self.optimizer,
            self.lr_scheduler,
        )
        self.sentence_bleu_reward = SentenceBLEUReward(max_ngram_order=4)

    @staticmethod
    def _validate_slt_setup(
        model: SltModel,
        processor: SignTranslationProcessor,
        args: SltGRPOConfig | None,
    ) -> None:
        if not isinstance(model, SltModel):
            raise TypeError("SltGRPOTrainer expects an instantiated SltModel")
        if not isinstance(processor, SignTranslationProcessor):
            raise TypeError(
                "processing_class must be an instantiated SignTranslationProcessor"
            )
        if model.config.video_soft_token_id != processor.video_soft_token_id:
            raise ValueError(
                "model and processor use different video placeholder token IDs"
            )
        if args is not None and getattr(args, "use_vllm", False):
            raise ValueError("The SLT scaffold does not support vLLM generation")
        if args is not None and getattr(args, "use_liger_kernel", False):
            raise ValueError(
                "SltGRPOTrainer does not support use_liger_kernel=True because "
                "the Liger loss path bypasses the multimodal SLT loss inputs"
            )
        if args is not None and not getattr(args, "disable_dropout", False):
            warnings.warn(
                "⚠️ SltGRPOTrainer recommends disable_dropout=True; active "
                "dropout can make old-policy and current-policy log probabilities "
                "differ before any parameter update.",
                UserWarning,
                stacklevel=2,
            )
        if args is not None and getattr(args, "beta", 0.0) != 0.0:
            raise ValueError("The first SLT GRPO version requires beta=0.0")
        if args is not None and getattr(args, "slt_disable_auxiliary_losses", True):
            previous_dsid_weight = float(model.config.dsid_loss_weight)
            previous_diversity_weight = float(
                model.config.attention_diversity_loss_weight
            )
            model.config.dsid_loss_weight = 0.0
            model.config.attention_diversity_loss_weight = 0.0
            model._current_dsid_loss_weight = 0.0
            logger.warning(
                "⚠️ slt_disable_auxiliary_losses=True: forced all SltModel "
                "auxiliary losses off for GRPO (dsid_loss_weight: %s -> 0.0, "
                "attention_diversity_loss_weight: %s -> 0.0, scheduled D-SID "
                "runtime weight -> 0.0).",
                previous_dsid_weight,
                previous_diversity_weight,
            )

    @property
    def slt_processor(self) -> SignTranslationProcessor:
        """Return the processor with a useful static/runtime type."""
        return self.processing_class

    def prepare_slt_batch(
        self,
        examples: Sequence[Mapping[str, Any]],
        *,
        training: bool = False,
    ) -> dict[str, Any]:
        """Convert raw dataset rows into one packed SLT prompt batch.

        Expected row keys are ``video``, ``src_lang``, and ``text``. The policy
        receives prompt-only ``generation_*`` tensors; processor labels and
        language IDs are retained exclusively for reward calculation.
        """
        if not examples:
            raise ValueError("cannot prepare an empty SLT batch")

        videos = [example["video"] for example in examples]
        source_languages = [example["lang"] for example in examples]
        texts = [example["text"] for example in examples]
        prompt_resolver = (
            self.train_prompt_resolver
            if self.model.training
            else self.eval_prompt_resolver
        )
        epoch = int(self.state.epoch or 0)
        prompt_templates = [
            prompt_resolver.resolve(example, epoch=epoch).template
            for example in examples
        ]
        batch = self.slt_processor(
            videos=videos,
            text=texts,
            src_lang=source_languages,
            prompt_templates=prompt_templates,
            training=training,
            add_eos_token=training,
            return_tensors="pt",
        )

        # GRPO generation consumes prompts only. The processor also exposes
        # prompt-prefixed tensors for the project's supervised/evaluation path.
        prompt_aliases = {
            "generation_input_ids": "input_ids",
            "generation_attention_mask": "attention_mask",
            "generation_token_type_ids": "token_type_ids",
        }
        result = dict(batch)
        for source_name, destination_name in prompt_aliases.items():
            if source_name in result:
                result[destination_name] = result[source_name]
        # Keep targets under a reward-only name. ``input_ids`` was replaced by
        # prompt-only generation IDs above, so gold text cannot leak to policy.
        result["reward_labels"] = result.pop("labels")
        return result

    @staticmethod
    def repeat_packed_video_inputs(
        pixel_values: torch.Tensor,
        pixel_values_length: torch.Tensor,
        repeats: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Repeat each packed video contiguously for GRPO completions.

        For lengths ``[2, 3]`` and ``repeats=2`` the returned frame order is
        ``video0, video0, video1, video1`` with lengths ``[2, 2, 3, 3]``.
        """
        if repeats < 1:
            raise ValueError("repeats must be positive")
        if pixel_values_length.ndim != 1:
            raise ValueError("pixel_values_length must have shape [batch]")
        if pixel_values.ndim != 4:
            raise ValueError("pixel_values must have shape [sum(frames), C, H, W]")
        lengths = [int(length) for length in pixel_values_length.tolist()]
        if any(length < 0 for length in lengths):
            raise ValueError("video lengths cannot be negative")
        if sum(lengths) != pixel_values.shape[0]:
            raise ValueError(
                "sum(pixel_values_length) must equal pixel_values.shape[0]"
            )

        videos = torch.split(pixel_values, lengths, dim=0)
        repeated_frames = [video for video in videos for _ in range(repeats)]
        repeated_pixel_values = torch.cat(repeated_frames, dim=0)
        repeated_lengths = pixel_values_length.repeat_interleave(repeats)
        return repeated_pixel_values, repeated_lengths

    @classmethod
    def repeat_slt_batch(
        cls,
        batch: Mapping[str, Any],
        repeats: int,
    ) -> dict[str, Any]:
        """Repeat an entire SLT prompt batch in per-example GRPO order."""
        required = {"pixel_values", "pixel_values_length"}
        missing = required.difference(batch)
        if missing:
            raise KeyError(f"SLT batch is missing: {sorted(missing)}")

        result = dict(batch)
        result["pixel_values"], result["pixel_values_length"] = (
            cls.repeat_packed_video_inputs(
                batch["pixel_values"], batch["pixel_values_length"], repeats
            )
        )
        batch_size = int(batch["pixel_values_length"].shape[0])
        for name, value in batch.items():
            if name in required:
                continue
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] == batch_size:
                    result[name] = value.repeat_interleave(repeats, dim=0)
        return result

    @staticmethod
    def build_policy_inputs(
        prompt_batch: Mapping[str, Any],
        completion_ids: torch.Tensor,
        completion_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Append sampled completions while preserving SLT multimodal inputs."""
        prompt_ids = prompt_batch["input_ids"]
        prompt_mask = prompt_batch["attention_mask"]
        input_ids = torch.cat((prompt_ids, completion_ids), dim=1)
        attention_mask = torch.cat((prompt_mask, completion_mask), dim=1)

        policy_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": prompt_batch["pixel_values"],
            "pixel_values_length": prompt_batch["pixel_values_length"],
            "use_cache": False,
        }
        if "token_type_ids" in prompt_batch:
            completion_types = torch.zeros_like(completion_ids)
            policy_inputs["token_type_ids"] = torch.cat(
                (prompt_batch["token_type_ids"], completion_types), dim=1
            )
        return policy_inputs

    def generate_slt_completions(
        self,
        model: nn.Module,
        prompt_batch: Mapping[str, torch.Tensor],
        **generation_kwargs: Any,
    ) -> torch.Tensor:
        """Generate completions through ``SltModel.prepare_inputs_for_generation``."""
        generation_inputs = {
            name: prompt_batch[name]
            for name in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "pixel_values",
                "pixel_values_length",
            )
            if name in prompt_batch
        }
        return model.generate(**generation_inputs, **generation_kwargs)

    @contextmanager
    def internal_lora_reference(self, model: nn.Module) -> Iterator[None]:
        """Temporarily turn off the LoRA nested inside ``SltModel.llm``."""
        unwrapped = self.accelerator.unwrap_model(model)
        llm = getattr(unwrapped, "llm", None)
        disable_adapter = getattr(llm, "disable_adapter", None)
        if disable_adapter is None:
            raise RuntimeError(
                "internal LoRA reference requested, but model.llm does not "
                "provide disable_adapter()"
            )
        was_training = model.training
        model.eval()
        try:
            with torch.no_grad(), disable_adapter():
                yield
        finally:
            model.train(was_training)

    def reference_context(self, model: nn.Module):
        """Return the configured reference-policy context manager."""
        if getattr(self.args, "slt_use_internal_lora_reference", False):
            return self.internal_lora_reference(model)
        return nullcontext()

    @staticmethod
    def _completion_mask(
        completion_ids: torch.Tensor,
        *,
        eos_token_id: int | list[int] | None,
        pad_token_id: int,
    ) -> torch.Tensor:
        """Keep tokens through the first EOS and mask padding/EOS suffixes."""
        if eos_token_id is None:
            return completion_ids.ne(pad_token_id).long()

        eos_ids = (
            torch.tensor(eos_token_id, device=completion_ids.device)
            if isinstance(eos_token_id, list)
            else completion_ids.new_tensor([eos_token_id])
        )
        is_eos = (completion_ids.unsqueeze(-1) == eos_ids).any(dim=-1)
        eos_seen_before = F.pad(is_eos.cumsum(dim=1)[:, :-1], (1, 0)) > 0
        mask = ~eos_seen_before

        # When PAD and EOS share an ID, the first occurrence is the generated
        # EOS and must remain valid; all later occurrences are already removed
        # by ``eos_seen_before``. With distinct IDs, remove padding normally.
        pad_is_eos = bool((eos_ids == pad_token_id).any().item())
        if not pad_is_eos:
            mask &= completion_ids.ne(pad_token_id)
        return mask.long()

    def _get_slt_per_token_logps(
        self,
        model: nn.Module,
        policy_inputs: Mapping[str, torch.Tensor],
        completion_length: int,
    ) -> torch.Tensor:
        """Return log p(completion token) without materializing full log-softmax."""
        outputs = model(**policy_inputs)
        logits = outputs.logits[:, :-1, :]
        logits = logits[:, -completion_length:, :] / self.temperature
        completion_ids = policy_inputs["input_ids"][:, -completion_length:]
        selected_logits = logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
        return selected_logits - torch.logsumexp(logits, dim=-1)

    def _calculate_slt_rewards(
        self,
        completion_texts: Sequence[str],
        reward_labels: torch.Tensor,
        language_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate sentence BLEU-4 from processor labels and languages."""
        if reward_labels.ndim != 2:
            raise ValueError("reward_labels must have shape [batch, sequence]")
        if language_ids.ndim != 1:
            raise ValueError("language_ids must have shape [batch]")
        if not (
            len(completion_texts) == reward_labels.shape[0] == language_ids.shape[0]
        ):
            raise ValueError(
                "completion texts, reward labels, and language IDs must have "
                "the same batch size"
            )

        # ---------------------------------------------------------------------
        # 1. Decode target-only labels emitted by SignTranslationProcessor.
        # ---------------------------------------------------------------------
        clean_labels = reward_labels.detach().cpu().clone()
        clean_labels.masked_fill_(clean_labels.eq(-100), self._tokenizer.pad_token_id)
        references = self.processing_class.batch_decode(
            clean_labels,
            skip_special_tokens=True,
        )

        # ---------------------------------------------------------------------
        # 2. Recover language codes from processor-produced language IDs.
        # ---------------------------------------------------------------------
        languages = [
            str(LANGUAGE_MAP.inverse[int(language_id)])
            for language_id in language_ids.detach().cpu().tolist()
        ]

        # ---------------------------------------------------------------------
        # 3. Return one smoothed sentence BLEU-4 reward per completion.
        # ---------------------------------------------------------------------
        scores = self.sentence_bleu_reward(
            predictions=completion_texts,
            references=references,
            languages=languages,
        )
        return torch.tensor(
            scores,
            device=self.accelerator.device,
            dtype=torch.float32,
        )

    @staticmethod
    def _normalize_grouped_rewards(
        rewards: torch.Tensor,
        num_generations: int,
    ) -> torch.Tensor:
        """Normalize globally ordered rewards within each prompt group."""
        if rewards.ndim != 1:
            raise ValueError("rewards must have shape [num_rollouts]")
        if num_generations < 1:
            raise ValueError("num_generations must be positive")
        if rewards.numel() % num_generations != 0:
            raise ValueError(
                "global rollout count must be divisible by num_generations"
            )

        grouped_rewards = rewards.view(-1, num_generations)
        grouped_mean = grouped_rewards.mean(dim=1).repeat_interleave(num_generations)
        if num_generations > 1:
            grouped_std = grouped_rewards.std(dim=1, unbiased=False)
            grouped_std = grouped_std.repeat_interleave(num_generations)
        else:
            grouped_std = torch.zeros_like(rewards)
        return (rewards - grouped_mean) / (grouped_std + 1e-4)

    def _gather_rewards_and_get_local_advantages(
        self,
        local_rewards: torch.Tensor,
        num_generations: int,
    ) -> torch.Tensor:
        """Gather reward scalars globally and return this rank's advantages."""

        # ---------------------------------------------------------------------
        # 1. Verify that every rank contributes an equally sized local batch.
        # ---------------------------------------------------------------------
        local_count = torch.tensor(
            [local_rewards.numel()],
            device=local_rewards.device,
            dtype=torch.long,
        )
        process_counts = self.accelerator.gather(local_count)
        if not torch.equal(
            process_counts, process_counts[:1].expand_as(process_counts)
        ):
            raise ValueError(
                "all processes must contribute the same number of GRPO rollouts"
            )

        # ---------------------------------------------------------------------
        # 2. Gather only BLEU scalars; videos and token tensors remain local.
        # Accelerator gathers tensors in process-rank order, matching TRL's
        # distributed RepeatSampler ordering.
        # ---------------------------------------------------------------------
        global_rewards = self.accelerator.gather(local_rewards.contiguous())
        global_advantages = self._normalize_grouped_rewards(
            global_rewards,
            num_generations,
        )

        # Keep rollout quality observable in both training and evaluation.
        # ``global_rewards`` already contains every rank, so each process
        # appends the same scalar and TRL's logger can average batches safely.
        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["reward"].append(global_rewards.mean().item())
        reward_std = (
            global_rewards.std(unbiased=False).item()
            if global_rewards.numel() > 1
            else 0.0
        )
        self._metrics[mode]["reward_std"].append(reward_std)

        # ---------------------------------------------------------------------
        # 3. Slice the global result back to the current process.
        # ---------------------------------------------------------------------
        local_size = int(local_rewards.numel())
        local_start = self.accelerator.process_index * local_size
        local_end = local_start + local_size
        return global_advantages[local_start:local_end]

    def _generate_and_score_completions(self, inputs: Any) -> dict[str, Any]:
        """Generate SLT rollouts and prepare the tensors consumed by GRPO loss."""

        # ---------------------------------------------------------------------
        # 1. Build one packed SLT prompt batch.
        # TRL's RepeatSampler has already repeated every dataset row in the
        # order [A0, A1, ..., B0, B1, ...], so do not repeat it again here.
        # ---------------------------------------------------------------------
        prompt_batch = self.prepare_slt_batch(inputs, training=False)
        # Call the Transformers implementation directly. Calling TRL's override
        # here would start another rollout and recurse back into this method.
        prompt_batch = Trainer._prepare_inputs(self, prompt_batch)
        prompt_ids = prompt_batch["input_ids"]
        prompt_mask = prompt_batch["attention_mask"]

        # ---------------------------------------------------------------------
        # 2. Sample completions with the model's native GenerationMixin path.
        # ---------------------------------------------------------------------
        with torch.no_grad():
            generated_ids = self.generate_slt_completions(
                self.model,
                prompt_batch,
                generation_config=self.generation_config,
            )
        completion_ids = generated_ids[:, prompt_ids.size(1) :]
        if completion_ids.size(1) == 0:
            raise RuntimeError("generation returned no completion tokens")

        # ---------------------------------------------------------------------
        # 3. Mask padding and every token after the first EOS.
        # ---------------------------------------------------------------------
        completion_mask = self._completion_mask(
            completion_ids,
            eos_token_id=self._tokenizer.eos_token_id,
            pad_token_id=self._tokenizer.pad_token_id,
        )
        num_items_in_batch = self.accelerator.gather(completion_mask.sum()).sum()

        # ---------------------------------------------------------------------
        # 4. Build the full video + prompt + completion policy input.
        # ---------------------------------------------------------------------
        policy_inputs = self.build_policy_inputs(
            prompt_batch,
            completion_ids,
            completion_mask,
        )

        # ---------------------------------------------------------------------
        # 5. Freeze rollout-time token probabilities as the old policy.
        # ---------------------------------------------------------------------
        with torch.no_grad():
            old_per_token_logps = self._get_slt_per_token_logps(
                self.model,
                policy_inputs,
                completion_ids.size(1),
            )

        # ---------------------------------------------------------------------
        # 6. Decode and calculate processor-aligned BLEU-4 rewards.
        # ---------------------------------------------------------------------
        completion_texts = self.processing_class.batch_decode(
            completion_ids,
            skip_special_tokens=True,
        )
        rewards = self._calculate_slt_rewards(
            completion_texts,
            prompt_batch["reward_labels"],
            prompt_batch["lang_ids"],
        )

        # ---------------------------------------------------------------------
        # 7. Gather rewards and normalize globally complete prompt groups.
        # ---------------------------------------------------------------------
        mode = "train" if self.model.training else "eval"
        num_generations = (
            self.num_generations if mode == "train" else self.num_generations_eval
        )
        advantages = self._gather_rewards_and_get_local_advantages(
            rewards,
            num_generations,
        )

        # ---------------------------------------------------------------------
        # 8. Return exactly the tensors needed by the minimal SLT GRPO loss.
        # ---------------------------------------------------------------------
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "num_items_in_batch": num_items_in_batch,
            "pixel_values": prompt_batch["pixel_values"],
            "pixel_values_length": prompt_batch["pixel_values_length"],
            "token_type_ids": policy_inputs.get("token_type_ids"),
        }

    def _prepare_inputs(self, inputs: Any) -> dict[str, Any]:
        """Generate, shuffle, split, and buffer packed-video GRPO rollouts."""

        # ---------------------------------------------------------------------
        # 1. Generate a fresh rollout batch at the configured cadence.
        # ---------------------------------------------------------------------
        mode = "train" if self.model.training else "eval"
        if mode == "eval":
            return self._generate_and_score_completions(inputs)

        generate_every = self.args.steps_per_generation * self.num_iterations
        should_generate = (
            self._step % generate_every == 0 or self._buffered_inputs is None
        )
        if should_generate:
            rollout_batch = self._generate_and_score_completions(inputs)

            # -----------------------------------------------------------------
            # 2. Shuffle rollout samples while keeping every video's frames
            # attached to its text tensors and scalar rollout statistics.
            # -----------------------------------------------------------------
            rollout_batch = self._shuffle_packed_slt_batch(rollout_batch)

            # -----------------------------------------------------------------
            # 3. Split samples into optimizer micro-batches. Packed frames are
            # sliced by cumulative video lengths instead of tensor dimension 0.
            # -----------------------------------------------------------------
            self._buffered_inputs = self._split_packed_slt_batch(
                rollout_batch,
                num_chunks=self.args.steps_per_generation,
            )

        # ---------------------------------------------------------------------
        # 4. Reuse the buffered rollouts for each policy iteration.
        # ---------------------------------------------------------------------
        buffer_index = self._step % self.args.steps_per_generation
        return self._buffered_inputs[buffer_index]

    @staticmethod
    def _shuffle_packed_slt_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
        """Shuffle samples and rebuild packed frames in the same order."""
        lengths = batch["pixel_values_length"]
        pixel_values = batch["pixel_values"]
        batch_size = int(lengths.numel())
        if int(lengths.sum().item()) != pixel_values.shape[0]:
            raise ValueError("packed frame count does not match pixel_values_length")

        permutation = torch.randperm(batch_size, device=lengths.device)
        videos = torch.split(pixel_values, lengths.tolist(), dim=0)
        shuffled_videos = [videos[index] for index in permutation.tolist()]

        result: dict[str, Any] = {}
        for name, value in batch.items():
            if name == "pixel_values":
                result[name] = torch.cat(shuffled_videos, dim=0)
            elif isinstance(value, torch.Tensor) and value.ndim > 0:
                if value.shape[0] == batch_size:
                    result[name] = value.index_select(0, permutation)
                else:
                    result[name] = value
            elif isinstance(value, list) and len(value) == batch_size:
                result[name] = [value[index] for index in permutation.tolist()]
            else:
                result[name] = value
        return result

    @staticmethod
    def _split_packed_slt_batch(
        batch: Mapping[str, Any],
        num_chunks: int,
    ) -> list[dict[str, Any]]:
        """Split by samples while slicing packed frames at video boundaries."""
        if num_chunks < 1:
            raise ValueError("num_chunks must be positive")

        lengths = batch["pixel_values_length"]
        pixel_values = batch["pixel_values"]
        batch_size = int(lengths.numel())
        if batch_size % num_chunks != 0:
            raise ValueError(
                "rollout batch size must be divisible by steps_per_generation"
            )
        if int(lengths.sum().item()) != pixel_values.shape[0]:
            raise ValueError("packed frame count does not match pixel_values_length")

        chunk_size = batch_size // num_chunks
        frame_offsets = torch.cat(
            (lengths.new_zeros(1), lengths.cumsum(dim=0)),
            dim=0,
        )
        chunks: list[dict[str, Any]] = []
        for chunk_index in range(num_chunks):
            sample_start = chunk_index * chunk_size
            sample_end = sample_start + chunk_size
            frame_start = int(frame_offsets[sample_start].item())
            frame_end = int(frame_offsets[sample_end].item())

            chunk: dict[str, Any] = {}
            for name, value in batch.items():
                if name == "pixel_values":
                    chunk[name] = value[frame_start:frame_end]
                elif isinstance(value, torch.Tensor) and value.ndim > 0:
                    if value.shape[0] == batch_size:
                        chunk[name] = value[sample_start:sample_end]
                    else:
                        chunk[name] = value
                elif isinstance(value, list) and len(value) == batch_size:
                    chunk[name] = value[sample_start:sample_end]
                else:
                    chunk[name] = value
            chunks.append(chunk)
        return chunks

    def _compute_loss(
        self,
        model: nn.Module,
        inputs: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Minimal clipped GRPO loss with SLT packed-video forwarding."""

        # ---------------------------------------------------------------------
        # 1. Reconstruct the complete policy input, including packed video.
        # ---------------------------------------------------------------------
        prompt_batch = {
            "input_ids": inputs["prompt_ids"],
            "attention_mask": inputs["prompt_mask"],
            "pixel_values": inputs["pixel_values"],
            "pixel_values_length": inputs["pixel_values_length"],
        }
        if inputs.get("token_type_ids") is not None:
            prompt_length = inputs["prompt_ids"].size(1)
            prompt_batch["token_type_ids"] = inputs["token_type_ids"][:, :prompt_length]
        policy_inputs = self.build_policy_inputs(
            prompt_batch,
            inputs["completion_ids"],
            inputs["completion_mask"],
        )

        # ---------------------------------------------------------------------
        # 2. Compute current-policy completion log probabilities with gradients.
        # ---------------------------------------------------------------------
        per_token_logps = self._get_slt_per_token_logps(
            model,
            policy_inputs,
            inputs["completion_ids"].size(1),
        )

        # ---------------------------------------------------------------------
        # 3. Apply the basic clipped GRPO surrogate objective.
        # ---------------------------------------------------------------------
        old_per_token_logps = inputs["old_per_token_logps"]
        advantages = inputs["advantages"].unsqueeze(1)
        ratio = torch.exp(per_token_logps - old_per_token_logps)
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.epsilon_low,
            1.0 + self.epsilon_high,
        )
        per_token_loss = -torch.minimum(
            ratio * advantages,
            clipped_ratio * advantages,
        )

        # ---------------------------------------------------------------------
        # 4. Average valid completion tokens, then average the batch.
        # ---------------------------------------------------------------------
        mask = inputs["completion_mask"].to(per_token_loss.dtype)
        loss = ((per_token_loss * mask).sum(1) / mask.sum(1).clamp_min(1)).mean()
        if self.model.training:
            loss = loss / self.current_gradient_accumulation_steps
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Any,
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[None, None, None]:
        """Generate eval rollouts and rewards without a redundant loss forward."""
        del model, prediction_loss_only, ignore_keys
        self._prepare_inputs(inputs)
        return None, None, None
