"""Structured requests, outputs, and selection for SLT forward information.

These containers describe which expensive intermediate values a caller wants
and carry the selected tensors out of the model. They intentionally contain no
logging, plotting, device transfer, or file-writing behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import torch

from .misc import packed_position_ids_to_padded

if TYPE_CHECKING:
    from .output_utils import PrepareForCausalLMOutput


@dataclass(frozen=True)
class InformationRequest:
    """Select optional intermediate information from an SLT forward pass.

    Args:
        llm_attentions: Return selected language-model attention maps.
        sample_indices: Batch entries retained in the information output.
        llm_layers: LLM layer indices retained when ``llm_attentions`` is true.
            Negative indices follow normal Python indexing semantics.
        reduce_heads: Average the head dimension before returning LLM maps.
    """

    llm_attentions: bool = False
    sample_indices: tuple[int, ...] = (0,)
    llm_layers: tuple[int, ...] = (-1,)
    reduce_heads: bool = True

    def __post_init__(self) -> None:
        for name in (
            "llm_attentions",
            "reduce_heads",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} must be a bool")

        self._validate_indices(
            self.sample_indices,
            name="sample_indices",
            allow_negative=False,
        )
        self._validate_indices(
            self.llm_layers,
            name="llm_layers",
            allow_negative=True,
        )

    @staticmethod
    def _validate_indices(
        indices: tuple[int, ...],
        *,
        name: str,
        allow_negative: bool,
    ) -> None:
        if not isinstance(indices, tuple):
            raise TypeError(f"{name} must be a tuple of integers")
        if not indices:
            raise ValueError(f"{name} must not be empty")
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in indices
        ):
            raise TypeError(f"{name} must contain only integers")
        if not allow_negative and any(index < 0 for index in indices):
            raise ValueError(f"{name} must contain only non-negative indices")
        if len(set(indices)) != len(indices):
            raise ValueError(f"{name} must not contain duplicate indices")

    @property
    def enabled(self) -> bool:
        """Whether the caller requested any intermediate information."""
        return self.llm_attentions


@dataclass
class InformationOutput:
    """Optional intermediate tensors selected from an SLT forward pass.

    Tensor values are expected to be detached by the model. Moving tensors to
    CPU, serializing them, and rendering plots remain the caller's
    responsibility, avoiding synchronization and I/O side effects in forward.
    """

    llm_attentions: Optional[tuple[torch.Tensor, ...]] = None
    llm_visual_mask: Optional[torch.Tensor] = None

    visual_lengths: Optional[torch.Tensor] = None
    visual_position_ids: Optional[torch.Tensor] = None

    def detach_to_cpu(self) -> InformationOutput:
        """Return a detached CPU copy suitable for storage or visualization.

        Tensor dtype, padding, and container structure are preserved. This
        method does not convert values to NumPy and does not mutate the current
        output, so the model-facing object can remain on its original device.
        """
        return InformationOutput(
            llm_attentions=_detach_to_cpu(self.llm_attentions),
            llm_visual_mask=_detach_to_cpu(self.llm_visual_mask),
            visual_lengths=_detach_to_cpu(self.visual_lengths),
            visual_position_ids=_detach_to_cpu(self.visual_position_ids),
        )


def _detach_to_cpu(value: Any) -> Any:
    """Recursively detach tensors and move them to CPU."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_detach_to_cpu(item) for item in value)
    if isinstance(value, list):
        return [_detach_to_cpu(item) for item in value]
    return value


def build_information_output(
    *,
    request: InformationRequest,
    batch_size: int,
    llm_attentions: Optional[tuple[torch.Tensor, ...]],
    prepare_output: Optional[PrepareForCausalLMOutput],
) -> InformationOutput:
    """Select and detach requested intermediate forward information."""
    device = (
        prepare_output.inputs_embeds.device
        if prepare_output is not None
        else (llm_attentions[0].device if llm_attentions else torch.device("cpu"))
    )
    sample_indices = torch.tensor(
        request.sample_indices, device=device, dtype=torch.long
    )
    if bool((sample_indices >= batch_size).any()):
        raise IndexError(
            "information sample index is out of range for batch size "
            f"{batch_size}: {request.sample_indices}"
        )

    information = InformationOutput()

    if prepare_output is not None:
        information.llm_visual_mask = prepare_output.visual_mask.index_select(
            0, sample_indices
        ).detach()
        information.visual_lengths = prepare_output.visual_lengths.index_select(
            0, sample_indices
        ).detach()
        if prepare_output.packed_visual_position_ids is not None:
            padded_positions, _ = packed_position_ids_to_padded(
                prepare_output.packed_visual_position_ids,
                prepare_output.visual_lengths,
            )
            information.visual_position_ids = padded_positions.index_select(
                0, sample_indices
            ).detach()

    if request.llm_attentions:
        if not llm_attentions:
            raise RuntimeError(
                "the language model did not return attention weights; verify "
                "that its attention implementation supports output_attentions"
            )
        num_layers = len(llm_attentions)
        normalized_layers = []
        for layer in request.llm_layers:
            normalized_layer = layer if layer >= 0 else num_layers + layer
            if not 0 <= normalized_layer < num_layers:
                raise IndexError(
                    f"LLM layer index {layer} is out of range for {num_layers} layers"
                )
            normalized_layers.append(normalized_layer)
        if len(set(normalized_layers)) != len(normalized_layers):
            raise ValueError("llm_layers resolve to duplicate layer indices")

        selected_attentions = []
        for layer in normalized_layers:
            attention = llm_attentions[layer]
            if attention is None:
                raise RuntimeError(
                    f"LLM layer {layer} did not return attention weights"
                )
            attention = attention.index_select(0, sample_indices.to(attention.device))
            if request.reduce_heads:
                attention = attention.float().mean(dim=1)
            selected_attentions.append(attention.detach())
        information.llm_attentions = tuple(selected_attentions)

    return information
