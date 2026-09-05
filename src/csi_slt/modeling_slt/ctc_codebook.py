"""Differentiable bridge from CTC logits to LLM-compatible embeddings."""

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import nn
from torch.nn import functional as F


SelectionMode = Literal["soft", "straight_through", "argmax"]
_SELECTION_MODES = frozenset({"soft", "straight_through", "argmax"})


@dataclass
class CTCCodebookOutput:
    """Outputs of :class:`CTCCodebookBridge`, kept in packed-token layout."""

    embeddings: torch.Tensor  # [sum(T_i), llm_hidden_size]
    lengths: torch.Tensor  # [B]
    token_distribution: torch.Tensor  # [sum(T_i), ctc_vocab_size]
    predicted_ids: torch.Tensor  # [sum(T_i)]
    blank_probability: torch.Tensor  # [sum(T_i)]
    logging_scalars: dict[str, torch.Tensor]


class CTCCodebookBridge(nn.Module):
    """Map packed CTC logits into the input-embedding space of the LLM.

    The CTC classifier and this codebook deliberately do not share weights.
    Training supports a stable soft path and a straight-through Gumbel path;
    evaluation defaults to deterministic argmax. The temporal layout is kept
    intact: blank and repeated slots are never physically collapsed.
    """

    def __init__(
        self,
        *,
        ctc_vocab_size: int,
        llm_hidden_size: int,
        blank_id: int,
        training_mode: SelectionMode = "soft",
        min_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self._validate_init_args(
            ctc_vocab_size=ctc_vocab_size,
            llm_hidden_size=llm_hidden_size,
            blank_id=blank_id,
            training_mode=training_mode,
            min_temperature=min_temperature,
        )

        self.ctc_vocab_size = ctc_vocab_size
        self.llm_hidden_size = llm_hidden_size
        self.blank_id = blank_id
        self.training_mode = training_mode
        self.min_temperature = float(min_temperature)

        # The codebook width intentionally equals the language model's hidden
        # size so each row can be initialized directly from its sub-token
        # embeddings. Nothing here is specific to one language model: the
        # bridge only ever sees an embedding table, sub-token ids and a pad id.
        self.codebook = nn.Embedding(ctc_vocab_size, llm_hidden_size)
        # Set by initialize_from_llm_embeddings. Keeping the original pad
        # vector lets logging expose whether the trainable blank row drifts.
        self.register_buffer(
            "initial_blank_embedding",
            torch.zeros(llm_hidden_size),
        )
        # Persist this state with the weights: a fresh model must be initialized
        # explicitly, while a checkpoint restores both the learned codebook and
        # proof that initialization happened. The Python cache avoids a device
        # synchronization on every forward.
        self.register_buffer(
            "codebook_initialized",
            torch.tensor([False]),
        )
        self._initialization_verified = False

    def forward(
        self,
        ctc_logits: torch.Tensor,
        lengths: torch.Tensor,
        *,
        temperature: float = 1.0,
    ) -> CTCCodebookOutput:
        """Convert packed ``[sum(T_i), V]`` CTC logits to packed embeddings."""
        self.assert_initialized()
        self._validate_forward_inputs(ctc_logits, lengths)
        mode = self.training_mode if self.training else "argmax"
        self._validate_temperature(temperature, mode)

        distribution = self._select_distribution(ctc_logits, mode, temperature)
        blank_probability = distribution[:, self.blank_id]
        predicted_ids = distribution.argmax(dim=-1)

        # Blank owns a normal, trainable codebook row initialized from the
        # LLM's pad embedding. It remains a real prefix slot rather than pretending
        # that an all-zero vector is invisible to the language model.
        embeddings = distribution @ self.codebook.weight

        return CTCCodebookOutput(
            embeddings=embeddings,
            lengths=lengths,
            token_distribution=distribution,
            predicted_ids=predicted_ids,
            blank_probability=blank_probability,
            logging_scalars=self._build_logging_scalars(distribution, predicted_ids),
        )

    def assert_initialized(self) -> None:
        """Fail before a random, uninitialized codebook can be used."""
        if self._initialization_verified:
            return
        if not bool(self.codebook_initialized.item()):
            raise RuntimeError(
                "CTC codebook has not been initialized from the LLM and CTC "
                "tokenizers. Initialize it before training or prediction."
            )
        self._initialization_verified = True

    @torch.no_grad()
    def initialize_from_llm_embeddings(
        self,
        llm_embeddings: nn.Embedding,
        llm_token_ids_by_ctc_id: Sequence[Sequence[int]],
        *,
        llm_pad_token_id: int,
    ) -> None:
        """Initialize each non-blank codebook row from mean LLM embeddings.

        ``llm_token_ids_by_ctc_id[i]`` contains the LLM sub-token ids for CTC
        token ``i``. The blank sequence is ignored and its row is initialized
        from the LLM's pad-token embedding.

        Averaging several sub-token embeddings shortens the result -- roughly
        by 1/sqrt(k) for k unrelated directions -- so an uncorrected row lands
        well inside the shell the language model's own embeddings occupy, by a
        factor that varies with how many pieces the token happened to split
        into. Each averaged row is therefore rescaled to the mean norm of the
        sub-token embeddings it was built from: a no-op for the single-piece
        rows, which stay exactly equal to the real embedding, and a restoration
        of that word's own natural scale for the rest. The pad-initialized
        blank row is left untouched for the same reason -- it is already a real
        embedding, and ``initial_blank_embedding`` records it as the reference
        for the drift diagnostics.
        """
        self._validate_llm_initialization(
            llm_embeddings,
            llm_token_ids_by_ctc_id,
            llm_pad_token_id,
        )
        source_weight = llm_embeddings.weight.detach()
        initialized = torch.empty_like(self.codebook.weight)
        for ctc_id, llm_ids in enumerate(llm_token_ids_by_ctc_id):
            if ctc_id == self.blank_id:
                initialized[ctc_id].copy_(source_weight[llm_pad_token_id])
                continue
            ids = torch.as_tensor(llm_ids, device=source_weight.device)
            sub_tokens = source_weight.index_select(0, ids)
            initialized[ctc_id].copy_(self._scaled_mean(sub_tokens))

        self.codebook.weight.copy_(
            initialized.to(
                device=self.codebook.weight.device,
                dtype=self.codebook.weight.dtype,
            )
        )
        self.initial_blank_embedding.copy_(
            source_weight[llm_pad_token_id].to(self.initial_blank_embedding)
        )
        self.codebook_initialized.fill_(True)
        self._initialization_verified = True

    @staticmethod
    def _scaled_mean(sub_tokens: torch.Tensor) -> torch.Tensor:
        """Mean of ``sub_tokens`` carrying their mean norm.

        Exactly the input row when there is only one sub-token. A degenerate
        mean (sub-token embeddings that cancel out) keeps the raw average
        rather than being rescaled by a near-zero divisor.
        """
        mean = sub_tokens.mean(dim=0)
        mean_norm = mean.norm()
        if mean_norm <= torch.finfo(mean.dtype).eps:
            return mean
        return mean * (sub_tokens.norm(dim=-1).mean() / mean_norm)

    def _build_logging_scalars(
        self, distribution: torch.Tensor, predicted_ids: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Diagnostics for the codebook bridge.

        Two families. Codebook-weight drift: the blank row's norm and its
        cosine to the pad embedding it was seeded from. Selection sharpness:
        the mean top-1 mass of the temperature-selected distribution, overall
        and over non-blank frames only. ``mean_top1_prob_nonblank`` is the one
        to watch during a temperature anneal -- it says how close the content
        frames are to the one-hot regime evaluation always uses, while blank
        frames (the majority, and already near-peaky) dominate the overall
        mean and hide that. In eval both figures are 1.0 by construction.

        Blank-frequency stats (probability mean, argmax ratio) live in
        ``SltModel`` instead: they are pure functions of the CTC head's raw
        logits and need no codebook state, so both ``ctc_only`` (which never
        builds a codebook distribution at all) and joint training compute
        them the same way without going through this bridge.
        """
        blank_embedding = self.codebook.weight[self.blank_id].float()
        scalars = {
            "blank_embedding_norm": blank_embedding.norm().detach(),
        }
        reference = self.initial_blank_embedding.float()
        scalars["blank_pad_cosine_similarity"] = F.cosine_similarity(
            blank_embedding.unsqueeze(0),
            reference.unsqueeze(0),
        ).squeeze(0).detach()

        top1_prob = distribution.max(dim=-1).values
        scalars["mean_top1_prob"] = top1_prob.mean().detach()
        non_blank = predicted_ids != self.blank_id
        scalars["mean_top1_prob_nonblank"] = (
            top1_prob[non_blank].mean().detach()
            if bool(non_blank.any())
            else top1_prob.new_zeros(())
        )
        return scalars

    def _select_distribution(
        self,
        logits: torch.Tensor,
        mode: SelectionMode,
        temperature: float,
    ) -> torch.Tensor:
        if mode == "soft":
            return F.softmax(logits / temperature, dim=-1)
        if mode == "straight_through":
            return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
        predicted_ids = logits.argmax(dim=-1)
        return F.one_hot(predicted_ids, num_classes=self.ctc_vocab_size).to(
            dtype=logits.dtype
        )

    def _validate_forward_inputs(
        self,
        logits: torch.Tensor,
        lengths: torch.Tensor,
    ) -> None:
        if logits.ndim != 2 or logits.shape[1] != self.ctc_vocab_size:
            raise ValueError(
                "ctc_logits must have shape [sum(T_i), ctc_vocab_size], got "
                f"{tuple(logits.shape)}"
            )
        if not logits.is_floating_point():
            raise TypeError("ctc_logits must be a floating-point tensor")
        if lengths.ndim != 1 or lengths.dtype == torch.bool:
            raise ValueError("lengths must be a 1D integer tensor")
        if lengths.is_floating_point() or lengths.is_complex():
            raise TypeError("lengths must be an integer tensor")
        if bool((lengths < 0).any()) or int(lengths.sum().item()) != logits.shape[0]:
            raise ValueError("lengths must be non-negative and sum to packed tokens")

    def _validate_temperature(self, temperature: float, mode: SelectionMode) -> None:
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
            raise TypeError("temperature must be a real number")
        if mode != "argmax" and temperature < self.min_temperature:
            raise ValueError(
                f"temperature must be >= {self.min_temperature} for {mode} mode"
            )

    def _validate_llm_initialization(
        self,
        embeddings: nn.Embedding,
        token_ids: Sequence[Sequence[int]],
        pad_token_id: int,
    ) -> None:
        if not isinstance(embeddings, nn.Embedding):
            raise TypeError("llm_embeddings must be an nn.Embedding")
        if embeddings.embedding_dim != self.llm_hidden_size:
            raise ValueError("LLM embedding width must equal llm_hidden_size")
        if len(token_ids) != self.ctc_vocab_size:
            raise ValueError("one LLM token-id sequence is required per CTC token")
        if isinstance(pad_token_id, bool) or not isinstance(pad_token_id, int):
            raise TypeError("llm_pad_token_id must be an int")
        if not 0 <= pad_token_id < embeddings.num_embeddings:
            raise ValueError("llm_pad_token_id is outside the embedding vocabulary")
        for ctc_id, ids in enumerate(token_ids):
            if ctc_id != self.blank_id and not ids:
                raise ValueError(f"CTC token {ctc_id} has no LLM token ids")
            if any(not isinstance(token_id, int) for token_id in ids):
                raise TypeError("LLM token ids must be integers")
            if any(token_id < 0 or token_id >= embeddings.num_embeddings for token_id in ids):
                raise ValueError("an LLM token id is outside the embedding vocabulary")

    @staticmethod
    def _validate_selection_mode(mode: str) -> None:
        if mode not in _SELECTION_MODES:
            raise ValueError(
                f"selection mode must be one of {sorted(_SELECTION_MODES)}, got {mode!r}"
            )

    @classmethod
    def _validate_init_args(
        cls,
        *,
        ctc_vocab_size: int,
        llm_hidden_size: int,
        blank_id: int,
        training_mode: str,
        min_temperature: float,
    ) -> None:
        for name, value in (
            ("ctc_vocab_size", ctc_vocab_size),
            ("llm_hidden_size", llm_hidden_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an int")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(blank_id, bool) or not isinstance(blank_id, int):
            raise TypeError("blank_id must be an int")
        if not 0 <= blank_id < ctc_vocab_size:
            raise ValueError("blank_id must be in [0, ctc_vocab_size)")
        cls._validate_selection_mode(training_mode)
        if isinstance(min_temperature, bool) or not isinstance(
            min_temperature, (int, float)
        ):
            raise TypeError("min_temperature must be a real number")
        if min_temperature <= 0:
            raise ValueError("min_temperature must be positive")
