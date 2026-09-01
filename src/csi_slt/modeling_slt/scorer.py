"""Linear scorer that ranks patches by how much they look like a hand.

One question, one affine map: given a patch feature, how hand-like is it?  The
caller decides what to do with the ranking -- this module never selects, masks
or pools.

The output is a raw **logit**: unbounded, uncalibrated, and deliberately not
passed through a sigmoid.  Fit it with ``BCEWithLogitsLoss``; rank with it
directly, since any monotone map leaves a ranking unchanged.

Fitting is done by a separate script, not by the SLT training loop, against
MediaPipe labels -- positive: a patch holding a hand landmark; negative: face
plus clean background; excluded: the ring of ambiguous patches around either.
The face must be in the negative class or the probe degenerates into a skin
detector that keeps the face and drops the hands.

``feature_mean`` and ``feature_scale`` are the fitting set's per-channel
statistics, held explicitly rather than folded into the weights.  Folding is
the same affine map and one fewer op, but it is invisible: a fitting script
that forgot to fold would produce a scorer that still runs, still returns
plausible logits, and silently ranks worse.  Kept as buffers, a mismatch is
something you can look at.  They are constants of the fitting set, not a
LayerNorm -- every patch of every frame is shifted and scaled by the same two
vectors, and the identity defaults leave the module a bare linear map.

Persistence is Hugging Face's: ``save_pretrained`` / ``from_pretrained`` carry
the coefficients, the statistics and the :class:`HandPatchScorerConfig`
provenance fields together, so a scorer can never be loaded without the record
of which backbone, layer and resolution it was fitted against.
"""

import torch
from torch import Tensor, nn
from transformers.modeling_utils import PreTrainedModel

from csi_slt.configuration_slt.configuration_scorer import HandPatchScorerConfig


class HandPatchScorer(PreTrainedModel):
    """Score patch features: ``[..., input_dim] -> [...]``."""

    config_class = HandPatchScorerConfig
    base_model_prefix = "hand_patch_scorer"

    def __init__(self, config: HandPatchScorerConfig) -> None:
        super().__init__(config)
        self.register_buffer("feature_mean", torch.zeros(config.input_dim))
        self.register_buffer("feature_scale", torch.ones(config.input_dim))
        self.linear = nn.Linear(config.input_dim, 1)
        self.post_init()

    @property
    def frozen(self) -> bool:
        """Whether the coefficients are held out of the gradient graph.

        Read off ``requires_grad`` rather than a separate flag, so it cannot
        disagree with what the optimizer will actually pick up. This module
        never sets it: which parameters train is the caller's policy, and
        ``from_pretrained`` leaves them trainable like any other model.
        """
        return not any(parameter.requires_grad for parameter in self.parameters())

    def set_frozen(self, frozen: bool = True) -> "HandPatchScorer":
        """Freeze or unfreeze the coefficients; returns self so it can chain."""
        self.requires_grad_(not frozen)
        return self

    @torch.no_grad()
    def set_feature_statistics(self, mean, scale) -> None:
        """Install the fitting set's per-channel statistics.

        Compute these in one pass over the fitting features and install them
        before training the linear layer.  Updating them while it trains makes
        it chase a moving target: the weights end up calibrated to statistics
        that differ from the ones their gradients were accumulated under.
        """
        mean = torch.as_tensor(
            mean, dtype=self.feature_mean.dtype, device=self.feature_mean.device
        ).reshape(-1)
        scale = torch.as_tensor(
            scale, dtype=self.feature_scale.dtype, device=self.feature_scale.device
        ).reshape(-1)
        for name, tensor in (("mean", mean), ("scale", scale)):
            if tensor.numel() != self.config.input_dim:
                raise ValueError(
                    f"{name} must have {self.config.input_dim} entries, got "
                    f"{tensor.numel()}"
                )
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains non-finite values")
        # A constant channel has zero variance, and dividing by it would turn
        # every logit into inf without raising anywhere. Clamp at the source.
        if bool(scale.le(0).any()):
            raise ValueError("scale entries must be positive; clamp them first")
        self.feature_mean.copy_(mean)
        self.feature_scale.copy_(scale)

    def forward(self, patch_features: Tensor) -> Tensor:
        standardized = (patch_features - self.feature_mean) / self.feature_scale
        return self.linear(standardized).squeeze(-1)
