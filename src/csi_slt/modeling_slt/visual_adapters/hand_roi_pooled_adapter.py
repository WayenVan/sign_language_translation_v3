"""Hand-ROI pooling: one token carrying both the whole frame and just the hands.

The pooled-linear baseline averages all 196 patches of a frame into one vector,
and measurements on real C-RADIO features say that vector has lost the hands:
a linear readout recovers R^2 = 0.59 of hand shape and orientation from a hard
crop of the hand's patches and **0.00** from the frame's global mean.  A fixed
3x2 spatial grid only reaches 0.09, and every soft re-weighting tried -- motion,
and even a near-perfect frozen hand probe used as softmax weights -- stays under
0.09 as well.  What works is hard selection followed by a plain mean over the
selected patches, which is exactly what this adapter does.

Data flow::

    patch features [F, P, D]
        |                    \\
        |                     v  frozen scorer -> top-k hard mask [F, P]
        |                     |
        v mean over P         v mean over the selected patches only
    global [F, D]          roi [F, D]
        \\                    /
         +-------- concat ---+
                   |
                   v  boundary-safe temporal mean over s frames
             [N, 2D]
                   v  LayerNorm -> Linear -> GELU -> Linear
             visual tokens [N, D_out]

The two halves are **concatenated, not emitted as separate tokens**: the adapter
still produces one token per ``temporal_scale_factor`` frames, so sequence
length, ``video_token_scale`` and the CTC head are all untouched and a run
against the pooled-linear baseline changes exactly one thing -- what each token
contains.

The global half is kept deliberately.  It carries torso, posture and scene, and
it is the fallback: if the ROI half turns out to be noise, the adapter can still
learn the baseline's behaviour.

Selection is scored on the **raw** backbone features.  The scorer's 1153
coefficients were fitted on unmodified patches of a specific backbone and layer
(recorded in its own config), so anything that transforms the patches first --
a motion residual, for instance -- must not sit in front of it.  Score on raw
features, pool whatever content you like under the resulting mask.
"""

import math

import torch
from torch import Tensor, nn

from csi_slt.modeling_slt.misc import (
    mark_module_tree_as_initialized,
    random_derangement,
)
from csi_slt.configuration_slt.configuration_scorer import HandPatchScorerConfig
from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.scorer import HandPatchScorer


class TopKRoiPool(nn.Module):
    """Score one set of patches, pool another under the resulting mask.

    The two inputs are kept separate on purpose.  The scorer's coefficients were
    fitted on unmodified patches of one backbone and layer, so anything that
    transforms the patches -- a motion residual, a context convolution -- must
    not sit in front of it or the ranking silently degrades.  The *content*
    being pooled has no such constraint.  Score on raw features, pool whatever
    is richest::

        roi = pool(score_features=raw, feature_patches=fused)

    Only the patch axis has to agree between the two; their widths may differ,
    since the mask is over patches rather than channels.

    Selection is hard and the pooled mean is uniform: the scores decide *which*
    patches survive and never how much a survivor counts.  Soft weighting was
    measured on real features and does not preserve handshape (R^2 0.03-0.08
    against 0.59 for a hard crop), because the surviving mass over ~170 non-hand
    patches averages the hand away regardless of how peaked the weights are.
    """

    def __init__(
        self,
        input_dim: int,
        top_k: int = 24,
        scorer_path: str | None = None,
        freeze_scorer: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")
        self.top_k = top_k
        self.input_dim = input_dim
        # Recorded, deliberately not read: construction stays pure so it works
        # on a meta device, needs no disk, and lets a checkpoint load without
        # the fitting directory still existing.
        self.scorer_path = scorer_path

        # Only input_dim shapes the module; the config's other fields are
        # provenance and arrive with the fitted weights.
        self.scorer = HandPatchScorer(HandPatchScorerConfig(input_dim=input_dim))
        # Fitted constants, not something to train alongside an adapter. The
        # scorer itself never touches requires_grad -- that policy lives here.
        self.scorer.set_frozen(freeze_scorer)
        # An instance marker rather than a class attribute, so ``freeze_scorer``
        # remains a real choice. The trainability policy re-freezes anything
        # carrying it after applying a plan, because a plan that trains the
        # adapter would otherwise recurse straight into these coefficients and
        # silently invalidate the provenance they were fitted under.
        self.scorer.always_frozen = freeze_scorer
        # Persistent: it asserts "the coefficients I carry are fitted", which is
        # true of a checkpoint's weights too. A non-persistent flag would come
        # back False after a resume and fire on a perfectly good model.
        self.register_buffer("scorer_loaded", torch.zeros(1, dtype=torch.bool))

    @property
    def scorer_is_loaded(self) -> bool:
        return bool(self.scorer_loaded.item())

    @torch.no_grad()
    def load_pretrained_components(self) -> None:
        """Install fitted coefficients from ``scorer_path``.

        Called once when a model is built from external pretrained sources.
        Resuming from a checkpoint must *not* call it: the weights come from the
        checkpoint, and the fitting directory may be long gone.
        """
        if self.scorer_path is None:
            raise ValueError(
                "no scorer_path was configured, so there is nothing to load; "
                "an unfitted scorer ranks patches at random"
            )
        fitted = HandPatchScorer.from_pretrained(self.scorer_path)
        if fitted.config.input_dim != self.input_dim:
            raise ValueError(
                f"scorer at {self.scorer_path} expects "
                f"{fitted.config.input_dim}-wide patches, but this pool is built "
                f"for {self.input_dim}; the scorer is only valid for the backbone "
                "and layer it was fitted on"
            )
        self.scorer.load_state_dict(fitted.state_dict())
        self.scorer.config = fitted.config
        self.scorer_loaded.fill_(True)

    def select(self, score_features: Tensor) -> Tensor:
        """Hard top-k mask over the patch axis: ``[F, P, D] -> [F, P]`` bool.

        Under ``no_grad`` because ``topk`` has no useful gradient.
        """
        if score_features.ndim != 3:
            raise ValueError(
                "score_features must have shape [F, P, D], got "
                f"{tuple(score_features.shape)}"
            )
        if score_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"score_features must be {self.input_dim}-wide, got "
                f"{score_features.shape[-1]}; the scorer is only valid for the "
                "backbone and layer it was fitted on"
            )
        if not self.scorer_is_loaded:
            raise RuntimeError(
                "the scorer holds unfitted coefficients: its ranking would be "
                "noise and nothing downstream would notice. Call "
                "load_pretrained_components() when building a fresh model, or "
                "load a checkpoint that carries fitted weights."
            )
        with torch.no_grad():
            scores = self.scorer(score_features)
            # Clamped rather than raised on, so a frame carrying fewer patches
            # than top_k still yields a mask.
            k = min(self.top_k, scores.shape[-1])
            mask = torch.zeros_like(scores, dtype=torch.bool)
            mask.scatter_(-1, scores.topk(k, dim=-1).indices, True)
        return mask

    def forward(
        self, score_features: Tensor, feature_patches: Tensor | None = None
    ) -> Tensor:
        """Mean of the top-k patches: ``[F, P, D] -> [F, D]``.

        ``feature_patches`` defaults to ``score_features``, which is the plain
        case of ranking and pooling the same tensor.
        """
        if feature_patches is None:
            feature_patches = score_features
        if feature_patches.shape[:2] != score_features.shape[:2]:
            raise ValueError(
                "score_features and feature_patches must share [F, P], got "
                f"{tuple(score_features.shape[:2])} and "
                f"{tuple(feature_patches.shape[:2])}"
            )
        mask = self.select(score_features)
        weights = mask.unsqueeze(-1).to(dtype=feature_patches.dtype)
        return (feature_patches * weights).sum(dim=1) / weights.sum(dim=1)


class HandRoiPooledAdapter(nn.Module):
    """Concatenate a global mean and a hand-ROI mean, then pool over time.

    ``top_k`` is the one knob that trades recall against purity.  Measured on
    held-out videos with the shipped scorer: k=16 keeps 91% of hand patches with
    7% of the budget spent on non-hand patches, k=24 keeps 98% at 11%, k=32
    keeps 99.6% at 18%.  Every non-hand patch kept is one more vector averaged
    into the ROI half, so this is not simply "larger is safer".
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        scorer_path: str | None = None,
        top_k: int = 24,
        projection_rank: int | None = None,
        use_layer_norm: bool = True,
        temporal_scale_factor: int = 2,
        freeze_scorer: bool = True,
    ) -> None:
        super().__init__()
        self._validate_dimension("input_dim", input_dim)
        self._validate_dimension("output_dim", output_dim)
        self._validate_dimension("top_k", top_k)
        if projection_rank is not None:
            self._validate_dimension("projection_rank", projection_rank)
        self._validate_dimension("temporal_scale_factor", temporal_scale_factor)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.top_k = top_k
        self.temporal_scale_factor = temporal_scale_factor
        # A ``None`` rank resolves to output_dim, so this attribute always
        # reports the hidden width the projection actually uses.
        self.projection_rank = (
            output_dim if projection_rank is None else projection_rank
        )

        self.roi_pool = TopKRoiPool(
            input_dim=input_dim,
            top_k=top_k,
            scorer_path=scorer_path,
            freeze_scorer=freeze_scorer,
        )

        # The projection sees the two halves concatenated.
        pooled_dim = 2 * input_dim
        self.norm = nn.LayerNorm(pooled_dim) if use_layer_norm else nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(pooled_dim, self.projection_rank),
            nn.GELU(),
            nn.Linear(self.projection_rank, output_dim),
        )

        self._reset_projection_parameters()
        mark_module_tree_as_initialized(self)

    @staticmethod
    def _validate_dimension(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")

    def _reset_projection_parameters(self) -> None:
        # Fan-in init on both layers, matching the pooled-linear baseline so the
        # two runs start from the same projection scale.
        input_projection, output_projection = self.projection[0], self.projection[2]
        for projection in (input_projection, output_projection):
            nn.init.kaiming_uniform_(projection.weight, a=math.sqrt(5))
            nn.init.zeros_(projection.bias)

    @property
    def trainable_parameter_count(self) -> int:
        """Trainable adapter parameters; excludes the frozen scorer."""
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def forward(
        self,
        visual_backbone_output: VisualBackboneOutput,
        permute_video_tokens: bool = False,
    ) -> VisualAdapterOutput:
        patch_features = visual_backbone_output.visual_features
        visual_length = visual_backbone_output.visual_length
        self._validate_inputs(patch_features, visual_length)

        # Scored and pooled on the same tensor for now; a motion residual would
        # go in as feature_patches while the scoring stays on the raw features.
        roi_features = self.roi_pool(patch_features)
        global_features = patch_features.mean(dim=1)
        frame_features = torch.cat([global_features, roi_features], dim=-1)

        # Temporal mean runs separately inside each packed video, so a window
        # can never cross a video boundary.
        video_features = torch.split(frame_features, visual_length.tolist(), dim=0)
        pooled_features = torch.cat(
            [
                features.unflatten(0, (-1, self.temporal_scale_factor)).mean(dim=1)
                for features in video_features
            ],
            dim=0,
        )
        pooled_length = visual_length // self.temporal_scale_factor
        visual_features = self.projection(self.norm(pooled_features))

        if permute_video_tokens:
            permutation = random_derangement(
                pooled_length, device=visual_features.device
            )
            visual_features = visual_features[permutation]

        return VisualAdapterOutput(
            visual_features=visual_features,
            visual_length=pooled_length,
        )

    def _validate_inputs(
        self,
        patch_features: Tensor | None,
        visual_length: Tensor | None,
    ) -> None:
        if patch_features is None:
            raise ValueError("visual_features must contain patch features")
        if patch_features.ndim != 3:
            raise ValueError(
                "visual_features must have shape [sum(T), P, input_dim], got "
                f"{tuple(patch_features.shape)}"
            )
        if patch_features.shape[1] == 0:
            raise ValueError("visual_features must contain at least one patch")
        if patch_features.shape[-1] != self.input_dim:
            raise ValueError(
                f"patch feature dimension must be {self.input_dim}, got "
                f"{patch_features.shape[-1]}"
            )
        if visual_length is None:
            raise ValueError("visual_length must be provided")
        if visual_length.ndim != 1 or visual_length.numel() == 0:
            raise ValueError("visual_length must be a non-empty 1D tensor")
        if visual_length.is_floating_point() or visual_length.is_complex():
            raise TypeError("visual_length must use an integer dtype")
        if bool((visual_length <= 0).any()):
            raise ValueError("all entries in visual_length must be positive")
        if int(visual_length.sum().item()) != patch_features.shape[0]:
            raise ValueError(
                "visual_length.sum() must equal the number of packed frames"
            )
        if bool(visual_length.remainder(self.temporal_scale_factor).ne(0).any()):
            raise ValueError(
                "every visual length must be divisible by temporal_scale_factor "
                f"{self.temporal_scale_factor}"
            )
