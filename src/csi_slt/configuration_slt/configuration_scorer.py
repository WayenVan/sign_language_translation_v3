from copy import deepcopy
from typing import Any

from transformers.configuration_utils import PretrainedConfig


class HandPatchScorerConfig(PretrainedConfig):
    """Configuration for the frozen hand-patch scorer.

    The scorer is one linear map over a single patch feature, so ``input_dim``
    is the only field it needs to run.  The rest is provenance, and it is not
    decoration: a scorer is valid only for the backbone, layer and input
    resolution that produced its fitting features, and a mismatched one degrades
    into a near-random patch ranking rather than failing.

    ``visual_backbone_class`` and ``visual_backbone_init_kwargs`` are enough to
    rebuild that feature extractor::

        module, _, name = config.visual_backbone_class.rpartition(".")
        backbone_class = getattr(importlib.import_module(module), name)
        backbone = backbone_class.from_pretrained_backbone(
            **config.visual_backbone_init_kwargs
        )

    Every backbone in ``csi_slt.modeling_slt.registry`` is built through
    ``from_pretrained_backbone``, so that constructor is a convention here
    rather than a stored field, and ``visual_backbone_init_kwargs`` holds its
    literal keyword arguments -- ``config`` and ``dtype`` included.

    ``patch_grid_size`` is unused by ``forward`` but needed by any consumer that
    treats the scores spatially: smoothing a score map or dilating a selection
    mask both require reshaping ``[P]`` back into a grid.
    """

    model_type = "hand_patch_scorer"

    def __init__(
        self,
        input_dim: int = 1152,
        patch_grid_size: tuple[int, int] | None = None,
        visual_backbone_class: str | None = None,
        visual_backbone_init_kwargs: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        if (
            isinstance(input_dim, bool)
            or not isinstance(input_dim, int)
            or input_dim <= 0
        ):
            raise ValueError(f"input_dim must be a positive integer, got {input_dim!r}")
        if patch_grid_size is not None:
            patch_grid_size = tuple(patch_grid_size)
            if len(patch_grid_size) != 2 or any(
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
                for size in patch_grid_size
            ):
                raise ValueError(
                    "patch_grid_size must be two positive integers, got "
                    f"{patch_grid_size!r}"
                )
        if visual_backbone_class is not None and not isinstance(
            visual_backbone_class, str
        ):
            raise TypeError(
                "visual_backbone_class must be a fully qualified class name or "
                f"None, got {visual_backbone_class!r}"
            )
        if visual_backbone_init_kwargs is not None and not isinstance(
            visual_backbone_init_kwargs, dict
        ):
            raise TypeError(
                "visual_backbone_init_kwargs must be a dictionary or None, got "
                f"{visual_backbone_init_kwargs!r}"
            )
        self.input_dim = input_dim
        self.patch_grid_size = patch_grid_size
        self.visual_backbone_class = visual_backbone_class
        self.visual_backbone_init_kwargs = deepcopy(
            visual_backbone_init_kwargs
            if visual_backbone_init_kwargs is not None
            else {}
        )
        super().__init__(**kwargs)
