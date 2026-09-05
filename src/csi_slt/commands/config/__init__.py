"""Composition-root helpers that translate Hydra config into typed objects.

Each module here resolves one section of the experiment config (``cfg.prompt``,
``cfg.engine``) into the plain Python/typed values that ``commands/*.py`` entry
scripts pass to constructors. Nothing here is itself runnable; it exists only
to keep that translation in one place instead of duplicated across scripts.
"""

from .engine import build_slt_trainer_kwargs
from .prompts import instantiate_prompt_resolvers

__all__ = [
    "build_slt_trainer_kwargs",
    "instantiate_prompt_resolvers",
]
