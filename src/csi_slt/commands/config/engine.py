"""Composition-root helpers for constructing ``SltTrainer``'s engine arguments.

``SltTrainer`` no longer mines ``hydra_config`` for values that drive training
behavior -- that config is a snapshot kept only for logging/checkpointing
(wandb, ``hydra_config.yaml``). The values that used to be pulled out of it
internally (``engine.forward_mode``, ``engine.optimization``,
``engine.ctc_codebook_temperature_schedule``, ``engine.eval_information``,
``engine.train_probe``) are resolved once here,
by the caller, into explicit, typed keyword arguments. Both ``train.py`` and
``evaluate.py`` build their trainer from the same helper so the two scripts
cannot resolve the same config key two different ways.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from csi_slt.engine.optimization import OptimizationPlan
from csi_slt.engine.schedule import ScalarAnnealSchedule


def _resolve_mapping(cfg: DictConfig, path: str) -> dict[str, Any]:
    """Read an optional mapping node, defaulting to empty when absent."""
    node = OmegaConf.select(cfg, path, default=None)
    if node is None:
        return {}
    if OmegaConf.is_config(node):
        return OmegaConf.to_container(node, resolve=True)
    return dict(node)


def resolve_forward_mode(cfg: DictConfig) -> str:
    """Read ``engine.forward_mode``, defaulting to ``"joint"`` when absent."""
    return OmegaConf.select(cfg, "engine.forward_mode", default="joint")


def build_slt_trainer_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Resolve every engine-config value ``SltTrainer`` needs, once.

    Returns a dict meant to be splatted straight into ``SltTrainer(...)``
    alongside ``hydra_config=cfg`` (kept separately, since that argument
    stays purely for logging).
    """
    return {
        "forward_mode": resolve_forward_mode(cfg),
        "optimization_plan": OptimizationPlan.from_mapping(
            _resolve_mapping(cfg, "engine.optimization")
        ),
        "ctc_temperature_schedule": ScalarAnnealSchedule.from_mapping(
            _resolve_mapping(cfg, "engine.ctc_codebook_temperature_schedule")
        ),
        "eval_information_kwargs": _resolve_mapping(cfg, "engine.eval_information"),
        "train_probe_kwargs": _resolve_mapping(cfg, "engine.train_probe"),
    }
