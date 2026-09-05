from omegaconf import OmegaConf

from csi_slt.commands.config import build_slt_trainer_kwargs
from csi_slt.engine.optimization import OptimizationPlan


def test_engine_config_resolves_to_typed_trainer_kwargs():
    cfg = OmegaConf.create(
        {
            "engine": {
                "optimization": {"visual_adapter": {"learning_rate": 1e-4}},
                "eval_information": {"every_n_evaluations": 3, "num_samples": 2},
                "train_probe": {"every_n_evaluations": 1, "num_samples": 800},
            }
        }
    )

    kwargs = build_slt_trainer_kwargs(cfg)

    assert isinstance(kwargs["optimization_plan"], OptimizationPlan)
    assert kwargs["optimization_plan"].get("visual_adapter").learning_rate == 1e-4
    assert kwargs["eval_information_kwargs"] == {
        "every_n_evaluations": 3,
        "num_samples": 2,
    }
    assert kwargs["train_probe_kwargs"] == {
        "every_n_evaluations": 1,
        "num_samples": 800,
    }


def test_absent_engine_sections_resolve_to_empty_defaults():
    kwargs = build_slt_trainer_kwargs(OmegaConf.create({"engine": {}}))

    assert kwargs["optimization_plan"].components == {}
    assert kwargs["eval_information_kwargs"] == {}
    assert kwargs["train_probe_kwargs"] == {}


def test_interpolations_are_resolved_before_the_trainer_sees_them():
    cfg = OmegaConf.create(
        {
            "engine": {
                "training_args": {"learning_rate": 5e-5},
                "optimization": {
                    "visual_adapter": {
                        "learning_rate": "${engine.training_args.learning_rate}"
                    }
                },
            }
        }
    )

    plan = build_slt_trainer_kwargs(cfg)["optimization_plan"]

    assert plan.get("visual_adapter").learning_rate == 5e-5
