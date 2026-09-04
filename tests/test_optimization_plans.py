import pytest

from csi_slt.engine.optimization import OptimizationPlan


def test_sparse_component_overrides_preserve_missing_values():
    plan = OptimizationPlan.from_mapping(
        {
            "visual_adapter": {"learning_rate": 5e-5},
            "ctc_codebook": {"weight_decay": 0.0},
        }
    )

    assert plan.get("visual_adapter").learning_rate == 5e-5
    assert plan.get("visual_adapter").weight_decay is None
    assert plan.get("ctc_codebook").learning_rate is None
    assert plan.get("ctc_codebook").weight_decay == 0.0
    assert plan.get("ctc_head").learning_rate is None


def test_parameter_group_overrides_are_sparse():
    plan = OptimizationPlan.from_mapping(
        {
            "visual_adapter": {
                "learning_rate": 1e-4,
                "parameter_groups": {"gates": {"learning_rate": 5e-4}},
            }
        }
    )

    gates = plan.get("visual_adapter").parameter_groups["gates"]
    assert gates.learning_rate == 5e-4
    assert gates.weight_decay is None


@pytest.mark.parametrize(
    "config",
    [
        {"unknown": {"learning_rate": 1e-4}},
        {"ctc_head": {"momentum": 0.9}},
        {"visual_adapter": {"parameter_groups": {"gates": {"momentum": 0.9}}}},
    ],
)
def test_optimization_plan_rejects_unknown_names(config):
    with pytest.raises(ValueError, match="unknown"):
        OptimizationPlan.from_mapping(config)


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", -1e-4),
        ("weight_decay", -0.1),
        ("learning_rate", True),
    ],
)
def test_optimization_plan_rejects_invalid_values(field, value):
    error = TypeError if isinstance(value, bool) else ValueError
    with pytest.raises(error):
        OptimizationPlan.from_mapping({"ctc_head": {field: value}})
