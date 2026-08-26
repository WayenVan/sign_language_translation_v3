import pytest

from csi_slt.engine.trainability import (
    ComponentTrainabilityPlan,
    LlmTrainabilityPlan,
    SltTrainabilityPlan,
    VisualBackboneTrainabilityPlan,
)


def test_slt_plan_defaults_to_frozen_components():
    plan = SltTrainabilityPlan()

    assert plan.llm.mode == "frozen"
    assert plan.visual_backbone.mode == "frozen"
    assert plan.visual_adapter.mode == "frozen"
    assert plan.visual_semantic_encoder.mode == "frozen"


def test_visual_last_n_layers_requires_a_positive_layer_count():
    with pytest.raises(ValueError, match="positive integer"):
        VisualBackboneTrainabilityPlan(mode="last_n_layers")

    with pytest.raises(ValueError, match="positive integer"):
        VisualBackboneTrainabilityPlan(mode="last_n_layers", n_layers=0)


def test_visual_layer_count_is_only_valid_for_last_n_layers():
    with pytest.raises(ValueError, match="only be set"):
        VisualBackboneTrainabilityPlan(mode="full", n_layers=2)


def test_plans_reject_unknown_modes():
    with pytest.raises(ValueError, match="component trainability mode"):
        ComponentTrainabilityPlan(mode="partial")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="visual backbone trainability mode"):
        VisualBackboneTrainabilityPlan(mode="partial")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="LLM trainability mode"):
        LlmTrainabilityPlan(mode="partial")  # type: ignore[arg-type]


def test_lora_mode_is_specific_to_llm_plan():
    assert LlmTrainabilityPlan(mode="lora").mode == "lora"

    with pytest.raises(ValueError, match="component trainability mode"):
        ComponentTrainabilityPlan(mode="lora")  # type: ignore[arg-type]
