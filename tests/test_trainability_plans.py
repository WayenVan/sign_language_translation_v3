import pytest

from csi_slt.engine.trainability import SltTrainabilityPlan


def _mapping(**overrides):
    mapping = {
        "llm": {"parameter_mode": "frozen"},
        "visual_backbone": {"parameter_mode": "frozen"},
        "visual_adapter": {"parameter_mode": "frozen"},
        "ctc_head": {"parameter_mode": "frozen"},
        "ctc_codebook": {"parameter_mode": "frozen"},
        "visual_position_embedding": {"parameter_mode": "frozen"},
        "visual_boundary_embeddings": {"parameter_mode": "frozen"},
    }
    mapping.update(overrides)
    return mapping


def test_slt_plan_defaults_frozen_runtime_components_to_eval():
    plan = SltTrainabilityPlan.from_mapping(_mapping())

    assert plan["llm"].parameter_mode == "frozen"
    assert plan["llm"].resolved_runtime_mode == "eval"
    assert plan["visual_backbone"].resolved_runtime_mode == "eval"
    assert plan["visual_adapter"].resolved_runtime_mode == "eval"
    assert plan["ctc_head"].resolved_runtime_mode is None


def test_derived_runtime_tracks_parameter_mode_and_can_be_overridden():
    plan = SltTrainabilityPlan.from_mapping(
        _mapping(
            llm={"parameter_mode": "lora"},
            visual_adapter={"parameter_mode": "full", "runtime_mode": "eval"},
            visual_backbone={"parameter_mode": "full", "runtime_mode": "follow"},
        )
    )

    assert plan["llm"].resolved_runtime_mode == "follow"
    assert plan["visual_adapter"].resolved_runtime_mode == "eval"
    assert plan["visual_backbone"].resolved_runtime_mode == "follow"


def test_frozen_component_can_still_be_asked_to_follow():
    plan = SltTrainabilityPlan.from_mapping(
        _mapping(llm={"parameter_mode": "frozen", "runtime_mode": "follow"})
    )
    assert plan["llm"].resolved_runtime_mode == "follow"


@pytest.mark.parametrize("retired", ["auto", "train"])
def test_retired_runtime_vocabulary_is_rejected(retired):
    """``auto`` is now spelled by omission, and ``train`` by ``follow``."""
    with pytest.raises(ValueError, match="must be follow or eval"):
        SltTrainabilityPlan.from_mapping(
            _mapping(
                visual_adapter={"parameter_mode": "full", "runtime_mode": retired}
            )
        )


@pytest.mark.parametrize("n_layers", [None, 0, True])
def test_visual_last_n_layers_requires_a_positive_layer_count(n_layers):
    with pytest.raises(ValueError, match="positive integer"):
        SltTrainabilityPlan.from_mapping(
            _mapping(
                visual_backbone={
                    "parameter_mode": "last_n_layers",
                    "options": {"n_layers": n_layers},
                }
            )
        )


def test_visual_layer_count_is_only_valid_for_last_n_layers():
    with pytest.raises(ValueError, match="only valid"):
        SltTrainabilityPlan.from_mapping(
            _mapping(
                visual_backbone={
                    "parameter_mode": "full",
                    "options": {"n_layers": 2},
                }
            )
        )


def test_component_rules_reject_unsupported_modes_and_runtime():
    with pytest.raises(ValueError, match="parameter_mode for ctc_head"):
        SltTrainabilityPlan.from_mapping(
            _mapping(ctc_head={"parameter_mode": "lora"})
        )
    with pytest.raises(ValueError, match="does not support runtime_mode"):
        SltTrainabilityPlan.from_mapping(
            _mapping(ctc_codebook={"parameter_mode": "frozen", "runtime_mode": "eval"})
        )
    with pytest.raises(ValueError, match="runtime_mode for llm"):
        SltTrainabilityPlan.from_mapping(
            _mapping(llm={"parameter_mode": "frozen", "runtime_mode": "sometimes"})
        )


def test_component_rules_reject_unknown_fields_and_options():
    with pytest.raises(ValueError, match="unknown fields"):
        SltTrainabilityPlan.from_mapping(
            _mapping(llm={"parameter_mode": "frozen", "mode": "full"})
        )
    with pytest.raises(ValueError, match="unsupported keys"):
        SltTrainabilityPlan.from_mapping(
            _mapping(
                visual_adapter={
                    "parameter_mode": "full",
                    "options": {"n_layers": 2},
                }
            )
        )


def test_plan_requires_exactly_the_known_components():
    missing = _mapping()
    missing.pop("ctc_head")
    with pytest.raises(ValueError, match="missing components"):
        SltTrainabilityPlan.from_mapping(missing)

    with pytest.raises(ValueError, match="unknown components"):
        SltTrainabilityPlan.from_mapping({**_mapping(), "extra": {}})
