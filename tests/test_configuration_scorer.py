import pytest

from csi_slt.configuration_slt.configuration_scorer import HandPatchScorerConfig


def test_visual_backbone_provenance_round_trips_through_pretrained_config(tmp_path):
    class_name = (
        "csi_slt.modeling_slt.visual_backbones.c_radio_v4_backbone."
        "CRadioV4Backbone"
    )
    init_kwargs = {
        "config": {
            "id": "nvidia/C-RADIOv4-SO400M",
            "output_layer": -1,
        }
    }

    config = HandPatchScorerConfig(
        visual_backbone_class=class_name,
        visual_backbone_init_kwargs=init_kwargs,
    )
    config.save_pretrained(tmp_path)
    restored = HandPatchScorerConfig.from_pretrained(tmp_path)

    assert restored.visual_backbone_class == class_name
    assert restored.visual_backbone_init_kwargs == init_kwargs


def test_visual_backbone_init_kwargs_are_copied():
    init_kwargs = {"config": {"id": "backbone-id"}}

    config = HandPatchScorerConfig(visual_backbone_init_kwargs=init_kwargs)
    init_kwargs["config"]["id"] = "changed"

    assert config.visual_backbone_init_kwargs == {
        "config": {"id": "backbone-id"}
    }


def test_visual_backbone_provenance_defaults_support_old_configs():
    config = HandPatchScorerConfig()

    assert config.visual_backbone_class is None
    assert config.visual_backbone_init_kwargs == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("visual_backbone_class", 123),
        ("visual_backbone_init_kwargs", []),
    ],
)
def test_visual_backbone_provenance_rejects_invalid_types(field, value):
    with pytest.raises(TypeError, match=field):
        HandPatchScorerConfig(**{field: value})
