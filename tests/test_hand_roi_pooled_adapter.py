import json

import pytest
import torch
from torch import nn

from csi_slt.configuration_slt.configuration_scorer import HandPatchScorerConfig
from csi_slt.modeling_slt.output_utils import VisualBackboneOutput
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS
from csi_slt.modeling_slt.scorer import HandPatchScorer
from csi_slt.modeling_slt.slt import _load_pretrained_submodule_components
from csi_slt.modeling_slt.visual_adapters.hand_roi_pooled_adapter import (
    HandRoiPooledAdapter,
    TopKRoiPool,
)

INPUT_DIM = 6
PATCHES = 9


@pytest.fixture
def scorer_dir(tmp_path):
    """A fitted scorer on disk, with coefficients that rank patch order."""
    torch.manual_seed(0)
    scorer = HandPatchScorer(HandPatchScorerConfig(input_dim=INPUT_DIM))
    with torch.no_grad():
        scorer.linear.weight.copy_(torch.arange(INPUT_DIM).float()[None])
        scorer.linear.bias.zero_()
    scorer.set_feature_statistics(torch.zeros(INPUT_DIM), torch.ones(INPUT_DIM))
    path = tmp_path / "scorer"
    scorer.save_pretrained(path)
    return str(path)


def _backbone_output(patches: torch.Tensor) -> VisualBackboneOutput:
    return VisualBackboneOutput(
        visual_features=patches,
        visual_length=torch.tensor([2, patches.shape[0] - 2]),
    )


def test_hand_roi_pooled_is_registered() -> None:
    assert VISUAL_ADAPTERS["hand_roi_pooled"] is HandRoiPooledAdapter


def test_construction_touches_no_disk_and_refuses_to_score(scorer_dir) -> None:
    # Construction has to stay pure: from_pretrained runs it before overwriting
    # the weights, and a checkpoint must load without the fitting directory.
    pool = TopKRoiPool(input_dim=INPUT_DIM, top_k=3, scorer_path="/does/not/exist")

    assert pool.scorer_is_loaded is False
    with pytest.raises(RuntimeError, match="unfitted coefficients"):
        pool(torch.randn(2, PATCHES, INPUT_DIM))


def test_pool_returns_the_mean_of_the_top_k_patches(scorer_dir) -> None:
    torch.manual_seed(0)
    pool = TopKRoiPool(input_dim=INPUT_DIM, top_k=3, scorer_path=scorer_dir)
    pool.load_pretrained_components()
    patches = torch.randn(4, PATCHES, INPUT_DIM)

    mask = pool.select(patches)
    scores = pool.scorer(patches)

    assert mask.sum(dim=1).tolist() == [3] * 4
    for frame in range(4):
        expected = set(scores[frame].topk(3).indices.tolist())
        assert set(mask[frame].nonzero().flatten().tolist()) == expected
        assert torch.allclose(
            pool(patches)[frame], patches[frame][mask[frame]].mean(dim=0), atol=1e-6
        )


def test_scoring_and_pooling_read_different_tensors(scorer_dir) -> None:
    # The scorer's coefficients were fitted on unmodified patches, so a motion
    # residual belongs in the pooled content and never in front of the ranking.
    pool = TopKRoiPool(input_dim=INPUT_DIM, top_k=3, scorer_path=scorer_dir)
    pool.load_pretrained_components()
    raw = torch.randn(4, PATCHES, INPUT_DIM)
    content = torch.randn(4, PATCHES, 2)

    mask = pool.select(raw)
    pooled = pool(raw, content)

    assert pooled.shape == (4, 2)
    for frame in range(4):
        assert torch.allclose(
            pooled[frame], content[frame][mask[frame]].mean(dim=0), atol=1e-6
        )


def test_pool_rejects_inputs_that_disagree_on_the_patch_axis(scorer_dir) -> None:
    pool = TopKRoiPool(input_dim=INPUT_DIM, top_k=3, scorer_path=scorer_dir)
    pool.load_pretrained_components()
    raw = torch.randn(4, PATCHES, INPUT_DIM)

    with pytest.raises(ValueError, match="share"):
        pool(raw, torch.randn(4, PATCHES + 1, 2))
    with pytest.raises(ValueError, match="wide"):
        pool(torch.randn(4, PATCHES, INPUT_DIM + 1))


def test_loaded_flag_survives_a_state_dict_round_trip(scorer_dir) -> None:
    # A non-persistent flag would come back False after a resume and fire on a
    # model whose weights are perfectly good, having come from the checkpoint.
    fitted = TopKRoiPool(input_dim=INPUT_DIM, top_k=3, scorer_path=scorer_dir)
    fitted.load_pretrained_components()

    resumed = TopKRoiPool(input_dim=INPUT_DIM, top_k=3)
    resumed.load_state_dict(fitted.state_dict())

    assert "scorer_loaded" in fitted.state_dict()
    assert resumed.scorer_is_loaded is True
    patches = torch.randn(2, PATCHES, INPUT_DIM)
    assert torch.equal(resumed(patches), fitted(patches))


def test_loading_rejects_a_scorer_fitted_for_another_width(scorer_dir) -> None:
    pool = TopKRoiPool(input_dim=INPUT_DIM + 1, top_k=3, scorer_path=scorer_dir)

    with pytest.raises(ValueError, match="only valid for the backbone"):
        pool.load_pretrained_components()


def test_adapter_concatenates_both_halves_and_keeps_the_baseline_token_count(
    scorer_dir,
) -> None:
    torch.manual_seed(0)
    adapter = HandRoiPooledAdapter(
        input_dim=INPUT_DIM,
        output_dim=5,
        scorer_path=scorer_dir,
        top_k=3,
        projection_rank=7,
        temporal_scale_factor=2,
    )
    adapter.roi_pool.load_pretrained_components()
    patches = torch.randn(6, PATCHES, INPUT_DIM)

    output = adapter(_backbone_output(patches))

    # One token per temporal_scale_factor frames, exactly as the pooled-linear
    # baseline emits, so token count and the CTC head are untouched.
    assert output.visual_features.shape == (3, 5)
    assert output.visual_length.tolist() == [1, 2]

    mask = adapter.roi_pool.select(patches)
    frame_features = torch.cat(
        [
            patches.mean(dim=1),
            torch.stack([patches[f][mask[f]].mean(dim=0) for f in range(6)]),
        ],
        dim=-1,
    )
    pooled = torch.cat(
        [
            frame_features[:2].unflatten(0, (-1, 2)).mean(dim=1),
            frame_features[2:].unflatten(0, (-1, 2)).mean(dim=1),
        ]
    )
    expected = adapter.projection(adapter.norm(pooled))
    assert torch.allclose(output.visual_features, expected, atol=1e-6)


def test_scorer_is_excluded_from_the_trainable_count(scorer_dir) -> None:
    adapter = HandRoiPooledAdapter(
        input_dim=INPUT_DIM, output_dim=5, scorer_path=scorer_dir, projection_rank=7
    )

    assert adapter.roi_pool.scorer.frozen is True
    assert adapter.roi_pool.scorer.always_frozen is True
    assert adapter.trainable_parameter_count == sum(
        parameter.numel()
        for module in (adapter.norm, adapter.projection)
        for parameter in module.parameters()
    )


def test_freeze_scorer_false_leaves_the_marker_off(scorer_dir) -> None:
    adapter = HandRoiPooledAdapter(
        input_dim=INPUT_DIM,
        output_dim=5,
        scorer_path=scorer_dir,
        projection_rank=7,
        freeze_scorer=False,
    )

    assert adapter.roi_pool.scorer.always_frozen is False
    assert adapter.roi_pool.scorer.frozen is False


def test_model_factory_hook_installs_the_scorer_once(scorer_dir) -> None:
    model = nn.Module()
    model.visual_adapter = HandRoiPooledAdapter(
        input_dim=INPUT_DIM, output_dim=5, scorer_path=scorer_dir, projection_rank=7
    )

    _load_pretrained_submodule_components(model)

    assert model.visual_adapter.roi_pool.scorer_is_loaded is True
    # The hook lives on the owner only; a parent that also declared it would
    # make the factory read the same weights twice.
    assert not hasattr(model.visual_adapter, "load_pretrained_components")


def test_saved_scorer_carries_the_backbone_it_was_fitted_against(tmp_path) -> None:
    config = HandPatchScorerConfig(
        input_dim=INPUT_DIM,
        patch_grid_size=(3, 3),
        visual_backbone_class="pkg.module.SomeBackbone",
        visual_backbone_init_kwargs={"config": {"id": "x"}, "dtype": "bfloat16"},
    )
    HandPatchScorer(config).save_pretrained(tmp_path / "scorer")

    stored = json.loads((tmp_path / "scorer" / "config.json").read_text())
    assert stored["visual_backbone_class"] == "pkg.module.SomeBackbone"
    assert stored["visual_backbone_init_kwargs"]["dtype"] == "bfloat16"
    assert tuple(stored["patch_grid_size"]) == (3, 3)
