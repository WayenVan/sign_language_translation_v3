import pytest

from csi_slt.modeling_slt.visual_adapters.hand_roi_pooled_adapter import (
    HandRoiPooledAdapter,
)
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_motion_adapter import (
    SpatiotemporalMotionAdapter,
)
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_adapter import (
    SpatiotemporalNextFrameAdapter,
)
from csi_slt.modeling_slt.visual_adapters.spatiotemporal_next_frame_motion_adapter import (
    SpatiotemporalNextFrameMotionAdapter,
)


def _registered_gates(adapter):
    return adapter.optimization_parameter_groups().get("gates", ())


@pytest.mark.parametrize(
    ("adapter", "expected_gates"),
    [
        (
            SpatiotemporalMotionAdapter(input_dim=4, output_dim=6),
            lambda adapter: (adapter.temporal_fusion.gate,),
        ),
        (
            SpatiotemporalNextFrameAdapter(
                input_dim=4, output_dim=6, patch_grid_size=(2, 2)
            ),
            lambda adapter: (adapter.next_frame_patch_fusion.fusion_gate,),
        ),
        (
            SpatiotemporalNextFrameMotionAdapter(
                input_dim=4, output_dim=6, patch_grid_size=(2, 2)
            ),
            lambda adapter: (
                adapter.next_frame_patch_fusion.fusion_gate,
                adapter.temporal_fusion.gate,
            ),
        ),
    ],
)
def test_motion_adapters_register_their_residual_gates(adapter, expected_gates):
    assert _registered_gates(adapter) == expected_gates(adapter)


def test_hand_roi_adapter_registers_gate_only_in_gated_mode():
    concat = HandRoiPooledAdapter(
        input_dim=4, output_dim=6, top_k=2, fusion_mode="concat"
    )
    gated = HandRoiPooledAdapter(
        input_dim=4, output_dim=6, top_k=2, fusion_mode="gated"
    )

    assert concat.optimization_parameter_groups() == {}
    assert _registered_gates(gated) == (gated.fusion_gate,)
