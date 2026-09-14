import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from csi_slt.modeling_slt.output_utils import VisualAdapterOutput, VisualBackboneOutput
from csi_slt.modeling_slt.registry import VISUAL_ADAPTERS, VISUAL_BACKBONES
from csi_slt.modeling_slt.slt import SltConfig, SltModel, get_llm_cls_by_model_name


def test_dense_qwen_selection_is_unchanged():
    assert get_llm_cls_by_model_name('Qwen/Qwen3-14B', Qwen3Config()) is Qwen3ForCausalLM


def test_moe_config_selects_native_model_and_backpropagates_to_inputs(tmp_path):
    config = Qwen3MoeConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2,
    )
    # Config-based dispatch works even when loading from an arbitrary local path.
    cls = get_llm_cls_by_model_name(str(tmp_path), config)
    assert cls is Qwen3MoeForCausalLM
    model = cls._from_config(config).eval().requires_grad_(False)
    embeds = torch.randn(1, 4, 16, requires_grad=True)
    output = model(inputs_embeds=embeds, use_cache=False)
    output.logits.square().mean().backward()
    assert torch.isfinite(embeds.grad).all()
    assert embeds.grad.abs().sum() > 0
    model.save_pretrained(tmp_path)
    restored = cls.from_pretrained(tmp_path)
    assert restored.config.model_type == 'qwen3_moe'


class _TinyBackbone(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    def forward(self, video, video_length, **kwargs):
        pooled = video.mean(dim=(1, 2, 3), keepdim=True).view(-1, 1).expand(-1, 4)
        return VisualBackboneOutput(
            visual_features=self.proj(pooled), visual_length=video_length, extras=None
        )


class _TinyAdapter(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(4, 16)

    def forward(self, backbone_output, permute_video_tokens=False):
        return VisualAdapterOutput(
            visual_features=self.proj(backbone_output.visual_features),
            visual_length=backbone_output.visual_length,
            position_ids=None,
            logging_scalars=None,
        )


def _tiny_moe_slt_model(*, video_bidirectional_attention: bool) -> SltModel:
    VISUAL_BACKBONES['_tiny_moe_test'] = _TinyBackbone
    VISUAL_ADAPTERS['_tiny_moe_test'] = _TinyAdapter
    llm_config = Qwen3MoeConfig(
        vocab_size=64, hidden_size=16, intermediate_size=32,
        moe_intermediate_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, head_dim=8,
        num_experts=4, num_experts_per_tok=2,
    )
    config = SltConfig(
        llm_config=llm_config,
        llm_model_name_or_path='dummy/qwen3-moe',
        video_soft_token_id=5,
        visual_backbone_type='_tiny_moe_test',
        visual_backbone_config={},
        visual_adapter_type='_tiny_moe_test',
        visual_adapter_kwargs={},
        video_bidirectional_attention=video_bidirectional_attention,
        attn_implementation='sdpa',
    )
    return SltModel(config)


@pytest.mark.parametrize('video_bidirectional_attention', [True, False])
def test_moe_backend_forward_backward_and_cached_decode(video_bidirectional_attention):
    """Qwen3-MoE's forward() has no dict-attention_mask branch (unlike dense
    Qwen3/Gemma3/Gemma4, which index a
    {"full_attention": ..., "sliding_attention": ...} mapping by
    config.layer_types). SltModel always builds that mapping to carry the
    bidirectional video-token overlay; passing it straight through to a
    Qwen3-MoE backend used to crash with
    ``AttributeError: 'dict' object has no attribute 'ndim'``.
    """
    model = _tiny_moe_slt_model(video_bidirectional_attention=video_bidirectional_attention)
    video = torch.randn(4, 3, 8, 8)
    video_length = torch.tensor([4])
    input_ids = torch.tensor([[0, 0, 5, 5, 5, 5, 5, 5, 1, 1]])
    token_type_ids = (input_ids == 5).long()
    labels = input_ids.clone()

    out = model(
        input_ids=input_ids,
        pixel_values=video,
        pixel_values_length=video_length,
        attention_mask=torch.ones_like(input_ids),
        token_type_ids=token_type_ids,
        labels=labels,
    )
    assert torch.isfinite(out.loss)
    out.loss.backward()

    model.eval()
    with torch.no_grad():
        decode_out = model(
            input_ids=torch.tensor([[7]]),
            attention_mask=torch.ones(1, 1, dtype=torch.long),
            use_cache=True,
        )
    assert decode_out.logits.shape == (1, 1, 64)
