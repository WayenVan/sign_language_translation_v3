import torch
from torch import nn

from csi_slt.modeling_slt.slt import SltModel


class _DummyLlm(nn.Module):
    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embedding


def _model_with_embedding(embedding: nn.Embedding) -> SltModel:
    model = SltModel.__new__(SltModel)
    nn.Module.__init__(model)
    model.llm = _DummyLlm(embedding)
    return model


def test_apply_llm_embedding_scale_uses_embedding_scale() -> None:
    embedding = nn.Embedding(8, 4)
    embedding.register_buffer("embed_scale", torch.tensor(4.0), persistent=False)
    model = _model_with_embedding(embedding)
    visual_embeddings = torch.ones(2, 3, 4)

    scaled = model._apply_llm_embedding_scale(visual_embeddings)

    assert torch.equal(scaled, torch.full_like(visual_embeddings, 4.0))


def test_apply_llm_embedding_scale_preserves_unscaled_backends() -> None:
    model = _model_with_embedding(nn.Embedding(8, 4))
    visual_embeddings = torch.randn(2, 3, 4)

    scaled = model._apply_llm_embedding_scale(visual_embeddings)

    assert scaled is visual_embeddings
