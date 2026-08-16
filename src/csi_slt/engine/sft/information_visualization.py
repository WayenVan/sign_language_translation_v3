"""Reusable rendering helpers for model information outputs."""

from pathlib import Path

import torch


def render_llm_attention(
    attention: torch.Tensor,
    visual_mask: torch.Tensor,
    output_path: str | Path,
) -> None:
    """Render a head-averaged LLM attention matrix as a PNG image.

    The attention matrix and visual mask must already be restricted to valid
    (non-padding) tokens. Red guide lines mark the visual-token span.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    output_path = Path(output_path)
    attention_array = attention.detach().cpu().float().numpy()
    visual_mask_array = visual_mask.detach().cpu().bool().numpy()

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(
        attention_array,
        interpolation="nearest",
        aspect="auto",
    )
    visual_indices = visual_mask_array.nonzero()[0]
    if visual_indices.size:
        start = int(visual_indices[0]) - 0.5
        end = int(visual_indices[-1]) + 0.5
        axis.axvline(start, color="tab:red", linewidth=0.8)
        axis.axvline(end, color="tab:red", linewidth=0.8)
        axis.axhline(start, color="tab:red", linewidth=0.8)
        axis.axhline(end, color="tab:red", linewidth=0.8)
    axis.set_xlabel("Key token index")
    axis.set_ylabel("Query token index")
    axis.set_title("Last-layer mean LLM attention")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
