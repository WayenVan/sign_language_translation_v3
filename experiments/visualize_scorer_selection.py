"""Visualize the frozen hand-patch scorer on one configured video.

The sibling of ``visualize_cradio_attention.py`` for the other selector: instead
of CLS-to-patch attention, every patch is scored by the fitted
:class:`HandPatchScorer` and everything outside the per-frame top-k is blacked
out, so what the overlay shows is exactly what the ROI branch of
``SpatiotemporalNextFrameHandRoiClsAdapter`` averages.

Usage::

    # default: the scorer in SCORER_PATH, top-k from TOP_K, val sample 3
    python experiments/visualize_scorer_selection.py

    # elsewhere
    python experiments/visualize_scorer_selection.py outputs/my_scorer_vis

Edit the constants below to steal a different existing Hydra configuration.
``SAMPLE_INDEX`` addresses the same video as in
``visualize_cradio_cross_frame_similarity.py`` and ``visualize_cradio_attention.py``:
the three scripts compose the same config and walk the same unshuffled loader, so
index 3 is one video and the attention map, the matching field and the scorer
selection can be read side by side.

Two things are deliberately not configured here:

* **The backbone is rebuilt from the scorer's own provenance**, not from the
  model config. A scorer's 1153 coefficients are only valid for the backbone and
  ``output_layer`` its fitting features came from, and ``HandPatchScorerConfig``
  records exactly that. Reading it here means the picture can never be of a
  layer the scorer was not fitted on; the Hydra config is used only for the data
  pipeline and the tokenizer.
* **Scoring and selection are delegated to the training-time**
  :class:`TopKRoiPool`, so ``TOP_K`` is the adapter's ``top_k`` kwarg and the
  mask is the one the adapter would pool under -- including its provenance check
  against the live backbone.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from csi_slt.commands.config import instantiate_prompt_resolvers
from csi_slt.data.datamodule import DataModule
from csi_slt.modeling_slt.scorer import HandPatchScorer
from csi_slt.modeling_slt.visual_adapters.hand_roi_pooled_adapter import TopKRoiPool


# Experiment knobs: deliberately kept here instead of adding another Hydra config.
# The config name, the overrides, SPLIT and the loader below are held identical
# to visualize_cradio_cross_frame_similarity.py and visualize_cradio_attention.py
# so that one SAMPLE_INDEX means one video across all three: the dataset order is
# fixed by the `data` group alone, and the `model` override only supplies the
# tokenizer. (The backbone here comes from the scorer's own provenance, not from
# this config, so the 1.7B entry is not a claim about which model is visualized.)
CONFIG_NAME = "train/pretrain_adapter/base"
CONFIG_OVERRIDES = [
    "model=qwen3-1.7b-cradio-l-dinoframecrossv3",
    "data=ph14t_*x224x224_qwen_multiling",
    "data.processor.video_processor.do_normalize=false",
]
SPLIT = "val"
SAMPLE_INDEX = 3
# The fitted scorer: its config names the backbone and output layer the patches
# must come from, and TopKRoiPool refuses a mismatch.
SCORER_PATH = "outputs/hand_patch_scorer_L8"
TOP_K = 24  # Mirrors the adapter's `top_k` kwarg.
# Logits are comparable across frames of one video, so the colour scale is fitted
# once over the whole video: a frame holding no hand then reads dim instead of
# being stretched back to full contrast. Set True for the per-frame view.
NORMALIZE_PER_FRAME = False
SCORE_PERCENTILE_CLIP = (1.0, 99.0)  # Robust ends of the video-wide scale.
BACKBONE_BATCH_SIZE = 32
VIDEO_FPS = 12.0
OVERLAY_ALPHA = 0.45
DRAW_PATCH_GRID = True
PATCH_GRID_COLOR = (255, 255, 255)  # BGR
PATCH_GRID_THICKNESS = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay hand-patch scores and the top-k mask on one video."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "outputs" / "scorer_selection",
    )
    return parser.parse_args()


def compose_config():
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        return hydra.compose(config_name=CONFIG_NAME, overrides=CONFIG_OVERRIDES)


def build_datamodule(cfg) -> DataModule:
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.config.llm_model_name_or_path)
    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(cfg.prompt, (SPLIT,)),
    )
    datamodule.setup("fit" if SPLIT in ("train", "val") else "predict")
    return datamodule


def load_one_video(datamodule: DataModule):
    dataset = getattr(datamodule, f"{SPLIT}_dataset")
    collator = getattr(datamodule, f"{SPLIT}_collator")
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
    )
    try:
        batch = (
            next(iter(loader))
            if SAMPLE_INDEX == 0
            else next(
                batch for index, batch in enumerate(loader) if index == SAMPLE_INDEX
            )
        )
    except StopIteration as error:
        raise IndexError(
            f"sample index {SAMPLE_INDEX} is outside the {SPLIT} dataset"
        ) from error

    length = int(batch["pixel_values_length"][0].item())
    return batch["pixel_values"][:length], batch["names"][0]


def build_backbone(scorer_config, device: torch.device):
    """Rebuild the exact feature extractor the scorer was fitted against.

    ``visual_backbone_class`` plus ``visual_backbone_init_kwargs`` are stored for
    this: every registry backbone is constructed through
    ``from_pretrained_backbone``, and the kwargs are that constructor's literal
    arguments.
    """
    if not scorer_config.visual_backbone_class:
        raise ValueError(
            f"the scorer at {SCORER_PATH} records no backbone provenance, so the "
            "features it must be scored on are unknown; re-fit it with "
            "preprocess/extract_scorer_features.py + preprocess/train_scorer.py"
        )
    module_name, _, class_name = scorer_config.visual_backbone_class.rpartition(".")
    backbone_class = getattr(importlib.import_module(module_name), class_name)

    init_kwargs = dict(scorer_config.visual_backbone_init_kwargs)
    dtype = init_kwargs.get("dtype", "float32")
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype.removeprefix("torch."))
    # bfloat16 on CPU is both slow and unnecessary for a single video.
    init_kwargs["dtype"] = dtype if device.type == "cuda" else torch.float32

    backbone = backbone_class.from_pretrained_backbone(**init_kwargs)
    return backbone.to(device).eval(), init_kwargs["dtype"]


@torch.no_grad()
def extract_patch_features(backbone, frames: torch.Tensor, device, dtype) -> torch.Tensor:
    """Raw patch features for every frame: ``[F, P, D]`` float32 on the CPU.

    Called exactly as ``preprocess/extract_scorer_features.py`` calls it -- one
    frame-independent pass, no ``t_lengths`` -- because the scorer's coefficients
    are calibrated to the features that script produced.
    """
    chunks = []
    for start in range(0, len(frames), BACKBONE_BATCH_SIZE):
        batch = frames[start : start + BACKBONE_BATCH_SIZE]
        batch = batch.to(device=device, dtype=dtype, non_blocking=True)
        chunks.append(backbone(batch).visual_features.float().cpu())
    return torch.cat(chunks, dim=0)


def infer_patch_grid(patch_count: int, frame_height: int, frame_width: int):
    grid_height = int(round(math.sqrt(patch_count * frame_height / frame_width)))
    if grid_height <= 0 or patch_count % grid_height != 0:
        raise ValueError(
            f"cannot infer a rectangular patch grid for {patch_count} tokens and "
            f"frame size {frame_height}x{frame_width}"
        )
    return grid_height, patch_count // grid_height


def to_rgb_uint8(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(frame * 255).astype(np.uint8)


def experiment_output_label() -> str:
    scale = "per_frame" if NORMALIZE_PER_FRAME else "per_video"
    return f"{Path(SCORER_PATH).name}_index_{SAMPLE_INDEX}_topk_{TOP_K}_{scale}"


def draw_patch_grid(image: np.ndarray, grid_height: int, grid_width: int) -> None:
    """Draw the exact inferred patch-cell boundaries in-place."""
    height, width = image.shape[:2]
    for y in np.rint(np.linspace(0, height, grid_height + 1)).astype(int)[1:-1]:
        cv2.line(image, (0, y), (width - 1, y), PATCH_GRID_COLOR, PATCH_GRID_THICKNESS)
    for x in np.rint(np.linspace(0, width, grid_width + 1)).astype(int)[1:-1]:
        cv2.line(image, (x, 0), (x, height - 1), PATCH_GRID_COLOR, PATCH_GRID_THICKNESS)


def score_scale(scores: np.ndarray) -> tuple[float, float]:
    """Low/high ends of the colour scale, over the whole video."""
    low, high = np.percentile(scores, SCORE_PERCENTILE_CLIP)
    return float(low), float(max(high, low + 1e-8))


def normalize(frame_scores: np.ndarray, scale: tuple[float, float]) -> np.ndarray:
    low, high = (
        (float(frame_scores.min()), float(frame_scores.max()))
        if NORMALIZE_PER_FRAME
        else scale
    )
    return np.clip((frame_scores - low) / max(high - low, 1e-8), 0.0, 1.0)


def create_video_writer(path: Path, frame_width: int, frame_height: int):
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (frame_width, frame_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create {path.name}")
    return writer


def render_outputs(
    frames: torch.Tensor,
    scores: torch.Tensor,
    mask: torch.Tensor,
    output_dir: Path,
) -> tuple[int, int]:
    raw_dir = output_dir / "raw_frames"
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    masked_dir = output_dir / "masked_frames"
    for directory in (raw_dir, heatmap_dir, overlay_dir, masked_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame_height, frame_width = frames.shape[-2:]
    grid_height, grid_width = infer_patch_grid(
        scores.shape[-1], frame_height, frame_width
    )
    score_array = scores.cpu().numpy()
    scale = score_scale(score_array)

    overlay_writer = create_video_writer(
        output_dir / "scorer_selection.mp4", frame_width, frame_height
    )
    # The same mask without the colour on top: the plainest way to judge whether
    # the hand actually survived the cut, since nothing tints the kept pixels.
    masked_writer = create_video_writer(
        output_dir / "masked_frames.mp4", frame_width, frame_height
    )

    try:
        for frame_index, (frame, frame_scores, keep_mask) in enumerate(
            zip(frames, score_array, mask.cpu().numpy(), strict=True)
        ):
            rgb = to_rgb_uint8(frame)
            heat = normalize(frame_scores, scale).reshape(grid_height, grid_width)
            heat = cv2.resize(
                heat,
                (frame_width, frame_height),
                # Preserve one constant value per patch. Smooth interpolation
                # can create rings/halos that are absent from the token map.
                interpolation=cv2.INTER_NEAREST,
            )
            heatmap_bgr = cv2.applyColorMap(
                np.rint(heat * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            overlay_bgr = cv2.addWeighted(
                frame_bgr,
                1.0 - OVERLAY_ALPHA,
                heatmap_bgr,
                OVERLAY_ALPHA,
                0.0,
            )
            pixel_keep_mask = cv2.resize(
                keep_mask.reshape(grid_height, grid_width).astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            # Everything the adapter's ROI mean never sees is blacked out.
            overlay_bgr[~pixel_keep_mask] = 0
            # Copy: frame_bgr is written out unmodified as the raw frame.
            masked_bgr = frame_bgr.copy()
            masked_bgr[~pixel_keep_mask] = 0
            if DRAW_PATCH_GRID:
                draw_patch_grid(overlay_bgr, grid_height, grid_width)
                draw_patch_grid(masked_bgr, grid_height, grid_width)
            filename = f"{frame_index:05d}.png"
            cv2.imwrite(str(raw_dir / filename), frame_bgr)
            cv2.imwrite(str(heatmap_dir / filename), heatmap_bgr)
            cv2.imwrite(str(overlay_dir / filename), overlay_bgr)
            cv2.imwrite(str(masked_dir / filename), masked_bgr)
            overlay_writer.write(overlay_bgr)
            masked_writer.write(masked_bgr)
    finally:
        overlay_writer.release()
        masked_writer.release()
    return grid_height, grid_width


def main() -> None:
    args = parse_args()
    cfg = compose_config()
    datamodule = build_datamodule(cfg)
    frames, sample_name = load_one_video(datamodule)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scorer_config = HandPatchScorer.from_pretrained(SCORER_PATH).config
    backbone, dtype = build_backbone(scorer_config, device)
    patch_features = extract_patch_features(backbone, frames, device, dtype)

    # Reuse the training-time pool so the scores and the mask are exactly what
    # the adapter would select under; it also verifies that the live backbone is
    # the one the coefficients were fitted against.
    roi_pool = TopKRoiPool(
        input_dim=patch_features.shape[-1],
        top_k=TOP_K,
        scorer_path=SCORER_PATH,
    )
    roi_pool.load_pretrained_components(visual_backbone=backbone)
    roi_pool.eval()
    with torch.no_grad():
        scores = roi_pool.scorer(patch_features)
        mask = roi_pool.select(patch_features)
        selection_margin = float(roi_pool.score_margin(patch_features).item())

    output_dir = args.output_dir / experiment_output_label()
    output_dir.mkdir(parents=True, exist_ok=True)
    grid_height, grid_width = render_outputs(frames, scores, mask, output_dir)

    selected = scores[mask]
    metadata = {
        "sample_name": str(sample_name),
        "split": SPLIT,
        "sample_index": SAMPLE_INDEX,
        "scorer_path": SCORER_PATH,
        "scorer_backbone": {
            "class": scorer_config.visual_backbone_class,
            "init_kwargs": scorer_config.visual_backbone_init_kwargs,
            "patch_grid_size": scorer_config.patch_grid_size,
        },
        "top_k": TOP_K,
        "selector": "TopKRoiPool",
        "frame_count": len(frames),
        "patch_grid": [grid_height, grid_width],
        "patch_feature_shape": list(patch_features.shape),
        "normalize_per_frame": NORMALIZE_PER_FRAME,
        "score_percentile_clip": list(SCORE_PERCENTILE_CLIP),
        "score_stats": {
            # Logits, not probabilities: only the within-frame ranking is used.
            "min": float(scores.min()),
            "max": float(scores.max()),
            "mean": float(scores.mean()),
            "selected_mean": float(selected.mean()),
            # The k-th to (k+1)-th gap: how decisively the cut is made. A frame
            # with no hand still yields exactly top_k patches.
            "selection_margin": selection_margin,
        },
        "config_name": CONFIG_NAME,
        "config_overrides": CONFIG_OVERRIDES,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir}")


if __name__ == "__main__":
    main()
