"""Visualize C-RADIO CLS-to-patch attention for one configured video.

This is intentionally a standalone experiment rather than a project command.
Edit the constants below to steal a different existing Hydra configuration.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from csi_slt.commands.prompt_setup import instantiate_prompt_resolvers
from csi_slt.data.datamodule import DataModule
from csi_slt.modeling_slt.visual_adapters.attnpool_adapter import (
    ClsAttentionTopKSelector,
)
from csi_slt.modeling_slt.visual_backbones.c_radio_v4_backbone import (
    CRadioV4Backbone,
)


# Experiment knobs: deliberately kept here instead of adding another Hydra config.
CONFIG_NAME = "train/base"
CONFIG_OVERRIDES = [
    "model=qwen3-1.7b-cradio-l-dinoframecrossv3",
    "data=ph14t_*x224x224_qwen_multiling",
    "data.processor.video_processor.do_normalize=false",
]
SPLIT = "val"
SAMPLE_INDEX = 3
ATTENTION_LAYER = -25
# Scoring and selection are delegated to the training-time
# ClsAttentionTopKSelector, so these two knobs mirror the adapter kwargs
# `top_k` and `attention_smooth_kernel_size`. The selector always zeros the four
# corner patches and always excludes them from the Top-K candidates; -1 keeps
# every non-corner patch.
TOP_K = 48
SPATIAL_SMOOTH_KERNEL = 3  # 1 disables smoothing; otherwise use an odd size.
VIDEO_FPS = 12.0
OVERLAY_ALPHA = 0.45
DRAW_PATCH_GRID = True
PATCH_GRID_COLOR = (255, 255, 255)  # BGR
PATCH_GRID_THICKNESS = 1

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay C-RADIO CLS attention on one video."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "outputs" / "cradio_attention",
    )
    return parser.parse_args()


def compose_config():
    with hydra.initialize_config_dir(
        version_base=None,
        config_dir=str(CONFIG_DIR),
    ):
        return hydra.compose(
            config_name=CONFIG_NAME,
            overrides=[
                *CONFIG_OVERRIDES,
                f"++model.config.visual_backbone_config.attention_layer={ATTENTION_LAYER}",
            ],
        )


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


def attention_layer_label(layer: int) -> str:
    return f"last_{abs(layer)}" if layer < 0 else f"layer_{layer}"


def experiment_output_label() -> str:
    top_k_label = "all" if TOP_K == -1 else str(TOP_K)
    return (
        f"{attention_layer_label(ATTENTION_LAYER)}_"
        f"index_{SAMPLE_INDEX}_topk_{top_k_label}_"
        f"smooth_{SPATIAL_SMOOTH_KERNEL}x{SPATIAL_SMOOTH_KERNEL}"
    )


def draw_patch_grid(
    image: np.ndarray,
    grid_height: int,
    grid_width: int,
) -> None:
    """Draw the exact inferred patch-cell boundaries in-place."""
    height, width = image.shape[:2]
    for y in np.rint(np.linspace(0, height, grid_height + 1)).astype(int)[1:-1]:
        cv2.line(
            image,
            (0, y),
            (width - 1, y),
            PATCH_GRID_COLOR,
            PATCH_GRID_THICKNESS,
        )
    for x in np.rint(np.linspace(0, width, grid_width + 1)).astype(int)[1:-1]:
        cv2.line(
            image,
            (x, 0),
            (x, height - 1),
            PATCH_GRID_COLOR,
            PATCH_GRID_THICKNESS,
        )


def render_outputs(
    frames: torch.Tensor,
    attention_maps: torch.Tensor,
    output_dir: Path,
) -> None:
    raw_dir = output_dir / "raw_frames"
    heatmap_dir = output_dir / "heatmaps"
    overlay_dir = output_dir / "overlays"
    for directory in (raw_dir, heatmap_dir, overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    frame_height, frame_width = frames.shape[-2:]
    grid_height, grid_width = infer_patch_grid(
        attention_maps.shape[-1], frame_height, frame_width
    )
    # Reuse the training-time selector so the rendered scores and mask are
    # exactly what AttnPoolAdapter would consume.
    selector = ClsAttentionTopKSelector(
        top_k=TOP_K,
        spatial_smooth_kernel=SPATIAL_SMOOTH_KERNEL,
        grid_size=(grid_height, grid_width),
    )
    selection = selector(attention_maps)
    video_writer = cv2.VideoWriter(
        str(output_dir / "attention.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        VIDEO_FPS,
        (frame_width, frame_height),
    )
    if not video_writer.isOpened():
        raise RuntimeError("OpenCV could not create attention.mp4")

    try:
        for frame_index, (frame, frame_scores, keep_mask) in enumerate(
            zip(frames, selection.scores, selection.mask, strict=True)
        ):
            rgb = to_rgb_uint8(frame)
            # Copy: the in-place min-max below must not touch selection.scores.
            attention = (
                frame_scores.reshape(grid_height, grid_width).cpu().numpy().copy()
            )
            attention -= attention.min()
            attention /= max(float(attention.max()), 1e-8)
            attention = cv2.resize(
                attention,
                (frame_width, frame_height),
                # Preserve one constant value per patch. Smooth interpolation
                # can create rings/halos that are absent from the token map.
                interpolation=cv2.INTER_NEAREST,
            )
            heatmap_bgr = cv2.applyColorMap(
                np.rint(attention * 255).astype(np.uint8), cv2.COLORMAP_JET
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
                keep_mask.reshape(grid_height, grid_width)
                .cpu()
                .numpy()
                .astype(np.uint8),
                (frame_width, frame_height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            overlay_bgr[~pixel_keep_mask] = 0
            if DRAW_PATCH_GRID:
                draw_patch_grid(overlay_bgr, grid_height, grid_width)
            filename = f"{frame_index:05d}.png"
            cv2.imwrite(str(raw_dir / filename), frame_bgr)
            cv2.imwrite(str(heatmap_dir / filename), heatmap_bgr)
            cv2.imwrite(str(overlay_dir / filename), overlay_bgr)
            video_writer.write(overlay_bgr)
    finally:
        video_writer.release()


def main() -> None:
    args = parse_args()
    cfg = compose_config()
    datamodule = build_datamodule(cfg)
    frames, sample_name = load_one_video(datamodule)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    backbone_config = OmegaConf.to_container(
        cfg.model.config.visual_backbone_config,
        resolve=True,
    )
    backbone = CRadioV4Backbone.from_pretrained_backbone(
        backbone_config,
        dtype=dtype,
    ).to(device)
    backbone.eval()

    lengths = torch.tensor([len(frames)], dtype=torch.long, device=device)
    with torch.inference_mode():
        output = backbone(
            frames.to(device=device, dtype=dtype),
            t_lengths=lengths,
            return_attention_maps=True,
        )
    attention_maps = output.extras["attention_maps"].cpu()

    layer_output_dir = args.output_dir / experiment_output_label()
    layer_output_dir.mkdir(parents=True, exist_ok=True)
    render_outputs(
        frames,
        attention_maps,
        layer_output_dir,
    )
    metadata = {
        "sample_name": str(sample_name),
        "split": SPLIT,
        "sample_index": SAMPLE_INDEX,
        "attention_layer": ATTENTION_LAYER,
        "attention_layer_label": attention_layer_label(ATTENTION_LAYER),
        "top_k": TOP_K,
        "spatial_smooth_kernel": SPATIAL_SMOOTH_KERNEL,
        "selector": "ClsAttentionTopKSelector",
        "frame_count": len(frames),
        "attention_shape": list(attention_maps.shape),
        "config_name": CONFIG_NAME,
        "config_overrides": [
            *CONFIG_OVERRIDES,
            f"++model.config.visual_backbone_config.attention_layer={ATTENTION_LAYER}",
        ],
    }
    (layer_output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
