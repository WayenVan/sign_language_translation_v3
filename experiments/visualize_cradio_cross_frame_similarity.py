"""Visualize temperature-softmax C-RADIO patch matching for one video.

This standalone experiment mirrors ``visualize_cradio_attention.py``: edit the
constants below to select another Hydra configuration, sample, matching window,
or query patch. For every adjacent frame pair it renders exactly four views:

1. one current-frame query patch and its next-frame probability heatmap;
2. the Top-1 matching probability of every current-frame patch;
3. the Top-1 patch displacement field from the current to the next frame;
4. the expected displacement under the complete softmax distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from csi_slt.commands.config import instantiate_prompt_resolvers
from csi_slt.data.datamodule import DataModule
from csi_slt.modeling_slt.visual_backbones.c_radio_v4_backbone import (
    CRadioV4Backbone,
)


# Experiment knobs: kept local rather than adding a Hydra configuration.
CONFIG_NAME = "train/base"
CONFIG_OVERRIDES = [
    "model=qwen3-1.7b-cradio-l-dinoframecrossv3",
    "data=ph14t_*x224x224_qwen_multiling",
    "data.processor.video_processor.do_normalize=false",
]
SPLIT = "val"
SAMPLE_INDEX = 3

# Match the NextFramePatchFusion default. None searches the complete next frame.
SPATIAL_WINDOW_RADIUS: int | None = 1
TEMPERATURE = 0.1
MATCHING_TOP_K = 1
# None selects the center patch. Set both values to explicit zero-based grid
# coordinates to inspect another current-frame patch.
QUERY_PATCH_ROW: int | None = None
QUERY_PATCH_COLUMN: int | None = None

VIDEO_FPS = 12.0
OVERLAY_ALPHA = 0.45
DRAW_PATCH_GRID = True
PATCH_GRID_COLOR = (255, 255, 255)  # BGR
PATCH_GRID_THICKNESS = 1
# Draw one arrow every N rows/columns to prevent dense grids becoming opaque.
ARROW_STRIDE = 1
ARROW_MIN_PROBABILITY = 0.0
ARROW_THICKNESS = 1
ARROW_TIP_LENGTH = 0.25
STATIONARY_DOT_RADIUS = 2
# Keep the full probability range visible: values above an artificial cap would
# otherwise become indistinguishable precisely when checking Top-K sharpness.
PROBABILITY_VMIN = 0.0
PROBABILITY_VMAX = 1.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize cross-frame C-RADIO patch matching."
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        nargs="?",
        default=PROJECT_ROOT / "outputs" / "cradio_cross_frame_similarity",
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


def infer_patch_grid(
    patch_count: int, frame_height: int, frame_width: int
) -> tuple[int, int]:
    grid_height = int(round(math.sqrt(patch_count * frame_height / frame_width)))
    if grid_height <= 0 or patch_count % grid_height != 0:
        raise ValueError(
            f"cannot infer a rectangular patch grid for {patch_count} tokens and "
            f"frame size {frame_height}x{frame_width}"
        )
    return grid_height, patch_count // grid_height


def resolve_query_patch(grid_height: int, grid_width: int) -> tuple[int, int]:
    if (QUERY_PATCH_ROW is None) != (QUERY_PATCH_COLUMN is None):
        raise ValueError(
            "QUERY_PATCH_ROW and QUERY_PATCH_COLUMN must either both be set or "
            "both be None"
        )
    row = grid_height // 2 if QUERY_PATCH_ROW is None else QUERY_PATCH_ROW
    column = grid_width // 2 if QUERY_PATCH_COLUMN is None else QUERY_PATCH_COLUMN
    if not 0 <= row < grid_height or not 0 <= column < grid_width:
        raise ValueError(
            f"query patch ({row}, {column}) is outside the "
            f"{grid_height}x{grid_width} grid"
        )
    return row, column


def to_rgb_uint8(frame: torch.Tensor) -> np.ndarray:
    frame = frame.detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    return np.rint(frame * 255).astype(np.uint8)


def draw_patch_grid(image: np.ndarray, grid_height: int, grid_width: int) -> None:
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


def patch_center(
    row: float,
    column: float,
    grid_height: int,
    grid_width: int,
    image_height: int,
    image_width: int,
) -> tuple[int, int]:
    x = round((column + 0.5) * image_width / grid_width)
    y = round((row + 0.5) * image_height / grid_height)
    return x, y


def draw_query_patch(
    image: np.ndarray,
    row: int,
    column: int,
    grid_height: int,
    grid_width: int,
) -> None:
    height, width = image.shape[:2]
    x_edges = np.rint(np.linspace(0, width, grid_width + 1)).astype(int)
    y_edges = np.rint(np.linspace(0, height, grid_height + 1)).astype(int)
    cv2.rectangle(
        image,
        (x_edges[column], y_edges[row]),
        (x_edges[column + 1] - 1, y_edges[row + 1] - 1),
        (0, 255, 0),
        2,
    )


def spatial_neighbourhood_mask(
    grid_height: int,
    grid_width: int,
    radius: int | None,
    device: torch.device,
) -> torch.Tensor | None:
    if radius is None:
        return None
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("SPATIAL_WINDOW_RADIUS must be a non-negative integer")
    patch_count = grid_height * grid_width
    indices = torch.arange(patch_count, device=device)
    rows = torch.div(indices, grid_width, rounding_mode="floor")
    columns = indices.remainder(grid_width)
    row_distance = (rows[:, None] - rows[None, :]).abs()
    column_distance = (columns[:, None] - columns[None, :]).abs()
    return torch.maximum(row_distance, column_distance).le(radius)


def compute_cross_frame_matching_distribution(
    patch_features: torch.Tensor,
    grid_height: int,
    grid_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return softmax probabilities, their mask, and Top-1 values/indices."""
    if patch_features.ndim != 3 or patch_features.shape[0] < 2:
        raise ValueError("patch_features must have shape [F >= 2, P, D]")
    normalized = F.normalize(patch_features.float(), dim=-1)
    similarity = torch.einsum("fpd,fqd->fpq", normalized[:-1], normalized[1:])
    spatial_mask = spatial_neighbourhood_mask(
        grid_height,
        grid_width,
        SPATIAL_WINDOW_RADIUS,
        similarity.device,
    )
    if spatial_mask is None:
        spatial_mask = torch.ones(
            similarity.shape[-2:], dtype=torch.bool, device=similarity.device
        )
    similarity = similarity.masked_fill(~spatial_mask, float("-inf"))
    matching_top_k = min(MATCHING_TOP_K, similarity.shape[-1])
    top_values, top_indices = similarity.topk(matching_top_k, dim=-1)
    top_weights = F.softmax(top_values / TEMPERATURE, dim=-1)
    probabilities = torch.zeros_like(similarity).scatter(-1, top_indices, top_weights)
    top1_values, top1_indices = probabilities.max(dim=-1)
    return probabilities, spatial_mask, top1_values, top1_indices


def probability_to_heatmap(
    values: np.ndarray,
    output_size: tuple[int, int],
    valid_mask: np.ndarray | None = None,
) -> np.ndarray:
    normalized = np.clip(
        (values - PROBABILITY_VMIN) / (PROBABILITY_VMAX - PROBABILITY_VMIN),
        0.0,
        1.0,
    )
    normalized = cv2.resize(
        normalized.astype(np.float32),
        output_size,
        interpolation=cv2.INTER_NEAREST,
    )
    heatmap = cv2.applyColorMap(
        np.rint(normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    if valid_mask is not None:
        pixel_mask = cv2.resize(
            valid_mask.astype(np.uint8),
            output_size,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        heatmap[~pixel_mask] = 0
    return heatmap


def create_video_writer(
    path: Path, frame_width: int, frame_height: int
) -> cv2.VideoWriter:
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
    patch_features: torch.Tensor,
    output_dir: Path,
) -> dict[str, object]:
    query_dir = output_dir / "query_patch_heatmaps"
    top1_dir = output_dir / "top1_probability_maps"
    displacement_dir = output_dir / "displacement_fields"
    expected_displacement_dir = output_dir / "expected_displacement_fields"
    for directory in (
        query_dir,
        top1_dir,
        displacement_dir,
        expected_displacement_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    frame_height, frame_width = frames.shape[-2:]
    grid_height, grid_width = infer_patch_grid(
        patch_features.shape[1], frame_height, frame_width
    )
    query_row, query_column = resolve_query_patch(grid_height, grid_width)
    query_index = query_row * grid_width + query_column
    probabilities, spatial_mask, top1_values, top1_indices = (
        compute_cross_frame_matching_distribution(
            patch_features, grid_height, grid_width
        )
    )
    probabilities = probabilities.cpu()
    spatial_mask = spatial_mask.cpu()
    top1_values = top1_values.cpu()
    top1_indices = top1_indices.cpu()

    query_writer = create_video_writer(
        output_dir / "query_patch_heatmaps.mp4", frame_width * 2, frame_height
    )
    top1_writer = create_video_writer(
        output_dir / "top1_probability_maps.mp4", frame_width, frame_height
    )
    displacement_writer = create_video_writer(
        output_dir / "displacement_fields.mp4", frame_width, frame_height
    )
    expected_displacement_writer = create_video_writer(
        output_dir / "expected_displacement_fields.mp4",
        frame_width,
        frame_height,
    )

    candidate_indices = torch.arange(grid_height * grid_width)
    candidate_rows = torch.div(
        candidate_indices, grid_width, rounding_mode="floor"
    ).float()
    candidate_columns = candidate_indices.remainder(grid_width).float()
    expected_rows = torch.einsum("fpq,q->fp", probabilities, candidate_rows)
    expected_columns = torch.einsum("fpq,q->fp", probabilities, candidate_columns)

    try:
        for frame_index in range(len(frames) - 1):
            current_bgr = cv2.cvtColor(
                to_rgb_uint8(frames[frame_index]), cv2.COLOR_RGB2BGR
            )
            next_bgr = cv2.cvtColor(
                to_rgb_uint8(frames[frame_index + 1]), cv2.COLOR_RGB2BGR
            )
            filename = f"{frame_index:05d}_to_{frame_index + 1:05d}.png"

            # View 1: one marked current-frame query and its next-frame map.
            query_current = current_bgr.copy()
            draw_query_patch(
                query_current,
                query_row,
                query_column,
                grid_height,
                grid_width,
            )
            query_values = probabilities[frame_index, query_index].numpy()
            query_valid = spatial_mask[query_index].numpy()
            query_heatmap = probability_to_heatmap(
                query_values.reshape(grid_height, grid_width),
                (frame_width, frame_height),
                query_valid.reshape(grid_height, grid_width),
            )
            query_next = cv2.addWeighted(
                next_bgr,
                1.0 - OVERLAY_ALPHA,
                query_heatmap,
                OVERLAY_ALPHA,
                0.0,
            )
            if DRAW_PATCH_GRID:
                draw_patch_grid(query_current, grid_height, grid_width)
                draw_patch_grid(query_next, grid_height, grid_width)
            query_canvas = np.concatenate((query_current, query_next), axis=1)
            cv2.imwrite(str(query_dir / filename), query_canvas)
            query_writer.write(query_canvas)

            # View 2: each current patch's largest next-frame match probability.
            top1_grid = (
                top1_values[frame_index].reshape(grid_height, grid_width).numpy()
            )
            top1_heatmap = probability_to_heatmap(
                top1_grid, (frame_width, frame_height)
            )
            top1_overlay = cv2.addWeighted(
                current_bgr,
                1.0 - OVERLAY_ALPHA,
                top1_heatmap,
                OVERLAY_ALPHA,
                0.0,
            )
            if DRAW_PATCH_GRID:
                draw_patch_grid(top1_overlay, grid_height, grid_width)
            cv2.imwrite(str(top1_dir / filename), top1_overlay)
            top1_writer.write(top1_overlay)

            # View 3: Top-1 displacement arrows, colored by match probability.
            displacement = current_bgr.copy()
            matches = top1_indices[frame_index].reshape(grid_height, grid_width).numpy()
            for row in range(0, grid_height, ARROW_STRIDE):
                for column in range(0, grid_width, ARROW_STRIDE):
                    probability = float(top1_grid[row, column])
                    if probability < ARROW_MIN_PROBABILITY:
                        continue
                    matched_index = int(matches[row, column])
                    matched_row, matched_column = divmod(matched_index, grid_width)
                    start = patch_center(
                        row,
                        column,
                        grid_height,
                        grid_width,
                        frame_height,
                        frame_width,
                    )
                    end = patch_center(
                        matched_row,
                        matched_column,
                        grid_height,
                        grid_width,
                        frame_height,
                        frame_width,
                    )
                    color_index = round(
                        255
                        * np.clip(
                            (probability - PROBABILITY_VMIN)
                            / (PROBABILITY_VMAX - PROBABILITY_VMIN),
                            0.0,
                            1.0,
                        )
                    )
                    color = cv2.applyColorMap(
                        np.array([[color_index]], dtype=np.uint8),
                        cv2.COLORMAP_TURBO,
                    )[0, 0]
                    color_tuple = tuple(int(channel) for channel in color)
                    if start == end:
                        cv2.circle(
                            displacement,
                            start,
                            STATIONARY_DOT_RADIUS,
                            color_tuple,
                            -1,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.arrowedLine(
                            displacement,
                            start,
                            end,
                            color_tuple,
                            ARROW_THICKNESS,
                            cv2.LINE_AA,
                            tipLength=ARROW_TIP_LENGTH,
                        )
            if DRAW_PATCH_GRID:
                draw_patch_grid(displacement, grid_height, grid_width)
            cv2.imwrite(str(displacement_dir / filename), displacement)
            displacement_writer.write(displacement)

            # View 4: probability-weighted expected next-patch coordinates.
            expected_displacement = current_bgr.copy()
            frame_expected_rows = expected_rows[frame_index].reshape(
                grid_height, grid_width
            )
            frame_expected_columns = expected_columns[frame_index].reshape(
                grid_height, grid_width
            )
            for row in range(0, grid_height, ARROW_STRIDE):
                for column in range(0, grid_width, ARROW_STRIDE):
                    probability = float(top1_grid[row, column])
                    if probability < ARROW_MIN_PROBABILITY:
                        continue
                    start = patch_center(
                        row,
                        column,
                        grid_height,
                        grid_width,
                        frame_height,
                        frame_width,
                    )
                    end = patch_center(
                        float(frame_expected_rows[row, column]),
                        float(frame_expected_columns[row, column]),
                        grid_height,
                        grid_width,
                        frame_height,
                        frame_width,
                    )
                    color_index = round(
                        255
                        * np.clip(
                            (probability - PROBABILITY_VMIN)
                            / (PROBABILITY_VMAX - PROBABILITY_VMIN),
                            0.0,
                            1.0,
                        )
                    )
                    color = cv2.applyColorMap(
                        np.array([[color_index]], dtype=np.uint8),
                        cv2.COLORMAP_TURBO,
                    )[0, 0]
                    color_tuple = tuple(int(channel) for channel in color)
                    if start == end:
                        cv2.circle(
                            expected_displacement,
                            start,
                            STATIONARY_DOT_RADIUS,
                            color_tuple,
                            -1,
                            cv2.LINE_AA,
                        )
                    else:
                        cv2.arrowedLine(
                            expected_displacement,
                            start,
                            end,
                            color_tuple,
                            ARROW_THICKNESS,
                            cv2.LINE_AA,
                            tipLength=ARROW_TIP_LENGTH,
                        )
            if DRAW_PATCH_GRID:
                draw_patch_grid(expected_displacement, grid_height, grid_width)
            cv2.imwrite(
                str(expected_displacement_dir / filename), expected_displacement
            )
            expected_displacement_writer.write(expected_displacement)
    finally:
        query_writer.release()
        top1_writer.release()
        displacement_writer.release()
        expected_displacement_writer.release()

    return {
        "patch_grid": [grid_height, grid_width],
        "query_patch": [query_row, query_column],
        "transition_count": len(frames) - 1,
        "temperature": TEMPERATURE,
        "top1_probability_min": float(top1_values.min().item()),
        "top1_probability_max": float(top1_values.max().item()),
        "top1_probability_mean": float(top1_values.mean().item()),
    }


def experiment_output_label() -> str:
    radius = (
        "global" if SPATIAL_WINDOW_RADIUS is None else f"radius_{SPATIAL_WINDOW_RADIUS}"
    )
    return f"index_{SAMPLE_INDEX}_{radius}_match_topk_{MATCHING_TOP_K}"


def main() -> None:
    if TEMPERATURE <= 0:
        raise ValueError("TEMPERATURE must be positive")
    if (
        isinstance(MATCHING_TOP_K, bool)
        or not isinstance(MATCHING_TOP_K, int)
        or MATCHING_TOP_K <= 0
    ):
        raise ValueError("MATCHING_TOP_K must be a positive integer")
    if PROBABILITY_VMAX <= PROBABILITY_VMIN:
        raise ValueError("PROBABILITY_VMAX must be greater than PROBABILITY_VMIN")
    if (
        isinstance(ARROW_STRIDE, bool)
        or not isinstance(ARROW_STRIDE, int)
        or ARROW_STRIDE < 1
    ):
        raise ValueError("ARROW_STRIDE must be a positive integer")

    args = parse_args()
    cfg = compose_config()
    datamodule = build_datamodule(cfg)
    frames, sample_name = load_one_video(datamodule)
    if len(frames) < 2:
        raise ValueError("cross-frame visualization requires at least two frames")

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
        )

    layer_output_dir = args.output_dir / experiment_output_label()
    layer_output_dir.mkdir(parents=True, exist_ok=True)
    statistics = render_outputs(
        frames,
        output.visual_features,
        layer_output_dir,
    )
    metadata = {
        "sample_name": str(sample_name),
        "split": SPLIT,
        "sample_index": SAMPLE_INDEX,
        "frame_count": len(frames),
        "visual_features_shape": list(output.visual_features.shape),
        "spatial_window_radius": SPATIAL_WINDOW_RADIUS,
        "matching_top_k": MATCHING_TOP_K,
        "probability_color_range": [PROBABILITY_VMIN, PROBABILITY_VMAX],
        "config_name": CONFIG_NAME,
        "config_overrides": CONFIG_OVERRIDES,
        **statistics,
    }
    (layer_output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Top-1 displacement: {layer_output_dir / 'displacement_fields.mp4'}")
    print(
        "Expected displacement: "
        f"{layer_output_dir / 'expected_displacement_fields.mp4'}"
    )


if __name__ == "__main__":
    main()
