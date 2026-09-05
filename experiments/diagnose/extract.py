"""Cache per-layer C-RADIO patch features, CLS attention, motion and labels.

One pass over a sample of validation frames writes everything the separability
probe needs.  Frames are sampled as consecutive pairs (t-1, t) so the motion
scorer has a previous frame; only frame t is kept and labelled.

Output npz:
    features      [N, L, P, D]  float16  per-layer patch features at frame t
    cls_attention [N, L, P]     float32  head-mean CLS -> patch attention
    motion        [N, L, P]     float32  ||f_t - f_{t-1}|| per patch, per layer
    layers        [L]           int32    negative block indices, e.g. -27..-1
    frames        [N, H, W, 3]  uint8    RGB frames for the overlay panel
    labels        [N, P]        int8     1 = hand/face, 0 = background, -1 = ignore
    video_ids     [N]           int32    the probe splits groups on this
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from csi_slt.commands.config import instantiate_prompt_resolvers  # noqa: E402
from csi_slt.data.datamodule import DataModule  # noqa: E402
from csi_slt.modeling_slt.visual_backbones.c_radio_v4_backbone import (  # noqa: E402
    CRadioV4Backbone,
)

PATCH_SIZE = 16
SPLIT = "val"


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def build_loader(args):
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(PROJECT_ROOT / "configs")
    ):
        cfg = hydra.compose(
            config_name="train/base",
            overrides=[
                f"model={args.model_config}",
                # C-RADIO applies its own input conditioner and rejects
                # externally normalized frames.
                "data.processor.video_processor.do_normalize=false",
            ],
        )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.config.llm_model_name_or_path)
    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(cfg.prompt, (SPLIT,)),
    )
    datamodule.setup("fit")
    loader = DataLoader(
        getattr(datamodule, f"{SPLIT}_dataset"),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=getattr(datamodule, f"{SPLIT}_collator"),
        generator=torch.Generator().manual_seed(args.seed),
    )
    return cfg, loader


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
def attention_blocks(radio):
    return [
        module
        for module in radio.modules()
        if isinstance(getattr(module, "qkv", None), torch.nn.Module)
        and isinstance(getattr(module, "num_heads", None), int)
    ]


def capture_cls_attention(blocks, indices):
    """Hook every probed block so one forward yields all its CLS attention maps.

    Reuses the backbone's own reconstruction, so these maps are identical to what
    ``CRadioV4Backbone(return_attention_maps=True)`` returns at that layer.
    """
    captured = {}

    def make_hook(index):
        def hook(_module, args):
            captured[index] = args[0]

        return hook

    handles = [
        blocks[index].register_forward_pre_hook(make_hook(index)) for index in indices
    ]

    def collect(patch_count):
        maps = [
            CRadioV4Backbone._compute_cls_patch_attention(
                blocks[index], captured[index], patch_count=patch_count
            ).float().mean(dim=1)  # mean over heads -> [F, P]
            for index in indices
        ]
        captured.clear()
        return torch.stack(maps, dim=1)  # [F, L, P]

    return handles, collect


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def build_landmarkers(model_dir):
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    def options(name, cls, **kwargs):
        return cls(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(Path(model_dir) / name)
            ),
            running_mode=vision.RunningMode.IMAGE,
            **kwargs,
        )

    hand = vision.HandLandmarker.create_from_options(
        options(
            "hand_landmarker.task",
            vision.HandLandmarkerOptions,
            num_hands=2,
            min_hand_detection_confidence=0.3,
            min_hand_presence_confidence=0.3,
        )
    )
    face = vision.FaceLandmarker.create_from_options(
        options(
            "face_landmarker.task",
            vision.FaceLandmarkerOptions,
            num_faces=1,
            min_face_detection_confidence=0.3,
        )
    )
    return hand, face


def label_frames(frames, grid_h, grid_w, model_dir, detect_size):
    """Mark patches holding a hand or face landmark, with an ignore ring.

    Patches straddling the hand border are the dominant source of label noise,
    so the 8-neighbourhood of every positive patch is excluded from the probe
    rather than forced to a side.  Frames where nothing is detected are dropped
    whole (all -1) instead of being labelled all-background.
    """
    import mediapipe as mp

    hand, face = build_landmarkers(model_dir)
    labels = np.zeros((len(frames), grid_h, grid_w), dtype=np.int8)
    detected = 0
    for index, frame in enumerate(frames):
        # 224px is small for MediaPipe's detectors; upsampling raises recall a
        # lot and landmarks come back normalized, so the mapping is unchanged.
        enlarged = cv2.resize(
            frame, (detect_size, detect_size), interpolation=cv2.INTER_CUBIC
        )
        image = mp.Image(
            image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(enlarged)
        )
        landmark_sets = list(hand.detect(image).hand_landmarks)
        landmark_sets += list(face.detect(image).face_landmarks)

        core = np.zeros((grid_h, grid_w), dtype=bool)
        for landmark_set in landmark_sets:
            for landmark in landmark_set:
                row = int(np.clip(landmark.y, 0, 0.999) * grid_h)
                col = int(np.clip(landmark.x, 0, 0.999) * grid_w)
                core[row, col] = True
        if not core.any():
            labels[index] = -1
            continue
        detected += 1

        padded = np.pad(core, 1)
        ring = np.zeros_like(core)
        for row_shift in range(3):
            for col_shift in range(3):
                ring |= padded[
                    row_shift : row_shift + grid_h, col_shift : col_shift + grid_w
                ]
        frame_labels = np.zeros((grid_h, grid_w), dtype=np.int8)
        frame_labels[ring & ~core] = -1
        frame_labels[core] = 1
        labels[index] = frame_labels
        if (index + 1) % 50 == 0:
            print(f"  labelled {index + 1}/{len(frames)}", flush=True)

    print(f"MediaPipe found landmarks in {detected}/{len(frames)} frames")
    return labels.reshape(len(frames), -1)


# --------------------------------------------------------------------------- #
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/diagnose/features.npz")
    parser.add_argument(
        "--model-config", default="qwen3-4b-cradio-l-spatiotemporal-pooled-linear"
    )
    parser.add_argument("--backbone-id", default="nvidia/C-RADIOv4-SO400M")
    parser.add_argument(
        "--mediapipe-dir", default=str(PROJECT_ROOT / ".cache" / "mediapipe")
    )
    parser.add_argument("--detect-size", type=int, default=512)
    parser.add_argument("--num-videos", type=int, default=40)
    parser.add_argument("--frames-per-video", type=int, default=6)
    parser.add_argument("--layer-stride", type=int, default=1)
    parser.add_argument("--chunk", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    _, loader = build_loader(args)
    backbone = (
        CRadioV4Backbone.from_pretrained_backbone(
            config={"id": args.backbone_id}, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    radio = getattr(backbone.visual_encoder, "radio_model", backbone.visual_encoder)

    blocks = attention_blocks(radio)
    num_blocks = len(blocks)
    layers = list(range(-num_blocks, 0, args.layer_stride))
    block_indices = [num_blocks + layer for layer in layers]
    print(f"{num_blocks} attention blocks; probing {len(layers)} layers: {layers}")

    rng = np.random.default_rng(args.seed)
    frame_list, feature_list, video_id_list = [], [], []
    attention_list, motion_list = [], []
    for video_id, batch in enumerate(loader):
        if video_id >= args.num_videos:
            break
        length = int(batch["pixel_values_length"][0].item())
        pixel_values = batch["pixel_values"][:length]
        candidates = np.arange(1, len(pixel_values))
        if len(candidates) == 0:
            continue
        picked = np.sort(
            rng.choice(
                candidates,
                size=min(args.frames_per_video, len(candidates)),
                replace=False,
            )
        )
        pair_index = np.stack([picked - 1, picked], axis=1).reshape(-1)
        frames = pixel_values[pair_index].cuda()

        per_layer, per_layer_attention = [], []
        for start in range(0, len(frames), args.chunk):
            chunk = frames[start : start + args.chunk].to(torch.bfloat16)
            handles, collect = capture_cls_attention(blocks, block_indices)
            try:
                outputs = radio.forward_intermediates(
                    chunk,
                    indices=layers,
                    return_prefix_tokens=True,
                    norm=True,
                    output_fmt="NLC",
                    intermediates_only=True,
                    aggregation="sparse",
                )
                stacked = torch.stack([out.features for out in outputs], dim=1)
                per_layer_attention.append(collect(stacked.shape[2]).cpu())
            finally:
                for handle in handles:
                    handle.remove()
            per_layer.append(stacked.float().cpu())

        stacked = torch.cat(per_layer, dim=0)  # [2K, L, P, D]
        attention = torch.cat(per_layer_attention, dim=0)  # [2K, L, P]
        previous, current = stacked[0::2], stacked[1::2]

        feature_list.append(current.to(torch.float16).numpy())
        attention_list.append(attention[1::2].numpy())
        motion_list.append((current - previous).norm(dim=-1).numpy())
        frame_list.append(
            (frames[1::2].float().clamp(0, 1) * 255)
            .permute(0, 2, 3, 1)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        video_id_list.append(np.full(len(picked), video_id, dtype=np.int32))
        print(f"video {video_id}: {len(picked)} frame pairs", flush=True)

    features = np.concatenate(feature_list, axis=0)
    frames = np.concatenate(frame_list, axis=0)
    video_ids = np.concatenate(video_id_list, axis=0)
    cls_attention = np.concatenate(attention_list, axis=0)
    motion = np.concatenate(motion_list, axis=0)

    height, width = frames.shape[1:3]
    grid_h, grid_w = height // PATCH_SIZE, width // PATCH_SIZE
    if grid_h * grid_w != features.shape[2]:
        raise RuntimeError(
            f"{features.shape[2]} patches does not match a {grid_h}x{grid_w} grid "
            f"for {height}x{width} input at patch size {PATCH_SIZE}"
        )

    print(f"labelling {len(frames)} frames with MediaPipe...", flush=True)
    labels = label_frames(
        frames, grid_h, grid_w, args.mediapipe_dir, args.detect_size
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        features=features,
        cls_attention=cls_attention,
        motion=motion,
        layers=np.array(layers, dtype=np.int32),
        frames=frames,
        labels=labels,
        video_ids=video_ids,
    )
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    print(
        f"wrote {args.out}: features {features.shape}, "
        f"{positives} positive / {negatives} negative patches "
        f"({positives / max(positives + negatives, 1):.1%} positive rate)"
    )


if __name__ == "__main__":
    main()
