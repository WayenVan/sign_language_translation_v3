"""Re-extract the separability cache's frames at a different backbone input size.

The eval transform centre-crops 224x224 out of the native 260x210 frame and does
not resize, so the cached features are near-native pixels at a 14x14 patch grid.
Raising the input size upsamples that crop: no new pixel detail, but a finer
patch grid, which is the variable this script isolates.

Frames must match ``features.npz`` exactly or the probes cannot be compared, so
the loader seed, video count and frame sampling are identical to ``extract.py``.
Verify with ``handshape_probe.py``'s frame check before trusting an A/B.

Usage:
    python experiments/diagnose/extract_resolution.py --resolution 512
"""

import argparse
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
sys.path.append(str(PROJECT_ROOT / "experiments/diagnose"))

from csi_slt.commands.prompt_setup import instantiate_prompt_resolvers  # noqa: E402
from csi_slt.data.datamodule import DataModule  # noqa: E402
from csi_slt.modeling_slt.visual_backbones.c_radio_v4_backbone import (  # noqa: E402
    CRadioV4Backbone,
)
from extract import attention_blocks  # noqa: E402

SPLIT = "val"


def build_loader(args):
    with hydra.initialize_config_dir(
        version_base=None, config_dir=str(PROJECT_ROOT / "configs")
    ):
        cfg = hydra.compose(
            config_name="train/base",
            overrides=[
                f"model={args.model_config}",
                "data.processor.video_processor.do_normalize=false",
                "data.processor.video_processor.do_resize=true",
                f"data.processor.video_processor.size.height={args.resolution}",
                f"data.processor.video_processor.size.width={args.resolution}",
                "+data.processor.video_processor.resample=3",
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
    return DataLoader(
        getattr(datamodule, f"{SPLIT}_dataset"),
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=getattr(datamodule, f"{SPLIT}_collator"),
        generator=torch.Generator().manual_seed(args.seed),
    )


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--model-config", default="qwen3-4b-cradio-l-spatiotemporal-pooled-linear"
    )
    parser.add_argument("--backbone-id", default="nvidia/C-RADIOv4-SO400M")
    # Only the layers the handshape probe reads: a full 27-layer cache at 1024
    # patches would be five times the size of the 224 one for no extra answer.
    parser.add_argument("--layers", type=int, nargs="+", default=[-6, -3, -1])
    parser.add_argument("--num-videos", type=int, default=40)
    parser.add_argument("--frames-per-video", type=int, default=6)
    parser.add_argument("--chunk", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = args.out or f"outputs/diagnose/features{args.resolution}.npz"

    loader = build_loader(args)
    backbone = (
        CRadioV4Backbone.from_pretrained_backbone(
            config={"id": args.backbone_id}, dtype=torch.bfloat16
        )
        .cuda()
        .eval()
    )
    radio = getattr(backbone.visual_encoder, "radio_model", backbone.visual_encoder)
    print(
        f"resolution {args.resolution}, {len(attention_blocks(radio))} blocks, "
        f"layers {args.layers}",
        flush=True,
    )

    rng = np.random.default_rng(args.seed)
    feature_list, frame_list, video_id_list = [], [], []
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
        frames = pixel_values[np.stack([picked - 1, picked], 1).reshape(-1)].cuda()

        per_chunk = []
        for start in range(0, len(frames), args.chunk):
            outputs = radio.forward_intermediates(
                frames[start : start + args.chunk].to(torch.bfloat16),
                indices=args.layers,
                return_prefix_tokens=True,
                norm=True,
                output_fmt="NLC",
                intermediates_only=True,
                aggregation="sparse",
            )
            per_chunk.append(
                torch.stack([out.features for out in outputs], dim=1).float().cpu()
            )
        stacked = torch.cat(per_chunk, dim=0)  # [2K, L, P, D]
        feature_list.append(stacked[1::2].to(torch.float16).numpy())
        frame_list.append(
            (frames[1::2].float().clamp(0, 1) * 255)
            .permute(0, 2, 3, 1)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        video_id_list.append(np.full(len(picked), video_id, dtype=np.int32))
        print(
            f"video {video_id}: {len(picked)} frames, P={stacked.shape[2]}", flush=True
        )

    features = np.concatenate(feature_list)
    np.savez(
        out,
        features=features,
        frames=np.concatenate(frame_list),
        video_ids=np.concatenate(video_id_list),
        layers=np.array(args.layers, dtype=np.int32),
    )
    print(f"wrote {out}: features {features.shape}")


if __name__ == "__main__":
    main()
