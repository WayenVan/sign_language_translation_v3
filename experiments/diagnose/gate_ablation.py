"""Ask which component gives the model room to memorize the training set.

The frozen-base runs overfit unevenly: at step 30k the pooled-linear baseline's
train-dev BLEU-4 gap was +0.004, the hand-ROI concat run's +0.031 and the
next-frame fusion run's +0.043 -- while all three carry the same 20M adapter
parameters, and the baseline's projection is in fact the *largest* of them
(rank 5385 against 4429).  So the gap does not track capacity; it tracks the
extra input pathway each adapter adds.

That suggests a division of labour worth testing rather than assuming:

    the fusion makes videos distinguishable -- frame-to-frame motion is a
    high-entropy, video-specific signal -- and the projection turns that
    distinguishability into sentences.

Per-component weight norms are consistent with it (over 6k steps the fusion's
encoders moved +0.3-0.6% while the projection moved +3.3%), but weight norms
are a weak proxy: a component can memorize by rotating directions without
growing.

This script tests the claim directly and without training anything.  It takes a
trained checkpoint and evaluates teacher-forced loss on train and dev with the
fusion gate as trained, then again with the gate forced shut (sigmoid ~ 0,
which recovers the pooled-linear baseline exactly), and reports how much each
split relies on the branch:

    train rises much more than dev  ->  the branch's contribution is spent
                                        disproportionately on fitting train,
                                        i.e. it is what enables memorization
    both rise equally               ->  the branch carries real signal and the
                                        overfitting comes from somewhere else

Closing the gate answers *whether* the branch is spent on fitting train, not
*what part of it* is.  The fusion sums three streams into its update, and two of
them are large: content, a second learnable transform of the same ``x`` the
residual path already carries, and delta, the genuinely new temporal signal
(69% and 74% of the fusion hidden vector respectively, measured).  Closing the
gate removes both at once.  ``--streams`` therefore also zeroes them one at a
time, which is what separates "the motion signal is a per-video fingerprint"
from "the extra transform of static appearance is".

Teacher-forced loss rather than BLEU on purpose: it needs no generation, so
every condition runs in minutes, and it measures fit directly instead of
through a decoding step that adds its own variance.

Usage::

    python experiments/diagnose/gate_ablation.py \
        --checkpoint outputs/v5.0-...-nextframe20m-.../checkpoint-42000

    # fewer batches for a quick look, or a different gate attribute
    python experiments/diagnose/gate_ablation.py --checkpoint ... --batches 50
"""

import argparse
import json
import math
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from csi_slt.commands.config import instantiate_prompt_resolvers  # noqa: E402
from csi_slt.data.datamodule import DataModule  # noqa: E402
from csi_slt.modeling_slt.slt import SltModel  # noqa: E402

# Large enough that sigmoid underflows to zero in float32, so "off" is exactly
# off rather than nearly off.
GATE_OFF = -40.0


def find_gates(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    """Every scalar gate in the visual adapter, by qualified name.

    Found by name rather than hard-coded: the adapters spell theirs
    ``fusion_gate``, ``gate`` and ``fusion_gate`` again, and a script that knows
    only one of them would silently ablate nothing on the others.
    """
    adapter = getattr(model, "visual_adapter", None)
    if adapter is None:
        raise RuntimeError("model has no visual_adapter")
    return {
        name: parameter
        for name, parameter in adapter.named_parameters()
        if name.split(".")[-1] in ("gate", "fusion_gate", "motion_gate")
        and parameter.numel() == 1
    }


def stream_silencers(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    """The fusion's individually silenceable input streams, by short name.

    Returned as modules so a forward hook can zero one stream's output while
    the rest of the branch, the gate included, runs untouched. Zeroing the
    output rather than the input keeps each stream's bias out of the sum too,
    so "off" means the stream contributes nothing at all.
    """
    found = {}
    for name, module in model.named_modules():
        short = name.split(".")[-1]
        if short in ("content_encoder", "delta_encoder", "displacement_projection"):
            found[short.replace("_encoder", "").replace("_projection", "")] = module
    return found


def build_loaders(hydra_config, tokenizer, batch_size: int, workers: int):
    """Train and dev loaders from the checkpoint's own data configuration.

    Read from the checkpoint rather than composed afresh so the pipeline is
    exactly the one the weights were fitted under.
    """
    datamodule = DataModule(
        hydra_config.data,
        hydra_config.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(
            hydra_config.prompt, ("train", "val")
        ),
    )
    datamodule.setup("fit")
    return {
        "train": DataLoader(
            datamodule.train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=datamodule.train_collator,
        ),
        "dev": DataLoader(
            datamodule.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=workers,
            collate_fn=datamodule.val_collator,
        ),
    }


@torch.no_grad()
def mean_losses(model, loader, batches: int, device: str, dtype, desc: str) -> dict:
    """Mean CE (and CTC) over a fixed prefix of the loader.

    Under autocast because training runs under it: ``accelerate launch
    --mixed_precision=bf16`` wraps the step, and ``_keep_in_fp32_modules``
    holds the adapter in fp32 while the backbone emits bf16. Without autocast
    that mismatch raises, and forcing either side to match would evaluate a
    numerically different model than the one that was trained.
    """
    totals, seen = {}, 0
    for index, batch in enumerate(tqdm(loader, total=batches, desc=desc, leave=False)):
        if index >= batches:
            break
        batch = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
        }
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype):
            output = model(**batch)
        scalars = {"loss": output.loss}
        scalars.update(output.logging_scalars or {})
        for name, value in scalars.items():
            if torch.is_tensor(value) and value.numel() == 1:
                totals[name] = totals.get(name, 0.0) + float(value)
        seen += 1
    if seen == 0:
        raise RuntimeError(f"{desc}: loader was empty")
    return {name: total / seen for name, total in totals.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument(
        "--gate-values",
        type=float,
        nargs="+",
        default=[0.5, 0.0],
        help=(
            "sigmoid values to force the gate to, beside the trained one. "
            "Fully off saturates both splits near the unconditional language "
            "model's loss, which compresses the very difference being measured, "
            "so a partial setting is usually the more sensitive probe."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument(
        "--streams",
        nargs="*",
        default=["content", "delta"],
        help=(
            "fusion streams to silence one at a time, beside the gate sweep. "
            "Empty disables this part."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype", default="bfloat16", help="autocast dtype, matching training"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    hydra_config = OmegaConf.load(args.checkpoint / "hydra_config.yaml")
    # "auto" keeps each component at its checkpoint dtype, the way evaluate.py
    # loads; the autocast in mean_losses supplies the mixed precision instead.
    model = SltModel.from_pretrained(args.checkpoint, dtype="auto")
    model = model.to(args.device).eval()
    autocast_dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model.config.llm_model_name_or_path, config=model.config.llm_config
    )
    loaders = build_loaders(hydra_config, tokenizer, args.batch_size, args.num_workers)

    gates = find_gates(model)
    if not gates:
        raise RuntimeError(
            "this adapter has no scalar gate, so there is nothing to ablate; "
            f"adapter type is {model.config.visual_adapter_type}"
        )
    print(f"checkpoint: {args.checkpoint}")
    print(f"adapter   : {model.config.visual_adapter_type}")
    for name, gate in gates.items():
        value = float(gate.detach())
        print(
            f"  gate {name}: {value:+.4f} -> sigmoid {1 / (1 + math.exp(-value)):.4f}"
        )
    print(f"batches   : {args.batches} per split at batch size {args.batch_size}\n")

    silencers = stream_silencers(model)
    missing = [name for name in args.streams if name not in silencers]
    if missing:
        raise RuntimeError(
            f"no such fusion stream(s): {missing}; this adapter exposes "
            f"{sorted(silencers)}"
        )
    if args.streams:
        print(f"streams   : {', '.join(args.streams)} (silenced one at a time)\n")

    original = {name: gate.detach().clone() for name, gate in gates.items()}
    settings = (
        ["as-trained"]
        + [f"sigmoid={value:g}" for value in args.gate_values]
        + [f"no-{name}" for name in args.streams]
    )
    results = {}
    for setting in settings:
        handle = None
        with torch.no_grad():
            for name, gate in gates.items():
                gate.copy_(original[name])
        if setting.startswith("no-"):
            # The gate stays as trained: this isolates one stream's
            # contribution rather than the branch's.
            handle = silencers[setting[3:]].register_forward_hook(
                lambda module, inputs, output: torch.zeros_like(output)
            )
        with torch.no_grad():
            for name, gate in gates.items():
                if setting == "as-trained" or setting.startswith("no-"):
                    continue
                target = float(setting.split("=")[1])
                logit = (
                    GATE_OFF
                    if target <= 0.0
                    else math.log(target / (1.0 - target))
                    if target < 1.0
                    else -GATE_OFF
                )
                gate.copy_(torch.full_like(gate, logit))
        results[setting] = {
            split: mean_losses(
                model,
                loader,
                args.batches,
                args.device,
                autocast_dtype,
                f"{setting}/{split}",
            )
            for split, loader in loaders.items()
        }
        if handle is not None:
            handle.remove()
    with torch.no_grad():
        for name, gate in gates.items():
            gate.copy_(original[name])

    ablated = settings[1:]
    header = f"{'metric':<12}{'split':<7}{'as-trained':>12}"
    header += "".join(f"{name:>14}" for name in ablated)
    print(header)
    print("-" * len(header))
    rises = {}
    for metric in ("ce_loss", "ctc_loss"):
        for split in ("train", "dev"):
            base = results["as-trained"][split].get(metric)
            if base is None:
                continue
            row = f"{metric:<12}{split:<7}{base:>12.4f}"
            for name in ablated:
                value = results[name][split].get(metric)
                rises[(metric, split, name)] = value - base
                row += f"{value:>9.4f}{value - base:>+5.2f}"
            print(row)

    print(
        f"\n{'metric':<12}{'setting':<14}{'train rise':>12}{'dev rise':>10}{'train-dev':>11}"
    )
    for metric in ("ce_loss",):
        for name in ablated:
            train_rise = rises.get((metric, "train", name))
            dev_rise = rises.get((metric, "dev", name))
            if train_rise is None or dev_rise is None:
                continue
            print(
                f"{metric:<12}{name:<14}{train_rise:>+12.4f}{dev_rise:>+10.4f}"
                f"{train_rise - dev_rise:>+11.4f}"
            )
    print(
        "\nA clearly larger rise on train means the gated branch is spent "
        "disproportionately on fitting the training set.\nComparable rises mean "
        "it carries real signal and the overfitting comes from elsewhere."
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "adapter": model.config.visual_adapter_type,
                    "gates": {n: float(g) for n, g in original.items()},
                    "batches": args.batches,
                    "results": results,
                },
                indent=2,
            )
            + "\n"
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
