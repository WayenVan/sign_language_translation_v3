"""GPU integration test for a full, non-LoRA SLT checkpoint round trip.

This test is intentionally opt-in because it loads and rewrites a multi-GB
checkpoint and decodes one real PHOENIX-2014-T sample.  Run it on a GPU node:

    SLT_RUN_CHECKPOINT_ROUNDTRIP=1 \
    TMPDIR="$SLURM_TMPDIR" \
    PYTHONPATH=src \
    pytest -s tests/test_checkpoint_roundtrip_integration.py

``SLT_ROUNDTRIP_CHECKPOINT``, ``SLT_DATA_ROOT``, and
``SLT_ROUNDTRIP_SAMPLE_INDEX`` may be used to override the defaults below.
No LoRA adapter is created or loaded anywhere in this test.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from csi_slt.commands.config import instantiate_prompt_resolvers
from csi_slt.commands.train import cast_module_dtype
from csi_slt.data.datamodule import DataModule
from csi_slt.engine.sft.trainer import SltTrainer
from csi_slt.modeling_slt.slt import SltModel
from csi_slt.utils.generation_config import merge_generation_config


DEFAULT_CHECKPOINT = Path(
    "/mnt/scratch/users/2533494w/slt_outputs/"
    "v4.0-qwen3-1.7b-cradio-l-crossshuffle-lite-0828.224x224-ctc-de/"
    "checkpoint-42000"
)


def _assert_clean_load(loading_info: dict[str, Any], source: Path) -> None:
    problems = {
        name: loading_info.get(name)
        for name in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        )
        if loading_info.get(name)
    }
    assert not problems, f"Checkpoint load from {source} was not clean: {problems}"


def _assert_no_lora(model: SltModel) -> None:
    lora_names = [name for name, _ in model.named_parameters() if "lora_" in name]
    assert not lora_names, f"This must be a non-LoRA test, found: {lora_names[:10]}"


def _tensor_sha256(tensor: torch.Tensor) -> str:
    raw_bytes = (
        tensor.detach()
        .contiguous()
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    return hashlib.sha256(raw_bytes).hexdigest()


def _state_fingerprints(model: SltModel) -> dict[str, tuple[str, tuple[int, ...], str]]:
    """Fingerprint every persistent tensor without retaining a second model."""
    return {
        name: (str(tensor.dtype), tuple(tensor.shape), _tensor_sha256(tensor))
        for name, tensor in model.state_dict().items()
    }


def _semantic_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    # Library bookkeeping may change when save_pretrained is called; it is not
    # model or generation state.
    value.pop("transformers_version", None)
    value.pop("_name_or_path", None)
    return value


def _move_generation_inputs(batch: dict[str, Any], device: torch.device):
    generation_inputs = SltTrainer._build_generation_inputs(batch)
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in generation_inputs.items()
    }


@torch.no_grad()
def _generate_one(
    model: SltModel,
    batch: dict[str, Any],
    generation_config,
    device: torch.device,
) -> torch.Tensor:
    model.to(device)
    model.eval()
    generation_inputs = _move_generation_inputs(batch, device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        generated = model.generate(
            **generation_inputs,
            generation_config=generation_config,
            synced_gpus=False,
        )
    return generated.cpu()


def test_full_checkpoint_save_reload_is_bitwise_stable(tmp_path: Path) -> None:
    if os.environ.get("SLT_RUN_CHECKPOINT_ROUNDTRIP") != "1":
        pytest.skip("set SLT_RUN_CHECKPOINT_ROUNDTRIP=1 to run the multi-GB GPU test")
    if not torch.cuda.is_available():
        pytest.skip("the checkpoint round-trip test requires CUDA")

    checkpoint = Path(os.environ.get("SLT_ROUNDTRIP_CHECKPOINT", DEFAULT_CHECKPOINT))
    assert checkpoint.is_dir(), f"Checkpoint directory does not exist: {checkpoint}"
    hydra_config_path = checkpoint / "hydra_config.yaml"
    assert hydra_config_path.is_file(), f"Missing {hydra_config_path}"

    cfg = OmegaConf.load(hydra_config_path)
    if data_root := os.environ.get("SLT_DATA_ROOT"):
        cfg.data.data_root = data_root
    assert cfg.data.language == "de", (
        f"Expected the saved German data config, got {cfg.data.language!r}"
    )

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    datamodule = DataModule(
        cfg.data,
        cfg.datamodule,
        tokenizer=tokenizer,
        prompt_resolvers=instantiate_prompt_resolvers(cfg.prompt, ("test",)),
    )
    datamodule.setup("test")
    sample_index = int(os.environ.get("SLT_ROUNDTRIP_SAMPLE_INDEX", "0"))
    assert 0 <= sample_index < len(datamodule.test_dataset)
    batch = datamodule.test_collator([datamodule.test_dataset[sample_index]])

    model, loading_info = SltModel.from_pretrained(
        checkpoint, output_loading_info=True
    )
    _assert_clean_load(loading_info, checkpoint)
    _assert_no_lora(model)
    cast_module_dtype(model.llm, cfg.engine.llm_dtype)
    cast_module_dtype(model.visual_backbone, cfg.engine.visual_backbone_dtype)
    generation_config = merge_generation_config(
        model.generation_config,
        OmegaConf.to_container(cfg.engine.generation_config, resolve=True),
    )

    print(f"checkpoint={checkpoint}")
    print(f"sample_index={sample_index}")
    print(f"sample_names={batch.get('sample_names')}")
    print(f"generation_config={generation_config.to_dict()}")

    original_fingerprints = _state_fingerprints(model)
    original_tokens = _generate_one(
        model, batch, generation_config, torch.device("cuda:0")
    )

    roundtrip_dir = tmp_path / "checkpoint-roundtrip"
    model.save_pretrained(roundtrip_dir, safe_serialization=True)
    tokenizer.save_pretrained(roundtrip_dir)
    OmegaConf.save(cfg, roundtrip_dir / "hydra_config.yaml")

    # Release the first model before loading the second one, so the test also
    # works on GPUs with substantially less memory than an H100.
    model.to("cpu")
    del model
    torch.cuda.empty_cache()

    reloaded, reloaded_info = SltModel.from_pretrained(
        roundtrip_dir, output_loading_info=True
    )
    _assert_clean_load(reloaded_info, roundtrip_dir)
    _assert_no_lora(reloaded)
    cast_module_dtype(reloaded.llm, cfg.engine.llm_dtype)
    cast_module_dtype(reloaded.visual_backbone, cfg.engine.visual_backbone_dtype)

    reloaded_fingerprints = _state_fingerprints(reloaded)
    differing_tensors = [
        name
        for name in original_fingerprints.keys() | reloaded_fingerprints.keys()
        if original_fingerprints.get(name) != reloaded_fingerprints.get(name)
    ]
    assert not differing_tensors, (
        "Persistent tensors changed during save/reload; first differences: "
        f"{sorted(differing_tensors)[:20]}"
    )

    assert _semantic_json(checkpoint / "config.json") == _semantic_json(
        roundtrip_dir / "config.json"
    ), "Model config changed during save/reload"
    assert _semantic_json(checkpoint / "generation_config.json") == _semantic_json(
        roundtrip_dir / "generation_config.json"
    ), "Generation config changed during save/reload"

    reloaded_generation_config = merge_generation_config(
        reloaded.generation_config,
        OmegaConf.to_container(cfg.engine.generation_config, resolve=True),
    )
    reloaded_tokens = _generate_one(
        reloaded, batch, reloaded_generation_config, torch.device("cuda:0")
    )

    if not torch.equal(original_tokens, reloaded_tokens):
        common_length = min(original_tokens.shape[1], reloaded_tokens.shape[1])
        positions = torch.nonzero(
            original_tokens[:, :common_length] != reloaded_tokens[:, :common_length]
        )
        first_position = positions[0].tolist() if positions.numel() else None
        pytest.fail(
            "Generated token IDs changed after save/reload: "
            f"original_shape={tuple(original_tokens.shape)}, "
            f"reloaded_shape={tuple(reloaded_tokens.shape)}, "
            f"first_differing_[batch,token]={first_position}, "
            f"original_text={tokenizer.batch_decode(original_tokens)!r}, "
            f"reloaded_text={tokenizer.batch_decode(reloaded_tokens)!r}"
        )

    print(f"state_tensors_checked={len(original_fingerprints)}")
    print(f"generated_shape={tuple(original_tokens.shape)}")
    print(f"decoded={tokenizer.batch_decode(original_tokens)!r}")
    print("PASS: full non-LoRA checkpoint is bitwise stable across save/reload")
