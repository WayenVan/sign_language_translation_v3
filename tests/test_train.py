from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType
from torch import nn

import csi_slt.commands.train as train_command
from csi_slt.commands.train import cast_module_dtype, initialize_model
from csi_slt.engine.trainability import (
    LlmTrainabilityPlan,
    SltTrainabilityPlan,
    VisualAdapterTrainabilityPlan,
    VisualBackboneTrainabilityPlan,
    apply_trainability_plan,
)


class _ModelWithLoRA(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = nn.Module()
        self.llm.base = nn.Linear(4, 4)
        self.llm.lora_A = nn.Linear(4, 2, bias=False)
        self.llm.lora_B = nn.Linear(2, 4, bias=False)
        self.visual_backbone = nn.Module()
        self.visual_backbone.base = nn.Linear(4, 4)
        self.visual_backbone.lora_A = nn.Linear(4, 2, bias=False)
        self.visual_backbone.lora_B = nn.Linear(2, 4, bias=False)
        self.visual_adapter = nn.Linear(3, 4)
        self.ctc_head = nn.Linear(4, 8)
        self.visual_position_embedding = nn.Embedding(16, 4)
        self.start_video_embds = nn.Parameter(torch.zeros(1, 4))
        self.end_video_embeds = nn.Parameter(torch.zeros(1, 4))
        self.visual_scale = nn.Parameter(torch.ones(1))


def test_trainability_plan_independently_selects_llm_lora():
    model = _ModelWithLoRA()

    trainable_count = apply_trainability_plan(
        model,
        SltTrainabilityPlan(llm=LlmTrainabilityPlan(mode="lora")),
    )

    assert trainable_count == sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("llm.") and "lora_" in name
    )
    assert all(
        parameter.requires_grad == (name.startswith("llm.") and "lora_" in name)
        for name, parameter in model.named_parameters()
    )


def test_trainability_plan_rejects_missing_requested_visual_lora():
    model = _ModelWithLoRA()
    del model.visual_backbone.lora_A
    del model.visual_backbone.lora_B

    with pytest.raises(ValueError, match="visual LoRA training was requested"):
        apply_trainability_plan(
            model,
            SltTrainabilityPlan(
                visual_backbone=VisualBackboneTrainabilityPlan(mode="lora")
            ),
        )


def test_trainability_plan_can_select_only_visual_adapter():
    model = _ModelWithLoRA()

    trainable_count = apply_trainability_plan(
        model,
        SltTrainabilityPlan(visual_adapter=VisualAdapterTrainabilityPlan(mode="full")),
    )

    assert trainable_count == sum(
        parameter.numel() for parameter in model.visual_adapter.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.visual_adapter.parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("visual_adapter.")
    )


@pytest.mark.parametrize(
    ("runtime_mode", "expected_training"),
    [("eval", False), ("train", True)],
)
def test_slt_model_applies_llm_runtime_mode_independently_of_gradients(
    runtime_mode, expected_training
):
    model = object.__new__(train_command.SltModel)
    nn.Module.__init__(model)
    model.llm = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
    model.llm_runtime_mode = "eval"

    model.set_llm_runtime_mode(runtime_mode)
    model.train()

    assert model.training
    assert model.llm.training is expected_training

    model.requires_grad_(False)
    model.llm[-1].weight.requires_grad_(True)
    model.llm(torch.ones(1, 2)).sum().backward()
    assert model.llm[-1].weight.grad is not None

    model.eval()
    assert not model.llm.training


@pytest.mark.parametrize(
    ("runtime_mode", "expected_training"),
    [("eval", False), ("train", True)],
)
def test_slt_model_applies_visual_adapter_runtime_mode(runtime_mode, expected_training):
    model = object.__new__(train_command.SltModel)
    nn.Module.__init__(model)
    model.llm = nn.Linear(2, 2)
    model.llm_runtime_mode = "eval"
    model.visual_adapter = nn.Sequential(nn.Dropout(0.5), nn.Linear(2, 2))
    model.visual_adapter_runtime_mode = "eval"

    model.set_visual_adapter_runtime_mode(runtime_mode)
    model.train()

    assert model.training
    assert model.visual_adapter.training is expected_training

    model.visual_adapter.requires_grad_(False)
    model.visual_adapter[-1].weight.requires_grad_(True)
    model.visual_adapter(torch.ones(1, 2)).sum().backward()
    assert model.visual_adapter[-1].weight.grad is not None

    model.eval()
    assert not model.visual_adapter.training


def test_cast_module_dtype_casts_floating_parameters():
    module = nn.Linear(3, 4)
    cast_module_dtype(module, "bfloat16")
    assert all(parameter.dtype == torch.bfloat16 for parameter in module.parameters())


def test_cast_module_dtype_auto_preserves_checkpoint_dtype():
    module = nn.Linear(3, 4).to(dtype=torch.float64)
    cast_module_dtype(module, "auto")
    assert all(parameter.dtype == torch.float64 for parameter in module.parameters())


def test_cast_module_dtype_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        cast_module_dtype(nn.Linear(3, 4), "not_a_dtype")


def test_initialize_model_loads_checkpoint_and_uses_its_tokenizer(monkeypatch):
    loaded_model = object()
    monkeypatch.setattr(
        train_command.SltModel,
        "from_pretrained",
        lambda checkpoint_dir: loaded_model,
    )
    model_cfg = OmegaConf.create(
        {"load_from_checkpoint": True, "checkpoint_dir": "/tmp/checkpoint-10"}
    )

    model, tokenizer_source = initialize_model(
        model_cfg,
        llm_lora_config=None,
        visual_lora_config=None,
        llm_dtype="auto",
        visual_backbone_dtype="auto",
    )

    assert model is loaded_model
    assert tokenizer_source == "/tmp/checkpoint-10"


def test_initialize_model_loads_before_injecting_new_lora(monkeypatch):
    events = []

    class _LoadedModel:
        config = OmegaConf.create({"llm_lora": False})

        def inject_llm_lora(self, config):
            events.append(("llm", config))

        def inject_visual_lora(self, config):
            events.append(("visual", config))

    loaded_model = _LoadedModel()

    def load(checkpoint_dir):
        events.append(("load", checkpoint_dir))
        return loaded_model

    monkeypatch.setattr(train_command.SltModel, "from_pretrained", load)
    model_cfg = OmegaConf.create(
        {"load_from_checkpoint": True, "checkpoint_dir": "/tmp/checkpoint-10"}
    )
    llm_lora = object()
    visual_lora = object()

    model, _ = initialize_model(
        model_cfg,
        llm_lora_config=llm_lora,
        visual_lora_config=visual_lora,
        llm_dtype="auto",
        visual_backbone_dtype="auto",
    )

    assert model is loaded_model
    assert events == [
        ("load", "/tmp/checkpoint-10"),
        ("llm", llm_lora),
        ("visual", visual_lora),
    ]


def test_initialize_model_creates_components_then_injects_lora(monkeypatch):
    events = []

    class _FakeConfig:
        def __init__(self, **kwargs):
            self.llm_model_name_or_path = kwargs["llm_model_name_or_path"]

    class _FakeModel:
        config = OmegaConf.create({"llm_lora": False})

        def inject_llm_lora(self, config):
            events.append(("llm", config))

        def inject_visual_lora(self, config):
            events.append(("visual", config))

    created_model = _FakeModel()
    monkeypatch.setattr(train_command, "SltConfig", _FakeConfig)
    monkeypatch.setattr(
        train_command.SltModel,
        "from_pretrained_components",
        lambda **kwargs: created_model,
    )
    model_cfg = OmegaConf.create(
        {
            "load_from_checkpoint": False,
            "checkpoint_dir": None,
            "config": {"llm_model_name_or_path": "Qwen/test-model"},
        }
    )
    llm_lora = object()
    visual_lora = object()

    model, tokenizer_source = initialize_model(
        model_cfg,
        llm_lora_config=llm_lora,
        visual_lora_config=visual_lora,
        llm_dtype="bfloat16",
        visual_backbone_dtype="auto",
    )

    assert model is created_model
    assert tokenizer_source == "Qwen/test-model"
    assert events == [("llm", llm_lora), ("visual", visual_lora)]


def test_initialize_model_requires_checkpoint_dir_when_loading():
    model_cfg = OmegaConf.create({"load_from_checkpoint": True, "checkpoint_dir": None})

    with pytest.raises(ValueError, match="checkpoint_dir is required"):
        initialize_model(
            model_cfg,
            llm_lora_config=None,
            visual_lora_config=None,
            llm_dtype="auto",
            visual_backbone_dtype="auto",
        )


def test_initialize_model_does_not_reinject_matching_checkpoint_lora(monkeypatch):
    events = []
    requested = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
    )

    class _LoadedModel:
        config = SimpleNamespace(
            llm_lora=True,
            llm_lora_config={
                "task_type": "CAUSAL_LM",
                "r": 4,
                "lora_alpha": 8,
                "target_modules": ["v_proj", "q_proj"],
            },
        )

        def inject_llm_lora(self, config):
            events.append(("llm", config))

    loaded_model = _LoadedModel()
    monkeypatch.setattr(
        train_command.SltModel,
        "from_pretrained",
        lambda checkpoint_dir: loaded_model,
    )
    model_cfg = OmegaConf.create(
        {"load_from_checkpoint": True, "checkpoint_dir": "/tmp/checkpoint-10"}
    )

    model, _ = initialize_model(
        model_cfg,
        llm_lora_config=requested,
        visual_lora_config=None,
        llm_dtype="auto",
        visual_backbone_dtype="auto",
    )

    assert model is loaded_model
    assert events == []


def test_initialize_model_rejects_mismatched_checkpoint_lora(monkeypatch):
    requested_lora = LoraConfig(r=8, target_modules=["q_proj"])

    class _LoadedModel:
        config = SimpleNamespace(
            llm_lora=True,
            llm_lora_config={
                "r": 4,
                "target_modules": ["q_proj"],
            },
        )

    monkeypatch.setattr(
        train_command.SltModel,
        "from_pretrained",
        lambda checkpoint_dir: _LoadedModel(),
    )
    model_cfg = OmegaConf.create(
        {"load_from_checkpoint": True, "checkpoint_dir": "/tmp/checkpoint-10"}
    )

    with pytest.raises(ValueError, match="does not match"):
        initialize_model(
            model_cfg,
            llm_lora_config=requested_lora,
            visual_lora_config=None,
            llm_dtype="auto",
            visual_backbone_dtype="auto",
        )


class _ModelWithAlwaysFrozenComponent(nn.Module):
    """A plan-trainable adapter holding a nested module that must never train."""

    def __init__(self):
        super().__init__()
        self.llm = nn.Linear(4, 4)
        self.visual_backbone = nn.Linear(4, 4)
        self.visual_adapter = nn.Module()
        self.visual_adapter.projection = nn.Linear(4, 4)
        self.visual_adapter.constant = nn.Linear(4, 1)
        self.visual_adapter.constant.always_frozen = True


def test_always_frozen_modules_survive_a_plan_that_trains_their_parent():
    # requires_grad_ recurses, so "train the adapter" would otherwise reach the
    # fitted constants nested inside it -- silently, and on the default plan.
    model = _ModelWithAlwaysFrozenComponent()

    trainable_count = apply_trainability_plan(
        model,
        SltTrainabilityPlan(visual_adapter=VisualAdapterTrainabilityPlan(mode="full")),
    )

    assert all(
        not parameter.requires_grad
        for parameter in model.visual_adapter.constant.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.visual_adapter.projection.parameters()
    )
    # The frozen module must also be left out of the reported count.
    assert trainable_count == sum(
        parameter.numel() for parameter in model.visual_adapter.projection.parameters()
    )


def test_always_frozen_is_an_instance_marker_so_it_can_be_turned_off():
    model = _ModelWithAlwaysFrozenComponent()
    model.visual_adapter.constant.always_frozen = False

    apply_trainability_plan(
        model,
        SltTrainabilityPlan(visual_adapter=VisualAdapterTrainabilityPlan(mode="full")),
    )

    assert all(
        parameter.requires_grad
        for parameter in model.visual_adapter.constant.parameters()
    )
