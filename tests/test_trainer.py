from types import SimpleNamespace

import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from transformers import GenerationConfig, Seq2SeqTrainingArguments

from csi_slt.engine.sft.callbacks import (
    EvalInformationVisualizationCallback,
    TrainSubsetMetricsCallback,
)
from csi_slt.engine.sft.trainer import SltTrainer
from csi_slt.engine.sft.training_args import SltTrainingArguments


class _MeanReducedModelWithKwargs(nn.Module):
    """Minimal model reproducing Transformers' ``**kwargs`` inference."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace()
        self.last_kwargs = None

    def forward(self, labels=None, **kwargs):
        self.last_kwargs = kwargs
        return {"loss": self.weight * 0 + 8.0}


class _AdapterWithGate(nn.Linear):
    def __init__(self):
        super().__init__(2, 2)
        self.gate = nn.Parameter(torch.zeros(1))

    def optimization_parameter_groups(self):
        return {"gates": (self.gate,)}


class _ComponentLearningRateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.llm = nn.Module()
        self.llm.lora_A = nn.Linear(2, 2)
        self.visual_backbone = nn.Module()
        self.visual_backbone.lora_A = nn.Linear(2, 2)
        self.visual_adapter = _AdapterWithGate()
        self.ctc_head = nn.Linear(2, 3)
        self.ctc_codebook = nn.Embedding(3, 2)
        self.visual_position_embedding = nn.Embedding(4, 2)
        self.start_video_embds = nn.Parameter(torch.zeros(1, 2))
        self.end_video_embeds = nn.Parameter(torch.zeros(1, 2))
        self.config = SimpleNamespace()

    def forward(self, **kwargs):
        return {"loss": sum(parameter.sum() for parameter in self.parameters())}


def test_optimizer_uses_separate_component_learning_rates(tmp_path):
    model = _ComponentLearningRateModel()
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        auto_output_dir=False,
        report_to="none",
        learning_rate=1e-4,
        weight_decay=0.1,
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {
                "engine": {
                    "optimization": {
                        "llm": {"learning_rate": 2e-5, "weight_decay": 0.01},
                        "visual_backbone": {
                            "learning_rate": 3e-5,
                            "weight_decay": 0.02,
                        },
                        "visual_adapter": {
                            "learning_rate": 4e-5,
                            "weight_decay": 0.03,
                        },
                    }
                }
            }
        ),
    )

    optimizer = trainer.create_optimizer()
    parameter_lrs = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    parameter_weight_decays = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert {
        parameter_lrs[id(parameter)] for parameter in model.llm.lora_A.parameters()
    } == {2e-5}
    assert {
        parameter_lrs[id(parameter)]
        for parameter in model.visual_backbone.lora_A.parameters()
    } == {3e-5}
    assert {
        parameter_lrs[id(parameter)] for parameter in model.visual_adapter.parameters()
    } == {4e-5}
    assert parameter_weight_decays[id(model.llm.lora_A.weight)] == 0.01
    assert parameter_weight_decays[id(model.visual_backbone.lora_A.weight)] == 0.02
    assert parameter_weight_decays[id(model.visual_adapter.weight)] == 0.03
    assert parameter_weight_decays[id(model.llm.lora_A.bias)] == 0.0
    assert parameter_weight_decays[id(model.visual_backbone.lora_A.bias)] == 0.0
    assert parameter_weight_decays[id(model.visual_adapter.bias)] == 0.0


def test_component_learning_rate_defaults_to_global_rate(tmp_path):
    model = _ComponentLearningRateModel()
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        auto_output_dir=False,
        report_to="none",
        learning_rate=1e-4,
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {"engine": {"optimization": {"visual_adapter": {"learning_rate": 4e-5}}}}
        ),
    )

    optimizer = trainer.create_optimizer()
    parameter_lrs = {
        id(parameter): group["lr"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert parameter_lrs[id(model.llm.lora_A.weight)] == 1e-4
    assert parameter_lrs[id(model.visual_backbone.lora_A.weight)] == 1e-4
    assert parameter_lrs[id(model.visual_adapter.weight)] == 4e-5


def test_semantic_group_overrides_component_and_global_defaults(tmp_path):
    model = _ComponentLearningRateModel()
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        auto_output_dir=False,
        report_to="none",
        learning_rate=1e-4,
        weight_decay=0.1,
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {
                "engine": {
                    "optimization": {
                        "visual_adapter": {
                            "learning_rate": 4e-5,
                            "weight_decay": 0.03,
                            "parameter_groups": {"gates": {"learning_rate": 5e-4}},
                        }
                    }
                }
            }
        ),
    )

    optimizer = trainer.create_optimizer()
    parameter_groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert parameter_groups[id(model.visual_adapter.gate)]["lr"] == 5e-4
    assert parameter_groups[id(model.visual_adapter.gate)]["weight_decay"] == 0.0
    assert (
        parameter_groups[id(model.visual_adapter.gate)]["slt_parameter_group"]
        == "gates"
    )
    assert parameter_groups[id(model.visual_adapter.weight)]["lr"] == 4e-5


def test_optimizer_rejects_unregistered_semantic_group(tmp_path):
    model = _ComponentLearningRateModel()
    args = SltTrainingArguments(
        output_dir=str(tmp_path), auto_output_dir=False, report_to="none"
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {
                "engine": {
                    "optimization": {
                        "visual_adapter": {
                            "parameter_groups": {"missing": {"learning_rate": 5e-4}}
                        }
                    }
                }
            }
        ),
    )

    with pytest.raises(ValueError, match="unregistered groups: missing"):
        trainer.create_optimizer()


@pytest.mark.parametrize(
    "invalid_registration,match",
    [
        ("outside", "outside the component"),
        ("overlap", "overlap"),
        ("frozen", "is frozen"),
    ],
)
def test_optimizer_rejects_invalid_semantic_group_registration(
    tmp_path, invalid_registration, match
):
    model = _ComponentLearningRateModel()
    if invalid_registration == "outside":
        model.visual_adapter.optimization_parameter_groups = lambda: {
            "gates": (model.ctc_head.weight,)
        }
    elif invalid_registration == "overlap":
        model.visual_adapter.optimization_parameter_groups = lambda: {
            "gates": (model.visual_adapter.gate,),
            "duplicate": (model.visual_adapter.gate,),
        }
    else:
        model.visual_adapter.gate.requires_grad_(False)

    groups = {"gates": {"learning_rate": 5e-4}}
    if invalid_registration == "overlap":
        groups["duplicate"] = {"learning_rate": 5e-4}
    args = SltTrainingArguments(
        output_dir=str(tmp_path), auto_output_dir=False, report_to="none"
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {
                "engine": {
                    "optimization": {"visual_adapter": {"parameter_groups": groups}}
                }
            }
        ),
    )

    with pytest.raises(ValueError, match=match):
        trainer.create_optimizer()


def test_component_weight_decay_defaults_to_global_value(tmp_path):
    model = _ComponentLearningRateModel()
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        auto_output_dir=False,
        report_to="none",
        weight_decay=0.1,
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {"engine": {"optimization": {"visual_adapter": {"weight_decay": 0.0}}}}
        ),
    )

    optimizer = trainer.create_optimizer()
    parameter_weight_decays = {
        id(parameter): group["weight_decay"]
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert parameter_weight_decays[id(model.llm.lora_A.weight)] == 0.1
    assert parameter_weight_decays[id(model.visual_backbone.lora_A.weight)] == 0.1
    assert parameter_weight_decays[id(model.visual_adapter.weight)] == 0.0


def test_ctc_components_support_independent_optimizer_overrides(tmp_path):
    model = _ComponentLearningRateModel()
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        auto_output_dir=False,
        report_to="none",
        learning_rate=1e-4,
        weight_decay=0.1,
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {
                "engine": {
                    "optimization": {
                        "ctc_head": {"learning_rate": 3e-4},
                        "ctc_codebook": {
                            "learning_rate": 2e-5,
                            "weight_decay": 0.0,
                        },
                    }
                }
            }
        ),
    )

    optimizer = trainer.create_optimizer()
    parameter_groups = {
        id(parameter): group
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert parameter_groups[id(model.ctc_head.weight)]["lr"] == 3e-4
    assert parameter_groups[id(model.ctc_head.weight)]["weight_decay"] == 0.1
    assert parameter_groups[id(model.ctc_codebook.weight)]["lr"] == 2e-5
    assert parameter_groups[id(model.ctc_codebook.weight)]["weight_decay"] == 0.0


def test_optimizer_rejects_override_for_frozen_component(tmp_path):
    model = _ComponentLearningRateModel()
    model.ctc_codebook.requires_grad_(False)
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        auto_output_dir=False,
        report_to="none",
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {"engine": {"optimization": {"ctc_codebook": {"learning_rate": 2e-5}}}}
        ),
    )

    with pytest.raises(ValueError, match="frozen or absent components: ctc_codebook"):
        trainer.create_optimizer()


def test_training_log_includes_each_trainable_component_learning_rate(tmp_path):
    model = _ComponentLearningRateModel()
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        auto_output_dir=False,
        report_to="none",
        learning_rate=1e-4,
    )
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create(
            {
                "engine": {
                    "optimization": {
                        "llm": {"learning_rate": 2e-5},
                        "visual_backbone": {"learning_rate": 3e-5},
                        "visual_adapter": {"learning_rate": 4e-5},
                    }
                }
            }
        ),
    )
    trainer.create_optimizer()

    trainer.log({"loss": 1.0, "learning_rate": 2e-5})

    logged = trainer.state.log_history[-1]
    assert logged["learning_rate/llm"] == 2e-5
    assert logged["learning_rate/visual_backbone"] == 3e-5
    assert logged["learning_rate/visual_adapter"] == 4e-5


def test_trainer_does_not_claim_model_handles_num_items_in_batch(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
    )

    model = _MeanReducedModelWithKwargs()
    trainer = SltTrainer(model=model, args=args)

    assert trainer.args.average_tokens_across_devices is True
    assert trainer.model_accepts_loss_kwargs is False

    loss = trainer.compute_loss(
        model,
        {"labels": torch.tensor([[1]])},
        num_items_in_batch=torch.tensor(1),
    )

    assert loss.item() == 8.0
    assert "num_items_in_batch" not in model.last_kwargs


def test_train_probe_test_dataloader_disables_persistent_workers(tmp_path, monkeypatch):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
        dataloader_persistent_workers=True,
        dataloader_num_workers=1,
    )
    trainer = SltTrainer(model=_MeanReducedModelWithKwargs(), args=args)
    captured = {}

    def capture_get_dataloader(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(trainer, "_get_dataloader", capture_get_dataloader)
    trainer._is_train_probe = True

    trainer.get_test_dataloader([{"labels": torch.tensor([1])}])

    assert captured["persistent_workers"] is False
    assert trainer.args.dataloader_persistent_workers is True


def test_trainer_builds_eval_information_callback_from_engine_config(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
    )
    hydra_config = OmegaConf.create(
        {
            "engine": {
                "eval_information": {
                    "every_n_evaluations": 3,
                    "num_samples": 2,
                }
            }
        }
    )
    trainer = SltTrainer(
        model=_MeanReducedModelWithKwargs(),
        args=args,
        hydra_config=hydra_config,
    )

    callback = next(
        callback
        for callback in trainer.callback_handler.callbacks
        if isinstance(callback, EvalInformationVisualizationCallback)
    )
    assert callback.every_n_evaluations == 3
    assert callback.num_samples == 2


def test_trainer_builds_train_probe_callback_from_engine_config(tmp_path):
    args = Seq2SeqTrainingArguments(output_dir=str(tmp_path), report_to="none")
    hydra_config = OmegaConf.create(
        {
            "engine": {
                "train_probe": {
                    "every_n_evaluations": 3,
                    "num_samples": 17,
                    "seed": 9,
                    "metric_key_prefix": "train_sample",
                }
            }
        }
    )
    trainer = SltTrainer(
        model=_MeanReducedModelWithKwargs(), args=args, hydra_config=hydra_config
    )

    callback = next(
        item
        for item in trainer.callback_handler.callbacks
        if isinstance(item, TrainSubsetMetricsCallback)
    )
    assert callback.every_n_evaluations == 3
    assert callback.num_samples == 17
    assert callback.seed == 9
    assert callback.metric_key_prefix == "train_sample"


def test_evaluate_train_subset_is_deterministic_and_restores_state(
    tmp_path, monkeypatch
):
    args = Seq2SeqTrainingArguments(output_dir=str(tmp_path), report_to="none")
    normal_metric = object()
    probe_metric = object()
    train_collator = object()
    eval_collator = object()
    test_collator = object()
    trainer = SltTrainer(
        model=_MeanReducedModelWithKwargs(),
        args=args,
        train_dataset=list(range(20)),
        compute_metrics=normal_metric,
        train_probe_compute_metrics=probe_metric,
        train_data_collator=train_collator,
        eval_data_collator=eval_collator,
        test_data_collator=test_collator,
    )
    observed = {}

    def fake_predict(test_dataset, **kwargs):
        observed["samples"] = [
            test_dataset[index] for index in range(len(test_dataset))
        ]
        observed["collator"] = trainer.test_data_collator
        observed["metric"] = trainer.compute_metrics
        observed["prefix"] = kwargs["metric_key_prefix"]
        trainer.model.eval()
        return SimpleNamespace(metrics={"train_probe_bleu": 1.5})

    monkeypatch.setattr(trainer, "predict", fake_predict)
    metrics = trainer.evaluate_train_subset(num_samples=5, seed=7)

    assert observed["samples"] == [10, 4, 12, 1, 2]
    assert observed["collator"] is eval_collator
    assert observed["metric"] is probe_metric
    assert observed["prefix"] == "train_probe"
    assert trainer.test_data_collator is test_collator
    assert trainer.compute_metrics is normal_metric
    assert trainer.model.training is True
    assert metrics == {"train_probe_bleu": 1.5}


def test_compute_loss_does_not_forward_generation_fields(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
    )
    model = _MeanReducedModelWithKwargs()
    trainer = SltTrainer(model=model, args=args)

    trainer.compute_loss(
        model,
        {
            "labels": torch.tensor([[1]]),
            "generation_input_ids": torch.tensor([[2]]),
            "generation_attention_mask": torch.tensor([[1]]),
            "generation_token_type_ids": torch.tensor([[0]]),
        },
    )

    assert not any(name.startswith("generation_") for name in model.last_kwargs)


class _GenerationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(pad_token_id=0)
        self.generation_config = GenerationConfig(
            pad_token_id=0,
            max_new_tokens=2,
        )
        self.generate_kwargs = None

    def forward(self, input_ids=None, **kwargs):
        return {"loss": self.weight * 0, "logits": torch.empty(0)}

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        input_ids = kwargs["input_ids"]
        new_token = torch.full(
            (input_ids.shape[0], 1),
            9,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, new_token), dim=-1)


class _TeacherForcingGenerationModel(_GenerationModel):
    def __init__(self):
        super().__init__()
        self.teacher_forcing_kwargs = None

    def forward(self, input_ids=None, **kwargs):
        self.teacher_forcing_kwargs = {
            "input_ids": input_ids,
            **kwargs,
        }
        return {"loss": self.weight * 0 + 2.5, "logits": torch.empty(0)}


class _CTCGenerationModel(_GenerationModel):
    def __init__(self):
        super().__init__()
        self.config.ctc_blank_id = 0
        self.forward_calls = 0

    def forward(self, input_ids=None, **kwargs):
        self.forward_calls += 1
        # Paths: [blank, 1, 1, blank] and [2, blank].
        paths = torch.tensor([0, 1, 1, 0, 2, 0])
        logits = torch.nn.functional.one_hot(paths, num_classes=3).float()
        return SimpleNamespace(
            loss=self.weight * 0 + 1.5,
            logits=torch.empty(0),
            ctc_logits=logits,
            ctc_lengths=torch.tensor([4, 2]),
        )


class _CTCOnlyModel(_CTCGenerationModel):
    def forward(self, pixel_values=None, pixel_values_length=None, **kwargs):
        assert kwargs["forward_mode"] == "ctc_only"
        assert "input_ids" not in kwargs
        output = super().forward(**kwargs)
        return SimpleNamespace(
            loss=output.loss,
            logits=output.ctc_logits,
            lengths=output.ctc_lengths,
        )

    def generate(self, **kwargs):
        raise AssertionError("ctc_only prediction must not call generate()")


class _RequiresCTCMetric:
    requires_ctc_outputs = True

    def __call__(self, output):
        return {}


class _ForwardModeModel(_GenerationModel):
    def __init__(self):
        super().__init__()
        self.forward_mode = None

    def forward(self, input_ids=None, forward_mode="joint", **kwargs):
        self.forward_mode = forward_mode
        return {"loss": self.weight * 0, "logits": torch.empty(0)}


def test_compute_loss_injects_explicit_engine_forward_mode(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
    )
    model = _ForwardModeModel()
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create({"engine": {"forward_mode": "ctc_only"}}),
    )

    trainer.compute_loss(model, {"input_ids": torch.tensor([[1]])})

    assert model.forward_mode == "ctc_only"


def test_prediction_step_uses_prompt_only_generation_fields(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
        predict_with_generate=True,
    )
    model = _GenerationModel()
    trainer = SltTrainer(model=model, args=args)
    full_input_ids = torch.tensor([[3, 4, 5, 6, 7]])
    prompt_input_ids = torch.tensor([[3, 4, 5]])
    labels = torch.tensor([[-100, -100, -100, 6, 7]])
    lang_ids = torch.tensor([1])

    _, predictions, label_output = trainer.prediction_step(
        model,
        {
            "input_ids": full_input_ids,
            "attention_mask": torch.ones_like(full_input_ids),
            "position_ids": torch.arange(5).unsqueeze(0),
            "token_type_ids": torch.zeros_like(full_input_ids),
            "labels": labels,
            "lang_ids": lang_ids,
            "pixel_values": torch.zeros(2, 3, 1, 1),
            "pixel_values_length": torch.tensor([2]),
            "generation_input_ids": prompt_input_ids,
            "generation_attention_mask": torch.ones_like(prompt_input_ids),
            "generation_token_type_ids": torch.zeros_like(prompt_input_ids),
        },
        prediction_loss_only=False,
    )

    assert torch.equal(
        model.generate_kwargs["input_ids"].cpu(),
        prompt_input_ids,
    )
    assert set(model.generate_kwargs).isdisjoint(
        {
            "labels",
            "position_ids",
            "lang_ids",
            "generation_input_ids",
            "generation_attention_mask",
            "generation_token_type_ids",
        }
    )
    generated_tokens, generated_lengths, prompt_lengths = predictions
    assert generated_tokens.shape == (1, 4)
    assert generated_lengths.tolist() == [4]
    assert prompt_lengths.tolist() == [3]
    output_labels, output_lang_ids = label_output
    assert torch.equal(output_labels.cpu(), labels)
    assert torch.equal(output_lang_ids.cpu(), lang_ids)


def test_prediction_step_optionally_computes_teacher_forcing_loss(tmp_path):
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
        predict_with_generate=True,
        predict_with_teacher_forcing=True,
        auto_output_dir=False,
    )
    model = _TeacherForcingGenerationModel()
    trainer = SltTrainer(model=model, args=args)
    trainer._is_predicting = True
    full_input_ids = torch.tensor([[3, 4, 5, 6]])
    prompt_input_ids = torch.tensor([[3, 4, 5]])

    loss, _, _ = trainer.prediction_step(
        model,
        {
            "input_ids": full_input_ids,
            "attention_mask": torch.ones_like(full_input_ids),
            "position_ids": torch.arange(4).unsqueeze(0),
            "token_type_ids": torch.zeros_like(full_input_ids),
            "labels": torch.tensor([[-100, -100, -100, 6]]),
            "pixel_values": torch.zeros(2, 3, 1, 1),
            "pixel_values_length": torch.tensor([2]),
            "generation_input_ids": prompt_input_ids,
            "generation_attention_mask": torch.ones_like(prompt_input_ids),
            "generation_token_type_ids": torch.zeros_like(prompt_input_ids),
        },
        prediction_loss_only=False,
    )

    assert loss.item() == 2.5
    assert torch.equal(model.teacher_forcing_kwargs["input_ids"], full_input_ids)
    assert torch.equal(
        model.teacher_forcing_kwargs["labels"],
        torch.tensor([[-100, -100, -100, 6]]),
    )


def test_prediction_step_returns_ctc_sequences_for_combined_metric(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
        predict_with_generate=True,
    )
    model = _CTCGenerationModel()
    trainer = SltTrainer(model=model, args=args, compute_metrics=_RequiresCTCMetric())
    full_input_ids = torch.tensor([[3, 4], [3, 4]])
    prompt_input_ids = torch.tensor([[3], [3]])

    loss, predictions, label_output = trainer.prediction_step(
        model,
        {
            "input_ids": full_input_ids,
            "attention_mask": torch.ones_like(full_input_ids),
            "position_ids": torch.arange(2).expand(2, -1),
            "token_type_ids": torch.zeros_like(full_input_ids),
            "labels": torch.tensor([[-100, 4], [-100, 4]]),
            "lang_ids": torch.tensor([1, 1]),
            "pixel_values": torch.zeros(6, 3, 1, 1),
            "pixel_values_length": torch.tensor([4, 2]),
            "generation_input_ids": prompt_input_ids,
            "generation_attention_mask": torch.ones_like(prompt_input_ids),
            "generation_token_type_ids": torch.zeros_like(prompt_input_ids),
            "pseudo_gloss_ids": torch.tensor([1, 3, 2]),
            "pseudo_gloss_length": torch.tensor([2, 1]),
        },
        prediction_loss_only=False,
    )

    assert loss.item() == 1.5
    assert model.forward_calls == 1
    ctc_ids, ctc_lengths = predictions[3:5]
    assert ctc_ids.tolist() == [[1], [2]]
    assert ctc_lengths.tolist() == [1, 1]
    reference_ids, reference_lengths = label_output[2:4]
    assert reference_ids.tolist() == [[1, 3], [2, 0]]
    assert reference_lengths.tolist() == [2, 1]


def test_ctc_only_prediction_step_skips_generation_and_joint_inputs(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
        predict_with_generate=True,
    )
    model = _CTCOnlyModel()
    trainer = SltTrainer(
        model=model,
        args=args,
        hydra_config=OmegaConf.create({"engine": {"forward_mode": "ctc_only"}}),
    )

    loss, predictions, references = trainer.prediction_step(
        model,
        {
            "pixel_values": torch.zeros(6, 3, 1, 1),
            "pixel_values_length": torch.tensor([4, 2]),
            "pseudo_gloss_ids": torch.tensor([1, 3, 2]),
            "pseudo_gloss_length": torch.tensor([2, 1]),
        },
        prediction_loss_only=False,
    )

    assert loss.item() == 1.5
    assert model.forward_calls == 1
    prediction_ids, prediction_lengths = predictions
    assert prediction_ids.tolist() == [[1], [2]]
    assert prediction_lengths.tolist() == [1, 1]
    reference_ids, reference_lengths = references
    assert reference_ids.tolist() == [[1, 3], [2, 0]]
    assert reference_lengths.tolist() == [2, 1]
