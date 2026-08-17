from types import SimpleNamespace

import torch
from torch import nn
from omegaconf import OmegaConf
from transformers import GenerationConfig, Seq2SeqTrainingArguments

from csi_slt.engine.sft.callbacks import (
    DSIDWeightSchedulerCallback,
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


def test_trainer_builds_dsid_scheduler_callback_from_engine_config(tmp_path):
    args = Seq2SeqTrainingArguments(
        output_dir=str(tmp_path),
        report_to="none",
    )
    hydra_config = OmegaConf.create(
        {"engine": {"dsid_scheduler": {"warmup_ratio": 0.2, "decay_ratio": 0.4}}}
    )
    trainer = SltTrainer(
        model=_MeanReducedModelWithKwargs(),
        args=args,
        hydra_config=hydra_config,
    )

    callback = next(
        callback
        for callback in trainer.callback_handler.callbacks
        if isinstance(callback, DSIDWeightSchedulerCallback)
    )
    assert callback.warmup_ratio == 0.2
    assert callback.decay_ratio == 0.4


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
