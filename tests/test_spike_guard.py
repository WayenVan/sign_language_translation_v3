import math
import socket
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from optimi import StableAdamW
from torch import nn
from transformers import TrainerCallback

from csi_slt.engine.sft.spike_guard import (
    SpikeGuard,
    install_optimizer_step_guard,
    synchronized_grad_norm,
)
from csi_slt.engine.sft.trainer import SltTrainer
from csi_slt.engine.sft.training_args import SltTrainingArguments


# --------------------------------------------------------------------------- #
# SpikeGuard decisions
# --------------------------------------------------------------------------- #


def _warm_guard(norms=(10.0, 11.0, 9.0, 10.0), **kwargs):
    kwargs = {"factor": 10.0, "window": 10, "min_history": len(norms), **kwargs}
    guard = SpikeGuard(**kwargs)
    for norm in norms:
        assert guard.observe(norm).skip is False
    return guard


def test_no_finite_step_is_skipped_before_min_history():
    guard = SpikeGuard(factor=10.0, window=10, min_history=3)

    assert guard.observe(1.0).skip is False
    assert guard.observe(1000.0).skip is False  # still warming up
    assert list(guard.history) == [1.0, 1000.0]


def test_spike_is_skipped_and_kept_out_of_the_history():
    guard = _warm_guard()

    decision = guard.observe(849.0)

    assert decision.skip is True
    assert decision.reason == "spike"
    assert decision.median == 10.0
    assert 849.0 not in guard.history
    assert guard.skipped_total == 1
    # The next ordinary step is accepted and resets the consecutive count.
    assert guard.observe(12.0).skip is False
    assert guard.consecutive == 0


def test_norm_at_exactly_factor_times_median_is_accepted():
    guard = _warm_guard()

    assert guard.observe(100.0).skip is False


def test_consecutive_limit_accepts_the_next_outlier_and_rebases():
    guard = _warm_guard(max_consecutive=3)

    assert [guard.observe(500.0).skip for _ in range(3)] == [True, True, True]
    decision = guard.observe(500.0)

    assert decision.skip is False
    assert decision.reason == "consecutive_limit"
    assert guard.history[-1] == 500.0
    assert guard.consecutive == 0
    assert guard.skipped_total == 3
    assert guard.consecutive_limit_hits == 1


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_non_finite_norm_is_always_skipped(value):
    guard = SpikeGuard(factor=10.0, window=10, min_history=5, max_consecutive=1)

    # Skipped even during warm-up and past the consecutive limit.
    for _ in range(3):
        decision = guard.observe(value)
        assert decision.skip is True
        assert decision.reason == "non_finite"
    assert len(guard.history) == 0
    assert guard.consecutive == 0


def test_history_is_bounded_by_the_window():
    guard = SpikeGuard(factor=10.0, window=3, min_history=1)
    for norm in [1.0, 2.0, 3.0, 4.0]:
        guard.observe(norm)

    assert list(guard.history) == [2.0, 3.0, 4.0]


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"factor": 1.0}, "spike_skip_factor"),
        ({"factor": 10.0, "window": 0}, "spike_skip_window"),
        ({"factor": 10.0, "window": 10, "min_history": 11}, "spike_skip_min_history"),
        ({"factor": 10.0, "min_history": 0}, "spike_skip_min_history"),
        ({"factor": 10.0, "max_consecutive": 0}, "spike_skip_max_consecutive"),
    ],
)
def test_invalid_guard_settings_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SpikeGuard(**kwargs)


def test_training_arguments_validate_spike_skip_settings(tmp_path):
    with pytest.raises(ValueError, match="spike_skip_factor"):
        SltTrainingArguments(
            output_dir=str(tmp_path),
            auto_output_dir=False,
            report_to="none",
            spike_skip_factor=0.5,
        )


# --------------------------------------------------------------------------- #
# Optimizer step guard
# --------------------------------------------------------------------------- #


def _optimizer_snapshot(model, optimizer):
    params = [p.detach().clone() for p in model.parameters()]
    state = {
        id(p): {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in optimizer.state[p].items()
        }
        for p in model.parameters()
    }
    steps = [
        group["step"].clone() if torch.is_tensor(group.get("step")) else group.get("step")
        for group in optimizer.param_groups
    ]
    return params, state, steps


def _assert_same_snapshot(before, after):
    for a, b in zip(before[0], after[0]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert before[1].keys() == after[1].keys()
    for key in before[1]:
        assert before[1][key].keys() == after[1][key].keys()
        for name in before[1][key]:
            a, b = before[1][key][name], after[1][key][name]
            if torch.is_tensor(a):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            else:
                assert a == b
    for a, b in zip(before[2], after[2]):
        if torch.is_tensor(a):
            assert torch.equal(a, b)
        else:
            assert a == b


def _backward(model, scale):
    model.zero_grad()
    (model(torch.ones(1, 4)).sum() * scale).backward()


@pytest.mark.parametrize(
    "make_optimizer",
    [
        lambda params: torch.optim.AdamW(params, lr=0.1, weight_decay=0.01),
        # optimi picks its Triton kernels even for CPU tensors when Triton is
        # importable; training runs on GPU, so pin the foreach path here.
        lambda params: StableAdamW(params, lr=0.1, weight_decay=0.01, triton=False),
    ],
    ids=["adamw", "stable_adamw"],
)
def test_skipped_step_leaves_parameters_moments_and_step_count_untouched(
    make_optimizer,
):
    torch.manual_seed(0)
    model = nn.Linear(4, 3)
    optimizer = make_optimizer(model.parameters())
    skip = {"value": False}
    install_optimizer_step_guard(optimizer, lambda: skip["value"])

    _backward(model, 1.0)
    optimizer.step()  # populate the moments first
    _backward(model, 1000.0)
    before = _optimizer_snapshot(model, optimizer)

    skip["value"] = True
    optimizer.step()
    _assert_same_snapshot(before, _optimizer_snapshot(model, optimizer))

    skip["value"] = False
    optimizer.step()
    after = _optimizer_snapshot(model, optimizer)
    assert not torch.equal(before[0][0], after[0][0])


def test_guard_is_installed_once():
    optimizer = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    install_optimizer_step_guard(optimizer, lambda: False)
    first = optimizer.step

    install_optimizer_step_guard(optimizer, lambda: True)

    assert optimizer.step.__func__ is first.__func__


def test_lr_scheduler_wraps_the_guard_without_warnings():
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1)
    install_optimizer_step_guard(optimizer, lambda: False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        _backward(model, 1.0)
        optimizer.step()
        scheduler.step()

    assert getattr(optimizer, "_opt_called", False) is True


# --------------------------------------------------------------------------- #
# SltTrainer end to end
# --------------------------------------------------------------------------- #


class _ScaledLinearModel(nn.Module):
    """loss = scale * sum(w * x): the gradient norm is proportional to scale."""

    def __init__(self):
        super().__init__()
        # Any registered optimizer component works; the CTC head is the simplest.
        self.ctc_head = nn.Linear(4, 1, bias=False)
        nn.init.ones_(self.ctc_head.weight)
        self.config = type("Config", (), {})()

    @property
    def weight(self):
        return self.ctc_head.weight

    def forward(self, x=None, scale=None, labels=None, **kwargs):
        return {"loss": (scale.reshape(-1, 1) * self.ctc_head(x)).sum()}


class _RecordStepCallback(TrainerCallback):
    def __init__(self):
        self.before = {}
        self.after = {}

    def on_step_begin(self, args, state, control, model=None, **kwargs):
        self.before[state.global_step + 1] = model.weight.detach().clone()

    def on_step_end(self, args, state, control, model=None, **kwargs):
        self.after[state.global_step] = model.weight.detach().clone()


def _run_trainer(tmp_path, scales, **spike_kwargs):
    model = _ScaledLinearModel()
    dataset = [
        {"x": torch.full((4,), 1.0 + 0.01 * index), "scale": torch.tensor(scale)}
        for index, scale in enumerate(scales)
    ]
    args = SltTrainingArguments(
        output_dir=str(tmp_path),
        auto_output_dir=False,
        report_to="none",
        use_cpu=True,
        max_steps=len(scales),
        per_device_train_batch_size=1,
        learning_rate=0.01,
        weight_decay=0.0,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        max_grad_norm=1.0,
        **spike_kwargs,
    )
    recorder = _RecordStepCallback()
    trainer = SltTrainer(model=model, args=args, train_dataset=dataset)
    trainer.add_callback(recorder)
    # The length-bucketing sampler needs video metadata; order is what matters.
    trainer._get_train_sampler = lambda dataset=None: torch.utils.data.SequentialSampler(
        dataset if dataset is not None else trainer.train_dataset
    )
    trainer.train()
    return trainer, recorder


def test_trainer_skips_only_the_spike_step(tmp_path):
    # Steps 1-4 warm the history, step 5 is a 1000x spike, steps 6-7 are normal.
    scales = [1.0, 1.0, 1.0, 1.0, 1000.0, 1.0, 1.0]
    trainer, recorder = _run_trainer(
        tmp_path,
        scales,
        spike_skip_factor=10.0,
        spike_skip_window=10,
        spike_skip_min_history=4,
    )

    for step in range(1, len(scales) + 1):
        unchanged = torch.equal(recorder.before[step], recorder.after[step])
        assert unchanged is (step == 5), f"step {step}"

    logs = [entry for entry in trainer.state.log_history if "loss" in entry]
    assert logs[-1]["spike_skip/total"] == 1.0
    assert logs[4]["grad_norm"] > 100  # the skipped spike is still logged
    assert trainer.state.global_step == len(scales)


def test_trainer_without_spike_skip_updates_every_step(tmp_path):
    scales = [1.0, 1.0, 1.0, 1.0, 1000.0, 1.0]
    trainer, recorder = _run_trainer(tmp_path, scales)

    for step in range(1, len(scales) + 1):
        assert not torch.equal(recorder.before[step], recorder.after[step])
    assert trainer._spike_guard is None
    assert not getattr(trainer.optimizer.optimizer, "_slt_step_guard_installed", False)
    assert all(
        "spike_skip/total" not in entry for entry in trainer.state.log_history
    )


# --------------------------------------------------------------------------- #
# Cross-process agreement
# --------------------------------------------------------------------------- #


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sync_worker(rank, world_size, port, local_norms, results):
    dist.init_process_group(
        "gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        guard = SpikeGuard(factor=10.0, window=10, min_history=3)
        decisions = []
        for norm in local_norms[rank]:
            synced = synchronized_grad_norm(torch.tensor(norm), torch.device("cpu"))
            decisions.append((synced, guard.observe(synced).skip))
        results[rank] = decisions
    finally:
        dist.destroy_process_group()


def test_ranks_agree_on_skipping_even_when_local_norms_differ():
    # Rank 1 alone sees a spike at step 4 and a NaN at step 5; both ranks must
    # still use the same norm and make the same decision at every step.
    local_norms = [
        [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
        [10.0, 10.0, 10.0, 900.0, math.nan, 10.0],
    ]
    manager = mp.Manager()
    results = manager.dict()
    mp.spawn(
        _sync_worker,
        args=(2, _free_port(), local_norms, results),
        nprocs=2,
        join=True,
    )

    assert results[0] == results[1]
    assert [skip for _, skip in results[0]] == [False, False, False, True, True, False]
    assert math.isinf(results[0][4][0])
