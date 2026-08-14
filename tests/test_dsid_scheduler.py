from types import SimpleNamespace

import pytest
from transformers.trainer_callback import ExportableState

from csi_slt.engine.callbacks import DSIDWeightSchedulerCallback
from csi_slt.engine.scheduler import DSIDScheduler


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, 0.0),
        (1, 0.5),
        (2, 1.0),
        (7, 1.0),
        (8, 0.5),
        (9, 0.0),
    ],
)
def test_dsid_scheduler_has_warmup_plateau_and_decay(step, expected):
    scheduler = DSIDScheduler(
        max_weight=1.0,
        total_steps=10,
        warmup_ratio=2 / 9,
        decay_ratio=2 / 9,
    )

    assert scheduler.value_at(step) == pytest.approx(expected)


def test_dsid_scheduler_state_round_trip():
    scheduler = DSIDScheduler(max_weight=0.8, total_steps=20)
    expected = scheduler.step(11)

    restored = DSIDScheduler(max_weight=0.8, total_steps=20)
    restored.load_state_dict(scheduler.state_dict())

    assert restored.current_step == 11
    assert restored.current_weight == pytest.approx(expected)


class _WeightTarget:
    def __init__(self):
        self.config = SimpleNamespace(dsid_loss_weight=1.0)
        self.weight = None

    def set_dsid_loss_weight(self, weight):
        self.weight = weight


def test_callback_synchronizes_scheduler_to_resumed_global_step():
    model = _WeightTarget()
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda wrapped: wrapped),
    )
    state = SimpleNamespace(global_step=8, max_steps=10)
    callback = DSIDWeightSchedulerCallback(
        warmup_ratio=2 / 9,
        decay_ratio=2 / 9,
    )

    callback.on_train_begin(None, state, None, trainer=trainer)

    assert callback.scheduler.current_step == 8
    assert model.weight == pytest.approx(0.5)


def test_callback_exports_and_restores_scheduler_state():
    model = _WeightTarget()
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda wrapped: wrapped),
    )
    state = SimpleNamespace(global_step=8, max_steps=10)
    callback = DSIDWeightSchedulerCallback(
        warmup_ratio=2 / 9,
        decay_ratio=2 / 9,
    )
    assert isinstance(callback, ExportableState)
    callback.on_train_begin(None, state, None, trainer=trainer)
    saved = callback.state()

    restored = DSIDWeightSchedulerCallback(**saved["args"])
    for name, value in saved["attributes"].items():
        setattr(restored, name, value)
    restored.on_train_begin(None, state, None, trainer=trainer)

    assert restored.scheduler.current_step == 8
    assert restored.scheduler.current_weight == pytest.approx(0.5)
    assert restored.scheduler_state is None
    assert restored.scheduler_parameters is None


def test_callback_state_matches_trainer_step_after_optimizer_update():
    model = _WeightTarget()
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda wrapped: wrapped),
    )
    state = SimpleNamespace(global_step=0, max_steps=10)
    callback = DSIDWeightSchedulerCallback()
    callback.on_train_begin(None, state, None, trainer=trainer)

    # Trainer increments global_step before dispatching on_step_end.
    state.global_step = 1
    callback.on_step_end(None, state, None, trainer=trainer)

    assert callback.state()["attributes"]["scheduler_state"] == {"current_step": 1}


@pytest.mark.parametrize(
    ("state_override", "message"),
    [
        ({"global_step": 7}, "checkpoint step does not match"),
        ({"max_steps": 12}, "parameters changed while resuming"),
    ],
)
def test_callback_rejects_inconsistent_restored_state(state_override, message):
    model = _WeightTarget()
    trainer = SimpleNamespace(
        model=model,
        accelerator=SimpleNamespace(unwrap_model=lambda wrapped: wrapped),
    )
    saved_state = SimpleNamespace(global_step=8, max_steps=10)
    callback = DSIDWeightSchedulerCallback()
    callback.on_train_begin(None, saved_state, None, trainer=trainer)
    exported = callback.state()

    restored = DSIDWeightSchedulerCallback(**exported["args"])
    for name, value in exported["attributes"].items():
        setattr(restored, name, value)
    resumed_state = SimpleNamespace(
        global_step=state_override.get("global_step", 8),
        max_steps=state_override.get("max_steps", 10),
    )

    with pytest.raises(RuntimeError, match=message):
        restored.on_train_begin(None, resumed_state, None, trainer=trainer)
