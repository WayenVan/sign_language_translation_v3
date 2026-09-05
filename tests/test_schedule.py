import pytest

from csi_slt.engine.schedule import ScalarAnnealSchedule, value_at


def test_from_mapping_returns_none_for_absent_schedule():
    assert ScalarAnnealSchedule.from_mapping(None) is None
    assert ScalarAnnealSchedule.from_mapping({}) is None


def test_from_mapping_applies_defaults():
    schedule = ScalarAnnealSchedule.from_mapping({"start": 1.0, "end": 0.3})
    assert schedule == ScalarAnnealSchedule(
        start=1.0, end=0.3, anneal_ratio=1.0, mode="linear"
    )


@pytest.mark.parametrize(
    "config",
    [
        {"start": 1.0},
        {"end": 0.3},
        {"start": 1.0, "end": 0.3, "unknown": 1},
    ],
)
def test_from_mapping_rejects_invalid_config(config):
    with pytest.raises(ValueError):
        ScalarAnnealSchedule.from_mapping(config)


def test_anneal_ratio_must_be_in_unit_interval():
    with pytest.raises(ValueError):
        ScalarAnnealSchedule(start=1.0, end=0.3, anneal_ratio=0.0)
    with pytest.raises(ValueError):
        ScalarAnnealSchedule(start=1.0, end=0.3, anneal_ratio=1.5)


def test_linear_value_at_interpolates_and_holds_after_anneal_ratio():
    schedule = ScalarAnnealSchedule(start=1.0, end=0.0, anneal_ratio=0.5, mode="linear")

    assert value_at(schedule, step=0, max_steps=1000) == 1.0
    assert value_at(schedule, step=250, max_steps=1000) == pytest.approx(0.5)
    assert value_at(schedule, step=500, max_steps=1000) == pytest.approx(0.0)
    # Past the anneal window the value holds at `end`.
    assert value_at(schedule, step=1000, max_steps=1000) == pytest.approx(0.0)


def test_cosine_value_at_matches_endpoints_and_midpoint():
    schedule = ScalarAnnealSchedule(start=1.0, end=0.0, anneal_ratio=1.0, mode="cosine")

    assert value_at(schedule, step=0, max_steps=1000) == pytest.approx(1.0)
    assert value_at(schedule, step=1000, max_steps=1000) == pytest.approx(0.0)
    assert value_at(schedule, step=500, max_steps=1000) == pytest.approx(0.5)


@pytest.mark.parametrize("max_steps", [None, 0])
def test_value_at_falls_back_to_start_without_known_progress(max_steps):
    schedule = ScalarAnnealSchedule(start=1.0, end=0.3)
    assert value_at(schedule, step=100, max_steps=max_steps) == 1.0
