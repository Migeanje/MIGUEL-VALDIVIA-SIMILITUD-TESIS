"""Stage 2 of the winner closed by D30 E3's full rule, with the labeled-set F1 in step 2, on
T19's candidates from ``conftest.py`` and hand-made F1 differences."""

from dataclasses import asdict, replace
from typing import Any

import pytest

from thematic_redundancy.redundancy.comparison import BootstrapSettings, F1Difference
from thematic_redundancy.redundancy.stage_two import (
    candidate_from_record,
    close_stage_two,
    f1_step,
)
from thematic_redundancy.topic_map.selection import stage_two_provisional

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
NAMES = (MINILM, MPNET)
SETTINGS = BootstrapSettings(resamples=2000, seed=7, interval=0.95)


def difference(
    first: float | None, second: float | None, lower: float | None, upper: float | None
) -> F1Difference:
    verdict = None
    if lower is not None and upper is not None:
        verdict = "first_higher" if lower > 0 else "second_higher" if upper < 0 else "tie"
    return F1Difference(
        first_tau=0.7,
        second_tau=0.8,
        first_f1=first,
        second_f1=second,
        difference=None if first is None or second is None else first - second,
        lower=lower,
        upper=upper,
        verdict=verdict,  # type: ignore[arg-type]
        undefined_resamples=0,
        settings=SETTINGS,
    )


FIRST_HIGHER = difference(0.9, 0.6, 0.1, 0.5)
SECOND_HIGHER = difference(0.6, 0.9, -0.5, -0.1)
TIE = difference(0.8, 0.75, -0.02, 0.12)


def winners(kit: Any, scenario: str = "npmi") -> tuple[Any, Any]:
    found = kit.candidates(scenario)
    return (found["C2"], found["C4"]) if scenario == "gates" else (found["C1"], found["C4"])


def close(kit: Any, delta: F1Difference, scenario: str = "npmi", **changes: Any) -> Any:
    first, second = winners(kit, scenario)
    first, second = (
        replace(first, **changes.get("first", {})),
        replace(second, **changes.get("second", {})),
    )
    fallback = kit.config.experiment.selection.fallbacks.model
    return close_stage_two(first, second, delta, gates=kit.gates, fallback_model=fallback)


# --- The F1 step -------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("delta", "outcome", "words"),
    [
        (FIRST_HIGHER, "first", "lies above 0"),
        (SECOND_HIGHER, "second", "lies below 0"),
        (TIE, "tie", "holds or touches 0"),
        (difference(0.8, 0.8, 0.0, 0.1), "tie", "holds or touches 0"),
    ],
)
def test_the_f1_step_follows_the_paired_interval(
    delta: F1Difference, outcome: str, words: str
) -> None:
    step = f1_step(delta, NAMES)

    assert step.outcome == outcome
    assert words in step.reason
    assert "95%" in step.reason
    assert (step.first, step.second) == NAMES
    assert (step.first_f1, step.second_f1, step.difference) == (
        delta.first_f1,
        delta.second_f1,
        delta.difference,
    )
    assert (step.lower, step.upper, step.verdict, step.interval) == (
        delta.lower,
        delta.upper,
        delta.verdict,
        0.95,
    )


@pytest.mark.parametrize(
    ("delta", "words"),
    [
        (difference(0.8, 0.7, None, None), "has no interval"),
        (replace(difference(0.6, 0.7, 0.01, 0.2)), "opposite"),
        (replace(difference(0.7, 0.9, -0.3, -0.1), first_f1=0.95, difference=0.05), "opposite"),
        (replace(difference(0.7, 0.7, 0.01, 0.2)), "opposite"),
        (replace(FIRST_HIGHER, first_f1=None, difference=None), "opposite"),
    ],
)
def test_an_f1_step_that_names_no_higher_f1_is_undefined(delta: F1Difference, words: str) -> None:
    step = f1_step(delta, NAMES)

    assert step.outcome == "undefined"
    assert words in step.reason
    assert "D30 does not name this case" in step.reason


# --- The full rule ---------------------------------------------------------------------------


def test_the_provisional_decision_is_t19_s(kit: Any) -> None:
    first, second = winners(kit)
    fallback = kit.config.experiment.selection.fallbacks.model

    stage = close(kit, FIRST_HIGHER)

    assert stage.provisional == stage_two_provisional(
        first, second, gates=kit.gates, fallback_model=fallback
    )
    assert stage.provisional.winner == MPNET


@pytest.mark.parametrize(
    ("delta", "winner", "outcome"),
    [(FIRST_HIGHER, MINILM, "changed"), (SECOND_HIGHER, MPNET, "confirmed")],
)
def test_a_deciding_f1_names_the_model_at_step_two(
    kit: Any, delta: F1Difference, winner: str, outcome: str
) -> None:
    stage = close(kit, delta)

    assert (stage.status, stage.winner, stage.outcome()) == ("closed", winner, outcome)
    decision = stage.decision
    assert (decision.deciding_step, decision.deciding_criterion) == (2, "f1")
    assert decision.steps[0] == stage.provisional.steps[0]
    assert decision.steps[1].outcome == ("first" if winner == MINILM else "second")
    assert (decision.steps[1].first, decision.steps[1].second) == (
        delta.first_f1,
        delta.second_f1,
    )
    assert decision.steps[1].reason == stage.f1.reason
    assert len(decision.steps) == 2
    assert decision.findings == ()


def test_a_tied_f1_goes_on_to_t19_s_npmi_step(kit: Any) -> None:
    stage = close(kit, TIE)

    assert (stage.winner, stage.outcome()) == (MPNET, "confirmed")
    assert stage.decision.deciding_step == 3
    assert stage.decision.steps[1].outcome == "tie"
    assert stage.decision.steps[2] == stage.provisional.steps[2]


def test_a_tie_through_npmi_goes_to_the_fallback_model(kit: Any) -> None:
    first, _ = winners(kit)
    stage = close(kit, TIE, second={"npmi": first.npmi})

    assert stage.provisional.winner == MINILM
    assert (stage.winner, stage.decision.deciding_step) == (MINILM, 4)
    assert stage.decision.steps[3].criterion == "fallback"
    assert stage.outcome() == "confirmed"


def test_the_gates_decide_before_the_f1(kit: Any) -> None:
    stage = close(kit, FIRST_HIGHER, scenario="gates")

    assert (stage.winner, stage.decision.deciding_step, stage.outcome()) == (MPNET, 1, "confirmed")
    assert len(stage.decision.steps) == 1
    assert stage.f1.outcome == "first"


def test_an_undefined_f1_step_leaves_stage_two_undefined(kit: Any) -> None:
    stage = close(kit, difference(0.8, 0.7, None, None))

    assert (stage.status, stage.winner, stage.decision, stage.outcome()) == (
        "undefined",
        None,
        None,
        "undefined",
    )
    assert stage.undefined_reason == stage.f1.reason


def test_an_undefined_f1_step_does_not_matter_when_the_gates_decide(kit: Any) -> None:
    stage = close(kit, difference(0.8, 0.7, None, None), scenario="gates")

    assert (stage.status, stage.winner) == ("closed", MPNET)
    assert stage.undefined_reason is None


def test_a_flip_under_the_npmi_sensitivity_is_reported(kit: Any) -> None:
    first, second = winners(kit)
    stage = close(kit, TIE, first={"npmi_sensitivity": second.npmi})

    (sensitivity,) = stage.sensitivities
    assert sensitivity.name == "npmi_sensitivity"
    assert (sensitivity.primary, sensitivity.value, sensitivity.flipped) == (MPNET, MINILM, True)
    assert "a flip, reported; the rule is not changed" in sensitivity.reason
    assert stage.winner == MPNET


def test_no_flip_when_the_f1_decides(kit: Any) -> None:
    first, second = winners(kit)
    stage = close(kit, SECOND_HIGHER, first={"npmi_sensitivity": second.npmi})

    (sensitivity,) = stage.sensitivities
    assert (sensitivity.primary, sensitivity.value, sensitivity.flipped) == (MPNET, MPNET, False)


# --- T19's records ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["C1", "C2", "C3", "C4", "B0"])
def test_a_candidate_comes_back_from_t19_s_record(kit: Any, name: str) -> None:
    record = kit.selection_record("npmi")["configurations"][name]

    assert candidate_from_record(record) == kit.candidates("npmi")[name]


def test_a_record_without_a_field_or_with_another_is_refused(kit: Any) -> None:
    record = dict(kit.selection_record("npmi")["configurations"]["C1"])

    with pytest.raises(ValueError, match="not a candidate"):
        candidate_from_record({key: value for key, value in record.items() if key != "npmi"})
    with pytest.raises(ValueError, match="not a candidate"):
        candidate_from_record({**record, "extra": 1})
    with pytest.raises(ValueError, match="not a candidate"):
        candidate_from_record({**record, "npmi": {"mean": 0.1}})


def test_a_candidate_record_is_t19_s_dataclass_as_json(kit: Any) -> None:
    candidate = kit.candidates("npmi")["C4"]
    record = {**asdict(candidate), "gates_failed": []}

    assert candidate_from_record(record) == candidate
