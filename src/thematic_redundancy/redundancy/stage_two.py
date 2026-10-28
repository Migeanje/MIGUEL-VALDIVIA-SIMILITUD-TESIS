"""Stage 2 of the winner selection closed by the full rule of D30 E3 (UD3, O03), with the
labeled-set F1 that T19 had to skip.

T19 closed stage 1 (the algorithm within each model's space) and named a provisional winner by
stage 2's steps 1, 3 and 4 (:func:`~thematic_redundancy.topic_map.selection.stage_two_provisional`).
D30 E3's stage 2 compares the two models' stage-1 configurations step by step, and the first
step that decides names the model:

1. a model whose stage-1 configuration passes every gate wins over one whose configuration
   does not;
2. the labeled-set F1 (:func:`f1_step`): each model's F1 at its fixed, cross-validated τ,
   compared by the paired bootstrap of
   :func:`~thematic_redundancy.redundancy.comparison.paired_f1_difference`; "if the 95%
   percentile interval of ΔF1 excludes 0, the higher F1 wins, otherwise it is a tie";
3. the NPMI of each model's stage-1 configuration, by "A beats B";
4. the fallback model, MiniLM.

:func:`close_stage_two` recomputes T19's provisional decision from the same candidates and puts
the F1 step in place of the skipped one, so steps 1, 3 and 4 are T19's own code and reasons.

**An undefined F1 step.** D30 names the winner of step 2 as "the higher F1" when the interval
excludes 0. Two cases leave that without a winner, and no rule here fills the gap: the interval
does not exist (no resample has both F1 values), or it excludes 0 on the side opposite to the
observed difference (or the observed difference is 0 or undefined), so "the higher F1" is not
the side that the interval names. Stage 2 is then **undefined**: no winner is named, the reason
is recorded, and the case is a decision for the user. Overlapping intervals of the two F1 values
are not a case of their own: D30 compares the paired interval of ΔF1, not the two intervals.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from typing import Any, Literal

from thematic_redundancy.redundancy.comparison import F1Difference, Verdict
from thematic_redundancy.topic_map.selection import (
    Candidate,
    Comparison,
    Decision,
    Gates,
    SeedSummary,
    Sensitivity,
    stage_two_provisional,
)

F1Outcome = Literal["first", "second", "tie", "undefined"]
Status = Literal["closed", "undefined"]
Outcome = Literal["confirmed", "changed", "undefined"]
"""Stage 2's outcome against T19's provisional winner."""

F1_STEP = 2
"""The step of stage 2 that the labeled-set F1 takes (D30 E3)."""


@dataclass(frozen=True)
class F1Step:
    """Stage 2's F1 step: the paired comparison of the two models' F1 and what it decides."""

    first: str
    second: str
    first_f1: float | None
    second_f1: float | None
    difference: float | None
    """ΔF1 = F1(:attr:`first`) − F1(:attr:`second`), each at its fixed τ."""
    lower: float | None
    upper: float | None
    interval: float
    """The coverage of the percentile interval, 0.95 (D30)."""
    verdict: Verdict | None
    """The interval's side, as :func:`~.comparison.paired_f1_difference` gives it."""
    undefined_resamples: int
    outcome: F1Outcome
    reason: str


@dataclass(frozen=True)
class StageTwo:
    """Stage 2 closed by the full rule, beside T19's provisional decision."""

    provisional: Decision
    """T19's decision, recomputed from the same candidates (the F1 step skipped)."""
    f1: F1Step
    decision: Decision | None
    """The full rule's decision; ``None`` when stage 2 is undefined."""
    undefined_reason: str | None
    sensitivities: tuple[Sensitivity, ...]
    """Stage 2 with NPMI at its sensitivity setting; a flip is reported, never used."""

    @property
    def status(self) -> Status:
        return "closed" if self.decision is not None else "undefined"

    @property
    def winner(self) -> str | None:
        """The winning model, or ``None`` when stage 2 is undefined."""
        return None if self.decision is None else self.decision.winner

    def outcome(self) -> Outcome:
        """``confirmed`` when the full rule names T19's provisional model, ``changed`` when it
        names the other one, ``undefined`` when it names none."""
        if self.decision is None:
            return "undefined"
        return "confirmed" if self.decision.winner == self.provisional.winner else "changed"


def f1_step(difference: F1Difference, names: tuple[str, str]) -> F1Step:
    """Return stage 2's F1 step for ``difference``, ΔF1 = F1(``names[0]``) − F1(``names[1]``)."""
    coverage = f"{difference.settings.interval:.0%}"
    lower, upper, observed = difference.lower, difference.upper, difference.difference
    outcome: F1Outcome
    if lower is None or upper is None or difference.verdict is None:
        outcome = "undefined"
        reason = (
            "the F1 step is undefined: ΔF1 has no interval, since no bootstrap resample has both "
            f"F1 values ({difference.undefined_resamples} undefined resamples), so no higher F1 "
            "is shown; D30 does not name this case"
        )
    elif difference.verdict == "tie":
        outcome = "tie"
        reason = (
            f"a tie: the {coverage} percentile interval of ΔF1, [{lower:.4f}, {upper:.4f}], "
            "holds or touches 0, so neither F1 is higher"
        )
    else:
        side: F1Outcome = "first" if difference.verdict == "first_higher" else "second"
        where = "above" if side == "first" else "below"
        bounds = f"[{lower:.4f}, {upper:.4f}]"
        if observed is not None and (observed > 0 if side == "first" else observed < 0):
            outcome = side
            winner, loser = names if side == "first" else (names[1], names[0])
            reason = (
                f"{winner} has the higher F1 and beats {loser}: the {coverage} percentile "
                f"interval of ΔF1, {bounds}, lies {where} 0 (ΔF1 {observed:.4f})"
            )
        else:
            outcome = "undefined"
            shown = "undefined" if observed is None else f"{observed:.4f}"
            reason = (
                f"the F1 step is undefined: the {coverage} percentile interval of ΔF1, {bounds}, "
                f"lies {where} 0, while the observed ΔF1 is {shown}, which is 0, undefined or on "
                "the opposite side, so the side the interval names is not the higher F1; D30 "
                "does not name this case"
            )
    return F1Step(
        first=names[0],
        second=names[1],
        first_f1=difference.first_f1,
        second_f1=difference.second_f1,
        difference=observed,
        lower=lower,
        upper=upper,
        interval=difference.settings.interval,
        verdict=difference.verdict,
        undefined_resamples=difference.undefined_resamples,
        outcome=outcome,
        reason=reason,
    )


def close_stage_two(
    first: Candidate,
    second: Candidate,
    difference: F1Difference,
    *,
    gates: Gates,
    fallback_model: str,
) -> StageTwo:
    """Return stage 2 of ``first`` and ``second``, the two models' stage-1 configurations in
    T19's order, closed with ``difference``, ΔF1 = F1(first's model) − F1(second's model).

    Raises:
        SelectionError: as
            :func:`~thematic_redundancy.topic_map.selection.stage_two_provisional` raises.
    """
    step = f1_step(difference, (first.representation, second.representation))
    provisional, decision = _close(first, second, step, gates=gates, fallback_model=fallback_model)
    _, under_sensitivity = _close(
        replace(first, npmi=first.npmi_sensitivity),
        replace(second, npmi=second.npmi_sensitivity),
        step,
        gates=gates,
        fallback_model=fallback_model,
    )
    primary = "undefined" if decision is None else decision.winner
    value = "undefined" if under_sensitivity is None else under_sensitivity.winner
    what = "Stage 2 with NPMI at its sensitivity"
    flipped = primary != value
    sensitivity = Sensitivity(
        name="npmi_sensitivity",
        primary=primary,
        value=value,
        flipped=flipped,
        reason=(
            f"{what} gives {value} instead of {primary}: a flip, reported; the rule is not changed"
            if flipped
            else f"{what} gives the same outcome, {primary}"
        ),
    )
    return StageTwo(
        provisional=provisional,
        f1=step,
        decision=decision,
        undefined_reason=None if decision is not None else step.reason,
        sensitivities=(sensitivity,),
    )


def _close(
    first: Candidate, second: Candidate, step: F1Step, *, gates: Gates, fallback_model: str
) -> tuple[Decision, Decision | None]:
    """T19's provisional decision, and the full rule's: T19's steps with the F1 step in place of
    the skipped one, up to the first step that decides; ``None`` when the F1 step is reached and
    undefined."""
    provisional = stage_two_provisional(first, second, gates=gates, fallback_model=fallback_model)
    gates_step, *after_gates = provisional.steps
    steps = [gates_step]
    if gates_step.outcome not in ("first", "second"):
        if step.outcome == "undefined":
            return provisional, None
        steps.append(
            Comparison(F1_STEP, "f1", step.first_f1, step.second_f1, step.outcome, step.reason)
        )
        if step.outcome == "tie":
            steps += after_gates[1:]  # T19's NPMI step, and its fallback step after a tie
    deciding = steps[-1]
    return provisional, Decision(
        first=provisional.first,
        second=provisional.second,
        winner=provisional.first if deciding.outcome == "first" else provisional.second,
        deciding_step=deciding.step,
        deciding_criterion=deciding.criterion,
        steps=tuple(steps),
        findings=(),
    )


def candidate_from_record(record: Mapping[str, Any]) -> Candidate:
    """Return the :class:`~thematic_redundancy.topic_map.selection.Candidate` that T19 wrote to
    ``selection.json`` as ``record`` (its fields, and ``gates_failed`` beside them).

    Raises:
        ValueError: if ``record`` is not a candidate's record.
    """
    names = {field.name for field in fields(Candidate)}
    given = set(record) - {"gates_failed"}
    if given != names:
        raise ValueError(
            f"the record is not a candidate's: it lacks {sorted(names - given)} and holds "
            f"{sorted(given - names)} besides"
        )
    try:
        return Candidate(**{name: _summary(record[name]) for name in names})
    except TypeError as error:
        raise ValueError(f"the record is not a candidate's: {error}") from error


def _summary(value: Any) -> Any:
    """A seed summary's record as a :class:`SeedSummary`; any other value as it is."""
    if isinstance(value, Mapping):
        if set(value) != {"values", "mean", "sd"}:
            raise TypeError(f"a seed summary holds values, mean and sd, not {sorted(value)}")
        return SeedSummary(values=tuple(value["values"]), mean=value["mean"], sd=value["sd"])
    return value
