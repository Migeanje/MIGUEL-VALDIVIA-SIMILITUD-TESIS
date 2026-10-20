"""The selection rules of the experiment (T19, D30 E2 and E3), as pure functions.

Every rule reads seed summaries: the values of one metric of one configuration, one per seed in
the order of ``seeds``, with their mean and sample SD (ddof 1, D30 E3). Nothing here clusters or
reads a file: the experiment runner gathers the summaries, calls these rules and records what
they return in ``selection.json`` (D30 E6), every decision with its reason.

- :func:`choose_k` (E2): K* is the K with the highest seed-mean silhouette; an exact tie goes to
  the smaller K. A K* at either end of the sweep is flagged as a finding (X2); the sweep is never
  extended.
- :func:`choose_min_cluster_size` (E2): among the sizes whose metrics are defined in every seed
  and whose mean noise share is at most the noise cap, the highest seed-mean silhouette; exact
  ties go to the lower mean noise share, then to the smaller size. If no size passes, HDBSCAN is
  ineligible for that model and is reported at its lowest-noise size.
- :func:`beats` (E3): A beats B on a seeded metric when ``mean_A - mean_B > max(SD_A, SD_B)`` in
  the better direction; otherwise the comparison is a tie.
- :func:`evaluate_gates` (E3): G1, the three internal metrics are defined in every seed; G2, the
  mean noise share is at most the cap; G3, the mean pairwise ARI is at least the floor.
- :func:`stage_one` (E3): the algorithm within one model's space; :func:`stage_two_provisional`
  (E3, UD3): the model, without the labeled-set F1 that T22 brings, so its winner is provisional.
- :func:`criterion_one_npmi` and :func:`criterion_four`: the NPMI part of acceptance criterion 1
  and criterion 4; :func:`matched_k` and :func:`representative_seed`.

The TF-IDF baseline is never a candidate: it enters :func:`criterion_one_npmi` only (UD1).

A comparison with an undefined value on either side is a tie: nothing beats an undefined metric,
and a configuration whose internal metrics are undefined in a seed already fails G1.
"""

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

Algorithm = Literal["kmeans", "hdbscan"]
Outcome = Literal["first", "second", "tie", "skipped"]
"""Which side a comparison favours; ``skipped`` when its data does not exist yet."""

GATE_NAMES = ("G1", "G2", "G3")


class SelectionError(ValueError):
    """The inputs of a rule break its contract, so nothing was decided."""


@dataclass(frozen=True)
class SeedSummary:
    """One metric of one configuration over the seeds, in the order of ``seeds``."""

    values: tuple[float | None, ...]
    """One value per seed; ``None`` where the metric is undefined in that seed."""
    mean: float | None
    """``None`` unless every value is defined."""
    sd: float | None
    """The sample SD (ddof 1); ``None`` unless every value is defined and there are two."""

    @property
    def defined(self) -> bool:
        return self.mean is not None


def summarize(values: Sequence[float | None]) -> SeedSummary:
    """Return the seed summary of ``values``."""
    kept = tuple(None if value is None else float(value) for value in values)
    defined = [value for value in kept if value is not None]
    if not kept or len(defined) < len(kept):
        return SeedSummary(values=kept, mean=None, sd=None)
    sd = statistics.stdev(defined) if len(defined) >= 2 else None
    return SeedSummary(values=kept, mean=statistics.fmean(defined), sd=sd)


def beats(first: SeedSummary, second: SeedSummary, *, higher_is_better: bool = True) -> bool:
    """Return whether ``first`` beats ``second``: the difference of their means, in the better
    direction, exceeds the larger of their SDs (D30 E3)."""
    if first.mean is None or second.mean is None or first.sd is None or second.sd is None:
        return False
    gap = first.mean - second.mean if higher_is_better else second.mean - first.mean
    return gap > max(first.sd, second.sd)


def compare_seeded(
    first: SeedSummary, second: SeedSummary, *, higher_is_better: bool = True
) -> Outcome:
    """Return ``first`` or ``second`` for the side that beats the other, else ``tie``."""
    if beats(first, second, higher_is_better=higher_is_better):
        return "first"
    if beats(second, first, higher_is_better=higher_is_better):
        return "second"
    return "tie"


@dataclass(frozen=True)
class SettingSummary:
    """One hyperparameter value of one configuration over the seeds."""

    parameter: int
    """K for K-means, ``min_cluster_size`` for HDBSCAN."""
    silhouette: SeedSummary
    noise_fraction: SeedSummary
    metrics_defined: bool
    """Whether the three internal metrics are defined in every seed."""


@dataclass(frozen=True)
class ParameterChoice:
    """The E2 choice of one configuration's hyperparameter."""

    rule: str
    parameter: int
    eligible: bool
    """``False`` when no value passes the rule's conditions (HDBSCAN only)."""
    at_bound: bool
    """Whether K* is the first or the last K of the sweep (K-means only)."""
    reason: str


def _check_parameters(settings: Sequence[SettingSummary], name: str) -> None:
    if not settings:
        raise SelectionError(f"there is no {name} value to choose from")
    parameters = [summary.parameter for summary in settings]
    if len(set(parameters)) != len(parameters):
        raise SelectionError(f"the {name} values repeat: {parameters}")


def choose_k(settings: Sequence[SettingSummary]) -> ParameterChoice:
    """Return K*, the K of the highest seed-mean silhouette; an exact tie goes to the smaller K.

    A K whose silhouette is undefined in a seed has no seed mean, so it cannot be K*.

    Raises:
        SelectionError: if ``settings`` is empty, repeats a K, or no K has a defined silhouette.
    """
    _check_parameters(settings, "K")
    ranked = sorted(
        (-summary.silhouette.mean, summary.parameter)
        for summary in settings
        if summary.silhouette.mean is not None
    )
    if not ranked:
        raise SelectionError("no K has a silhouette defined in every seed")
    best_mean, best = -ranked[0][0], ranked[0][1]
    bounds = (min(s.parameter for s in settings), max(s.parameter for s in settings))
    at_bound = best in bounds
    reason = (
        f"K {best} has the highest seed-mean silhouette, {best_mean:.6f}, of the {len(ranked)} "
        "values of K defined in every seed; an exact tie goes to the smaller K"
    )
    if at_bound:
        reason += (
            f"; it lies at a bound of the sweep ({bounds[0]} to {bounds[1]}), which is reported "
            "as a finding, and the sweep is not extended (D30 E2, X2)"
        )
    return ParameterChoice(
        rule="max_seed_mean_silhouette",
        parameter=best,
        eligible=True,
        at_bound=at_bound,
        reason=reason,
    )


def choose_min_cluster_size(
    settings: Sequence[SettingSummary], *, max_mean_noise_fraction: float
) -> ParameterChoice:
    """Return the chosen ``min_cluster_size`` (see the module), or the lowest-noise size,
    ineligible, when no size passes.

    Raises:
        SelectionError: if ``settings`` is empty or repeats a size.
    """
    _check_parameters(settings, "min_cluster_size")
    rule = "max_seed_mean_silhouette_under_noise_cap"
    passing = sorted(
        (-summary.silhouette.mean, summary.noise_fraction.mean, summary.parameter)
        for summary in settings
        if summary.metrics_defined
        and summary.silhouette.mean is not None
        and summary.noise_fraction.mean is not None
        and summary.noise_fraction.mean <= max_mean_noise_fraction
    )
    if passing:
        silhouette, _, best = passing[0]
        return ParameterChoice(
            rule=rule,
            parameter=best,
            eligible=True,
            at_bound=False,
            reason=(
                f"min_cluster_size {best} has the highest seed-mean silhouette, {-silhouette:.6f}, "
                f"of the {len(passing)} of {len(settings)} sizes whose metrics are defined in "
                f"every seed and whose mean noise share is at most {max_mean_noise_fraction}; "
                "exact ties go to the lower mean noise share, then to the smaller size"
            ),
        )
    noise, lowest = min(
        (math.inf if s.noise_fraction.mean is None else s.noise_fraction.mean, s.parameter)
        for s in settings
    )
    return ParameterChoice(
        rule=rule,
        parameter=lowest,
        eligible=False,
        at_bound=False,
        reason=(
            f"no size has its metrics defined in every seed and a mean noise share at most "
            f"{max_mean_noise_fraction}, so HDBSCAN is ineligible for this model; it is reported "
            f"at its lowest-noise size, {lowest} (mean noise share {noise:.6f}) (D30 E2)"
        ),
    )


@dataclass(frozen=True)
class Gates:
    """The eligibility thresholds of D30 E3 (``experiment.selection.gates``)."""

    min_mean_ari: float
    max_mean_noise_fraction: float


@dataclass(frozen=True)
class GateResult:
    metrics_defined: bool
    """G1."""
    noise_within_cap: bool
    """G2."""
    stable: bool
    """G3."""

    @property
    def failed(self) -> tuple[str, ...]:
        """The names of the gates that failed, in order."""
        passed = (self.metrics_defined, self.noise_within_cap, self.stable)
        return tuple(name for name, ok in zip(GATE_NAMES, passed, strict=True) if not ok)

    @property
    def passed(self) -> bool:
        return not self.failed


BASELINE_REPRESENTATION = "tfidf"
"""The representation of the TF-IDF baseline, which is never a candidate (UD1)."""


@dataclass(frozen=True)
class Candidate:
    """A configuration at its E2 hyperparameter, with the seed summaries the rules read."""

    configuration: str
    """C1 to C4, or B0 for the baseline."""
    representation: str
    """The embedding model id, or :data:`BASELINE_REPRESENTATION`."""
    algorithm: Algorithm
    parameter: int
    eligible: bool
    metrics_defined: bool
    silhouette: SeedSummary
    noise_fraction: SeedSummary
    clusters: SeedSummary
    npmi: SeedSummary
    """At the primary NPMI setting."""
    ari_mean: float | None
    """Mean pairwise ARI under the primary noise handling."""
    npmi_sensitivity: SeedSummary
    ari_mean_sensitivity: float | None
    silhouette_noise_reassigned: SeedSummary | None = None
    """HDBSCAN only: the silhouette with each noise thesis sent to its nearest centroid."""


def evaluate_gates(candidate: Candidate, gates: Gates) -> GateResult:
    """Return the three gates of ``candidate``."""
    noise = candidate.noise_fraction.mean
    return GateResult(
        metrics_defined=candidate.metrics_defined,
        noise_within_cap=noise is not None and noise <= gates.max_mean_noise_fraction,
        stable=candidate.ari_mean is not None and candidate.ari_mean >= gates.min_mean_ari,
    )


@dataclass(frozen=True)
class Comparison:
    """One step of a decision, with the values it compared."""

    step: int
    criterion: str
    first: Any
    second: Any
    outcome: Outcome
    reason: str


@dataclass(frozen=True)
class Decision:
    """The outcome of comparing two candidates step by step, up to the first decisive step."""

    first: str
    second: str
    winner: str
    deciding_step: int
    deciding_criterion: str
    steps: tuple[Comparison, ...]
    findings: tuple[str, ...]


@dataclass(frozen=True)
class Sensitivity:
    """An outcome recomputed under a sensitivity setting, which never changes the rule."""

    name: str
    primary: str
    value: str
    flipped: bool
    reason: str


Names = tuple[str, str]
Step = Callable[[], Comparison]


def _winner_name(outcome: Outcome, names: Names) -> str:
    return names[0] if outcome == "first" else names[1]


def _seeded_step(
    step: int,
    criterion: str,
    first: SeedSummary,
    second: SeedSummary,
    names: Names,
    *,
    higher_is_better: bool = True,
) -> Comparison:
    outcome = compare_seeded(first, second, higher_is_better=higher_is_better)
    if first.mean is None or second.mean is None or first.sd is None or second.sd is None:
        reason = "a tie: the metric is undefined on at least one side"
    else:
        gap, bar = abs(first.mean - second.mean), max(first.sd, second.sd)
        if outcome == "tie":
            reason = (
                f"a tie: the difference of the seed means, {gap:.6f}, does not exceed the larger "
                f"seed SD, {bar:.6f}"
            )
        else:
            loser = names[1] if outcome == "first" else names[0]
            reason = (
                f"{_winner_name(outcome, names)} beats {loser}: the difference of the seed means, "
                f"{gap:.6f}, exceeds the larger seed SD, {bar:.6f}"
            )
    return Comparison(
        step=step,
        criterion=criterion,
        first={"mean": first.mean, "sd": first.sd},
        second={"mean": second.mean, "sd": second.sd},
        outcome=outcome,
        reason=reason,
    )


def _scalar_step(
    step: int,
    criterion: str,
    first: float | None,
    second: float | None,
    names: Names,
    *,
    higher_is_better: bool,
) -> Comparison:
    direction = "higher" if higher_is_better else "lower"
    if first is None or second is None or first == second:
        outcome: Outcome = "tie"
        reason = f"a tie: the values are equal or undefined, so neither is {direction}"
    else:
        outcome = "first" if (first > second) == higher_is_better else "second"
        reason = f"{_winner_name(outcome, names)} has the {direction} value"
    return Comparison(step, criterion, first, second, outcome, reason)


def _fallback_step(step: int, values: Names, fallback: str, names: Names) -> Comparison:
    if fallback not in values:
        raise SelectionError(f"the fallback {fallback} is neither {values[0]} nor {values[1]}")
    outcome: Outcome = "first" if values[0] == fallback else "second"
    return Comparison(
        step,
        "fallback",
        values[0],
        values[1],
        outcome,
        f"a tie through every earlier step goes to the declared fallback, {fallback}",
    )


def _decide(names: Names, steps: Sequence[Step], findings: Sequence[str]) -> Decision:
    done: list[Comparison] = []
    for make in steps:
        comparison = make()
        done.append(comparison)
        if comparison.outcome in ("first", "second"):
            return Decision(
                first=names[0],
                second=names[1],
                winner=_winner_name(comparison.outcome, names),
                deciding_step=comparison.step,
                deciding_criterion=comparison.criterion,
                steps=tuple(done),
                findings=tuple(findings),
            )
    raise SelectionError("no step decided")  # the fallback step always decides


def _gate_findings(candidates: Sequence[Candidate], gates: Gates) -> list[str]:
    results = [(c.configuration, evaluate_gates(c, gates)) for c in candidates]
    findings = [f"{name} fails {', '.join(r.failed)}" for name, r in results if not r.passed]
    if all(not result.passed for _, result in results):
        findings.append("neither configuration passes every gate")
    return findings


def _eligibility_step(kmeans: Candidate, hdbscan: Candidate, names: Names) -> Comparison:
    if kmeans.eligible == hdbscan.eligible:
        outcome: Outcome = "tie"
        reason = "both configurations have a hyperparameter chosen by D30 E2"
    else:
        outcome = "first" if kmeans.eligible else "second"
        reason = f"{_winner_name(outcome, names)} is the only eligible configuration (D30 E2)"
    return Comparison(0, "eligibility", kmeans.eligible, hdbscan.eligible, outcome, reason)


def _gate_count_step(first: Candidate, second: Candidate, gates: Gates, names: Names) -> Comparison:
    a, b = evaluate_gates(first, gates).failed, evaluate_gates(second, gates).failed
    if len(a) == len(b):
        outcome: Outcome = "tie"
        reason = (
            "a tie: both pass every gate"
            if not a
            else f"a tie: each fails {len(a)} of the three gates, a finding"
        )
    else:
        outcome = "first" if len(a) < len(b) else "second"
        reason = f"{_winner_name(outcome, names)} fails fewer gates ({min(len(a), len(b))})"
    return Comparison(1, "gates", list(a), list(b), outcome, reason)


def stage_one(
    kmeans: Candidate, hdbscan: Candidate, *, gates: Gates, fallback: Algorithm
) -> Decision:
    """Return the algorithm within one model's space (D30 E3, stage 1).

    Steps, in order, up to the first that is not a tie: (0) eligibility; (1) the gates, fewer
    failures win; (2) the silhouette and (3) NPMI, by :func:`beats`; (4) the higher mean ARI;
    (5) the lower mean noise share; (6) the ``fallback`` algorithm.

    Raises:
        SelectionError: if the candidates are not one K-means and one HDBSCAN of one model.
    """
    if (kmeans.algorithm, hdbscan.algorithm) != ("kmeans", "hdbscan"):
        raise SelectionError("stage 1 compares one K-means and one HDBSCAN configuration")
    if kmeans.representation != hdbscan.representation:
        raise SelectionError("stage 1 compares two configurations of one model")
    if kmeans.representation == BASELINE_REPRESENTATION:
        raise SelectionError("the TF-IDF baseline is never a candidate (UD1)")
    names = (kmeans.configuration, hdbscan.configuration)
    findings = _gate_findings((kmeans, hdbscan), gates)
    findings += [
        f"{c.configuration} is ineligible (D30 E2)" for c in (kmeans, hdbscan) if not c.eligible
    ]
    steps: list[Step] = [
        lambda: _eligibility_step(kmeans, hdbscan, names),
        lambda: _gate_count_step(kmeans, hdbscan, gates, names),
        lambda: _seeded_step(2, "silhouette", kmeans.silhouette, hdbscan.silhouette, names),
        lambda: _seeded_step(3, "npmi", kmeans.npmi, hdbscan.npmi, names),
        lambda: _scalar_step(
            4, "mean_ari", kmeans.ari_mean, hdbscan.ari_mean, names, higher_is_better=True
        ),
        lambda: _scalar_step(
            5,
            "noise_fraction",
            kmeans.noise_fraction.mean,
            hdbscan.noise_fraction.mean,
            names,
            higher_is_better=False,
        ),
        lambda: _fallback_step(6, ("kmeans", "hdbscan"), fallback, names),
    ]
    return _decide(names, steps, findings)


def _sensitivity(name: str, primary: str, value: str, what: str) -> Sensitivity:
    flipped = primary != value
    reason = (
        f"{what} gives {value} instead of {primary}: a flip, reported; the rule is not changed"
        if flipped
        else f"{what} gives the same outcome, {primary}"
    )
    return Sensitivity(name=name, primary=primary, value=value, flipped=flipped, reason=reason)


def stage_one_sensitivities(
    kmeans: Candidate, hdbscan: Candidate, *, gates: Gates, fallback: Algorithm
) -> tuple[Sensitivity, ...]:
    """Return stage 1 under the NPMI and ARI sensitivity settings, and step 2 under the
    reassigned-noise silhouette; a flip is reported, and the rule is not changed (D30 E3, E4)."""
    primary = stage_one(kmeans, hdbscan, gates=gates, fallback=fallback).winner
    npmi = stage_one(
        replace(kmeans, npmi=kmeans.npmi_sensitivity),
        replace(hdbscan, npmi=hdbscan.npmi_sensitivity),
        gates=gates,
        fallback=fallback,
    ).winner
    ari = stage_one(
        replace(kmeans, ari_mean=kmeans.ari_mean_sensitivity),
        replace(hdbscan, ari_mean=hdbscan.ari_mean_sensitivity),
        gates=gates,
        fallback=fallback,
    ).winner
    results = [
        _sensitivity("npmi_sensitivity", primary, npmi, "Stage 1 with NPMI at its sensitivity"),
        _sensitivity("ari_sensitivity", primary, ari, "Stage 1 with the sensitivity ARI handling"),
    ]
    if hdbscan.silhouette_noise_reassigned is not None:
        results.append(
            _sensitivity(
                "silhouette_noise_reassigned",
                compare_seeded(kmeans.silhouette, hdbscan.silhouette),
                compare_seeded(kmeans.silhouette, hdbscan.silhouette_noise_reassigned),
                "Step 2 with the HDBSCAN silhouette over every thesis, each noise thesis sent to "
                "its nearest centroid by cosine (D30 E5),",
            )
        )
    return tuple(results)


def _check_models(first: Candidate, second: Candidate, fallback_model: str) -> None:
    if BASELINE_REPRESENTATION in (first.representation, second.representation):
        raise SelectionError("the TF-IDF baseline is never a candidate (UD1)")
    if first.representation == second.representation:
        raise SelectionError("stage 2 compares the configurations of two models")
    if fallback_model not in (first.representation, second.representation):
        raise SelectionError("the fallback model is neither of the two models")


def _gate_pass_step(first: Candidate, second: Candidate, gates: Gates, names: Names) -> Comparison:
    a, b = evaluate_gates(first, gates), evaluate_gates(second, gates)
    if a.passed == b.passed:
        outcome: Outcome = "tie"
        reason = "a tie: both pass every gate" if a.passed else "a tie: both fail a gate"
    else:
        outcome = "first" if a.passed else "second"
        reason = f"{_winner_name(outcome, names)} alone passes every gate"
    return Comparison(1, "gates", list(a.failed), list(b.failed), outcome, reason)


def stage_two_provisional(
    first: Candidate, second: Candidate, *, gates: Gates, fallback_model: str
) -> Decision:
    """Return the model (D30 E3, stage 2) without the labeled-set F1: (1) a candidate that passes
    every gate over one that does not; (2) F1, skipped until T22; (3) NPMI, by :func:`beats`;
    (4) the ``fallback_model``. The winner is provisional (UD3).

    Raises:
        SelectionError: if the two candidates share a model, a candidate is the baseline, or
            ``fallback_model`` is neither model.
    """
    _check_models(first, second, fallback_model)
    names = (first.representation, second.representation)
    findings = [
        "provisional: stage 2 ran without the labeled-set F1 (step 2), which waits for the "
        "labeled set (U3, T21); T22 closes stage 2 by the full rule (UD3)"
    ]
    steps: list[Step] = [
        lambda: _gate_pass_step(first, second, gates, names),
        lambda: Comparison(
            2,
            "f1",
            None,
            None,
            "skipped",
            "skipped: there is no labeled set yet; T22 compares the F1 by a paired bootstrap",
        ),
        lambda: _seeded_step(3, "npmi", first.npmi, second.npmi, names),
        lambda: _fallback_step(4, names, fallback_model, names),
    ]
    return _decide(names, steps, findings)


def stage_two_sensitivities(
    first: Candidate, second: Candidate, *, gates: Gates, fallback_model: str
) -> tuple[Sensitivity, ...]:
    """Return stage 2 under the NPMI sensitivity setting."""
    primary = stage_two_provisional(first, second, gates=gates, fallback_model=fallback_model)
    value = stage_two_provisional(
        replace(first, npmi=first.npmi_sensitivity),
        replace(second, npmi=second.npmi_sensitivity),
        gates=gates,
        fallback_model=fallback_model,
    )
    what = "Stage 2 with NPMI at its sensitivity"
    return (_sensitivity("npmi_sensitivity", primary.winner, value.winner, what),)


def matched_k(winner: Candidate) -> int:
    """Return the baseline's matched K: the winner's K* when it is K-means, else the rounded
    mean number of HDBSCAN clusters over its seeds (D30 E3).

    Raises:
        SelectionError: if the winner is the baseline, or its cluster counts are undefined.
    """
    if winner.representation == BASELINE_REPRESENTATION:
        raise SelectionError("the matched K comes from the semantic winner, not the baseline")
    if winner.algorithm == "kmeans":
        return winner.parameter
    if winner.clusters.mean is None:
        raise SelectionError(f"{winner.configuration} has no mean number of clusters")
    return math.floor(winner.clusters.mean + 0.5)


@dataclass(frozen=True)
class CriterionOne:
    """The NPMI part of acceptance criterion 1 (UD1); its F1 part waits for T22."""

    own_k: Comparison
    matched_k: Comparison
    npmi_met: bool
    f1: str
    met: bool | None
    """``False`` when the NPMI part fails; ``None`` while F1 is pending."""
    reason: str


def criterion_one_npmi(winner: Candidate, baseline: Candidate, matched: Candidate) -> CriterionOne:
    """Return whether the winner beats the baseline on NPMI at the baseline's K* and at the
    matched K, both by :func:`beats`."""
    if (baseline.representation, matched.representation) != (BASELINE_REPRESENTATION,) * 2:
        raise SelectionError("criterion 1 compares the winner with the TF-IDF baseline")
    own = _seeded_step(
        1,
        "npmi_at_baseline_k",
        winner.npmi,
        baseline.npmi,
        (winner.configuration, f"{baseline.configuration} at K {baseline.parameter}"),
    )
    at_matched = _seeded_step(
        2,
        "npmi_at_matched_k",
        winner.npmi,
        matched.npmi,
        (winner.configuration, f"{matched.configuration} at K {matched.parameter}"),
    )
    npmi_met = own.outcome == at_matched.outcome == "first"
    return CriterionOne(
        own_k=own,
        matched_k=at_matched,
        npmi_met=npmi_met,
        f1="pending: the labeled-set F1 comparison waits for T22",
        met=None if npmi_met else False,
        reason=(
            "the winner beats the baseline on NPMI at both K; criterion 1 waits for the F1 of T22"
            if npmi_met
            else "the winner does not beat the baseline on NPMI at both K, so criterion 1 is "
            "not met, a finding"
        ),
    )


def coefficient_of_variation(summary: SeedSummary) -> float | None:
    """Return ``SD / |mean|``, or ``None`` when it is undefined or the mean is 0."""
    if summary.mean is None or summary.sd is None or summary.mean == 0:
        return None
    return summary.sd / abs(summary.mean)


@dataclass(frozen=True)
class CriterionFour:
    """Acceptance criterion 4: G3, and a coefficient of variation at most the cap for the
    silhouette and for NPMI."""

    ari_mean: float | None
    stable: bool
    cv_silhouette: float | None
    cv_npmi: float | None
    max_cv: float
    met: bool
    reason: str


def criterion_four(winner: Candidate, *, gates: Gates, max_cv: float) -> CriterionFour:
    """Return acceptance criterion 4 for the winner."""
    stable = evaluate_gates(winner, gates).stable
    cv_silhouette = coefficient_of_variation(winner.silhouette)
    cv_npmi = coefficient_of_variation(winner.npmi)
    within = [cv is not None and cv <= max_cv for cv in (cv_silhouette, cv_npmi)]
    words = ["at most" if ok else "above, or undefined against," for ok in within]
    return CriterionFour(
        ari_mean=winner.ari_mean,
        stable=stable,
        cv_silhouette=cv_silhouette,
        cv_npmi=cv_npmi,
        max_cv=max_cv,
        met=stable and all(within),
        reason=(
            f"G3 {'passes' if stable else 'fails'}; the coefficient of variation is {words[0]} "
            f"{max_cv} for the silhouette and {words[1]} {max_cv} for NPMI"
        ),
    )


@dataclass(frozen=True)
class Representative:
    """The winner's representative run: the seed run with the highest mean ARI to the others."""

    seed: int
    mean_ari_to_others: tuple[float, ...]
    """One per seed, in the order of ``seeds``."""
    reason: str


def representative_seed(
    matrix: Sequence[Sequence[float | None]], seeds: Sequence[int]
) -> Representative:
    """Return the seed whose run has the highest mean ARI to the other runs; a tie goes to the
    seed listed first.

    Raises:
        SelectionError: if the matrix is not square over the seeds, or a value is undefined.
    """
    size = len(seeds)
    if size < 2 or len(matrix) != size or any(len(row) != size for row in matrix):
        raise SelectionError(f"the ARI matrix must be {size} × {size}, one row per seed")
    means = []
    for index, row in enumerate(matrix):
        others = [row[column] for column in range(size) if column != index]
        values = [value for value in others if value is not None]
        if len(values) < len(others):
            raise SelectionError("an ARI value of the winner's runs is undefined")
        means.append(statistics.fmean(values))
    best = max(range(size), key=lambda index: (means[index], -index))
    return Representative(
        seed=seeds[best],
        mean_ari_to_others=tuple(means),
        reason=(
            f"seed {seeds[best]} has the highest mean ARI to the other runs, {means[best]:.6f}; "
            "a tie goes to the seed listed first"
        ),
    )
