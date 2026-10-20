"""The selection rules of T19 (D30 E2 and E3) on hand-built per-seed metric tables.

Seed values follow the pattern ``(m - d, m - d, m, m + d, m + d)``, whose mean is ``m`` and whose
sample SD (ddof 1) is exactly ``d``; the population SD would be ``0.894 d``. With ``d`` a power of
two, both are exact in floating point, so the boundaries of the rules can be tested exactly.
"""

import json
import math
import statistics
from dataclasses import asdict

import pytest

from thematic_redundancy.topic_map.selection import (
    Candidate,
    Gates,
    SelectionError,
    SettingSummary,
    beats,
    choose_k,
    choose_min_cluster_size,
    coefficient_of_variation,
    compare_seeded,
    criterion_four,
    criterion_one_npmi,
    evaluate_gates,
    matched_k,
    representative_seed,
    stage_one,
    stage_one_sensitivities,
    stage_two_provisional,
    stage_two_sensitivities,
    summarize,
)

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
GATES = Gates(min_mean_ari=0.65, max_mean_noise_fraction=0.30)
SEEDS = (7, 13, 21, 42, 73)


def spread(mean: float, sd: float = 0.0) -> tuple[float, ...]:
    """Five seed values of exactly this mean and sample SD."""
    return (mean - sd, mean - sd, mean, mean + sd, mean + sd)


NO_NOISE = spread(0.0)
SILHOUETTE = spread(0.30, 0.0078125)
NPMI = spread(0.10, 0.0078125)
TEN_CLUSTERS = spread(10.0)


def setting(parameter: int, silhouette, noise=NO_NOISE, defined: bool = True) -> SettingSummary:
    return SettingSummary(
        parameter=parameter,
        silhouette=summarize(silhouette),
        noise_fraction=summarize(noise),
        metrics_defined=defined,
    )


def candidate(
    configuration: str = "C1",
    algorithm: str = "kmeans",
    *,
    representation: str = MINILM,
    parameter: int = 10,
    silhouette=SILHOUETTE,
    npmi=NPMI,
    noise=NO_NOISE,
    clusters=TEN_CLUSTERS,
    ari: float | None = 0.9,
    npmi_sensitivity=None,
    ari_sensitivity: float | None = None,
    reassigned=None,
    eligible: bool = True,
    metrics_defined: bool = True,
) -> Candidate:
    return Candidate(
        configuration=configuration,
        representation=representation,
        algorithm=algorithm,
        parameter=parameter,
        eligible=eligible,
        metrics_defined=metrics_defined,
        silhouette=summarize(silhouette),
        noise_fraction=summarize(noise),
        clusters=summarize(clusters),
        npmi=summarize(npmi),
        ari_mean=ari,
        npmi_sensitivity=summarize(npmi if npmi_sensitivity is None else npmi_sensitivity),
        ari_mean_sensitivity=ari if ari_sensitivity is None else ari_sensitivity,
        silhouette_noise_reassigned=None if reassigned is None else summarize(reassigned),
    )


def hdbscan(**kwargs) -> Candidate:
    kwargs.setdefault("noise", spread(0.10))
    kwargs.setdefault("parameter", 10)
    return candidate("C2", "hdbscan", **kwargs)


# Seed summaries and "A beats B"


def test_a_summary_takes_the_mean_and_the_sample_sd() -> None:
    summary = summarize((2.5, 2.5, 3.0, 3.5, 3.5))

    assert summary.mean == 3.0
    assert summary.sd == 0.5  # ddof 1; the population SD would be 0.447
    assert summary.sd == statistics.stdev(summary.values)
    assert summary.defined


def test_a_summary_with_an_undefined_seed_has_no_mean_and_no_sd() -> None:
    summary = summarize((0.2, None, 0.3, 0.3, 0.3))

    assert (summary.mean, summary.sd, summary.defined) == (None, None, False)
    assert summary.values == (0.2, None, 0.3, 0.3, 0.3)


def test_a_beats_b_when_the_gap_exceeds_the_larger_sd() -> None:
    a, b = summarize(spread(3.5, 0.5)), summarize(spread(2.875))

    assert beats(a, b)
    assert compare_seeded(a, b) == "first"
    assert compare_seeded(b, a) == "second"


def test_a_gap_equal_to_the_larger_sd_is_a_tie() -> None:
    a, b = summarize(spread(3.5, 0.5)), summarize(spread(3.0))

    assert not beats(a, b)
    assert compare_seeded(a, b) == "tie"


def test_the_bar_is_the_larger_of_the_two_sds() -> None:
    # The gap 0.375 exceeds a's SD (0.25) but not b's (0.5).
    a, b = summarize(spread(3.375, 0.25)), summarize(spread(3.0, 0.5))

    assert not beats(a, b)
    assert compare_seeded(a, b) == "tie"


def test_the_bar_uses_the_sample_sd_not_the_population_sd() -> None:
    # The gap 0.46875 is below the sample SD 0.5 and above the population SD 0.447.
    a, b = summarize(spread(3.46875, 0.5)), summarize(spread(3.0))

    assert not beats(a, b)


def test_lower_is_better_reverses_the_direction() -> None:
    low, high = summarize(spread(0.10)), summarize(spread(0.50, 0.125))

    assert beats(low, high, higher_is_better=False)
    assert not beats(high, low, higher_is_better=False)
    assert compare_seeded(low, high, higher_is_better=False) == "first"


def test_a_comparison_with_an_undefined_side_is_a_tie() -> None:
    defined, undefined = summarize(spread(0.9)), summarize((None, 0.1, 0.1, 0.1, 0.1))

    assert not beats(defined, undefined)
    assert compare_seeded(defined, undefined) == "tie"
    assert compare_seeded(undefined, defined) == "tie"


# E2: K* and min_cluster_size*


def test_k_star_is_the_highest_seed_mean_silhouette() -> None:
    # K = 10 has the best single seed, but K = 12 the best mean.
    choice = choose_k(
        [
            setting(8, spread(0.20)),
            setting(10, (0.90, 0.10, 0.10, 0.10, 0.10)),
            setting(12, spread(0.30)),
            setting(14, spread(0.25)),
        ]
    )

    assert choice.parameter == 12
    assert choice.eligible
    assert not choice.at_bound
    assert "12" in choice.reason


def test_an_exact_tie_goes_to_the_smaller_k() -> None:
    choice = choose_k(
        [setting(10, spread(0.30)), setting(12, spread(0.30)), setting(14, (0.1,) * 5)]
    )

    assert choice.parameter == 10


def test_an_exact_tie_goes_to_the_smaller_k_whatever_the_order() -> None:
    choice = choose_k(
        [setting(14, (0.1,) * 5), setting(12, spread(0.30)), setting(10, spread(0.30))]
    )

    assert choice.parameter == 10


@pytest.mark.parametrize(("best", "at_bound"), [(8, True), (50, True), (30, False)])
def test_a_k_star_at_either_end_of_the_sweep_is_flagged(best: int, at_bound: bool) -> None:
    settings = [setting(k, spread(0.5 if k == best else 0.2)) for k in range(8, 51, 2)]

    choice = choose_k(settings)

    assert choice.parameter == best
    assert choice.at_bound is at_bound
    assert ("bound" in choice.reason) is at_bound


def test_a_k_undefined_in_one_seed_cannot_be_k_star() -> None:
    choice = choose_k([setting(8, (0.9, 0.9, 0.9, 0.9, None)), setting(10, spread(0.3))])

    assert choice.parameter == 10


def test_k_star_needs_a_k_with_a_defined_silhouette() -> None:
    with pytest.raises(SelectionError):
        choose_k([setting(8, (None,) * 5)])
    with pytest.raises(SelectionError):
        choose_k([])
    with pytest.raises(SelectionError):
        choose_k([setting(8, spread(0.3)), setting(8, spread(0.4))])


def test_the_size_with_the_best_silhouette_under_the_noise_cap_wins() -> None:
    choice = choose_min_cluster_size(
        [
            setting(5, spread(0.60), noise=spread(0.40)),  # best silhouette, above the cap
            setting(10, spread(0.40), noise=spread(0.20)),
            setting(15, spread(0.35), noise=spread(0.10)),
        ],
        max_mean_noise_fraction=0.30,
    )

    assert choice.parameter == 10
    assert choice.eligible


def test_a_mean_noise_share_at_the_cap_passes() -> None:
    choice = choose_min_cluster_size(
        [setting(5, spread(0.60), noise=(0.30,) * 5), setting(10, spread(0.40))],
        max_mean_noise_fraction=0.30,
    )

    assert choice.parameter == 5


def test_a_size_with_metrics_undefined_in_a_seed_is_skipped() -> None:
    choice = choose_min_cluster_size(
        [setting(5, spread(0.60), defined=False), setting(10, spread(0.40))],
        max_mean_noise_fraction=0.30,
    )

    assert choice.parameter == 10


def test_size_ties_go_to_the_lower_noise_then_to_the_smaller_size() -> None:
    by_noise = choose_min_cluster_size(
        [setting(5, spread(0.4), noise=spread(0.2)), setting(10, spread(0.4), noise=spread(0.1))],
        max_mean_noise_fraction=0.30,
    )
    by_size = choose_min_cluster_size(
        [setting(10, spread(0.4), noise=spread(0.1)), setting(5, spread(0.4), noise=spread(0.1))],
        max_mean_noise_fraction=0.30,
    )

    assert by_noise.parameter == 10
    assert by_size.parameter == 5


def test_no_size_under_the_cap_makes_hdbscan_ineligible_at_its_lowest_noise_size() -> None:
    choice = choose_min_cluster_size(
        [
            setting(5, spread(0.6), noise=spread(0.5)),
            setting(10, spread(0.2), noise=spread(0.35)),
            setting(15, spread(0.5), noise=spread(0.35)),
            setting(20, spread(0.4), noise=spread(0.45)),
        ],
        max_mean_noise_fraction=0.30,
    )

    assert choice.parameter == 10  # the lowest noise, a tie going to the smaller size
    assert not choice.eligible
    assert "ineligible" in choice.reason


# E3: the gates


def test_each_gate_fails_on_its_own() -> None:
    assert evaluate_gates(candidate(), GATES).passed
    assert evaluate_gates(candidate(metrics_defined=False), GATES).failed == ("G1",)
    assert evaluate_gates(candidate(noise=spread(0.31)), GATES).failed == ("G2",)
    assert evaluate_gates(candidate(ari=0.64), GATES).failed == ("G3",)
    assert evaluate_gates(candidate(ari=None), GATES).failed == ("G3",)


def test_the_gate_thresholds_are_inclusive() -> None:
    gates = evaluate_gates(candidate(noise=(0.30,) * 5, ari=0.65), GATES)

    assert gates.passed


# E3: stage 1


def test_stage_one_a_configuration_that_alone_passes_every_gate_wins() -> None:
    # K-means fails G3 while it has the better silhouette.
    decision = stage_one(
        candidate(silhouette=spread(0.9), ari=0.5), hdbscan(), gates=GATES, fallback="kmeans"
    )

    assert decision.winner == "C2"
    assert decision.deciding_criterion == "gates"
    assert decision.steps[-1].second == []
    assert decision.steps[-1].first == ["G3"]
    assert any("C1 fails G3" in finding for finding in decision.findings)


def test_stage_one_when_neither_passes_the_one_failing_fewer_goes_on() -> None:
    decision = stage_one(
        candidate(ari=0.5),
        hdbscan(silhouette=spread(0.9), noise=spread(0.5), ari=0.5),
        gates=GATES,
        fallback="kmeans",
    )

    assert decision.winner == "C1"
    assert decision.deciding_criterion == "gates"
    assert any("neither" in finding for finding in decision.findings)


def test_stage_one_an_equal_gate_failure_continues_to_the_silhouette() -> None:
    decision = stage_one(
        candidate(ari=0.5), hdbscan(silhouette=spread(0.9), ari=0.5), gates=GATES, fallback="kmeans"
    )

    assert decision.winner == "C2"
    assert decision.deciding_criterion == "silhouette"


def test_stage_one_the_silhouette_decides_before_npmi() -> None:
    decision = stage_one(
        candidate(silhouette=spread(0.30), npmi=spread(0.50)),
        hdbscan(silhouette=spread(0.60), npmi=spread(0.10)),
        gates=GATES,
        fallback="kmeans",
    )

    assert decision.winner == "C2"
    assert [step.criterion for step in decision.steps] == ["eligibility", "gates", "silhouette"]
    assert decision.deciding_step == 2
    assert decision.steps[-1].first == {"mean": 0.30, "sd": 0.0}
    assert decision.steps[-1].second == {"mean": 0.60, "sd": 0.0}


def test_stage_one_npmi_decides_a_silhouette_tie() -> None:
    decision = stage_one(
        candidate(silhouette=spread(0.30, 0.125), npmi=spread(0.10)),
        hdbscan(silhouette=spread(0.35, 0.125), npmi=spread(0.20)),
        gates=GATES,
        fallback="kmeans",
    )

    assert decision.winner == "C2"
    assert decision.deciding_criterion == "npmi"
    assert decision.steps[2].outcome == "tie"


def test_stage_one_the_higher_mean_ari_decides_a_npmi_tie() -> None:
    decision = stage_one(candidate(ari=0.80), hdbscan(ari=0.85), gates=GATES, fallback="kmeans")

    assert decision.winner == "C2"
    assert decision.deciding_step == 4
    assert decision.deciding_criterion == "mean_ari"


def test_stage_one_the_lower_mean_noise_decides_an_ari_tie() -> None:
    decision = stage_one(candidate(), hdbscan(), gates=GATES, fallback="hdbscan")

    assert decision.winner == "C1"  # K-means leaves no noise
    assert decision.deciding_criterion == "noise_fraction"


@pytest.mark.parametrize(("fallback", "winner"), [("kmeans", "C1"), ("hdbscan", "C2")])
def test_stage_one_an_exact_tie_goes_to_the_fallback(fallback: str, winner: str) -> None:
    decision = stage_one(candidate(), hdbscan(noise=spread(0.0)), gates=GATES, fallback=fallback)

    assert decision.winner == winner
    assert decision.deciding_step == 6
    assert decision.deciding_criterion == "fallback"
    assert [step.outcome for step in decision.steps[:-1]] == ["tie"] * 6


def test_stage_one_an_ineligible_hdbscan_loses_whatever_its_numbers() -> None:
    decision = stage_one(
        candidate(ari=0.1),
        hdbscan(silhouette=spread(0.9), noise=spread(0.5), eligible=False),
        gates=GATES,
        fallback="hdbscan",
    )

    assert decision.winner == "C1"
    assert decision.deciding_step == 0
    assert any("ineligible" in finding for finding in decision.findings)


def test_stage_one_uses_the_primary_ari_handling() -> None:
    # Under the sensitivity handling, HDBSCAN would fail G3.
    km = candidate()
    hdb = hdbscan(silhouette=spread(0.9), ari=0.70, ari_sensitivity=0.60)

    decision = stage_one(km, hdb, gates=GATES, fallback="kmeans")
    flips = {s.name: s for s in stage_one_sensitivities(km, hdb, gates=GATES, fallback="kmeans")}

    assert decision.winner == "C2"
    assert flips["ari_sensitivity"].value == "C1"
    assert flips["ari_sensitivity"].flipped


def test_stage_one_uses_the_primary_npmi_and_reports_a_flip_under_the_sensitivity() -> None:
    km = candidate(npmi=spread(0.10), npmi_sensitivity=spread(0.30))
    hdb = hdbscan(npmi=spread(0.20), npmi_sensitivity=spread(0.10))

    decision = stage_one(km, hdb, gates=GATES, fallback="kmeans")
    flips = {s.name: s for s in stage_one_sensitivities(km, hdb, gates=GATES, fallback="kmeans")}

    assert decision.winner == "C2"
    assert (flips["npmi_sensitivity"].primary, flips["npmi_sensitivity"].value) == ("C2", "C1")
    assert flips["npmi_sensitivity"].flipped
    assert not flips["ari_sensitivity"].flipped


def test_stage_one_reports_a_step_two_flip_under_the_reassigned_silhouette() -> None:
    km = candidate(silhouette=spread(0.30))
    hdb = hdbscan(silhouette=spread(0.50), reassigned=spread(0.20))

    decision = stage_one(km, hdb, gates=GATES, fallback="kmeans")
    flips = {s.name: s for s in stage_one_sensitivities(km, hdb, gates=GATES, fallback="kmeans")}

    assert decision.winner == "C2"  # the rule is not changed
    sensitivity = flips["silhouette_noise_reassigned"]
    assert (sensitivity.primary, sensitivity.value, sensitivity.flipped) == (
        "second",
        "first",
        True,
    )


def test_stage_one_refuses_candidates_of_two_models_or_two_k_means() -> None:
    with pytest.raises(SelectionError):
        stage_one(candidate(), hdbscan(representation=MPNET), gates=GATES, fallback="kmeans")
    with pytest.raises(SelectionError):
        stage_one(candidate(), candidate("C2"), gates=GATES, fallback="kmeans")


def test_a_decision_is_plain_json_data_that_repeats() -> None:
    # selection.json records every comparison with its values (D30 E3, E6).
    km, hdb = candidate(ari=0.5), hdbscan(silhouette=spread(0.9), noise=spread(0.5))

    first = json.dumps(asdict(stage_one(km, hdb, gates=GATES, fallback="kmeans")))
    again = json.dumps(asdict(stage_one(km, hdb, gates=GATES, fallback="kmeans")))

    assert first == again
    record = json.loads(first)
    assert record["winner"] == "C2"
    assert record["deciding_criterion"] == "silhouette"
    assert record["steps"][1] == {
        "step": 1,
        "criterion": "gates",
        "first": ["G3"],
        "second": ["G2"],
        "outcome": "tie",
        "reason": "a tie: each fails 1 of the three gates, a finding",
    }


# E3: stage 2, provisional


def test_stage_two_a_model_that_alone_passes_every_gate_wins() -> None:
    decision = stage_two_provisional(
        candidate(npmi=spread(0.9), ari=0.5),
        candidate("C3", representation=MPNET),
        gates=GATES,
        fallback_model=MINILM,
    )

    assert decision.winner == MPNET
    assert decision.deciding_criterion == "gates"


def test_stage_two_both_failing_is_a_gate_tie_whatever_the_counts() -> None:
    # Stage 1 counts the failures; stage 2 asks only whether each passes every gate.
    decision = stage_two_provisional(
        candidate(npmi=spread(0.9), ari=0.5, noise=spread(0.5)),
        candidate("C3", representation=MPNET, ari=0.5),
        gates=GATES,
        fallback_model=MPNET,
    )

    assert decision.winner == MINILM
    assert decision.deciding_criterion == "npmi"


def test_stage_two_skips_f1_and_lets_npmi_decide() -> None:
    decision = stage_two_provisional(
        candidate(npmi=spread(0.10)),
        candidate("C4", "hdbscan", representation=MPNET, npmi=spread(0.20), noise=spread(0.1)),
        gates=GATES,
        fallback_model=MINILM,
    )

    assert decision.winner == MPNET
    assert [step.criterion for step in decision.steps] == ["gates", "f1", "npmi"]
    assert decision.steps[1].outcome == "skipped"
    assert "T22" in decision.steps[1].reason
    assert any("provisional" in finding for finding in decision.findings)


@pytest.mark.parametrize("fallback", [MINILM, MPNET])
def test_stage_two_a_npmi_tie_goes_to_the_fallback_model(fallback: str) -> None:
    decision = stage_two_provisional(
        candidate(), candidate("C3", representation=MPNET), gates=GATES, fallback_model=fallback
    )

    assert decision.winner == fallback
    assert decision.deciding_criterion == "fallback"


def test_stage_two_never_takes_the_baseline_as_a_candidate() -> None:
    baseline = candidate("B0", representation="tfidf", npmi=spread(0.9))

    with pytest.raises(SelectionError):
        stage_two_provisional(baseline, candidate(), gates=GATES, fallback_model=MINILM)
    with pytest.raises(SelectionError):
        stage_two_provisional(candidate(), baseline, gates=GATES, fallback_model=MINILM)


def test_stage_two_refuses_a_fallback_model_that_is_neither_model() -> None:
    with pytest.raises(SelectionError):
        stage_two_provisional(
            candidate(), candidate("C3", representation=MPNET), gates=GATES, fallback_model="x"
        )


def test_stage_two_reports_a_flip_under_the_npmi_sensitivity() -> None:
    first = candidate(npmi=spread(0.10), npmi_sensitivity=spread(0.40))
    second = candidate("C3", representation=MPNET, npmi=spread(0.20), npmi_sensitivity=spread(0.1))

    (sensitivity,) = stage_two_sensitivities(first, second, gates=GATES, fallback_model=MINILM)

    assert (sensitivity.primary, sensitivity.value, sensitivity.flipped) == (MPNET, MINILM, True)


# The matched K, the acceptance criteria and the representative run


def test_the_matched_k_of_a_k_means_winner_is_its_k_star() -> None:
    assert matched_k(candidate(parameter=24, clusters=spread(24.0))) == 24


@pytest.mark.parametrize(
    ("clusters", "expected"),
    [((11.0, 12.0, 12.0, 13.0, 13.0), 12), ((12.0, 12.0, 13.0, 13.0, 13.0), 13)],
)
def test_the_matched_k_of_an_hdbscan_winner_is_its_rounded_mean_cluster_count(
    clusters: tuple[float, ...], expected: int
) -> None:
    assert matched_k(hdbscan(parameter=15, clusters=clusters)) == expected


def test_the_baseline_has_no_matched_k_of_its_own() -> None:
    with pytest.raises(SelectionError):
        matched_k(candidate("B0", representation="tfidf"))


def test_criterion_one_needs_the_npmi_win_at_both_baseline_ks() -> None:
    winner = candidate(npmi=spread(0.30))
    own = candidate("B0", representation="tfidf", parameter=20, npmi=spread(0.10))
    matched_low = candidate("B0", representation="tfidf", parameter=10, npmi=spread(0.10))
    matched_high = candidate("B0", representation="tfidf", parameter=10, npmi=spread(0.30))

    met = criterion_one_npmi(winner, own, matched_low)
    failed = criterion_one_npmi(winner, own, matched_high)

    assert (met.npmi_met, met.met) == (True, None)  # F1 is pending until T22
    assert met.own_k.outcome == met.matched_k.outcome == "first"
    assert (failed.npmi_met, failed.met) == (False, False)
    assert failed.matched_k.outcome == "tie"
    assert "T22" in met.f1


def test_the_coefficient_of_variation_uses_the_sample_sd() -> None:
    assert coefficient_of_variation(summarize(spread(2.0, 0.5))) == 0.25
    assert coefficient_of_variation(summarize(spread(-2.0, 0.5))) == 0.25
    assert coefficient_of_variation(summarize(spread(0.0, 0.5))) is None
    assert coefficient_of_variation(summarize((None, 1.0, 1.0, 1.0, 1.0))) is None


def test_criterion_four_needs_g3_and_both_coefficients_at_most_the_cap() -> None:
    stable = candidate(silhouette=spread(0.5, 0.0625), npmi=spread(0.25, 0.015625), ari=0.65)
    unstable = candidate(silhouette=spread(0.5, 0.0625), npmi=spread(0.25, 0.03125))
    weak = candidate(ari=0.6)

    met = criterion_four(stable, gates=GATES, max_cv=0.125)

    assert met.met
    assert (met.cv_silhouette, met.cv_npmi) == (0.125, 0.0625)
    assert not criterion_four(unstable, gates=GATES, max_cv=0.10).met
    assert not criterion_four(weak, gates=GATES, max_cv=0.10).met


def test_the_representative_run_has_the_highest_mean_ari_to_the_others() -> None:
    matrix = [
        [1.0, 0.7, 0.7, 0.7, 0.7],
        [0.7, 1.0, 0.9, 0.9, 0.9],
        [0.7, 0.9, 1.0, 0.8, 0.8],
        [0.7, 0.9, 0.8, 1.0, 0.8],
        [0.7, 0.9, 0.8, 0.8, 1.0],
    ]

    representative = representative_seed(matrix, SEEDS)

    assert representative.seed == 13
    assert math.isclose(representative.mean_ari_to_others[1], 0.85)
    assert math.isclose(representative.mean_ari_to_others[0], 0.7)


def test_a_representative_tie_goes_to_the_seed_listed_first() -> None:
    matrix = [[1.0, 0.8, 0.6], [0.8, 1.0, 0.6], [0.6, 0.6, 1.0]]

    assert representative_seed(matrix, (42, 7, 13)).seed == 42
    with pytest.raises(SelectionError):
        representative_seed([[1.0, None], [None, 1.0]], (7, 13))
