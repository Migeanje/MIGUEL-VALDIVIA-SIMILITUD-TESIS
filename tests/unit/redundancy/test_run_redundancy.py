"""T22's run on the labeled set through its command line, on the synthetic project of
``conftest.py``: T20a's real draw of the labeling tests' corpus, T19's records built with T19's
rules, and synthetic consensus labels derived from the pairs' scores.

Only the input loader (the document vectors and the snapshot's sensitive strings) and git are
replaced; the configuration is the real file.
"""

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from thematic_redundancy.labeling import draw_pairs
from thematic_redundancy.labeling.draw_pairs import DrawInputs, PairRow
from thematic_redundancy.labeling.pairs import PILOT, TFIDF
from thematic_redundancy.redundancy.comparison import (
    BootstrapSettings,
    F1Difference,
    bootstrap_f1,
    compare_representations,
)
from thematic_redundancy.redundancy.labels import (
    ADJUDICATION,
    AGREEMENT,
    REDUNDANT,
    RELATED,
    UNRELATED,
    ConsensusLabel,
)
from thematic_redundancy.redundancy.results_store import (
    COMPARISONS_FILE_NAME,
    CURVES_FILE_NAME,
    FOLDS_FILE_NAME,
    MANIFEST_FILE_NAME,
    PAIR_SCORES_FILE_NAME,
    REPRESENTATIONS_FILE_NAME,
    SELECTION_FILE_NAME,
    SUMMARY_FILE_NAME,
    ComparisonRow,
    CurveRow,
    FoldRow,
    PairScoreRow,
    RedundancyManifest,
    RepresentationRow,
    write_outputs,
)
from thematic_redundancy.redundancy.run_redundancy import (
    CRITERION_TWO_MIN_F1,
    RUN_COMMAND,
    compute,
    main,
)
from thematic_redundancy.redundancy.threshold import (
    CvSettings,
    calibrate_threshold,
    max_f1_threshold,
)
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.shared.filesystem import exclusive_run, write_text_atomically
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.results_store import cpu_record, library_versions
from thematic_redundancy.topic_map.run_experiment import GitState

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config(PROJECT_ROOT / "config" / "default.yaml")
MODELS = CONFIG.embedding.models
MINILM, MPNET = MODELS
SNAPSHOT = "S1"
RESULTS = Path("results") / "experiments" / SNAPSHOT / "t22"
PROCESSED = Path("data") / "processed" / SNAPSHOT / "experiments" / "t22"
TRACKED = {
    REPRESENTATIONS_FILE_NAME,
    FOLDS_FILE_NAME,
    CURVES_FILE_NAME,
    COMPARISONS_FILE_NAME,
    SELECTION_FILE_NAME,
    SUMMARY_FILE_NAME,
    MANIFEST_FILE_NAME,
}
REPRESENTATIONS = (*MODELS, TFIDF)
CV = CvSettings(folds=5, seed=7, stratified=True, f1_tie="highest_tau", final_tau="all_pairs")
BOOTSTRAP = BootstrapSettings(resamples=2000, seed=7, interval=0.95)
STORE = ParquetTableStore()


def run(kit: Any, root: Path, inputs: DrawInputs, *extra: str, **overrides: Any) -> int:
    arguments: dict[str, Any] = {
        "load": kit.loader(inputs),
        "git": kit.clean,
        "clock": kit.clock(),
        "timer": kit.timer(),
    }
    arguments.update(overrides)
    return main(["--project-root", str(root), "--snapshot-id", SNAPSHOT, *extra], **arguments)


def label_by(kit: Any, root: Path, policy: str) -> list[ConsensusLabel]:
    """Consensus labels whose redundant pairs are mpnet's or MiniLM's top 20, or the top 25 of
    both models' mean score."""
    pairs = kit.labeled_pairs(root)
    if policy == "mpnet":
        redundant = kit.top_ids(pairs, 1, 20)
    elif policy == "minilm":
        redundant = kit.top_ids(pairs, 0, 20)
    else:
        ranked = sorted(pairs, key=lambda row: -(row.scores[0] + row.scores[1]))
        redundant = {row.pair_id for row in ranked[:25]}
    return kit.consensus(pairs, redundant)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_of(root: Path) -> RedundancyManifest:
    content = (root / RESULTS / MANIFEST_FILE_NAME).read_bytes()
    return RedundancyManifest.model_validate_json(content)


def tracked_text(root: Path) -> str:
    """Every tracked output as text, a table as the JSON of its rows."""
    parts = []
    for name in sorted(TRACKED):
        path = root / RESULTS / name
        if name.endswith(".parquet"):
            parts.append(json.dumps(pq.read_table(path).to_pylist(), ensure_ascii=False))
        else:
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def scores_and_labels(
    kit: Any, root: Path, labels: list[ConsensusLabel]
) -> tuple[np.ndarray, np.ndarray]:
    """T20a's stored scores of the labeled pairs, one column per representation, and the
    binary labels, in draw order."""
    pairs = kit.labeled_pairs(root)
    by_id = {label.pair_id: label.label for label in labels}
    scores = np.array([row.scores for row in pairs], dtype=np.float64)
    return scores, np.array([by_id[row.pair_id] for row in pairs], dtype=np.int64)


# --- A run and its records ---------------------------------------------------------------------


def test_a_real_run_writes_the_tracked_aggregates_and_the_gitignored_pair_scores(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    labels = label_by(kit, project, "mpnet")
    kit.write_labels(project, labels)
    before = kit.files_under(project)

    assert run(kit, project, inputs) == 0

    written = set(kit.files_under(project)) - set(before)
    assert written == {
        *(f"{RESULTS.as_posix()}/{name}" for name in TRACKED),
        f"{RESULTS.as_posix()}/.lock",
        f"{PROCESSED.as_posix()}/{PAIR_SCORES_FILE_NAME}",
    }
    assert all(before[path] == kit.sha256_of(project / path) for path in before)
    manifest = manifest_of(project)
    draw = project / "results" / "labeled_set" / SNAPSHOT / "draw"
    records = draw_pairs.records_dir(CONFIG, project, SNAPSHOT)
    t19 = project / "results" / "experiments" / SNAPSHOT / "t19"
    assert manifest.snapshot_id == SNAPSHOT
    assert manifest.git_commit == "c" * 40
    assert manifest.settings == {
        "cv": {
            "folds": 5,
            "seed": 7,
            "stratified": True,
            "f1_tie": "highest_tau",
            "final_tau": "all_pairs",
        },
        "bootstrap": {"resamples": 2000, "seed": 7, "interval": 0.95},
        "criterion_two": {"min_f1": 0.70, "judged_on": "cv_f1"},
        "redundant_level": REDUNDANT,
        "embedding_document_vector": "section_mean",
        "tfidf_document_vector": "section_mean",
    }
    assert manifest.representations == REPRESENTATIONS
    assert manifest.labeled_set == {
        "manifest.json": kit.sha256_of(draw / "manifest.json"),
        "pairs.parquet": kit.sha256_of(records / "pairs.parquet"),
        "consensus.parquet": kit.sha256_of(kit.labels_path(project)),
    }
    assert manifest.experiment == {
        "manifest.json": kit.sha256_of(t19 / "manifest.json"),
        "selection.json": kit.sha256_of(t19 / "selection.json"),
    }
    assert manifest.inputs == {
        **inputs.input_sha256,
        "default.yaml": kit.sha256_of(project / "config" / "default.yaml"),
    }
    assert (manifest.provisional_winner, manifest.stage_two) == ("C4", "closed")
    assert (manifest.outcome, manifest.winner) == ("confirmed", "C4")
    levels = Counter(label.label_3 for label in labels)
    sources = Counter(label.source for label in labels)
    assert manifest.counts == {
        "labeled_pairs": 120,
        "pilot_pairs": 10,
        "redundant": 20,
        "related": levels[RELATED],
        "unrelated": levels[UNRELATED],
        "agreement": sources[AGREEMENT],
        "adjudication": sources[ADJUDICATION],
        "known_duplicates": 0,
    }
    assert levels[RELATED] + levels[UNRELATED] == 100 and sources[ADJUDICATION] == 18
    assert manifest.versions == library_versions()
    assert manifest.cpu == cpu_record()
    assert manifest.files == {
        name: kit.sha256_of(project / RESULTS / name) for name in TRACKED - {MANIFEST_FILE_NAME}
    }
    assert manifest.data_files == {
        PAIR_SCORES_FILE_NAME: kit.sha256_of(project / PROCESSED / PAIR_SCORES_FILE_NAME)
    }
    assert set(manifest.timings) == {"load", "check", "compute"}
    out = capsys.readouterr().out
    assert "the provisional winner C4 is confirmed" in out
    assert f"Wrote {RESULTS.as_posix()}/" in out


def test_no_tracked_output_names_a_thesis_or_a_pair(
    kit: Any, project: Path, inputs: DrawInputs, corpus: Any
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))

    assert run(kit, project, inputs) == 0

    text = tracked_text(project).casefold()
    every_pair = STORE.read(
        PairRow, draw_pairs.records_dir(CONFIG, project, SNAPSHOT) / "pairs.parquet"
    )
    assert not [code for code, _ in corpus.theses if code.casefold() in text]
    assert not [row.pair_id for row in every_pair if row.pair_id.casefold() in text]


def test_the_pair_scores_are_t20a_s_scores_with_the_labels_folds_and_predictions(
    kit: Any, project: Path, inputs: DrawInputs
) -> None:
    labels = label_by(kit, project, "mpnet")
    kit.write_labels(project, labels)

    assert run(kit, project, inputs) == 0

    pairs = kit.labeled_pairs(project)
    rows = STORE.read(PairScoreRow, project / PROCESSED / PAIR_SCORES_FILE_NAME)
    taus = {
        row.representation: row.tau
        for row in STORE.read(RepresentationRow, project / RESULTS / REPRESENTATIONS_FILE_NAME)
    }
    scores, binary = scores_and_labels(kit, project, labels)
    folds = calibrate_threshold(scores[:, 0], binary, CV).fold_of_pair
    by_id = {label.pair_id: label for label in labels}
    assert [row.pair_id for row in rows] == [pair.pair_id for pair in pairs]
    for row, pair, fold in zip(rows, pairs, folds, strict=True):
        label = by_id[pair.pair_id]
        assert (row.label_3, row.label, row.source) == (label.label_3, label.label, label.source)
        assert row.representations == REPRESENTATIONS
        assert row.scores == pair.scores
        assert row.predicted == tuple(
            score >= taus[name] for name, score in zip(REPRESENTATIONS, pair.scores, strict=True)
        )
        assert row.fold == fold
        assert row.known_duplicate == pair.known_duplicate


def test_every_number_comes_from_t22_s_functions_with_the_declared_values(
    kit: Any, project: Path, inputs: DrawInputs
) -> None:
    labels = label_by(kit, project, "mpnet")
    kit.write_labels(project, labels)

    assert run(kit, project, inputs) == 0

    scores, binary = scores_and_labels(kit, project, labels)
    rows = STORE.read(RepresentationRow, project / RESULTS / REPRESENTATIONS_FILE_NAME)
    folds = STORE.read(FoldRow, project / RESULTS / FOLDS_FILE_NAME)
    curves = STORE.read(CurveRow, project / RESULTS / CURVES_FILE_NAME)
    assert [row.representation for row in rows] == list(REPRESENTATIONS)
    for column, row in enumerate(rows):
        calibration = calibrate_threshold(scores[:, column], binary, CV)
        interval = bootstrap_f1(scores[:, column], binary, tau=calibration.tau, settings=BOOTSTRAP)
        at_tau, pooled = calibration.at_tau, calibration.pooled
        assert (row.pairs, row.positives, row.tau) == (120, 20, calibration.tau)
        assert (
            row.true_positives,
            row.false_positives,
            row.false_negatives,
            row.true_negatives,
        ) == (
            at_tau.true_positives,
            at_tau.false_positives,
            at_tau.false_negatives,
            at_tau.true_negatives,
        )
        assert (row.precision, row.recall, row.f1) == (at_tau.precision, at_tau.recall, at_tau.f1)
        assert (row.f1_lower, row.f1_upper) == (interval.lower, interval.upper)
        assert row.undefined_resamples == interval.undefined_resamples
        assert (
            row.cv_true_positives,
            row.cv_false_positives,
            row.cv_false_negatives,
            row.cv_true_negatives,
        ) == (
            pooled.true_positives,
            pooled.false_positives,
            pooled.false_negatives,
            pooled.true_negatives,
        )
        assert (row.cv_f1, row.mean_fold_f1) == (calibration.cv_f1, calibration.mean_fold_f1)
        assert (row.cv_precision, row.cv_recall) == (pooled.precision, pooled.recall)
        own_folds = [fold for fold in folds if fold.representation == row.representation]
        assert [(fold.fold, fold.tau, fold.train_f1, fold.test_f1) for fold in own_folds] == [
            (outcome.fold, outcome.tau, outcome.train_f1, outcome.test_f1)
            for outcome in calibration.folds
        ]
        assert sum(fold.test_pairs for fold in own_folds) == 120
        assert sum(fold.test_positives for fold in own_folds) == 20
        curve = max_f1_threshold(scores[:, column], binary, f1_tie="highest_tau").curve
        own_curve = [point for point in curves if point.representation == row.representation]
        assert [(point.point, point.tau, point.f1) for point in own_curve] == [
            (index, p.tau, p.f1) for index, p in enumerate(curve)
        ]
        assert [point.tau for point in own_curve if point.selected] == [calibration.tau]
    comparisons = STORE.read(ComparisonRow, project / RESULTS / COMPARISONS_FILE_NAME)
    expected = [
        ("stage_two", MINILM, MPNET, 0, 1),
        ("criterion_one", MINILM, TFIDF, 0, 2),
        ("criterion_one", MPNET, TFIDF, 1, 2),
    ]
    assert [(row.purpose, row.first, row.second) for row in comparisons] == [
        item[:3] for item in expected
    ]
    for row, (_, first, second, a, b) in zip(comparisons, expected, strict=True):
        oracle = compare_representations(
            scores[:, a], scores[:, b], binary, names=(first, second), cv=CV, bootstrap=BOOTSTRAP
        ).difference
        assert (row.first_f1, row.second_f1, row.difference) == (
            oracle.first_f1,
            oracle.second_f1,
            oracle.difference,
        )
        assert (row.lower, row.upper, row.verdict) == (oracle.lower, oracle.upper, oracle.verdict)
        assert (row.first_tau, row.second_tau) == (oracle.first_tau, oracle.second_tau)
        assert row.undefined_resamples == oracle.undefined_resamples


# --- Stage 2 of the winner ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "f1_outcome", "outcome", "winner", "step", "criterion"),
    [
        ("mpnet", "second", "confirmed", "C4", 2, "f1"),
        ("minilm", "first", "changed", "C1", 2, "f1"),
        ("both", "tie", "confirmed", "C4", 3, "npmi"),
    ],
)
def test_stage_two_closes_by_the_full_rule(
    kit: Any,
    project: Path,
    inputs: DrawInputs,
    capsys: pytest.CaptureFixture[str],
    policy: str,
    f1_outcome: str,
    outcome: str,
    winner: str,
    step: int,
    criterion: str,
) -> None:
    kit.write_labels(project, label_by(kit, project, policy))

    assert run(kit, project, inputs) == 0

    selection = read_json(project / RESULTS / SELECTION_FILE_NAME)
    stage = selection["stage_two"]
    decision = stage["decision"]
    assert (stage["status"], stage["outcome"]) == ("closed", outcome)
    assert (decision["deciding_step"], decision["deciding_criterion"]) == (step, criterion)
    assert [item["step"] for item in decision["steps"]] == list(range(1, step + 1))
    assert decision["steps"][1]["criterion"] == "f1"
    assert decision["steps"][1]["outcome"] == f1_outcome
    assert stage["f1"]["outcome"] == f1_outcome
    assert decision["steps"][0] == stage["provisional"]["steps"][0]
    assert selection["winner"]["configuration"] == winner
    assert selection["winner"]["provisional"] is False
    assert selection["provisional_winner"]["configuration"] == "C4"
    manifest = manifest_of(project)
    assert (manifest.outcome, manifest.winner) == (outcome, winner)
    summary = (project / RESULTS / SUMMARY_FILE_NAME).read_text(encoding="utf-8")
    out = capsys.readouterr().out
    sentence = (
        "the provisional winner C4 is confirmed"
        if outcome == "confirmed"
        else "the provisional winner C4 changes to C1"
    )
    assert sentence in summary
    assert sentence in out
    assert (
        "T19's provisional winner: C4, paraphrase-multilingual-mpnet-base-v2 with hdbscan at 10"
        in (summary)
    )
    assert f"the full rule's winner: {winner}, " in summary


def test_a_changed_winner_leaves_criterion_one_s_npmi_pending_and_names_what_to_rebuild(
    kit: Any, project: Path, inputs: DrawInputs
) -> None:
    kit.write_labels(project, label_by(kit, project, "minilm"))

    assert run(kit, project, inputs) == 0

    selection = read_json(project / RESULTS / SELECTION_FILE_NAME)
    one = selection["criterion_one"]
    assert one["representation"] == MINILM
    assert (one["npmi_met"], one["f1_met"], one["met"]) == (None, True, None)
    assert any("T23 to T26" in item for item in selection["consequences"])


def test_a_confirmed_winner_keeps_criterion_one_s_npmi_from_t19(
    kit: Any, project: Path, inputs: DrawInputs
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))

    assert run(kit, project, inputs) == 0

    one = read_json(project / RESULTS / SELECTION_FILE_NAME)["criterion_one"]
    t19 = kit.selection_record("npmi")["criterion_one"]
    assert (one["representation"], one["npmi_met"], one["f1_met"]) == (
        MPNET,
        t19["npmi_met"],
        True,
    )
    assert one["met"] is t19["npmi_met"]


def test_criterion_two_judges_the_winner_s_cross_validated_f1(
    kit: Any, project: Path, inputs: DrawInputs
) -> None:
    labels = label_by(kit, project, "mpnet")
    kit.write_labels(project, labels)

    assert run(kit, project, inputs) == 0

    two = read_json(project / RESULTS / SELECTION_FILE_NAME)["criterion_two"]
    scores, binary = scores_and_labels(kit, project, labels)
    calibration = calibrate_threshold(scores[:, 1], binary, CV)
    assert two["representation"] == MPNET
    assert (two["min_f1"], two["judged_on"]) == (CRITERION_TWO_MIN_F1, "cv_f1")
    assert (two["cv_f1"], two["f1"]) == (calibration.cv_f1, calibration.f1_at_tau)
    assert two["met"] is (calibration.cv_f1 >= 0.70)


def test_the_gates_decide_before_the_f1(kit: Any, project: Path, inputs: DrawInputs) -> None:
    kit.write_t19(project, scenario="gates")
    kit.write_labels(project, label_by(kit, project, "minilm"))

    assert run(kit, project, inputs) == 0

    stage = read_json(project / RESULTS / SELECTION_FILE_NAME)["stage_two"]
    assert (stage["status"], stage["outcome"]) == ("closed", "confirmed")
    assert (stage["decision"]["deciding_step"], stage["decision"]["winner"]) == (1, MPNET)
    assert stage["f1"]["outcome"] == "first"


def altered(change: Callable[[F1Difference], F1Difference]) -> Callable[..., Any]:
    """A comparison port that changes the difference of the two embedding models only."""

    def compare(first: Any, second: Any, labels: Any, *, names: Any, **settings: Any) -> Any:
        result = compare_representations(first, second, labels, names=names, **settings)
        if TFIDF in names:
            return result
        return replace(result, difference=change(result.difference))

    return compare


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda d: replace(d, lower=None, upper=None, verdict=None), "has no interval"),
        (lambda d: replace(d, lower=0.01, upper=0.2, verdict="first_higher"), "opposite"),
    ],
)
def test_an_undefined_f1_step_leaves_stage_two_undefined_and_names_no_winner(
    kit: Any,
    project: Path,
    inputs: DrawInputs,
    capsys: pytest.CaptureFixture[str],
    change: Callable[[F1Difference], F1Difference],
    reason: str,
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))

    assert run(kit, project, inputs, compare=altered(change)) == 0

    selection = read_json(project / RESULTS / SELECTION_FILE_NAME)
    stage = selection["stage_two"]
    assert (stage["status"], stage["outcome"], stage["decision"]) == (
        "undefined",
        "undefined",
        None,
    )
    assert stage["f1"]["outcome"] == "undefined"
    assert reason in stage["undefined_reason"]
    assert selection["winner"] is None
    assert selection["criterion_two"]["met"] is None
    assert any("decision for the user" in item for item in selection["consequences"])
    manifest = manifest_of(project)
    assert (manifest.stage_two, manifest.outcome, manifest.winner) == (
        "undefined",
        "undefined",
        None,
    )
    sentence = "stage 2 is undefined, so the provisional winner C4 stays provisional"
    summary = (project / RESULTS / SUMMARY_FILE_NAME).read_text(encoding="utf-8")
    assert sentence in summary
    assert "the full rule names no winner" in summary
    assert sentence in capsys.readouterr().out


# --- Refusals of the labels --------------------------------------------------------------------


def _pilot_id(root: Path) -> str:
    records = draw_pairs.records_dir(CONFIG, root, SNAPSHOT)
    rows = STORE.read(PairRow, records / "pairs.parquet")
    return next(row.pair_id for row in rows if row.phase == PILOT)


def _first(rows: list[ConsensusLabel], **changes: Any) -> list[ConsensusLabel]:
    data = {**rows[0].model_dump(), **changes}
    return [ConsensusLabel.model_construct(**data), *rows[1:]]


LABEL_CHANGES: dict[str, tuple[Callable[[Path, list[ConsensusLabel]], Any], str]] = {
    "a missing pair": (lambda root, rows: rows[1:], "1 of the 120 labeled pairs has no label"),
    "an extra pair": (
        lambda root, rows: [*rows, rows[0].model_copy(update={"pair_id": "L-00000000"})],
        "1 pair id that is not a pair of the labeled set",
    ),
    "a duplicate pair": (lambda root, rows: [*rows, rows[5]], "1 pair id given more than once"),
    "a pilot pair": (
        lambda root, rows: [*rows, rows[0].model_copy(update={"pair_id": _pilot_id(root)})],
        "1 pilot pair",
    ),
    "a label that is not the level's": (
        lambda root, rows: _first(rows, label_3=REDUNDANT, label=0),
        "label must be 1 when label_3 is Redundante and 0 otherwise",
    ),
    "a redundant label on a lower level": (
        lambda root, rows: _first(rows, label_3=RELATED, label=1),
        "label must be 1 when label_3 is Redundante and 0 otherwise",
    ),
    "an unknown level": (
        lambda root, rows: _first(rows, label_3="Redundant", label=1),
        "label_3: Input should be",
    ),
    "an unknown source": (
        lambda root, rows: _first(rows, source="majority"),
        "source: Input should be",
    ),
    "a label outside 0 and 1": (
        lambda root, rows: _first(rows, label_3=RELATED, label=2),
        "label: Input should be less than or equal to 1",
    ),
}


@pytest.mark.parametrize("case", sorted(LABEL_CHANGES))
def test_labels_that_break_the_contract_refuse_the_run_and_write_nothing(
    kit: Any,
    project: Path,
    inputs: DrawInputs,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    change, message = LABEL_CHANGES[case]
    kit.write_labels(project, change(project, label_by(kit, project, "mpnet")))
    before = kit.files_under(project)

    assert run(kit, project, inputs) == 1

    err = capsys.readouterr().err.splitlines()
    assert err[-1].startswith("error: ")
    assert message in err[-1]
    assert set(kit.files_under(project)) - set(before) == {f"{RESULTS.as_posix()}/.lock"}


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"note": pa.array(["x"] * 120, type=pa.string())}, "unexpected columns: note"),
        ({"source": None}, "missing columns: source"),
        ({"label": pa.array(["0"] * 120, type=pa.string())}, "label has type string"),
    ],
)
def test_a_labels_file_with_other_columns_is_refused(
    kit: Any,
    project: Path,
    inputs: DrawInputs,
    capsys: pytest.CaptureFixture[str],
    columns: dict[str, Any],
    message: str,
) -> None:
    rows = label_by(kit, project, "mpnet")
    table = {
        "pair_id": pa.array([row.pair_id for row in rows], type=pa.string()),
        "label_3": pa.array([row.label_3 for row in rows], type=pa.string()),
        "label": pa.array([row.label for row in rows], type=pa.int64()),
        "source": pa.array([row.source for row in rows], type=pa.string()),
    }
    for name, column in columns.items():
        if column is None:
            del table[name]
        else:
            table[name] = column
    kit.labels_path(project).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(table), kit.labels_path(project))

    assert run(kit, project, inputs) == 1

    assert message in capsys.readouterr().err
    assert not (project / RESULTS / MANIFEST_FILE_NAME).exists()


def test_missing_labels_refuse_the_run(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(kit, project, inputs) == 1

    err = capsys.readouterr().err
    assert "the consensus labels are missing" in err
    assert "labels/consensus.parquet" in err.replace("\\", "/")
    assert not (project / RESULTS / MANIFEST_FILE_NAME).exists()


# --- Refusals of the other inputs --------------------------------------------------------------


def test_scores_that_differ_from_t20a_s_refuse_the_run(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    pair = kit.labeled_pairs(project)[0]
    mpnet = inputs.semantic[1]
    row = mpnet.doc_codes.index(pair.doc_code_a)
    vectors = np.array(mpnet.vectors, dtype=np.float64)
    vectors[row, 0] += 1e-9  # a change of direction that a tolerance would hide
    changed = replace(inputs, semantic=(inputs.semantic[0], replace(mpnet, vectors=vectors)))

    assert run(kit, project, changed) == 1

    assert f"the scores of {MPNET} differ from those T20a stored" in capsys.readouterr().err
    assert not (project / RESULTS / MANIFEST_FILE_NAME).exists()


def test_the_guard_holds_every_doc_code_and_every_drawn_pair_id(
    kit: Any, project: Path, inputs: DrawInputs, corpus: Any
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))

    run_ = compute(
        CONFIG,
        project,
        SNAPSHOT,
        labels_path=kit.labels_path(project),
        load=kit.loader(inputs),
        compare=compare_representations,
        report=lambda line: None,
        timer=kit.timer(),
    )

    every_pair = STORE.read(
        PairRow, draw_pairs.records_dir(CONFIG, project, SNAPSHOT) / "pairs.parquet"
    )
    assert {code for code, _ in corpus.theses} <= run_.guarded
    assert {row.pair_id for row in every_pair} <= run_.guarded
    assert kit.sensitive <= run_.guarded


def test_the_scores_of_an_unchanged_representation_are_accepted_as_float64(
    kit: Any, project: Path, inputs: DrawInputs
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    mpnet = inputs.semantic[1]
    widened = replace(
        inputs,
        semantic=(
            inputs.semantic[0],
            replace(mpnet, vectors=np.array(mpnet.vectors, dtype=np.float64)),
        ),
    )

    assert run(kit, project, widened) == 0


def test_a_changed_pairs_file_is_refused(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    path = draw_pairs.records_dir(CONFIG, project, SNAPSHOT) / "pairs.parquet"
    rows = STORE.read(PairRow, path)
    STORE.write([rows[1], rows[0], *rows[2:]], path)

    assert run(kit, project, inputs) == 1

    assert "pairs.parquet differs from the SHA-256 that T20a's manifest records" in (
        capsys.readouterr().err
    )


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("embeddings_manifest.json", "embeddings_manifest.json differs from the one T20a drew"),
        ("metadata.jsonl", "metadata.jsonl differs from the one T20a drew"),
    ],
)
def test_inputs_that_are_not_the_draw_s_are_refused(
    kit: Any,
    project: Path,
    inputs: DrawInputs,
    capsys: pytest.CaptureFixture[str],
    name: str,
    message: str,
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    other = replace(inputs, input_sha256={**inputs.input_sha256, name: "0" * 64})

    assert run(kit, project, other) == 1

    assert message in capsys.readouterr().err


def test_t19_records_of_other_inputs_are_refused(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    path = project / "results" / "experiments" / SNAPSHOT / "t19" / "manifest.json"
    content = read_json(path)
    content["inputs"]["tfidf_manifest.json"] = "0" * 64
    path.write_text(json.dumps(content), encoding="utf-8")

    assert run(kit, project, inputs) == 1

    assert "tfidf_manifest.json differs from the one T19 ran on" in capsys.readouterr().err


def test_a_changed_t19_selection_is_refused(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    path = project / "results" / "experiments" / SNAPSHOT / "t19" / "selection.json"
    path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")

    assert run(kit, project, inputs) == 1

    assert "selection.json differs from the SHA-256 that T19's manifest records" in (
        capsys.readouterr().err
    )


def test_a_t19_stage_two_that_its_candidates_do_not_reproduce_is_refused(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    selection = kit.selection_record("npmi")
    selection["stage_two"]["decision"]["deciding_step"] = 4
    kit.write_t19(project, selection=selection)

    assert run(kit, project, inputs) == 1

    assert "T19's provisional stage 2 is not what its candidates give" in (capsys.readouterr().err)


def test_t19_records_of_another_declaration_are_refused(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    path = project / "results" / "experiments" / SNAPSHOT / "t19" / "manifest.json"
    content = read_json(path)
    content["declaration_sha256"] = "0" * 64
    path.write_text(json.dumps(content), encoding="utf-8")

    assert run(kit, project, inputs) == 1

    assert "T19 ran under the experiment declaration" in capsys.readouterr().err


def test_too_few_redundant_pairs_to_calibrate_refuse_the_run(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = kit.labeled_pairs(project)
    kit.write_labels(project, kit.consensus(pairs, {pairs[0].pair_id}))

    with pytest.warns(UserWarning, match="least populated class"):
        assert run(kit, project, inputs) == 1

    err = capsys.readouterr().err
    assert "tau cannot be calibrated by 5-fold cross-validation on 1 redundant pair" in err
    assert "hold no positive pair" in err
    assert not (project / RESULTS / MANIFEST_FILE_NAME).exists()


# --- The command -------------------------------------------------------------------------------


def test_the_dry_run_computes_everything_and_writes_nothing(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    before = kit.files_under(project)

    assert run(kit, project, inputs, "--dry-run") == 0

    assert kit.files_under(project) == before
    out = capsys.readouterr().out
    assert out.startswith(f"Dry run of T22 on snapshot {SNAPSHOT}: everything is computed")
    assert "the provisional winner C4 is confirmed" in out
    assert "Privacy guard: 6 tracked outputs checked" in out
    assert f"Command of the real run: {RUN_COMMAND} --snapshot-id {SNAPSHOT}" in out
    assert out.isascii()


def test_the_dry_run_reads_labels_from_another_path(
    kit: Any,
    project: Path,
    inputs: DrawInputs,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels = label_by(kit, project, "minilm")
    elsewhere = kit.write_labels(project, labels, path=tmp_path / "x.parquet")
    before = kit.files_under(project)

    assert run(kit, project, inputs, "--dry-run", "--labels", str(elsewhere)) == 0

    assert kit.files_under(project) == before
    assert not kit.labels_path(project).exists()
    out = capsys.readouterr().out
    assert "the provisional winner C4 changes to C1" in out
    assert f"Labels: {elsewhere.resolve()}" in out


def test_a_real_run_refuses_labels_from_another_path(
    kit: Any,
    project: Path,
    inputs: DrawInputs,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    labels = label_by(kit, project, "mpnet")
    elsewhere = kit.write_labels(project, labels, path=tmp_path / "x.parquet")
    before = kit.files_under(project)

    assert run(kit, project, inputs, "--labels", str(elsewhere)) == 1

    assert "--labels is for dry runs only" in capsys.readouterr().err
    assert kit.files_under(project) == before


def test_an_earlier_run_is_never_overwritten(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    assert run(kit, project, inputs) == 0
    first = kit.files_under(project)
    capsys.readouterr()

    assert run(kit, project, inputs) == 1

    assert f"a run of T22 on snapshot {SNAPSHOT} already exists" in capsys.readouterr().err
    assert kit.files_under(project) == first


def test_a_partial_run_is_never_overwritten(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    (project / PROCESSED).mkdir(parents=True)
    (project / PROCESSED / PAIR_SCORES_FILE_NAME).write_bytes(b"partial")

    assert run(kit, project, inputs) == 1

    assert "already exists" in capsys.readouterr().err
    assert not (project / RESULTS / MANIFEST_FILE_NAME).exists()


def test_the_dry_run_says_when_a_run_exists(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    assert run(kit, project, inputs) == 0
    capsys.readouterr()

    assert run(kit, project, inputs, "--dry-run") == 0

    assert "a real run refuses to overwrite it" in capsys.readouterr().out


def test_a_dirty_tree_refuses_the_real_run(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    before = kit.files_under(project)

    def dirty(root: Path, path: str) -> GitState:
        assert path == RESULTS.as_posix()
        return GitState(commit="c" * 40, dirty=("README.md",))

    assert run(kit, project, inputs, git=dirty) == 1

    assert "the git tree has 1 changed or untracked paths outside" in capsys.readouterr().err
    assert kit.files_under(project) == before


def test_a_run_refuses_while_another_holds_the_lock(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    results = project / RESULTS
    results.mkdir(parents=True)

    with exclusive_run(results, RuntimeError("held")):
        assert run(kit, project, inputs) == 1

    assert "another run of T22 holds the lock" in capsys.readouterr().err
    assert not (results / MANIFEST_FILE_NAME).exists()


@pytest.mark.parametrize("dry_run", [False, True])
def test_a_snapshot_string_in_an_output_refuses_every_write(
    kit: Any,
    project: Path,
    inputs: DrawInputs,
    capsys: pytest.CaptureFixture[str],
    dry_run: bool,
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    leaky = replace(inputs, sensitive=kit.sensitive | {"highest_tau"})
    before = kit.files_under(project)

    assert run(kit, project, leaky, *(["--dry-run"] if dry_run else [])) == 1

    assert "outputs hold strings taken from the snapshot" in capsys.readouterr().err
    assert set(kit.files_under(project)) - set(before) <= {f"{RESULTS.as_posix()}/.lock"}


def test_the_gitignored_pair_scores_come_first_and_the_manifest_last(
    kit: Any, project: Path, inputs: DrawInputs, tmp_path: Path
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    assert run(kit, project, inputs) == 0
    results, processed = project / RESULTS, project / PROCESSED
    order: list[str] = []

    class Store(ParquetTableStore):
        def write(self, models: Any, path: Path, *, model_cls: Any = None) -> None:
            order.append(path.name)
            super().write(models, path, model_cls=model_cls)

    def write_text(path: Path, text: str) -> None:
        order.append(path.name)
        write_text_atomically(path, text)

    manifest = write_outputs(
        results=tmp_path / "results",
        processed=tmp_path / "processed",
        representations=STORE.read(RepresentationRow, results / REPRESENTATIONS_FILE_NAME),
        folds=STORE.read(FoldRow, results / FOLDS_FILE_NAME),
        curves=STORE.read(CurveRow, results / CURVES_FILE_NAME),
        comparisons=STORE.read(ComparisonRow, results / COMPARISONS_FILE_NAME),
        pair_scores=STORE.read(PairScoreRow, processed / PAIR_SCORES_FILE_NAME),
        selection=read_json(results / SELECTION_FILE_NAME),
        summary=(results / SUMMARY_FILE_NAME).read_text(encoding="utf-8"),
        manifest=manifest_of(project).model_copy(update={"files": {}, "data_files": {}}),
        sensitive=frozenset(),
        store=Store(),
        write_text=write_text,
    )

    assert order == [
        PAIR_SCORES_FILE_NAME,
        REPRESENTATIONS_FILE_NAME,
        FOLDS_FILE_NAME,
        CURVES_FILE_NAME,
        COMPARISONS_FILE_NAME,
        SELECTION_FILE_NAME,
        SUMMARY_FILE_NAME,
        MANIFEST_FILE_NAME,
    ]
    assert manifest.files == manifest_of(project).files
    assert manifest == manifest_of(project)


def test_a_snapshot_string_in_the_manifest_alone_refuses_every_write(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    # "section_mean" is a value of the manifest's settings, and of no other tracked output.
    kit.write_labels(project, label_by(kit, project, "mpnet"))
    leaky = replace(inputs, sensitive=kit.sensitive | {"section_mean"})
    before = kit.files_under(project)

    assert run(kit, project, leaky) == 1

    assert "manifest.json (1)" in capsys.readouterr().err
    assert set(kit.files_under(project)) - set(before) == {f"{RESULTS.as_posix()}/.lock"}


def test_an_interruption_exits_130(
    kit: Any, project: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    kit.write_labels(project, label_by(kit, project, "mpnet"))

    def interrupted(*arguments: Any) -> DrawInputs:
        raise KeyboardInterrupt

    assert run(kit, project, inputs, load=interrupted) == 130

    assert "interrupted" in capsys.readouterr().err


def test_the_help_names_the_inputs_the_outputs_and_the_labels_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--help"])

    assert caught.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    for fragment in (
        "--dry-run",
        "--snapshot-id",
        "--labels PATH",
        "labels/consensus.parquet",
        "results/experiments/<snapshot>/t22/",
        "dry runs only",
    ):
        assert fragment in out
