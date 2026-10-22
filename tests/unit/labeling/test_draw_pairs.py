"""The labeled set's draw (T20a) through its command line, on the synthetic corpus of
``conftest.py``, in a project root under pytest's temporary directory.

Only the input loader (the fichas, the document vectors and the snapshot's sensitive strings)
and git are replaced; the configuration is the real file, read as on the real snapshot.
"""

import hashlib
import itertools
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from thematic_redundancy.gaps.results_store import PrivacyError
from thematic_redundancy.labeling.draw_pairs import (
    COMPOSITION_FILE_NAME,
    CUTS_FILE_NAME,
    MANIFEST_FILE_NAME,
    MEAN,
    PAIRS_FILE_NAME,
    RANGES_FILE_NAME,
    RUN_COMMAND,
    SUMMARY_FILE_NAME,
    CompositionRow,
    DrawInputs,
    DrawManifest,
    PairRow,
    RangeRow,
    compute,
    main,
    write_outputs,
)
from thematic_redundancy.labeling.pairs import (
    LABELED,
    PILOT,
    SBERT_HIGH,
    LabeledSetDraw,
    PairUniverse,
)
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.shared.filesystem import exclusive_run
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.results_store import cpu_record, library_versions
from thematic_redundancy.topic_map.run_experiment import GitState

TRACKED = {COMPOSITION_FILE_NAME, RANGES_FILE_NAME, CUTS_FILE_NAME, SUMMARY_FILE_NAME}
RESULTS = Path("results") / "labeled_set" / "S1" / "draw"
RECORDS = Path("data") / "labels" / "labeled_set" / "S1"
STARTED = datetime(2026, 10, 21, 9, 0, tzinfo=UTC)


def clean(root: Path, path: str) -> GitState:
    return GitState(commit="c" * 40, dirty=())


def clock() -> Any:
    moments = iter((STARTED, datetime(2026, 10, 21, 9, 1, tzinfo=UTC)))
    return lambda: next(moments)


def timer() -> Any:
    ticks = itertools.count(10.0, 0.5)
    return lambda: next(ticks)


def loader(inputs: DrawInputs, calls: list[Any] | None = None) -> Any:
    def load(*arguments: Any) -> DrawInputs:
        if calls is not None:
            calls.append(arguments)
        return inputs

    return load


def run(root: Path, inputs: DrawInputs, *extra: str, **overrides: Any) -> int:
    arguments: dict[str, Any] = {
        "load": loader(inputs),
        "git": clean,
        "clock": clock(),
        "timer": timer(),
    }
    arguments.update(overrides)
    return main(["--project-root", str(root), "--snapshot-id", "S1", *extra], **arguments)


def files_under(root: Path) -> set[str]:
    if not root.exists():
        return set()
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def written(root: Path) -> set[str]:
    """Return the draw's files, both directories together."""
    found = {f"results/{name}" for name in files_under(root / RESULTS)}
    return found | {f"data/{name}" for name in files_under(root / RECORDS)}


def digests(root: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files_under(root)
    }


def read_pairs(root: Path) -> tuple[PairRow, ...]:
    return ParquetTableStore().read(PairRow, root / RECORDS / PAIRS_FILE_NAME)


def tracked_texts(root: Path) -> dict[str, str]:
    """Return each tracked output as text, a table as the JSON of its rows."""
    store = ParquetTableStore()
    results = root / RESULTS
    return {
        COMPOSITION_FILE_NAME: json.dumps(
            [
                row.model_dump()
                for row in store.read(CompositionRow, results / COMPOSITION_FILE_NAME)
            ]
        ),
        RANGES_FILE_NAME: json.dumps(
            [row.model_dump() for row in store.read(RangeRow, results / RANGES_FILE_NAME)]
        ),
        **{
            name: (results / name).read_text("utf-8")
            for name in (CUTS_FILE_NAME, SUMMARY_FILE_NAME, MANIFEST_FILE_NAME)
        },
    }


def with_known_duplicate(root: Path, first: str, second: str) -> None:
    path = root / "config" / "default.yaml"
    text = path.read_text("utf-8")
    old = "- [DOC-1b29cde4, DOC-b2303c69]"
    assert old in text
    path.write_text(text.replace(old, f"- [{second}, {first}]"), encoding="utf-8")


# A real run


def test_a_real_run_writes_the_tracked_aggregates_and_the_gitignored_pairs(
    project_root: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(project_root, inputs)

    out = capsys.readouterr().out
    assert code == 0
    assert written(project_root) == {
        *(f"results/{name}" for name in TRACKED | {MANIFEST_FILE_NAME, ".lock"}),
        f"data/{PAIRS_FILE_NAME}",
    }
    for text in ("8,385 pairs", "Top 1 %", "Pilot", "Labeled set", RESULTS.as_posix()):
        assert text in out
    assert not any(code in out for code, _ in inputs.theses)


def test_each_drawn_pair_is_a_gitignored_row_with_its_theses_scores_and_pool(
    project_root: Path, inputs: DrawInputs, universe: PairUniverse, draw: LabeledSetDraw
) -> None:
    run(project_root, inputs)
    rows = read_pairs(project_root)

    assert [row.draw_order for row in rows] == list(range(1, 131))
    expected = [
        (step.phase, step.pool, step.source, pair) for step in draw.steps for pair in step.pairs
    ]
    programs = dict(inputs.theses)
    for row, (phase, pool, source, pair) in zip(rows, expected, strict=True):
        first, second = universe.codes(pair)
        assert (row.phase, row.pool, row.source) == (phase, pool, source)
        assert (row.doc_code_a, row.doc_code_b) == (first, second)
        assert (row.program_a, row.program_b) == (programs[first], programs[second])
        assert row.stratum == universe.stratum_of(pair)
        assert row.representations == universe.representations
        assert row.scores == tuple(float(scores[pair]) for scores in universe.scores)
        assert row.ranks == tuple(float(ranks[pair]) for ranks in universe.ranks)
        assert row.mean_rank == pytest.approx(float(universe.mean_ranks()[pair]))
        assert row.pair_id[:2] == ("P-" if phase == PILOT else "L-")
        assert row.known_duplicate is False
    assert len({row.pair_id for row in rows}) == 130
    assert [row.phase for row in rows] == [PILOT] * 10 + [LABELED] * 120


def test_the_tracked_records_hold_aggregates_and_no_doc_code(
    project_root: Path, inputs: DrawInputs, universe: PairUniverse, draw: LabeledSetDraw
) -> None:
    run(project_root, inputs)
    texts = tracked_texts(project_root)
    pair_ids = {row.pair_id for row in read_pairs(project_root)}

    for name, text in texts.items():
        assert not any(code in text for code, _ in inputs.theses), name
        assert not any(identifier in text for identifier in pair_ids), name
    cuts = json.loads(texts[CUTS_FILE_NAME])
    assert (cuts["theses"], cuts["pairs"], cuts["top_count"]) == (130, 8385, 84)
    assert cuts["universe_strata"] == universe.counts(range(universe.size))
    for record, top, scores in zip(
        cuts["representations"], draw.tops, universe.scores, strict=True
    ):
        assert record["cut_score"] == float(scores[top[-1]])
        assert record["highest_score"] == float(scores[top[0]])
        assert record["top_strata"] == universe.counts(top)
    assert cuts["mid_range"]["candidates"] == draw.mid_range.size
    assert cuts["mid_range"]["strata"] == universe.counts(draw.mid_range)


def test_the_composition_counts_each_pools_candidates_quota_and_draw_per_stratum(
    project_root: Path, inputs: DrawInputs, universe: PairUniverse, draw: LabeledSetDraw
) -> None:
    run(project_root, inputs)
    rows = ParquetTableStore().read(CompositionRow, project_root / RESULTS / COMPOSITION_FILE_NAME)

    assert len(rows) == len(draw.steps) * len(universe.strata)
    by_key = {(row.phase, row.pool, row.source, row.stratum): row for row in rows}
    for step in draw.steps:
        drawn = universe.counts(step.pairs)
        for stratum in universe.strata:
            row = by_key[(step.phase, step.pool, step.source, stratum)]
            assert row.candidates == step.candidates[stratum]
            assert row.drawn == drawn[stratum]
            assert row.quota == (None if step.quotas is None else step.quotas[stratum])


def test_the_score_ranges_of_each_pool_are_those_of_its_pairs(
    project_root: Path, inputs: DrawInputs, universe: PairUniverse, draw: LabeledSetDraw
) -> None:
    run(project_root, inputs)
    rows = ParquetTableStore().read(RangeRow, project_root / RESULTS / RANGES_FILE_NAME)

    assert len(rows) == len(draw.steps) * (len(universe.representations) + 1)
    by_key = {(row.phase, row.pool, row.source, row.representation): row for row in rows}
    means = universe.mean_ranks()
    for step in draw.steps:
        pairs = list(step.pairs)
        for name, scores, ranks in zip(
            universe.representations, universe.scores, universe.ranks, strict=True
        ):
            row = by_key[(step.phase, step.pool, step.source, name)]
            assert row.pairs == len(pairs)
            assert (row.score_min, row.score_max) == (
                float(scores[pairs].min()),
                float(scores[pairs].max()),
            )
            assert (row.rank_min, row.rank_max) == (
                float(ranks[pairs].min()),
                float(ranks[pairs].max()),
            )
        mean = by_key[(step.phase, step.pool, step.source, MEAN)]
        assert (mean.score_min, mean.score_max) == (None, None)
        assert (mean.rank_min, mean.rank_max) == pytest.approx(
            (means[pairs].min(), means[pairs].max())
        )


def test_the_manifest_records_provenance_values_and_every_digest(
    project_root: Path, inputs: DrawInputs, kit: SimpleNamespace
) -> None:
    run(project_root, inputs)
    manifest = DrawManifest.model_validate_json(
        (project_root / RESULTS / MANIFEST_FILE_NAME).read_text("utf-8")
    )

    assert (manifest.snapshot_id, manifest.git_commit, manifest.seed) == ("S1", "c" * 40, 20261021)
    assert manifest.settings == {
        **kit.config.labeling.model_dump(mode="json"),
        "records_dir": "data/labels/labeled_set",
        "embedding_document_vector": "section_mean",
        "tfidf_document_vector": "section_mean",
    }
    assert manifest.representations == (*kit.models, "tfidf")
    assert manifest.inputs == {
        **kit.input_sha256,
        "default.yaml": hashlib.sha256(
            (project_root / "config" / "default.yaml").read_bytes()
        ).hexdigest(),
    }
    assert manifest.counts == {
        "theses": 130,
        "pairs": 8385,
        "top_count": 84,
        "mid_range_candidates": manifest.counts["mid_range_candidates"],
        "pilot": 10,
        "labeled": 120,
        "known_duplicates_listed": 1,
        "known_duplicates_in_universe": 0,
        "known_duplicates_drawn": 0,
    }
    assert manifest.counts["mid_range_candidates"] > 0
    assert manifest.files == {
        name: hashlib.sha256((project_root / RESULTS / name).read_bytes()).hexdigest()
        for name in TRACKED
    }
    assert manifest.data_files == {
        PAIRS_FILE_NAME: hashlib.sha256(
            (project_root / RECORDS / PAIRS_FILE_NAME).read_bytes()
        ).hexdigest()
    }
    assert manifest.versions == library_versions()
    assert manifest.cpu == cpu_record()
    assert manifest.started_at == STARTED
    assert set(manifest.timings) == {"load", "universe", "draw"}
    assert manifest.seconds > 0


def test_a_known_duplicate_among_the_drawn_pairs_is_flagged(
    project_root: Path, inputs: DrawInputs, universe: PairUniverse, draw: LabeledSetDraw
) -> None:
    step = next(s for s in draw.steps if s.phase == LABELED and s.pool == SBERT_HIGH)
    first, second = universe.codes(step.pairs[0])
    with_known_duplicate(project_root, first, second)

    assert run(project_root, inputs) == 0

    flagged = [row for row in read_pairs(project_root) if row.known_duplicate]
    assert [(row.doc_code_a, row.doc_code_b) for row in flagged] == [(first, second)]
    manifest = json.loads((project_root / RESULTS / MANIFEST_FILE_NAME).read_text("utf-8"))
    assert manifest["counts"]["known_duplicates_in_universe"] == 1
    assert manifest["counts"]["known_duplicates_drawn"] == 1


# Refusals


def test_an_existing_draw_is_never_overwritten(
    project_root: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(project_root, inputs) == 0
    before = digests(project_root)
    capsys.readouterr()

    code = run(project_root, inputs)

    err = capsys.readouterr().err
    assert code == 1
    assert "already exists" in err and "one-time" in err
    assert digests(project_root) == before


@pytest.mark.parametrize("left", ["pairs", "manifest"])
def test_either_record_of_an_earlier_draw_refuses_a_new_one(
    project_root: Path, inputs: DrawInputs, left: str, capsys: pytest.CaptureFixture[str]
) -> None:
    path = (
        project_root / RECORDS / PAIRS_FILE_NAME
        if left == "pairs"
        else project_root / RESULTS / MANIFEST_FILE_NAME
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(b"an earlier draw")
    calls: list[Any] = []

    code = run(project_root, inputs, load=loader(inputs, calls))

    assert code == 1
    assert "already exists" in capsys.readouterr().err
    assert not calls
    assert path.read_bytes() == b"an earlier draw"


def test_a_dirty_tree_stops_a_real_run_before_anything_is_read(
    project_root: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[Any] = []
    seen: list[str] = []

    def dirty(root: Path, path: str) -> GitState:
        seen.append(path)
        return GitState(commit="c" * 40, dirty=("src/x.py", "odd/y.md"))

    code = run(project_root, inputs, load=loader(inputs, calls), git=dirty)

    assert code == 1
    assert "2 changed or untracked paths" in capsys.readouterr().err
    assert seen == [RESULTS.as_posix()]
    assert not calls
    assert written(project_root) == set()


def test_a_second_draw_is_refused_while_one_holds_the_lock(
    project_root: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    results = project_root / RESULTS
    results.mkdir(parents=True)
    calls: list[Any] = []

    with exclusive_run(results, OSError("held")):
        code = run(project_root, inputs, load=loader(inputs, calls))

    assert code == 1
    assert "another draw" in capsys.readouterr().err
    assert not calls
    assert written(project_root) == {"results/.lock"}


def test_the_dry_run_computes_everything_and_writes_nothing(
    project_root: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    before = files_under(project_root)
    calls: list[Any] = []
    dirty = lambda root, path: GitState(commit="c" * 40, dirty=("odd/notes.md",))  # noqa: E731

    code = run(project_root, inputs, "--dry-run", load=loader(inputs, calls), git=dirty)

    out = capsys.readouterr().out
    assert code == 0
    assert files_under(project_root) == before
    assert len(calls) == 1
    for text in (
        "Dry run",
        "nothing is written",
        "1 changed path",
        "8,385 pairs",
        "Top 1 %",
        "Mid-range",
        "Pilot",
        "Privacy guard",
        f"{RUN_COMMAND} --snapshot-id S1",
    ):
        assert text in out
    assert not any(code in out for code, _ in inputs.theses)


def test_the_dry_run_tells_that_a_draw_exists(
    project_root: Path, inputs: DrawInputs, capsys: pytest.CaptureFixture[str]
) -> None:
    assert run(project_root, inputs) == 0
    before = digests(project_root)
    capsys.readouterr()

    code = run(project_root, inputs, "--dry-run")

    assert code == 0
    assert "a real run refuses to overwrite it" in capsys.readouterr().out
    assert digests(project_root) == before


def test_a_string_of_the_snapshot_refuses_the_whole_write(
    project_root: Path, kit: SimpleNamespace, corpus: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    # The program key, which the summary shows, is taken here for a string of the snapshot.
    leaky = kit.make_inputs(corpus, sensitive=frozenset({"Industrial"}))

    code = run(project_root, leaky)

    err = capsys.readouterr().err
    assert code == 1
    assert SUMMARY_FILE_NAME in err and "nothing was written" in err
    assert written(project_root) == {"results/.lock"}


def test_the_dry_run_applies_the_privacy_guard_too(
    project_root: Path, kit: SimpleNamespace, corpus: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    leaky = kit.make_inputs(corpus, sensitive=frozenset({"industrial"}))

    code = run(project_root, leaky, "--dry-run")

    assert code == 1
    assert SUMMARY_FILE_NAME in capsys.readouterr().err
    assert written(project_root) == set()


@pytest.mark.parametrize(("error", "code"), [(KeyboardInterrupt(), 130), (OSError("missing"), 1)])
def test_exit_codes_and_the_rerun_hint(
    project_root: Path, inputs: DrawInputs, error: BaseException, code: int, capsys: Any
) -> None:
    def load(*_: Any) -> Any:
        raise error

    assert run(project_root, inputs, load=load) == code
    err = capsys.readouterr().err
    assert ("move the partial records away" in err) == (code == 130)
    assert MANIFEST_FILE_NAME not in {Path(p).name for p in written(project_root)}


def computed(root: Path, inputs: DrawInputs) -> Any:
    config = load_config(root / "config" / "default.yaml")
    return compute(config, root, "S1", load=loader(inputs), report=lambda _: None, timer=timer())


def blank_manifest() -> DrawManifest:
    return DrawManifest(
        snapshot_id="S1",
        git_commit="c" * 40,
        seed=1,
        settings={},
        representations=(),
        inputs={},
        counts={},
        versions={},
        cpu={},
        files={},
        data_files={},
        started_at=STARTED,
        finished_at=STARTED,
        seconds=1.0,
        timings={},
    )


def test_a_doc_code_in_a_tracked_output_refuses_the_whole_write(
    project_root: Path, inputs: DrawInputs
) -> None:
    run = computed(project_root, inputs)
    code = run.universe.doc_codes[0]

    with pytest.raises(PrivacyError, match=SUMMARY_FILE_NAME):
        write_outputs(
            run,
            results=project_root / RESULTS,
            records=project_root / RECORDS,
            summary=f"# A summary that names {code}",
            manifest=blank_manifest(),
        )

    assert written(project_root) == set()


def test_the_manifest_is_written_last(project_root: Path, inputs: DrawInputs) -> None:
    run = computed(project_root, inputs)
    texts: list[str] = []

    def write_text(path: Path, text: str) -> None:
        texts.append(path.name)
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        write_outputs(
            run,
            results=project_root / RESULTS,
            records=project_root / RECORDS,
            summary="# A summary",
            manifest=blank_manifest(),
            write_text=write_text,
        )

    assert texts == [CUTS_FILE_NAME]
    assert written(project_root) == {
        f"results/{COMPOSITION_FILE_NAME}",
        f"results/{RANGES_FILE_NAME}",
        f"data/{PAIRS_FILE_NAME}",
    }


def test_the_help_names_the_outputs(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["--help"])

    out = " ".join(capsys.readouterr().out.split())
    assert stopped.value.code == 0
    for text in (
        "results/labeled_set/<snapshot>/draw/",
        "labeling.records_dir",
        "--dry-run",
        "once",
    ):
        assert text in out
