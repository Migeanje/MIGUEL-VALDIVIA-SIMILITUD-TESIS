"""Draw the pairs of the labeled set (T20a, D34) once, or forecast the draw with ``--dry-run``::

    uv run python -m thematic_redundancy.labeling.draw_pairs [--project-root PATH]
        [--snapshot-id ID] [--dry-run]

The command reads the included fichas (D24), for each thesis's program; the document vectors of
each embedding model and the TF-IDF matrix, the ``document_vector`` of ``embedding`` and
``tfidf``, each checked against its manifest by its own loader; and the snapshot, for the strings
that no tracked output may hold. It builds the universe of pairs and draws the pilot and the
labeled set with :mod:`.pairs` and the values of ``labeling`` in the configuration.
``--dry-run`` stops there: it prints the draw and passes the outputs through the privacy guard,
writing nothing.

A draw is a one-time event: a real run refuses to overwrite an earlier draw of the snapshot (its
``pairs.parquet`` or its ``manifest.json``), and a dirty git tree (changes outside
``results/labeled_set/<snapshot>/draw/``), since the manifest records the commit. Then it holds
the run lock of that directory and writes, the gitignored pairs first and ``manifest.json``
last, as the commit point:

- under ``labeling.records_dir/<snapshot>/``, inside the gitignored ``data/``: ``pairs.parquet``,
  one row per drawn pair (:class:`PairRow`), with its two doc codes;
- under ``results/labeled_set/<snapshot>/draw/``, aggregates only (D33's rule, no doc code):
  ``composition.parquet``, each pool's candidates, quota and draw per stratum;
  ``ranges.parquet``, each pool's score and rank ranges per representation; ``cuts.json``, the
  universe, each representation's top cut and the mid-range; ``summary.md``; and
  ``manifest.json``: the snapshot, the git commit, the seed and the values used, the SHA-256 of
  every input read and every output, the library versions and the times.

Every tracked output passes the privacy guard of the gap run
(:func:`~thematic_redundancy.gaps.results_store.check_privacy`: the snapshot's titles, abstracts
and their long sentences, person names and ORCID iDs), with the theses' doc codes added, before
anything is written; a hit refuses every write. Progress goes to stderr, a summary without thesis
text to stdout; the exit code is 0 on success, 1 after an error, and 130 after an interruption.
"""

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict

from thematic_redundancy.corpus.build_fichas import FICHAS_MANIFEST_FILE_NAME, load_fichas
from thematic_redundancy.corpus.profile import sensitive_strings
from thematic_redundancy.corpus.snapshot import (
    METADATA_FILE_NAME,
    RAW_DIR_NAME,
    check_snapshot_id,
    latest_snapshot_id,
    read_snapshot,
)
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.gaps.results_store import check_privacy
from thematic_redundancy.labeling.pairs import (
    LABELED,
    PILOT,
    TFIDF,
    DocumentVectors,
    LabeledSetDraw,
    PairUniverse,
    Step,
    build_universe,
    draw_labeled_set,
    pair_id,
)
from thematic_redundancy.redundancy.known_duplicates import known_duplicate_flags
from thematic_redundancy.representation.build_embeddings import (
    EMBEDDINGS_DIR_NAME,
    EMBEDDINGS_MANIFEST_FILE_NAME,
    load_embeddings,
)
from thematic_redundancy.representation.build_tfidf import (
    TFIDF_DIR_NAME,
    TFIDF_MANIFEST_FILE_NAME,
    load_tfidf,
)
from thematic_redundancy.shared.config import AppConfig, LabelingConfig, load_config
from thematic_redundancy.shared.filesystem import exclusive_run, write_text_atomically
from thematic_redundancy.shared.storage import ParquetTableStore, TableStore
from thematic_redundancy.topic_map.results_store import cpu_record, library_versions
from thematic_redundancy.topic_map.run_experiment import GitState, read_git_state

CONFIG_PATH = Path("config") / "default.yaml"
RUN_COMMAND = "uv run python -m thematic_redundancy.labeling.draw_pairs"
LABELED_SET_DIR_NAME = "labeled_set"
DRAW_DIR_NAME = "draw"
PAIRS_FILE_NAME = "pairs.parquet"
COMPOSITION_FILE_NAME = "composition.parquet"
RANGES_FILE_NAME = "ranges.parquet"
CUTS_FILE_NAME = "cuts.json"
SUMMARY_FILE_NAME = "summary.md"
MANIFEST_FILE_NAME = "manifest.json"
MEAN = "mean"
"""The name of the mean percentile rank over the representations, in ``ranges.parquet``."""


class DrawError(ValueError):
    """The draw cannot go on: an input disagrees with the configuration or the git tree."""


def results_dir(results_root: Path, snapshot_id: str) -> Path:
    """Return ``<results_root>/labeled_set/<snapshot_id>/draw``."""
    return results_root / LABELED_SET_DIR_NAME / snapshot_id / DRAW_DIR_NAME


def records_dir(config: AppConfig, project_root: Path, snapshot_id: str) -> Path:
    """Return ``labeling.records_dir/<snapshot_id>``, resolved inside ``paths.data_dir``.

    Raises:
        ValueError: if it resolves outside the project root or the data directory.
    """
    directory = config.labeling.resolve_records_dir(project_root, config.paths.data_dir)
    return directory / check_snapshot_id(snapshot_id)


# --- Records --------------------------------------------------------------------------------


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class PairRow(_Row):
    """One row of the gitignored ``pairs.parquet``: one drawn pair, with its doc codes."""

    pair_id: str
    """``P-`` (pilot) or ``L-`` (labeled set) and 8 hex digits of the pair's doc codes; it tells
    neither the pool nor the draw order (:func:`~.pairs.pair_id`)."""
    phase: str
    """``pilot`` or ``labeled``."""
    draw_order: int
    """1 for the first pair drawn, the pilot's first, up to the number of pairs."""
    pool: str
    source: str | None
    """The embedding model whose top an SBERT-high pair came from; ``None`` otherwise."""
    stratum: str
    doc_code_a: str
    """The earlier doc code of the two."""
    doc_code_b: str
    program_a: str
    program_b: str
    representations: tuple[str, ...]
    """The order of :attr:`scores` and :attr:`ranks`: the embedding models, then ``tfidf``."""
    scores: tuple[float, ...]
    """The pair's cosine under each representation."""
    ranks: tuple[float, ...]
    """The pair's percentile rank under each representation."""
    mean_rank: float
    known_duplicate: bool
    """Whether the pair is a known true duplicate (O09, ``redundancy.known_duplicates``)."""


class CompositionRow(_Row):
    """One row of ``composition.parquet``: one stratum of one pool's draw in one phase."""

    phase: str
    pool: str
    source: str | None
    stratum: str
    candidates: int
    """The pool's pairs of the stratum that no earlier step had drawn."""
    quota: int | None
    """The stratum's quota, for a stratified pool."""
    drawn: int


class RangeRow(_Row):
    """One row of ``ranges.parquet``: the range of one representation's scores and percentile
    ranks over the pairs of one pool's draw; for :data:`MEAN`, of the mean percentile ranks."""

    phase: str
    pool: str
    source: str | None
    representation: str
    pairs: int
    score_min: float | None
    score_max: float | None
    rank_min: float | None
    rank_max: float | None


class DrawManifest(BaseModel):
    """Content of ``manifest.json``: provenance, values, counts and time, without a doc code."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    snapshot_id: str
    git_commit: str
    seed: int
    settings: dict[str, Any]
    """``labeling`` as declared, and the document vectors read."""
    representations: tuple[str, ...]
    inputs: dict[str, str]
    """SHA-256 of every input read, by file name: the manifests of the fichas, the embeddings
    and the TF-IDF matrices (whose loaders check every file against them), the snapshot's
    metadata and the configuration."""
    counts: dict[str, int]
    versions: dict[str, str]
    cpu: dict[str, Any]
    files: dict[str, str]
    """SHA-256 of each file written to the results directory, by name."""
    data_files: dict[str, str]
    """SHA-256 of each file written to the records directory, by name."""
    started_at: datetime
    finished_at: datetime
    seconds: float
    timings: dict[str, float]
    """Seconds spent loading the inputs, scoring the universe and drawing."""


# --- Inputs ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawInputs:
    """What the draw reads besides the configuration."""

    theses: tuple[tuple[str, str], ...]
    """``(doc_code, program)`` of each included thesis."""
    semantic: tuple[DocumentVectors, ...]
    """Each embedding model's document vectors, in the configured order."""
    tfidf: DocumentVectors
    sensitive: frozenset[str]
    """The snapshot's strings that no tracked output may hold (``corpus/profile.py``)."""
    input_sha256: dict[str, str]
    """The SHA-256 of the inputs read, by file name."""


def load_inputs(config: AppConfig, project_root: Path, snapshot_id: str) -> DrawInputs:
    """Load the included fichas, the document vectors and the snapshot of ``snapshot_id``, each
    through its own loader.

    Raises:
        FileNotFoundError: if an input is missing.
        ValueError: if an input is stale or damaged.
    """
    data_dir = config.paths.resolve_against(project_root).data_dir
    interim = data_dir / INTERIM_DIR_NAME / snapshot_id
    fichas = load_fichas(interim)
    embeddings = [
        load_embeddings(interim, model, variant=config.embedding.document_vector)
        for model in config.embedding.models
    ]
    tfidf = load_tfidf(interim, project_root, variant=config.tfidf.document_vector)
    snapshot, records = read_snapshot(data_dir / RAW_DIR_NAME / snapshot_id)
    manifests = (
        interim / FICHAS_MANIFEST_FILE_NAME,
        interim / EMBEDDINGS_DIR_NAME / EMBEDDINGS_MANIFEST_FILE_NAME,
        interim / TFIDF_DIR_NAME / TFIDF_MANIFEST_FILE_NAME,
    )
    return DrawInputs(
        theses=tuple((f.doc_code, str(f.program_key)) for f in fichas if f.include),
        semantic=tuple(DocumentVectors(e.model_id, e.doc_codes, e.vectors) for e in embeddings),
        tfidf=DocumentVectors(TFIDF, tfidf.doc_codes, tfidf.matrix),
        sensitive=sensitive_strings(records),
        input_sha256={
            **{path.name: _sha256(path) for path in manifests},
            METADATA_FILE_NAME: snapshot.metadata_sha256,
        },
    )


# --- The draw -------------------------------------------------------------------------------


@dataclass(frozen=True)
class DrawRun:
    """Everything a draw computed, ready to describe and to write."""

    snapshot_id: str
    config: AppConfig
    inputs: DrawInputs
    draw: LabeledSetDraw
    pairs: tuple[PairRow, ...]
    composition: tuple[CompositionRow, ...]
    ranges: tuple[RangeRow, ...]
    cuts: dict[str, Any]
    input_sha256: dict[str, str]
    counts: dict[str, int]
    timings: dict[str, float]

    @property
    def settings(self) -> LabelingConfig:
        return self.config.labeling

    @property
    def universe(self) -> PairUniverse:
        return self.draw.universe

    @property
    def guarded(self) -> frozenset[str]:
        """The strings no tracked output may hold: the snapshot's and the theses' doc codes."""
        return self.inputs.sensitive | frozenset(self.universe.doc_codes)


def pair_rows(
    draw: LabeledSetDraw, known_duplicates: Sequence[tuple[str, str]]
) -> tuple[PairRow, ...]:
    """Return one row per drawn pair, in draw order, each flagged when it is one of
    ``known_duplicates`` (O09).

    Raises:
        DrawError: if two drawn pairs share a pair id.
    """
    universe = draw.universe
    means = universe.mean_ranks()
    entries = [(step, pair) for step in draw.steps for pair in step.pairs]
    codes = [universe.codes(pair) for _, pair in entries]
    flags = known_duplicate_flags(codes, known_duplicates)
    rows = []
    for order, ((step, pair), (first, second), flag) in enumerate(
        zip(entries, codes, flags, strict=True), start=1
    ):
        rows.append(
            PairRow(
                pair_id=pair_id(step.phase, first, second),
                phase=step.phase,
                draw_order=order,
                pool=step.pool,
                source=step.source,
                stratum=universe.stratum_of(pair),
                doc_code_a=first,
                doc_code_b=second,
                program_a=universe.programs[int(universe.first[pair])],
                program_b=universe.programs[int(universe.second[pair])],
                representations=universe.representations,
                scores=tuple(float(scores[pair]) for scores in universe.scores),
                ranks=tuple(float(ranks[pair]) for ranks in universe.ranks),
                mean_rank=float(means[pair]),
                known_duplicate=flag,
            )
        )
    if len({row.pair_id for row in rows}) != len(rows):
        raise DrawError("two drawn pairs share a pair id; the ids need more hex digits")
    return tuple(rows)


def composition_rows(draw: LabeledSetDraw) -> tuple[CompositionRow, ...]:
    """Return, for each step of the draw and each stratum, the candidates left, the quota of a
    stratified pool and the pairs drawn."""
    universe = draw.universe
    rows = []
    for step in draw.steps:
        drawn = universe.counts(step.pairs)
        rows += [
            CompositionRow(
                phase=step.phase,
                pool=step.pool,
                source=step.source,
                stratum=stratum,
                candidates=step.candidates[stratum],
                quota=None if step.quotas is None else step.quotas[stratum],
                drawn=drawn[stratum],
            )
            for stratum in universe.strata
        ]
    return tuple(rows)


def range_rows(draw: LabeledSetDraw) -> tuple[RangeRow, ...]:
    """Return, for each step of the draw, the range of each representation's scores and ranks
    over its pairs, then that of their mean percentile ranks."""
    universe = draw.universe
    means = universe.mean_ranks()
    rows = []
    for step in draw.steps:
        pairs = np.asarray(step.pairs, dtype=np.int64)
        rows += [
            RangeRow(
                phase=step.phase,
                pool=step.pool,
                source=step.source,
                representation=name,
                pairs=int(pairs.size),
                score_min=_extreme(scores, pairs, min),
                score_max=_extreme(scores, pairs, max),
                rank_min=_extreme(ranks, pairs, min),
                rank_max=_extreme(ranks, pairs, max),
            )
            for name, scores, ranks in zip(
                universe.representations, universe.scores, universe.ranks, strict=True
            )
        ]
        rows.append(
            RangeRow(
                phase=step.phase,
                pool=step.pool,
                source=step.source,
                representation=MEAN,
                pairs=int(pairs.size),
                score_min=None,
                score_max=None,
                rank_min=_extreme(means, pairs, min),
                rank_max=_extreme(means, pairs, max),
            )
        )
    return tuple(rows)


def _extreme(values: np.ndarray, pairs: np.ndarray, pick: Callable[..., Any]) -> float | None:
    return None if pairs.size == 0 else float(pick(values[pairs]))


def cuts_record(draw: LabeledSetDraw, settings: LabelingConfig) -> dict[str, Any]:
    """Return the content of ``cuts.json``: the universe per stratum, each representation's top
    with its cut score, and the mid-range, with their counts per stratum."""
    universe = draw.universe
    return {
        "theses": len(universe.doc_codes),
        "pairs": universe.size,
        "universe_strata": universe.counts(np.arange(universe.size)),
        "top_share": settings.top_share,
        "top_count": draw.top_count,
        "representations": [
            {
                "name": name,
                "cut_score": float(scores[top[-1]]),
                "highest_score": float(scores[top[0]]),
                "top_strata": universe.counts(top),
            }
            for name, scores, top in zip(
                universe.representations, universe.scores, draw.tops, strict=True
            )
        ],
        "mid_range": {
            "low": settings.mid_low,
            "high": settings.mid_high,
            "candidates": int(draw.mid_range.size),
            "strata": universe.counts(draw.mid_range),
        },
    }


def compute(
    config: AppConfig,
    project_root: Path,
    snapshot_id: str,
    *,
    load: Callable[[AppConfig, Path, str], DrawInputs],
    report: Callable[[str], None],
    timer: Callable[[], float],
) -> DrawRun:
    """Read the inputs, build the universe and draw the pilot and the labeled set.

    Raises:
        OSError: if an input cannot be read.
        ValueError: if an input is damaged or disagrees with the configuration (a
            :class:`DrawError` or a :class:`~.pairs.SamplerError`).
    """
    started = timer()
    report("loading the fichas, the document vectors and the snapshot")
    inputs = load(config, project_root, snapshot_id)
    names = tuple(vectors.name for vectors in inputs.semantic)
    if names != config.embedding.models or inputs.tfidf.name != TFIDF:
        raise DrawError(
            f"the document vectors are of {[*names, inputs.tfidf.name]}, not of the configured "
            f"models {list(config.embedding.models)} and {TFIDF}"
        )
    loaded = timer()
    report(f"scoring every pair of {len(inputs.theses)} theses under {len(names) + 1} vectors")
    universe = build_universe(
        inputs.theses,
        inputs.semantic,
        inputs.tfidf,
        programs=[program.key for program in config.snapshot.programs],
    )
    built = timer()
    report(f"drawing the pilot and the labeled set from {universe.size:,} pairs")
    draw = draw_labeled_set(universe, config.labeling)
    drawn = timer()
    known = config.redundancy.known_duplicates
    pairs = pair_rows(draw, known)
    present = set(universe.doc_codes)
    return DrawRun(
        snapshot_id=snapshot_id,
        config=config,
        inputs=inputs,
        draw=draw,
        pairs=pairs,
        composition=composition_rows(draw),
        ranges=range_rows(draw),
        cuts=cuts_record(draw, config.labeling),
        input_sha256={**inputs.input_sha256, CONFIG_PATH.name: _sha256(project_root / CONFIG_PATH)},
        counts={
            "theses": len(universe.doc_codes),
            "pairs": universe.size,
            "top_count": draw.top_count,
            "mid_range_candidates": int(draw.mid_range.size),
            "pilot": sum(row.phase == PILOT for row in pairs),
            "labeled": sum(row.phase != PILOT for row in pairs),
            "known_duplicates_listed": len(known),
            "known_duplicates_in_universe": sum(set(pair) <= present for pair in known),
            "known_duplicates_drawn": sum(row.known_duplicate for row in pairs),
        },
        timings={
            "load": round(loaded - started, 3),
            "universe": round(built - loaded, 3),
            "draw": round(drawn - built, 3),
        },
    )


# --- Text -----------------------------------------------------------------------------------


def render_summary(run: DrawRun, *, git_commit: str | None) -> str:
    """Return ``summary.md``: the universe, the tops and the mid-range, the composition of every
    pool by stratum, the score ranges, and the values used. It holds no doc code."""
    universe, settings, cuts = run.universe, run.settings, run.cuts
    strata = universe.strata
    lines = [
        f"# Labeled-set draw of snapshot {run.snapshot_id}",
        "",
        f"The pilot and the labeled set of T20a (D34), drawn once at commit "
        f"`{git_commit or 'unknown'}` with seed {settings.seed} from every pair of the "
        f"{cuts['theses']:,} included theses. Which theses form each pair is in the gitignored "
        f"`{PAIRS_FILE_NAME}` only; the tracked records hold counts and scores.",
        "",
        "## Universe",
        "",
        _row(("Stratum", *strata, "All")),
        _rule(len(strata) + 2),
        _row(("Pairs", *_cells(cuts["universe_strata"], strata), f"{universe.size:,}")),
        "",
        f"## Top {_percent(settings.top_share)} of each representation",
        "",
        f"Each representation's top is its {run.draw.top_count:,} highest-ranked pairs, by rank "
        "(D28 C10); the cut is the lowest score in it.",
        "",
        _row(("Representation", "Cut", "Highest", *strata)),
        _rule(len(strata) + 3),
        *(
            _row(
                (
                    record["name"],
                    f"{record['cut_score']:.4f}",
                    f"{record['highest_score']:.4f}",
                    *_cells(record["top_strata"], strata),
                )
            )
            for record in cuts["representations"]
        ),
        "",
        "## Mid-range",
        "",
        f"Mean percentile rank over the representations at least {settings.mid_low:g} and below "
        f"{settings.mid_high:g}: {cuts['mid_range']['candidates']:,} candidates.",
        "",
        _row(("Stratum", *strata)),
        _rule(len(strata) + 1),
        _row(("Candidates", *_cells(cuts["mid_range"]["strata"], strata))),
        "",
        "## Pilot and labeled set by pool and stratum",
        "",
        "Each cell is the pairs drawn over the pool's candidates left in the stratum when the "
        "pool was drawn. The labeled set's high and mid-range pools are stratified (a stratum's "
        "quota is its drawn count); the pilot and the random pool are not.",
        "",
        _row(("Phase", "Pool", "Model", "Pairs", *strata)),
        _rule(len(strata) + 4),
        *(_step_row(run, step) for step in run.draw.steps),
        "",
        "## Score ranges per pool",
        "",
        "Each cell is the range of the pool's scores, then of its percentile ranks.",
        "",
        _row(("Phase", "Pool", "Model", *universe.representations, "Mean rank")),
        _rule(len(universe.representations) + 4),
        *_range_lines(run),
        "",
        "## Known duplicates (O09)",
        "",
        f"{run.counts['known_duplicates_listed']} listed, "
        f"{run.counts['known_duplicates_in_universe']} in the universe, "
        f"{run.counts['known_duplicates_drawn']} drawn; a drawn one is flagged in "
        f"`{PAIRS_FILE_NAME}`.",
        "",
        f"Declared values: {_settings_text(settings)}.",
    ]
    return "\n".join(lines)


def describe(run: DrawRun, *, seconds: float) -> str:
    """Describe a draw in counts, cuts and compositions, without thesis text."""
    universe, settings, cuts = run.universe, run.settings, run.cuts
    lines = [
        f"Universe: {cuts['theses']:,} theses, {universe.size:,} pairs; per stratum: "
        f"{_by_stratum(cuts['universe_strata'])}",
        *(f"  {name} SHA-256 {digest}" for name, digest in run.input_sha256.items()),
        f"Top {_percent(settings.top_share)}: the {run.draw.top_count:,} highest-ranked pairs of "
        "each representation",
        *(
            f"  {_short(record['name'])}: cut {record['cut_score']:.4f}, highest "
            f"{record['highest_score']:.4f}; per stratum: {_by_stratum(record['top_strata'])}"
            for record in cuts["representations"]
        ),
        f"Mid-range (mean percentile rank at least {settings.mid_low:g} and below "
        f"{settings.mid_high:g}): {cuts['mid_range']['candidates']:,} candidates; per stratum: "
        f"{_by_stratum(cuts['mid_range']['strata'])}",
    ]
    for phase, title in ((PILOT, "Pilot"), (LABELED, "Labeled set")):
        steps = [step for step in run.draw.steps if step.phase == phase]
        total = sum(len(step.pairs) for step in steps)
        lines.append(f"{title}: {total} pairs; drawn/candidates per stratum")
        for step in steps:
            drawn = universe.counts(step.pairs)
            cells = ", ".join(
                f"{stratum} {drawn[stratum]}/{step.candidates[stratum]:,}"
                for stratum in universe.strata
            )
            lines.append(f"  {_step_name(step)}: {len(step.pairs)} ({cells})")
    lines += [
        f"Known duplicates (O09): {run.counts['known_duplicates_listed']} listed, "
        f"{run.counts['known_duplicates_in_universe']} in the universe, "
        f"{run.counts['known_duplicates_drawn']} drawn",
        f"Seconds: {seconds:.1f} ("
        + ", ".join(f"{name} {value:.1f}" for name, value in run.timings.items())
        + ")",
    ]
    return "\n".join(lines)


def _step_row(run: DrawRun, step: Step) -> str:
    drawn = run.universe.counts(step.pairs)
    return _row(
        (
            step.phase,
            step.pool,
            _short(step.source) if step.source else "",
            str(len(step.pairs)),
            *(f"{drawn[s]} / {step.candidates[s]:,}" for s in run.universe.strata),
        )
    )


def _range_lines(run: DrawRun) -> list[str]:
    cells: dict[tuple[str, str, str | None], list[str]] = {}
    for row in run.ranges:
        if row.rank_min is None or row.rank_max is None:
            text = ""
        elif row.score_min is None or row.score_max is None:
            text = f"{row.rank_min:.4f} to {row.rank_max:.4f}"
        else:
            text = (
                f"{row.score_min:.4f} to {row.score_max:.4f} "
                f"({row.rank_min:.4f} to {row.rank_max:.4f})"
            )
        cells.setdefault((row.phase, row.pool, row.source), []).append(text)
    return [
        _row((phase, pool, _short(source) if source else "", *texts))
        for (phase, pool, source), texts in cells.items()
    ]


def _step_name(step: Step) -> str:
    return f"{step.pool} {_short(step.source)}" if step.source else step.pool


def _short(name: str | None) -> str:
    """Return a model id without its organization, such as ``paraphrase-multilingual-...``."""
    return "" if name is None else name.rsplit("/", 1)[-1]


def _percent(share: float) -> str:
    return f"{share * 100:g} %"


def _by_stratum(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{stratum} {count:,}" for stratum, count in counts.items())


def _cells(counts: Mapping[str, int], strata: Sequence[str]) -> list[str]:
    return [f"{counts[stratum]:,}" for stratum in strata]


def _row(cells: Sequence[str]) -> str:
    return f"| {' | '.join(cells)} |"


def _rule(columns: int) -> str:
    return "|" + "---|" * columns


def _settings_text(settings: LabelingConfig) -> str:
    values = settings.model_dump(exclude={"records_dir"})
    return ", ".join(f"{name} {value:g}" for name, value in values.items())


# --- Writing --------------------------------------------------------------------------------


def tracked_texts(run: DrawRun, summary: str) -> dict[str, str]:
    """Return the text of each tracked output but the manifest, by file name: a table as the
    JSON of its rows, one per line."""
    return {
        COMPOSITION_FILE_NAME: "\n".join(row.model_dump_json() for row in run.composition),
        RANGES_FILE_NAME: "\n".join(row.model_dump_json() for row in run.ranges),
        CUTS_FILE_NAME: json.dumps(run.cuts, indent=2, sort_keys=True, ensure_ascii=False),
        SUMMARY_FILE_NAME: summary,
    }


def write_outputs(
    run: DrawRun,
    *,
    results: Path,
    records: Path,
    summary: str,
    manifest: DrawManifest,
    store: TableStore | None = None,
    write_text: Callable[[Path, str], None] = write_text_atomically,
) -> DrawManifest:
    """Write the records once every tracked output passes the privacy guard: the gitignored
    pairs first, then the tracked tables, ``cuts.json`` and ``summary.md``, then
    ``manifest.json`` last, and return the manifest with the SHA-256 of every file written.

    Raises:
        PrivacyError: if a tracked output, the manifest included, holds a string of the snapshot
            or a doc code; nothing is written then.
        OSError: if writing fails.
    """
    store = store or ParquetTableStore()
    texts = tracked_texts(run, summary)
    check_privacy({**texts, MANIFEST_FILE_NAME: manifest.model_dump_json(indent=2)}, run.guarded)
    results.mkdir(parents=True, exist_ok=True)
    records.mkdir(parents=True, exist_ok=True)
    store.write(run.pairs, records / PAIRS_FILE_NAME, model_cls=PairRow)
    store.write(run.composition, results / COMPOSITION_FILE_NAME, model_cls=CompositionRow)
    store.write(run.ranges, results / RANGES_FILE_NAME, model_cls=RangeRow)
    write_text(results / CUTS_FILE_NAME, texts[CUTS_FILE_NAME])
    write_text(results / SUMMARY_FILE_NAME, summary.rstrip("\n"))
    tracked = (COMPOSITION_FILE_NAME, RANGES_FILE_NAME, CUTS_FILE_NAME, SUMMARY_FILE_NAME)
    final = manifest.model_copy(
        update={
            "files": {name: _sha256(results / name) for name in tracked},
            "data_files": {PAIRS_FILE_NAME: _sha256(records / PAIRS_FILE_NAME)},
        }
    )
    text = final.model_dump_json(indent=2)
    check_privacy({MANIFEST_FILE_NAME: text}, run.guarded)
    # The commit point: without it, the draw is not complete.
    write_text(results / MANIFEST_FILE_NAME, text)
    return final


def existing_draw(results: Path, records: Path) -> Path | None:
    """Return the record of an earlier draw, ``pairs.parquet`` or ``manifest.json``, if any."""
    for path in (records / PAIRS_FILE_NAME, results / MANIFEST_FILE_NAME):
        if path.exists():
            return path
    return None


# --- The command ----------------------------------------------------------------------------


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.labeling.draw_pairs",
        description=(
            "Draw the pilot and the labeled set of a snapshot (D34) once, with the values of "
            "labeling in the configuration, and write the drawn pairs under "
            "labeling.records_dir/<snapshot>/ (doc codes, gitignored) and the aggregates under "
            "results/labeled_set/<snapshot>/draw/ (tracked by git). An existing draw is never "
            "overwritten."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument("--snapshot-id", help="snapshot to use (default: the harvested last)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print the draw, apply the privacy guard, and write nothing",
    )
    return parser.parse_args(argv)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[AppConfig, Path, str], DrawInputs] = load_inputs,
    git: Callable[[Path, str], GitState] = read_git_state,
    clock: Callable[[], datetime] = _now,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and 130
    after an interruption."""
    arguments = _parse_arguments(argv)
    root = arguments.project_root.resolve()
    try:
        config = load_config(root / CONFIG_PATH)
        paths = config.paths.resolve_against(root)
        snapshot_id = check_snapshot_id(
            arguments.snapshot_id or latest_snapshot_id(paths.data_dir / RAW_DIR_NAME)
        )
        results = results_dir(paths.results_dir, snapshot_id)
        records = records_dir(config, root, snapshot_id)
        relative = results.relative_to(root).as_posix()
        if arguments.dry_run:
            return _dry_run(
                config, root, snapshot_id, results, records, load=load, git=git, timer=timer
            )
        state = git(root, relative)
        if state.dirty:
            raise DrawError(
                f"the git tree has {len(state.dirty)} changed or untracked paths outside "
                f"{relative}; commit them first, since the manifest records the commit"
            )
        _refuse_an_earlier_draw(snapshot_id, results, records)
        results.mkdir(parents=True, exist_ok=True)
        busy = DrawError(f"another draw holds the lock of {relative}; let it finish first")
        with exclusive_run(results, busy):
            _refuse_an_earlier_draw(snapshot_id, results, records)
            started_at, started = clock(), timer()
            run = compute(config, root, snapshot_id, load=load, report=_progress, timer=timer)
            summary = render_summary(run, git_commit=state.commit)
            _progress(f"writing the records, {MANIFEST_FILE_NAME} last")
            finished_at, seconds = clock(), timer() - started
            manifest = write_outputs(
                run,
                results=results,
                records=records,
                summary=summary,
                manifest=_manifest(
                    run,
                    git_commit=state.commit,
                    started_at=started_at,
                    finished_at=finished_at,
                    seconds=seconds,
                ),
            )
    except KeyboardInterrupt:
        print(
            f"interrupted: a draw is complete only once {MANIFEST_FILE_NAME} is written; move the "
            "partial records away, if any, and run the command again",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    print(f"Snapshot {snapshot_id}: commit {manifest.git_commit}, seed {manifest.seed}")
    print(describe(run, seconds=manifest.seconds))
    print(f"Wrote {relative}/ and the gitignored {PAIRS_FILE_NAME}, {MANIFEST_FILE_NAME} last")
    return 0


def _refuse_an_earlier_draw(snapshot_id: str, results: Path, records: Path) -> None:
    earlier = existing_draw(results, records)
    if earlier is not None:
        raise FileExistsError(
            f"a draw of snapshot {snapshot_id} already exists ({earlier}); a draw is a one-time "
            "event, so it is never overwritten: move its records away first to draw again"
        )


def _dry_run(
    config: AppConfig,
    root: Path,
    snapshot_id: str,
    results: Path,
    records: Path,
    *,
    load: Callable[[AppConfig, Path, str], DrawInputs],
    git: Callable[[Path, str], GitState],
    timer: Callable[[], float],
) -> int:
    relative = results.relative_to(root).as_posix()
    try:
        state: GitState | None = git(root, relative)
    except OSError:
        state = None
    started = timer()
    run = compute(config, root, snapshot_id, load=load, report=_progress, timer=timer)
    texts = tracked_texts(
        run, render_summary(run, git_commit=None if state is None else state.commit)
    )
    check_privacy(texts, run.guarded)
    seconds = timer() - started
    if state is None:
        tree = "not readable here; a real run needs a repository with a commit"
    elif not state.dirty:
        tree = f"commit {state.commit}, clean"
    else:
        count = len(state.dirty)
        tree = (
            f"commit {state.commit}, {count} changed path{'s' if count > 1 else ''} outside "
            f"{relative}; a real run refuses them until they are committed"
        )
    earlier = existing_draw(results, records)
    print(
        "\n".join(
            [
                f"Dry run of the labeled-set draw on snapshot {snapshot_id}: everything is "
                "computed and nothing is written",
                f"Git: {tree}",
                *(
                    []
                    if earlier is None
                    else [
                        f"A draw of this snapshot exists ({earlier.name}); a real run refuses to "
                        "overwrite it"
                    ]
                ),
                describe(run, seconds=seconds),
                f"Privacy guard: {len(texts)} tracked outputs checked against "
                f"{len(run.guarded):,} strings (the snapshot's, and the theses' doc codes), no hit "
                "(the manifest is checked when it is written)",
                f"Command of the real run: {RUN_COMMAND} --snapshot-id {snapshot_id}",
            ]
        )
    )
    return 0


def _manifest(
    run: DrawRun, *, git_commit: str, started_at: datetime, finished_at: datetime, seconds: float
) -> DrawManifest:
    settings = run.settings
    return DrawManifest(
        snapshot_id=run.snapshot_id,
        git_commit=git_commit,
        seed=settings.seed,
        settings={
            **settings.model_dump(mode="json"),
            "records_dir": settings.records_dir.as_posix(),
            "embedding_document_vector": run.config.embedding.document_vector,
            "tfidf_document_vector": run.config.tfidf.document_vector,
        },
        representations=run.universe.representations,
        inputs=run.input_sha256,
        counts=run.counts,
        versions=library_versions(),
        cpu=cpu_record(),
        files={},
        data_files={},
        started_at=started_at,
        finished_at=finished_at,
        seconds=round(max(seconds, 0.0), 3),
        timings=run.timings,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
