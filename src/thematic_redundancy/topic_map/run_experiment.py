"""Run the experiment of T19 on a snapshot, or forecast it with ``--dry-run``::

    uv run python -m thematic_redundancy.topic_map.run_experiment [--project-root PATH]
        [--snapshot-id ID] [--dry-run]

Before anything else the command checks the D30 declaration against its pin and the fallback
model against ``embedding.models``. It then loads and verifies the inputs: the document vectors
of both models (:func:`~thematic_redundancy.representation.build_embeddings.load_embeddings`),
the TF-IDF matrix and its terms (:func:`~thematic_redundancy.representation.build_tfidf.
load_tfidf`) and the full-clean section tokens (:func:`~thematic_redundancy.representation.
build_tfidf.load_full_clean_sections`), all in one order of doc codes.

``--dry-run`` stops there: it prints the counts of UMAP fits, clusterings, c-TF-IDF fits and
metric evaluations, the input hashes and the state of the git tree, and fits, clusters and
writes nothing. A real run first refuses a dirty git tree (changes outside its own results
directory), since the manifest records the commit; then it runs
:func:`~.experiment.run_experiment` and writes the records of :mod:`.results_store`,
``manifest.json`` last. Progress goes to stderr, a summary without text to stdout; the exit
code is 0 on success, 1 after an error, and 130 after an interruption.
"""

import argparse
import hashlib
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from thematic_redundancy.corpus.build_fichas import FICHAS_MANIFEST_FILE_NAME
from thematic_redundancy.corpus.snapshot import RAW_DIR_NAME, check_snapshot_id, latest_snapshot_id
from thematic_redundancy.extraction.text_manifest import INTERIM_DIR_NAME
from thematic_redundancy.representation.build_embeddings import (
    EMBEDDINGS_DIR_NAME,
    EMBEDDINGS_MANIFEST_FILE_NAME,
    load_embeddings,
)
from thematic_redundancy.representation.build_tfidf import (
    TFIDF_DIR_NAME,
    TFIDF_MANIFEST_FILE_NAME,
    CleanedSection,
    load_full_clean_sections,
    load_tfidf,
)
from thematic_redundancy.shared.config import AppConfig, load_config
from thematic_redundancy.topic_map.build import TopicMapPorts
from thematic_redundancy.topic_map.experiment import (
    ExperimentError,
    ExperimentInputs,
    ExperimentPlan,
    ExperimentResult,
    ExperimentSettings,
    SemanticInput,
    plan_experiment,
    run_experiment,
)
from thematic_redundancy.topic_map.results_store import (
    UMAP_DIR_NAME,
    ExperimentManifest,
    FileReductionCache,
    cpu_record,
    library_versions,
    processed_dir,
    results_dir,
    write_outputs,
)

CONFIG_PATH = Path("config") / "default.yaml"
RUN_COMMAND = "uv run python -m thematic_redundancy.topic_map.run_experiment"


@dataclass(frozen=True)
class GitState:
    commit: str
    dirty: tuple[str, ...]
    """Changed or untracked paths outside the run's own results directory."""


def read_git_state(project_root: Path, results_path: str) -> GitState:
    """Return the commit of ``project_root`` and its changed paths, ``results_path`` (relative,
    with ``/``) left out.

    Raises:
        OSError: if git cannot tell, such as outside a repository with a commit.
    """

    def git(*arguments: str) -> str:
        done = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
        if done.returncode != 0:
            raise OSError(f"git {arguments[0]} failed in {project_root}: {done.stderr.strip()}")
        return done.stdout

    commit = git("rev-parse", "HEAD").strip()
    entries = iter(git("status", "--porcelain", "-z", "--untracked-files=all").split("\0"))
    dirty = []
    for entry in entries:
        if not entry:
            continue
        if entry[0] in "RC":
            next(entries, None)  # a rename or copy also names its source
        path = entry[3:]
        if path != results_path and not path.startswith(f"{results_path}/"):
            dirty.append(path)
    return GitState(commit=commit, dirty=tuple(dirty))


def documents_by_row(
    sections: Sequence[CleanedSection], doc_codes: Sequence[str]
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    """Return each thesis's section tokens, in the order of ``doc_codes``.

    Raises:
        ValueError: if a section's row or doc code does not match ``doc_codes``, or a thesis has
            no section.
    """
    grouped: list[list[tuple[str, ...]]] = [[] for _ in doc_codes]
    for section in sections:
        row = section.document_row
        if not 0 <= row < len(doc_codes) or doc_codes[row] != section.doc_code:
            raise ValueError("the full-clean sections do not follow the rows of the TF-IDF matrix")
    for section in sections:
        grouped[section.document_row].append(tuple(section.tokens))
    empty = sum(not document for document in grouped)
    if empty:
        raise ValueError(f"{empty} theses have no full-clean section")
    return tuple(tuple(document) for document in grouped)


def load_inputs(config: AppConfig, project_root: Path, snapshot_id: str) -> ExperimentInputs:
    """Load and verify the inputs of ``snapshot_id`` through the T14 and T15 readers.

    Raises:
        FileNotFoundError: if an input is missing.
        ValueError: if an input is stale or damaged, or the inputs disagree on the theses.
    """
    data_dir = config.paths.resolve_against(project_root).data_dir
    interim = data_dir / INTERIM_DIR_NAME / snapshot_id
    embeddings = [
        load_embeddings(interim, model_id, variant=config.embedding.document_vector)
        for model_id in config.embedding.models
    ]
    tfidf = load_tfidf(interim, project_root, variant=config.tfidf.document_vector)
    for embedded in embeddings:
        if embedded.doc_codes != tfidf.doc_codes:
            raise ValueError(f"the theses of {embedded.model_id} differ from the TF-IDF matrix's")
    manifests = (
        interim / EMBEDDINGS_DIR_NAME / EMBEDDINGS_MANIFEST_FILE_NAME,
        interim / TFIDF_DIR_NAME / TFIDF_MANIFEST_FILE_NAME,
        interim / FICHAS_MANIFEST_FILE_NAME,
    )
    return ExperimentInputs(
        snapshot_id=snapshot_id,
        doc_codes=tfidf.doc_codes,
        models=tuple(SemanticInput(e.model_id, e.revision, e.vectors) for e in embeddings),
        tfidf=tfidf.matrix,
        vocabulary=tfidf.terms,
        documents=documents_by_row(
            load_full_clean_sections(interim, project_root), tfidf.doc_codes
        ),
        embedding_variant=config.embedding.document_vector,
        tfidf_variant=tfidf.variant,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in manifests},
    )


def describe_plan(
    plan: ExperimentPlan, inputs: ExperimentInputs, *, declaration: str, git: GitState | None
) -> str:
    """Describe the dry run: counts and hashes only."""
    low, sweep, new = (plan.counts(source) for source in ("baseline", "sweep", "new"))
    if git is None:
        state = "not readable here; a real run needs a repository with a commit"
    elif not git.dirty:
        state = f"commit {git.commit}, clean"
    else:
        dirty = len(git.dirty)
        state = (
            f"commit {git.commit}, {dirty} changed path{'s' if dirty > 1 else ''} outside the "
            "results directory; a real run refuses them until they are committed"
        )
    models = ", ".join(f"{m.model_id} ({m.vectors.shape[1]} dimensions)" for m in inputs.models)
    return "\n".join(
        [
            f"Dry run of T19 on snapshot {inputs.snapshot_id}: nothing is fitted, clustered or "
            "written",
            f"Declaration SHA-256 {declaration} (the D30 pin)",
            f"Git: {state}",
            f"Inputs: {plan.documents} theses; {models}; TF-IDF {inputs.tfidf.shape[0]} x "
            f"{inputs.tfidf.shape[1]}, {inputs.tfidf.nnz} nonzeros",
            *(f"  {name} SHA-256 {digest}" for name, digest in inputs.input_sha256.items()),
            f"Grid: {plan.k_values} values of K, {plan.min_cluster_sizes} of min_cluster_size, "
            f"{plan.seeds} seeds, {plan.models} models and the TF-IDF baseline",
            f"UMAP fits: {low.umap_fits} ({plan.reductions} reductions, {plan.reductions_cached} "
            "of them cached; 1 repeatability refit; the 2-D map)",
            f"PCA fits: {low.pca_fits} (the inspection figure)",
            f"Clusterings: {low.clusterings} (the sweeps and 1 repeatability rerun), plus "
            f"{new.clusterings - low.clusterings} if the matched K lies outside the sweep",
            f"c-TF-IDF fits: {low.ctfidf_fits} (5 final configurations), plus "
            f"{sweep.ctfidf_fits - low.ctfidf_fits} if the matched K is not B0's K*",
            f"Metric evaluations: {low.internal_metrics} internal (plus "
            f"{new.internal_metrics - low.internal_metrics}), {low.reassigned_silhouettes} "
            f"reassigned-noise silhouettes, {low.npmi_evaluations} NPMI (plus "
            f"{sweep.npmi_evaluations - low.npmi_evaluations}), {low.ari_evaluations} ARI (plus "
            f"{new.ari_evaluations - low.ari_evaluations})",
            f"Command of the real run: {RUN_COMMAND} --snapshot-id {inputs.snapshot_id}",
        ]
    )


def describe(result: ExperimentResult, manifest: ExperimentManifest) -> str:
    """Describe a run in counts, choices and outcomes, without text."""
    selection, counts = result.selection, manifest.counts
    lines = [
        f"Snapshot {result.snapshot_id}: {len(result.doc_codes)} theses; commit "
        f"{manifest.git_commit}; declaration {manifest.declaration_sha256}",
        f"{counts['clusterings']} clusterings, {counts['umap_fits']} UMAP fits "
        f"({counts['reductions_reused']} reductions reused), {counts['ctfidf_fits']} c-TF-IDF "
        f"fits, {counts['npmi_evaluations']} NPMI and {counts['ari_evaluations']} ARI evaluations",
    ]
    for representation, choices in selection["choices"].items():
        parts = [
            f"{algorithm} {choice['parameter']}"
            + ("" if choice["eligible"] else " (ineligible)")
            + (" (at a bound)" if choice["at_bound"] else "")
            for algorithm, choice in choices.items()
        ]
        stage = selection["stage_one"].get(representation)
        decided = " (the baseline)"
        if stage:
            decision = stage["decision"]
            decided = (
                f"; stage 1: {decision['winner']} at step {decision['deciding_step']} "
                f"({decision['deciding_criterion']})"
            )
        lines.append(f"{representation}: {', '.join(parts)}{decided}")
    winner, stage_two = selection["winner"], selection["stage_two"]["decision"]
    one, four = selection["criterion_one"], selection["criterion_four"]
    lines += [
        f"Provisional winner: {winner['configuration']} ({winner['representation']}, "
        f"{winner['algorithm']} {winner['parameter']}), stage 2 at step "
        f"{stage_two['deciding_step']} ({stage_two['deciding_criterion']})",
        f"Matched K {selection['matched_k']['k']} ({selection['matched_k']['source']}); "
        f"representative run {selection['representative']['run_id']}",
        f"Criterion 1, NPMI part: {'met' if one['npmi_met'] else 'not met'}; F1 pending (T22)",
        f"Criterion 4: {'met' if four['met'] else 'not met'}",
        f"Findings: {len(selection['findings'])} (selection.json)",
        f"Repeatability: UMAP {'identical' if result.repeatability.umap_identical else 'differs'}, "
        f"K-means {'identical' if result.repeatability.kmeans_identical else 'differs'}",
        f"Seconds: {manifest.seconds:.1f}",
    ]
    return "\n".join(lines)


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m thematic_redundancy.topic_map.run_experiment",
        description=(
            "Run the T19 experiment of the D30 declaration on a snapshot and write its records "
            "under results/experiments/<snapshot>/t19/ and data/processed/<snapshot>/."
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
        help="check the declaration and the inputs and print the forecast; fit nothing",
    )
    return parser.parse_args(argv)


def _progress(line: str) -> None:
    print(line, file=sys.stderr, flush=True)


def _now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def main(
    argv: Sequence[str] | None = None,
    *,
    load: Callable[[AppConfig, Path, str], ExperimentInputs] = load_inputs,
    git: Callable[[Path, str], GitState] = read_git_state,
    ports: TopicMapPorts | None = None,
    clock: Callable[[], datetime] = _now,
    timer: Callable[[], float] = time.perf_counter,
) -> int:
    """Run the command line and return its exit code: 0 on success, 1 after an error, and 130
    after an interruption."""
    arguments = _parse_arguments(argv)
    root = arguments.project_root.resolve()
    try:
        config = load_config(root / CONFIG_PATH)
        settings = ExperimentSettings.from_config(config)
        declaration = config.experiment_declaration_sha256()
        paths = config.paths.resolve_against(root)
        snapshot_id = check_snapshot_id(
            arguments.snapshot_id or latest_snapshot_id(paths.data_dir / RAW_DIR_NAME)
        )
        results = results_dir(paths.results_dir, snapshot_id)
        processed = processed_dir(paths.data_dir, snapshot_id)
        relative = results.relative_to(root).as_posix()
        versions = library_versions()
        cache = FileReductionCache(processed / UMAP_DIR_NAME, versions)
        if arguments.dry_run:
            try:
                state: GitState | None = git(root, relative)
            except OSError:
                state = None
            inputs = load(config, root, snapshot_id)
            plan = plan_experiment(inputs, settings, cache=cache)
            print(describe_plan(plan, inputs, declaration=declaration, git=state))
            return 0
        state = git(root, relative)
        if state.dirty:
            raise ExperimentError(
                f"the git tree has {len(state.dirty)} changed or untracked paths outside "
                f"{relative}; commit them first, since the manifest records the commit"
            )
        started_at, started = clock(), timer()
        inputs = load(config, root, snapshot_id)
        result = run_experiment(
            inputs, settings, ports=ports, cache=cache, timer=timer, report=_progress
        )
        _progress("writing the records, manifest.json last")
        manifest = write_outputs(
            result,
            results=results,
            processed=processed,
            declaration_sha256=declaration,
            git_commit=state.commit,
            input_sha256=inputs.input_sha256,
            started_at=started_at,
            finished_at=clock(),
            seconds=timer() - started,
            versions=versions,
            cpu=cpu_record(),
        )
    except KeyboardInterrupt:
        print(
            "interrupted: the UMAP reductions saved so far are reused by the next run, and "
            "records that differ from manifest.json are stale; run the command again",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {' '.join(str(error).split())}", file=sys.stderr)
        return 1
    print(describe(result, manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
