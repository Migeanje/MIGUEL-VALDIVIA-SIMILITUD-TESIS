"""Rank the lemmas of the theses by document frequency, to curate the domain stopwords (T11).

Run it from the project root, offline::

    uv run python experiments/domain_stopwords.py [--snapshot-id ID]

It reads ``data/raw/<snapshot_id>/`` (by default the snapshot harvested last) through
:func:`read_snapshot`, which opens only ``manifest.json`` and ``metadata.jsonl``, and keeps the
theses: the items with the exact ``renati.type`` fragment of the configuration. Exact duplicate
records, which repeat both the title and the abstract of an earlier one, count once (D24).

Each thesis gives one document: its title, light-cleaned and without its place and year
suffix, followed by its light-cleaned abstract. The full cleaner turns each document into
lemmas, with spaCy's Spanish stopwords removed but no domain stopwords, and the document
frequency of a lemma is the number of documents that hold it.

The output is ``results/eda/<snapshot_id>/stopword_candidates.json``: every lemma held by at
least :data:`MIN_DOCUMENT_RATIO` of the documents, with its document count and share, plus the
provenance and the method. It holds aggregates only. Before it is written, the output is
searched for every title, abstract, long abstract sentence, person's name and ORCID of the
snapshot, and each candidate must be a single lemma; one failure stops the run with nothing
written. While it writes, the script holds the lock ``results/eda/.lock``
(:func:`exclusive_run`), which ``experiments/eda_metadata.py`` also takes, so two runs never
stage the same file at once; a run that finds the lock taken writes nothing and exits with 1.

``config/stopwords_domain_es.txt`` is then curated by hand from the candidates.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from thematic_redundancy.corpus.profile import (
    ABSTRACT_KEY,
    PERSON_KEYS,
    TITLE_KEY,
    leaked_strings,
    sensitive_strings,
)
from thematic_redundancy.corpus.snapshot import (
    RAW_DIR_NAME,
    SnapshotManifest,
    SnapshotRecord,
    check_snapshot_id,
    latest_snapshot_id,
    read_snapshot,
)
from thematic_redundancy.corpus.thesis_files import is_thesis
from thematic_redundancy.preprocessing.full_cleaner import (
    DEFAULT_BATCH_SIZE,
    SPACY_MODEL,
    clean_full_batch,
    spanish_pipeline,
)
from thematic_redundancy.preprocessing.light_cleaner import clean_light, normalize_text
from thematic_redundancy.preprocessing.stopwords import stopword_candidates
from thematic_redundancy.preprocessing.title_suffix import strip_title_suffix
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.shared.filesystem import exclusive_run, write_bytes_atomically

CONFIG_PATH = Path("config") / "default.yaml"
SCRIPT_PATH = "experiments/domain_stopwords.py"
OUTPUT_DIR_NAME = "eda"
OUTPUT_FILE_NAME = "stopword_candidates.json"

MIN_DOCUMENT_RATIO = 0.05
"""Share of the documents that a lemma must reach to be a candidate: 5%, or 38 of the 744
distinct theses of snapshot ``20261002T224412Z``. Genre and boilerplate words lie far above it
(``presente`` is in two thirds of the theses), and a rarer word weighs little in TF-IDF."""

MAX_LEMMA_LENGTH = 40
"""Longest candidate that the output may hold; a longer one is no single lemma."""


class OutputsInUseError(RuntimeError):
    """Another run is writing under ``results/eda/``, the directory that this script locks."""


def _in_use_message(eda_dir: Path) -> str:
    return (
        f"another run is writing the outputs in {eda_dir}; wait until it ends, then run the "
        "script again"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Rank the lemmas, check the output for leaks, write it, and return the exit code."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    config_path = project_root / CONFIG_PATH
    try:
        config = load_config(config_path)
        paths = config.paths.resolve_against(project_root)
        raw_dir = paths.data_dir / RAW_DIR_NAME
        if arguments.snapshot_id is None:
            snapshot_id = latest_snapshot_id(raw_dir)
        else:
            snapshot_id = check_snapshot_id(arguments.snapshot_id)
        manifest, records = read_snapshot(raw_dir / snapshot_id)
        load_started = time.perf_counter()
        nlp = spanish_pipeline()
        load_seconds = time.perf_counter() - load_started
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    theses = [
        record for record in records if is_thesis(record.metadata, config.snapshot.thesis_type)
    ]
    distinct = distinct_theses(theses)
    documents = [thesis_document(record) for record in distinct]
    started = time.perf_counter()
    tokens = clean_full_batch(documents, domain_stopwords=frozenset())
    seconds = time.perf_counter() - started
    candidates = stopword_candidates(tokens, min_ratio=MIN_DOCUMENT_RATIO)
    document = {
        "provenance": _provenance(project_root, config_path, manifest, nlp.meta),
        "method": {
            "theses": len(theses),
            "documents": len(documents),
            "duplicates_counted_once": len(theses) - len(distinct),
            "document_text": (
                "title, light-cleaned and without its place and year suffix, followed by the "
                "light-cleaned abstract"
            ),
            "tokens": "full cleaner with spaCy's Spanish stopwords and no domain stopwords",
            "min_document_ratio": MIN_DOCUMENT_RATIO,
            "candidates": len(candidates),
        },
        "candidates": [candidate.model_dump() for candidate in candidates],
    }
    content = f"{json.dumps(document, ensure_ascii=False, indent=2)}\n".encode()
    problems = privacy_problems(content, [c.lemma for c in candidates], records)
    if problems:
        for problem in problems:
            print(f"error: {problem}", file=sys.stderr)
        print("error: nothing was written", file=sys.stderr)
        return 1
    eda_dir = paths.results_dir / OUTPUT_DIR_NAME
    output_path = eda_dir / snapshot_id / OUTPUT_FILE_NAME
    try:
        eda_dir.mkdir(parents=True, exist_ok=True)
        with exclusive_run(eda_dir, OutputsInUseError(_in_use_message(eda_dir))):
            _write_atomically(output_path, content)
    except OutputsInUseError as error:
        print(f"error: {error}", file=sys.stderr)
        print("error: nothing was written", file=sys.stderr)
        return 1
    name_words = _person_name_words(records)
    print(
        f"Ranked the lemmas of snapshot {snapshot_id}: {len(theses)} theses, "
        f"{len(documents)} distinct documents"
    )
    print(
        f"Full cleaner: {len(documents)} documents in {seconds:.1f} s "
        f"({_throughput(len(documents), seconds)}, batch size {DEFAULT_BATCH_SIZE}); "
        f"model load {load_seconds:.1f} s"
    )
    print(
        f"Candidates: {len(candidates)} lemmas in at least {MIN_DOCUMENT_RATIO:.0%} of the "
        "documents"
    )
    print(
        "Privacy check: the output was searched for the titles, abstracts, abstract sentences, "
        "person names and ORCIDs of the snapshot (0 hits); every candidate is a single lemma. "
        f"Candidates equal to a word of a person's name, for review: "
        f"{sum(c.lemma in name_words for c in candidates)}"
    )
    print(f"Wrote {output_path} ({len(content):,} bytes)")
    return 0


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python {SCRIPT_PATH}",
        description=(
            "Rank the lemmas of the theses by document frequency into "
            f"results/eda/<snapshot_id>/{OUTPUT_FILE_NAME}."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument("--snapshot-id", help="snapshot to read (default: the one harvested last)")
    return parser.parse_args(argv)


def distinct_theses(theses: Iterable[SnapshotRecord]) -> list[SnapshotRecord]:
    """Return ``theses`` without the records that repeat an earlier title and abstract.

    The texts are compared as the metadata profile compares them: whitespace collapsed and
    case folded, accents kept. The dataset builder (T12) fixes which record of a group stays.
    """
    seen: set[tuple[str, str]] = set()
    distinct = []
    for record in theses:
        key = (_first_text(record, TITLE_KEY), _first_text(record, ABSTRACT_KEY))
        if all(key) and key in seen:
            continue
        seen.add(key)
        distinct.append(record)
    return distinct


def thesis_document(record: SnapshotRecord) -> str:
    """Return the title without its suffix and the abstract of a thesis, light-cleaned."""
    title = strip_title_suffix(clean_light(_first_value(record, TITLE_KEY))).title
    abstract = clean_light(_first_value(record, ABSTRACT_KEY))
    return f"{title}\n\n{abstract}" if abstract else title


def privacy_problems(
    content: bytes, lemmas: Sequence[str], records: Sequence[SnapshotRecord]
) -> list[str]:
    """Return why ``content`` may not be published, or nothing when it may."""
    problems = []
    text = content.decode("utf-8")
    leaks = leaked_strings(text, sensitive_strings(records))
    if leaks:
        problems.append(f"the output holds {len(leaks)} string(s) taken from the snapshot")
    malformed = [
        lemma
        for lemma in lemmas
        if not lemma or len(lemma) > MAX_LEMMA_LENGTH or any(char.isspace() for char in lemma)
    ]
    if malformed:
        problems.append(f"{len(malformed)} candidate(s) are not a single lemma")
    return problems


def _throughput(documents: int, seconds: float) -> str:
    """Return the documents cleaned per second, or say it is unavailable.

    It is unavailable when the clock measured no time, as an empty snapshot or a coarse
    clock can give. The output is written by then, so the run still ends normally.
    """
    if seconds > 0:
        return f"{documents / seconds:.1f} documents/s"
    return "throughput unavailable"


def _person_name_words(records: Iterable[SnapshotRecord]) -> frozenset[str]:
    """Return the lower-case words of every author, advisor and juror name."""
    words = set()
    for record in records:
        for key in PERSON_KEYS:
            for name in _values(record, key):
                words.update(normalize_text(name).lower().replace(",", " ").split())
    return frozenset(words)


def _values(record: SnapshotRecord, key: str) -> list[str]:
    return [
        text.strip()
        for entry in record.metadata.get(key, ())
        if isinstance(text := entry.get("value"), str) and text.strip()
    ]


def _first_value(record: SnapshotRecord, key: str) -> str:
    values = _values(record, key)
    return values[0] if values else ""


def _first_text(record: SnapshotRecord, key: str) -> str:
    return normalize_text(_first_value(record, key)).casefold()


def _provenance(
    project_root: Path, config_path: Path, manifest: SnapshotManifest, meta: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "script": SCRIPT_PATH,
        "snapshot_id": manifest.snapshot_id,
        "harvested_at": manifest.harvested_at.isoformat().replace("+00:00", "Z"),
        "metadata_sha256": manifest.metadata_sha256,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_commit": _git(project_root, "rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(
            _git(project_root, "status", "--porcelain", "--untracked-files=no")
        ),
        "spacy_model": f"{SPACY_MODEL} {meta.get('version', 'unknown')}",
        "spacy_version": metadata.version("spacy"),
    }


def _git(project_root: Path, *arguments: str) -> str | None:
    """Return the output of a git command, or ``None`` when git is unavailable or fails."""
    try:
        run = subprocess.run(
            ["git", *arguments], cwd=project_root, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return run.stdout.strip() if run.returncode == 0 else None


def _write_atomically(path: Path, content: bytes) -> None:
    """Create the directory of ``path``, then put ``content`` there atomically and durably.

    :func:`write_bytes_atomically` flushes a staged file beside ``path`` before it moves it
    into place, and the directory after it, so readers and a crash find the previous output
    or the new one, never a partial one. A directory that cannot be opened (Windows) or
    flushed (EINVAL on some network, FUSE or overlay file systems) is skipped, since the
    output is already in place by then.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomically(path, content)


if __name__ == "__main__":
    raise SystemExit(main())
