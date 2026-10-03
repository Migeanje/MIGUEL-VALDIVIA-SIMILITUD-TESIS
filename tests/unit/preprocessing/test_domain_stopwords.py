"""Domain stopwords: the curated list of genre and boilerplate lemmas, how it is read, and the
script that ranks its candidates (``experiments/domain_stopwords.py``).

The list files, records and outputs written here are synthetic; one group of tests reads the
curated list of the project. The script tests load no language model: the lemmas and the
clock are stand-ins.
"""

import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from thematic_redundancy.corpus.snapshot import (
    ProgramSummary,
    SnapshotManifest,
    SnapshotRecord,
    encode_records,
    write_snapshot,
)
from thematic_redundancy.preprocessing.stopwords import (
    DOMAIN_STOPWORDS_FILE,
    StopwordCandidate,
    load_domain_stopwords,
    spacy_stopwords,
    stopword_candidates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CURATED_LIST = PROJECT_ROOT / DOMAIN_STOPWORDS_FILE
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "domain_stopwords.py"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_script() -> ModuleType:
    """Import the candidates script, which lives outside the package, as a module."""
    spec = importlib.util.spec_from_file_location("domain_stopwords_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()


def write_list(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "stopwords.txt"
    path.write_text(content, encoding="utf-8")
    return path


def test_load_domain_stopwords_reads_one_lemma_per_line_and_skips_comments_and_blanks(
    tmp_path: Path,
) -> None:
    path = write_list(tmp_path, "# Géneros de la tesis\n\ntesis\n   investigación  \n#otro\nperú\n")

    assert load_domain_stopwords(path) == {"tesis", "investigación", "perú"}


def test_load_domain_stopwords_puts_each_lemma_in_nfc_and_lower_case(tmp_path: Path) -> None:
    path = write_list(tmp_path, "Peru\u0301\nAREQUIPA\nInvestigacio\u0301n\n")

    assert load_domain_stopwords(path) == {"perú", "arequipa", "investigación"}


def test_load_domain_stopwords_refuses_a_line_with_more_than_one_word(tmp_path: Path) -> None:
    path = write_list(tmp_path, "tesis\nmejora continua\n")

    with pytest.raises(ValueError, match="line 2"):
        load_domain_stopwords(path)


def test_load_domain_stopwords_reads_an_empty_list(tmp_path: Path) -> None:
    path = write_list(tmp_path, "# Sin palabras todavía.\n\n")

    assert load_domain_stopwords(path) == frozenset()


def test_stopword_candidates_rank_lemmas_by_the_documents_that_hold_them() -> None:
    documents = [
        ["tesis", "planta", "tesis", "voladura"],
        ["tesis", "planta"],
        ["tesis", "costo"],
        ["proceso"],
    ]

    candidates = stopword_candidates(documents, min_ratio=0.5)

    assert candidates == [
        StopwordCandidate(lemma="tesis", documents=3, ratio=0.75),
        StopwordCandidate(lemma="planta", documents=2, ratio=0.5),
    ]


def test_stopword_candidates_break_ties_in_lemma_order_and_round_ratios() -> None:
    documents = [["voladura", "calidad"], ["calidad", "voladura"], ["costo"]]

    candidates = stopword_candidates(documents, min_ratio=0.5)

    assert [candidate.lemma for candidate in candidates] == ["calidad", "voladura"]
    assert candidates[0].ratio == 0.6667


def test_stopword_candidates_of_no_documents_are_none() -> None:
    assert stopword_candidates([], min_ratio=0.1) == []


@pytest.mark.parametrize("min_ratio", [0.0, -0.1, 1.5])
def test_stopword_candidates_refuse_a_ratio_outside_zero_to_one(min_ratio: float) -> None:
    with pytest.raises(ValueError, match="min_ratio"):
        stopword_candidates([["tesis"]], min_ratio=min_ratio)


def test_spacy_stopwords_are_the_lower_case_spanish_list() -> None:
    stopwords = spacy_stopwords()

    assert {"de", "la", "los", "que", "para"} <= stopwords
    assert all(word == word.lower() for word in stopwords)


# The curated list

GENRE_AND_BOILERPLATE = {
    "tesis",
    "investigación",
    "objetivo",
    "presente",
    "propuesta",
    "universidad",
    "arequipa",
    "perú",
}
FREQUENT_WORDS_LEFT_OUT = {"empresa", "mejora", "mejorar", "desarrollo", "desarrollar"}
"""Frequent words that fit no criterion of the list, so they stay out of it (T11a)."""
TOPIC_WORDS = {
    "productividad",
    "mantenimiento",
    "calidad",
    "costo",
    "seguridad",
    "voladura",
    "energía",
    "sistema",
    "proceso",
    "lean",
}


def test_the_curated_list_holds_genre_and_boilerplate_words_and_no_topic_words() -> None:
    stopwords = load_domain_stopwords(CURATED_LIST)

    assert stopwords >= GENRE_AND_BOILERPLATE
    assert stopwords.isdisjoint(TOPIC_WORDS)


def test_the_curated_list_leaves_out_frequent_words_that_fit_no_criterion() -> None:
    stopwords = load_domain_stopwords(CURATED_LIST)

    assert stopwords.isdisjoint(FREQUENT_WORDS_LEFT_OUT)
    assert "universidad" in stopwords


def test_the_curated_list_writes_each_lemma_once_in_nfc_lower_case() -> None:
    lines = [
        line.strip()
        for line in CURATED_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert len(lines) == len(set(lines))
    assert all(line == unicodedata.normalize("NFC", line).lower() for line in lines)


def test_the_curated_list_repeats_no_spacy_stopword() -> None:
    assert load_domain_stopwords(CURATED_LIST).isdisjoint(spacy_stopwords())


def test_the_curated_list_starts_with_its_curation_criteria() -> None:
    first_line = CURATED_LIST.read_text(encoding="utf-8").splitlines()[0]

    assert first_line.startswith("#")


# The candidates script: synthetic snapshot records

TESIS = "https://purl.org/pe-repo/renati/type#tesis"
HARVEST_TIME = datetime(2026, 10, 2, 22, 44, 12, tzinfo=UTC)
SNAPSHOT_ID = "20261002T224412Z"
COLLECTION = UUID("00000000-0000-4000-8000-0000000000c1")
FACULTY = UUID("00000000-0000-4000-8000-0000000000f1")

FieldValue = str | Sequence[str] | None


def entry(value: str) -> dict[str, Any]:
    """Return one metadata value in the shape that DSpace serves."""
    return {"value": value, "language": None, "authority": None, "confidence": -1, "place": 0}


def record(number: int, fields: Mapping[str, FieldValue] | None = None) -> SnapshotRecord:
    """Return a synthetic thesis with a title and an abstract of its own.

    ``fields`` replaces or adds metadata: a string gives one value, a sequence several, and
    ``None`` drops the field.
    """
    values: dict[str, FieldValue] = {
        "renati.type": TESIS,
        "dc.title": f"Título sintético {number:03d}",
        "dc.description.abstract": f"Resumen sintético {number:03d}.",
        **(fields or {}),
    }
    metadata = {
        key: [entry(value)] if isinstance(value, str) else [entry(item) for item in value]
        for key, value in values.items()
        if value is not None
    }
    return SnapshotRecord(
        uuid=UUID(f"00000000-0000-4000-8000-{number:012d}"),
        handle=f"123456789/{1000 + number}",
        program_key="sistemas",
        collection_uuid=COLLECTION,
        harvested_at=HARVEST_TIME,
        metadata=metadata,
    )


# The candidates script: the privacy gate

SENSITIVE_TITLE = "Diseño de un secador solar de bajo costo para granos andinos"
SENSITIVE_SENTENCE = "El estudio diseña un secador solar de bajo costo para granos andinos."
SENSITIVE_ABSTRACT = f"{SENSITIVE_SENTENCE} Se validó con tres prototipos en campo."
SENSITIVE_RECORDS = (
    record(
        1,
        {
            "dc.title": SENSITIVE_TITLE,
            "dc.description.abstract": SENSITIVE_ABSTRACT,
            "dc.contributor.author": "Ficticio Inventado, Ana Lucía",
            "dc.contributor.advisor": "Supuesto Imaginario, Juan",
            "renati.advisor.orcid": "https://orcid.org/0000-0002-1825-0097",
        },
    ),
    record(2),
)


def payload(lemmas: Iterable[str], note: str = "") -> bytes:
    """Return an output shaped like the script's: a method note and the candidate lemmas."""
    document = {
        "method": {"note": note},
        "candidates": [{"lemma": lemma, "documents": 40, "ratio": 0.5} for lemma in lemmas],
    }
    return f"{json.dumps(document, ensure_ascii=False, indent=2)}\n".encode()


def test_privacy_problems_pass_an_output_of_aggregates_and_single_lemmas() -> None:
    # Single words of a title or a name are no leak; only whole sensitive strings are.
    lemmas = ["tesis", "secador", "granos", "ficticio", "a" * script.MAX_LEMMA_LENGTH]

    assert script.privacy_problems(payload(lemmas), lemmas, SENSITIVE_RECORDS) == []


@pytest.mark.parametrize(
    "leak",
    [
        pytest.param(SENSITIVE_TITLE, id="title"),
        pytest.param(SENSITIVE_ABSTRACT, id="abstract"),
        pytest.param(SENSITIVE_SENTENCE.upper(), id="abstract sentence"),
        pytest.param("Ana Lucía Ficticio Inventado", id="person name"),
        pytest.param("0000-0002-1825-0097", id="orcid"),
    ],
)
def test_privacy_problems_refuse_an_output_that_holds_a_string_of_the_snapshot(
    leak: str,
) -> None:
    lemmas = ["tesis"]

    problems = script.privacy_problems(payload(lemmas, note=leak), lemmas, SENSITIVE_RECORDS)

    assert len(problems) == 1
    assert problems[0].endswith("string(s) taken from the snapshot")
    assert leak.casefold() not in problems[0].casefold()


@pytest.mark.parametrize(
    "lemma",
    [
        pytest.param("", id="empty"),
        pytest.param("mejora continua", id="space"),
        pytest.param("mejora\tcontinua", id="tab"),
        pytest.param("a" * (script.MAX_LEMMA_LENGTH + 1), id="too long"),
    ],
)
def test_privacy_problems_refuse_a_candidate_that_is_not_a_single_lemma(lemma: str) -> None:
    lemmas = ["tesis", lemma]

    problems = script.privacy_problems(payload(["tesis"]), lemmas, SENSITIVE_RECORDS)

    assert problems == ["1 candidate(s) are not a single lemma"]


def test_privacy_problems_report_a_leak_and_a_malformed_lemma_together() -> None:
    lemmas = ["tesis", ""]

    problems = script.privacy_problems(
        payload(lemmas, note=SENSITIVE_TITLE), lemmas, SENSITIVE_RECORDS
    )

    assert len(problems) == 2
    assert problems[1] == "1 candidate(s) are not a single lemma"


# The candidates script: exact duplicates count once (D24)

REPEATED = {"dc.title": "Título repetido", "dc.description.abstract": "Resumen repetido."}


def test_distinct_theses_keep_the_first_of_records_with_the_same_title_and_abstract() -> None:
    first, other, copy, last = record(1, REPEATED), record(2), record(3, REPEATED), record(4)

    assert script.distinct_theses([first, other, copy, last]) == [first, other, last]


def test_distinct_theses_compare_texts_in_nfc_with_whitespace_collapsed_and_case_folded() -> None:
    first = record(
        1, {"dc.title": " Ti\u0301tulo  Repetido", "dc.description.abstract": "Resumen\nrepetido."}
    )
    copy = record(
        2, {"dc.title": "TÍTULO repetido ", "dc.description.abstract": "resumen \t REPETIDO."}
    )

    assert script.distinct_theses([first, copy]) == [first]


def test_distinct_theses_keep_texts_that_differ_only_in_their_accents() -> None:
    first = record(1, REPEATED)
    other = record(
        2, {"dc.title": "Titulo repetido", "dc.description.abstract": "Resumen repetido."}
    )

    assert script.distinct_theses([first, other]) == [first, other]


def test_distinct_theses_compare_only_the_first_title_and_abstract() -> None:
    first = record(1, {**REPEATED, "dc.title": ["Título repetido", "Otro título"]})
    copy = record(2, {**REPEATED, "dc.title": ["Título repetido", "Título alterno"]})

    assert script.distinct_theses([first, copy]) == [first]


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({**REPEATED, "dc.description.abstract": None}, id="no abstract"),
        pytest.param({**REPEATED, "dc.description.abstract": "  \n "}, id="blank abstract"),
        pytest.param({**REPEATED, "dc.title": None}, id="no title"),
        pytest.param({"dc.title": None, "dc.description.abstract": None}, id="neither"),
    ],
)
def test_distinct_theses_never_drop_a_record_without_both_a_title_and_an_abstract(
    fields: Mapping[str, FieldValue],
) -> None:
    first, second = record(1, fields), record(2, fields)

    assert script.distinct_theses([first, second]) == [first, second]


# The candidates script: the document of a thesis

WRAPPED_TITLE = "Mejora de procesos en una plan-\nta textil, Arequipa 2025"
CLEAN_TITLE = "Mejora de procesos en una planta textil"


def test_thesis_document_is_the_title_without_its_suffix_then_the_abstract_light_cleaned() -> None:
    thesis = record(
        1,
        {
            "dc.title": WRAPPED_TITLE,
            "dc.description.abstract": (
                "El estudio mejora\nlos procesos de una planta.\n\nSe midió la productividad."
            ),
        },
    )

    assert script.thesis_document(thesis) == (
        f"{CLEAN_TITLE}\n\nEl estudio mejora los procesos de una planta.\n\n"
        "Se midió la productividad."
    )


@pytest.mark.parametrize(
    "abstract", [pytest.param(None, id="none"), pytest.param(" \n ", id="blank")]
)
def test_thesis_document_without_an_abstract_is_the_title_alone(abstract: str | None) -> None:
    thesis = record(1, {"dc.title": WRAPPED_TITLE, "dc.description.abstract": abstract})

    assert script.thesis_document(thesis) == CLEAN_TITLE


# The candidates script: a whole run


def write_test_snapshot(project_root: Path, records: Sequence[SnapshotRecord]) -> None:
    """Write ``records`` as snapshot :data:`SNAPSHOT_ID` under ``project_root``."""
    content = encode_records(records)
    manifest = SnapshotManifest(
        snapshot_id=SNAPSHOT_ID,
        harvested_at=HARVEST_TIME,
        base_url="https://repo.example.edu",
        search_url="https://repo.example.edu/server/api/discover/search/objects",
        query_parameters={},
        programs={"sistemas": ProgramSummary(collection_uuid=COLLECTION, items=len(records))},
        total_items=len(records),
        faculty_community_uuid=FACULTY,
        faculty_total=len(records),
        items_by_type={},
        future_dated_items=0,
        unparsed_issue_dates=0,
        dropped_metadata_keys=(),
        metadata_sha256=hashlib.sha256(content).hexdigest(),
    )
    write_snapshot(project_root / "data" / "raw" / SNAPSHOT_ID, content, manifest)


def stand_in_lemmas(texts: Iterable[str], *, domain_stopwords: frozenset[str]) -> list[list[str]]:
    """Stand-in for the full cleaner: the same two lemmas for every text."""
    return [["tesis", "planta"] for _ in texts]


@pytest.fixture
def cli_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a project root that holds a copy of the declared configuration.

    The script then loads no language model, asks git nothing, and reads a clock that
    stands still, so every timing that it measures is zero.
    """
    (tmp_path / "config").mkdir()
    shutil.copyfile(DEFAULT_CONFIG_PATH, tmp_path / "config" / "default.yaml")
    monkeypatch.setattr(script, "spanish_pipeline", lambda: SimpleNamespace(meta={}))
    monkeypatch.setattr(script, "clean_full_batch", stand_in_lemmas)
    monkeypatch.setattr(script, "_git", lambda *arguments: None)
    monkeypatch.setattr(script, "time", SimpleNamespace(perf_counter=lambda: 0.0))
    return tmp_path


@pytest.mark.parametrize("theses", [pytest.param(2, id="two theses"), pytest.param(0, id="none")])
def test_main_reports_the_throughput_as_unavailable_when_no_time_was_measured(
    cli_root: Path, capsys: pytest.CaptureFixture[str], theses: int
) -> None:
    write_test_snapshot(cli_root, [record(number) for number in range(1, theses + 1)])

    code = script.main(["--project-root", str(cli_root)])

    out = capsys.readouterr().out
    assert code == 0
    assert f"Full cleaner: {theses} documents in 0.0 s (throughput unavailable" in out
    output = cli_root / "results" / "eda" / SNAPSHOT_ID / "stopword_candidates.json"
    assert json.loads(output.read_text(encoding="utf-8"))["method"]["documents"] == theses
    assert f"Wrote {output}" in out


def test_main_reports_the_documents_cleaned_per_second(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, [record(1), record(2)])
    ticks = itertools.count()  # Each reading of the clock is one second after the last one.
    monkeypatch.setattr(script, "time", SimpleNamespace(perf_counter=lambda: float(next(ticks))))

    code = script.main(["--project-root", str(cli_root)])

    assert code == 0
    assert "Full cleaner: 2 documents in 1.0 s (2.0 documents/s, " in capsys.readouterr().out


# The candidates script: publishing the output durably


def record_file_calls(
    monkeypatch: pytest.MonkeyPatch, directory: Path, *, directory_opens: bool = True
) -> list[str]:
    """Record the flushes, moves and directory opens of a write, in order, and return the log.

    Opening ``directory`` gives a stand-in descriptor that is never flushed for real, since
    Windows cannot open a directory; with ``directory_opens`` false it fails as on Windows.
    Every other call goes through to :mod:`os`.
    """
    calls: list[str] = []
    real_open, real_fsync, real_close, real_replace = os.open, os.fsync, os.close, os.replace
    stand_in = directory.parent / "directory-stand-in"
    stand_in.write_bytes(b"")
    stand_ins: set[int] = set()

    def open_(path: Any, flags: int, *args: Any, **kwargs: Any) -> int:
        if Path(path) != directory:
            return real_open(path, flags, *args, **kwargs)
        calls.append("open directory")
        if not directory_opens:
            raise PermissionError(13, "Permission denied", str(path))
        descriptor = real_open(stand_in, os.O_RDONLY)
        stand_ins.add(descriptor)
        return descriptor

    def fsync(descriptor: int) -> None:
        if descriptor in stand_ins:
            calls.append("fsync directory")
            return
        calls.append(f"fsync file of {os.fstat(descriptor).st_size} bytes")
        real_fsync(descriptor)

    def close(descriptor: int) -> None:
        if descriptor in stand_ins:
            calls.append("close directory")
        real_close(descriptor)

    def replace(source: Any, target: Any) -> None:
        calls.append("replace")
        real_replace(source, target)

    monkeypatch.setattr(os, "open", open_)
    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "close", close)
    monkeypatch.setattr(os, "replace", replace)
    return calls


def test_write_atomically_flushes_the_file_then_moves_it_then_flushes_the_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "eda" / "candidates.json"
    path.parent.mkdir()
    calls = record_file_calls(monkeypatch, path.parent)

    script._write_atomically(path, b"{}\n")

    assert calls == [
        "fsync file of 3 bytes",
        "replace",
        "open directory",
        "fsync directory",
        "close directory",
    ]
    assert path.read_bytes() == b"{}\n"


def test_write_atomically_publishes_where_a_directory_cannot_be_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "eda" / "candidates.json"
    path.parent.mkdir()
    path.write_bytes(b"old\n")
    calls = record_file_calls(monkeypatch, path.parent, directory_opens=False)

    script._write_atomically(path, b"{}\n")

    assert calls == ["fsync file of 3 bytes", "replace", "open directory"]
    assert path.read_bytes() == b"{}\n"
    assert [entry.name for entry in path.parent.iterdir()] == ["candidates.json"]
