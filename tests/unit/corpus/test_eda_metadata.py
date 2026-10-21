"""The metadata profile script (``experiments/eda_metadata.py``): its observations, its notes
on the second tokenizer, the privacy gate that guards its outputs, and how it writes them.

Every record is synthetic, and no tokenizer is loaded: token counts come from a stand-in that
gives one token per word plus the two special tokens of a sequence. The tests of the privacy
gate and of the writing replace the figures with small stand-in outputs.
"""

import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from types import ModuleType
from typing import IO, Any
from uuid import UUID

import pytest

from thematic_redundancy.corpus.profile import TOP_ADVISORS, MetadataProfile, TokenCounter
from thematic_redundancy.corpus.snapshot import (
    ProgramSummary,
    SnapshotManifest,
    SnapshotRecord,
    encode_records,
    write_snapshot,
)
from thematic_redundancy.shared.filesystem import LOCK_FILE_NAME, exclusive_run

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "eda_metadata.py"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_script() -> ModuleType:
    """Import the profile script, which lives outside the package, as a module."""
    spec = importlib.util.spec_from_file_location("eda_metadata_script", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = load_script()
REAL_GIT = script._git
"""The script's own git helper, which the command-line fixture replaces with a stub."""

MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MPNET = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
OUTPUT_NAMES = (
    "metadata_profile.json",
    "summary.md",
    "program_year.png",
    "abstract_tokens.png",
    "title_tokens.png",
    "top_keywords.png",
)
PROGRAMS = {"sistemas": "Ingeniería de Sistemas", "industrial": "Ingeniería Industrial"}

TESIS = "https://purl.org/pe-repo/renati/type#tesis"
SUFICIENCIA = "https://purl.org/pe-repo/renati/type#trabajoDeSuficienciaProfesional"
OPEN = "https://purl.org/coar/access_right/c_abf2"
SNAPSHOT_ID = "20261002T224412Z"
SNAPSHOT_DATE = date(2026, 10, 2)
HARVEST_TIME = datetime(2026, 10, 2, 22, 44, 12, tzinfo=UTC)
COLLECTION = UUID("00000000-0000-4000-8000-0000000000c1")
FACULTY = UUID("00000000-0000-4000-8000-0000000000f1")

FieldValue = str | Sequence[str] | None


def count_tokens(text: str) -> int:
    """Stand-in tokenizer: one token per word, plus the two special tokens of a sequence."""
    return len(text.split()) + 2


def entry(value: str) -> dict[str, Any]:
    """Return one metadata value in the shape that DSpace serves."""
    return {"value": value, "language": None, "authority": None, "confidence": -1, "place": 0}


def record(
    number: int, fields: Mapping[str, FieldValue] | None = None, *, program: str = "sistemas"
) -> SnapshotRecord:
    """Return a synthetic open-access thesis issued on 2023-05-10.

    ``fields`` replaces or adds metadata: a string gives one value, a sequence several, and
    ``None`` drops the field.
    """
    values: dict[str, FieldValue] = {
        "renati.type": TESIS,
        "dc.date.issued": "2023-05-10",
        "dc.title": f"Título sintético {number:03d}",
        "dc.description.abstract": f"Resumen sintético {number:03d}. Otra oración breve.",
        "dc.subject": ["calidad", "costos"],
        "dc.contributor.advisor": f"Asesor Sintético, Número {number % 3}",
        "dc.rights": OPEN,
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
        program_key=program,
        collection_uuid=COLLECTION,
        harvested_at=HARVEST_TIME,
        metadata=metadata,
    )


def profile_of(records: Sequence[SnapshotRecord]) -> MetadataProfile:
    return script.profile_metadata(
        records,
        snapshot_id=SNAPSHOT_ID,
        snapshot_date=SNAPSHOT_DATE,
        thesis_type="tesis",
        program_keys=list(PROGRAMS),
        count_tokens=count_tokens,
        token_limit=128,
    )


def facts(model: str, **found: Any) -> dict[str, Any]:
    """Return what :func:`compare_tokenizer` reports for a cached tokenizer of ``model``."""
    return {
        "model": model,
        "class": "StandInTokenizer",
        "vocabulary_size": 1000,
        "special_tokens_per_sequence": 2,
        "cached": True,
        **found,
    }


NOT_CACHED = {"model": MPNET, "cached": False}


def provenance(other_tokenizers: Sequence[Mapping[str, Any]] = (NOT_CACHED,)) -> dict[str, Any]:
    """Return a provenance shaped like the script's, with the other tokenizers given."""
    return {
        "script": "experiments/eda_metadata.py",
        "snapshot_id": SNAPSHOT_ID,
        "harvested_at": "2026-10-02T22:44:12Z",
        "metadata_sha256": "0" * 64,
        "config_sha256": "1" * 64,
        "git_commit": None,
        "git_uncommitted_changes": False,
        "tokenizer": facts(MINILM),
        "other_tokenizers": list(other_tokenizers),
    }


THESES = [
    record(1),
    record(2),
    record(3),
    record(4, {"dc.date.issued": "2024"}, program="industrial"),
]


# Observations


def programs_observation(found: Iterable[str]) -> str:
    return next(text for text in found if text.startswith("**The programs are unbalanced.**"))


def test_observations_name_the_largest_and_smallest_program_and_the_busiest_year() -> None:
    found = script.observations(profile_of(THESES), PROGRAMS)

    assert programs_observation(found) == (
        "**The programs are unbalanced.** Ingeniería de Sistemas holds 75.0% of the theses and "
        "Ingeniería Industrial 25.0%, so topic sizes and gap rules need per-program "
        "normalization (R5). The busiest year is 2023 (3 theses)."
    )


def test_observations_of_theses_without_a_readable_issue_date_leave_out_the_busiest_year() -> None:
    theses = [record(1, {"dc.date.issued": None}), record(2, {"dc.date.issued": "sin fecha"})]

    found = script.observations(profile_of(theses), PROGRAMS)

    assert programs_observation(found).endswith("per-program normalization (R5).")
    assert not any("busiest year" in text for text in found)


@pytest.mark.parametrize(
    "records",
    [
        pytest.param([], id="no items"),
        pytest.param([record(1, {"renati.type": SUFICIENCIA})], id="no theses"),
    ],
)
def test_observations_of_a_snapshot_without_theses_leave_out_the_programs(
    records: list[SnapshotRecord],
) -> None:
    found = script.observations(profile_of(records), PROGRAMS)

    assert found
    assert not any("programs are unbalanced" in text for text in found)


def test_every_output_of_a_snapshot_without_theses_renders() -> None:
    outputs = script.render_outputs(profile_of([]), provenance(), PROGRAMS)

    assert tuple(outputs) == OUTPUT_NAMES
    assert all(outputs.values())
    assert outputs["program_year.png"].startswith(b"\x89PNG")


# Report notes


def test_the_advisor_note_counts_the_top_advisors_of_the_profile_constant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = profile_of(THESES)

    assert f"The {TOP_ADVISORS} advisors with the most theses supervised" in (
        script.render_summary(profile, provenance(), PROGRAMS)
    )
    monkeypatch.setattr(script, "TOP_ADVISORS", 3)
    assert "The 3 advisors with the most theses supervised" in (
        script.render_summary(profile, provenance(), PROGRAMS)
    )


def test_the_suffix_note_points_to_the_module_that_defines_the_patterns() -> None:
    summary = script.render_summary(profile_of(THESES), provenance(), PROGRAMS)

    assert "`src/thematic_redundancy/preprocessing/title_suffix.py` define the patterns" in summary
    assert (
        PROJECT_ROOT / "src" / "thematic_redundancy" / "preprocessing" / "title_suffix.py"
    ).is_file()


def stand_in_loader(
    counters: Mapping[str, TokenCounter], vocabulary_sizes: Mapping[str, int] | None = None
) -> Callable[[str], tuple[TokenCounter, dict[str, Any]]]:
    """Return a tokenizer loader that knows only the models of ``counters``, as a cache would.

    Each tokenizer has 1,000 tokens unless ``vocabulary_sizes`` gives another size.
    """

    def load(model_id: str) -> tuple[TokenCounter, dict[str, Any]]:
        if model_id not in counters:
            raise OSError(f"{model_id} is not in the local cache")
        count = counters[model_id]
        description = facts(
            model_id,
            vocabulary_size=(vocabulary_sizes or {}).get(model_id, 1000),
            special_tokens_per_sequence=count(""),
        )
        del description["cached"]
        return count, description

    return load


PARTIAL_CACHE = {MPNET: 5}
"""The vocabulary size of the stub tokenizer that a partly downloaded mpnet cache loads: its
configuration files without ``tokenizer.json`` or ``sentencepiece.bpe.model``."""


def counts_one_more_for_numbers(text: str) -> int:
    """Stand-in for another tokenizer: it agrees except on texts that hold a digit."""
    return count_tokens(text) + any(char.isdigit() for char in text)


REFERENCE = facts(MINILM)
"""What :func:`load_token_counter` describes for the tokenizer that counted the profile."""


def test_compare_tokenizer_reports_a_tokenizer_missing_from_the_cache() -> None:
    counted = {"": 2, "uno dos": 4}

    found = script.compare_tokenizer(MPNET, counted, stand_in_loader({}), REFERENCE)

    assert found == NOT_CACHED


@pytest.mark.parametrize(
    ("other", "differing"),
    [
        pytest.param(count_tokens, 0, id="same counts"),
        pytest.param(counts_one_more_for_numbers, 2, id="other counts"),
    ],
)
def test_compare_tokenizer_counts_the_texts_that_the_other_tokenizer_counts_differently(
    other: TokenCounter, differing: int
) -> None:
    counted = {text: count_tokens(text) for text in ("", "uno dos", "año 2023", "tres 4")}

    found = script.compare_tokenizer(MPNET, counted, stand_in_loader({MPNET: other}), REFERENCE)

    assert found == facts(MPNET, texts_compared=4, texts_counted_differently=differing)


def test_compare_tokenizer_compares_no_counts_when_the_vocabularies_differ() -> None:
    # A partly downloaded cache loads a stub tokenizer whose counts would mean nothing, even
    # where they agree, as they do here.
    counted = {text: count_tokens(text) for text in ("", "uno dos", "año 2023")}
    loader = stand_in_loader({MPNET: count_tokens}, PARTIAL_CACHE)

    found = script.compare_tokenizer(MPNET, counted, loader, REFERENCE)

    assert found == facts(MPNET, vocabulary_size=5)
    assert "texts_compared" not in found


DAMAGED_CACHE_ERRORS = [
    pytest.param(json.JSONDecodeError("Expecting value", "", 0), id="file-not-json"),
    pytest.param(KeyError("model_type"), id="key-missing-from-config"),
    pytest.param(Exception("data did not match any variant"), id="tokenizers-library"),
]
"""Errors that a damaged cache raises while a tokenizer loads: the Rust tokenizers library
raises a bare Exception for a ``tokenizer.json`` that it cannot parse."""


def failing_loader(model: str, error: BaseException) -> Callable[[str], Any]:
    """Return a loader whose tokenizer of ``model`` fails with ``error``; MiniLM's loads."""
    known = stand_in_loader({MINILM: count_tokens, MPNET: count_tokens})

    def load(model_id: str) -> Any:
        if model_id == model:
            raise error
        return known(model_id)

    return load


def failed_to_load(error: BaseException) -> dict[str, Any]:
    return {"model": MPNET, "cached": True, "load_error": type(error).__name__}


@pytest.mark.parametrize("error", DAMAGED_CACHE_ERRORS)
def test_compare_tokenizer_reports_a_tokenizer_that_fails_to_load_without_comparing(
    error: Exception,
) -> None:
    counted = {"": 2, "uno dos": 4}

    found = script.compare_tokenizer(MPNET, counted, failing_loader(MPNET, error), REFERENCE)

    assert found == failed_to_load(error)


def test_compare_tokenizer_lets_an_interruption_through() -> None:
    loader = failing_loader(MPNET, KeyboardInterrupt())

    with pytest.raises(KeyboardInterrupt):
        script.compare_tokenizer(MPNET, {"": 2}, loader, REFERENCE)


@pytest.mark.parametrize(
    ("other", "note"),
    [
        pytest.param(
            NOT_CACHED,
            "- **Second model.** The tokenizer of `paraphrase-multilingual-mpnet-base-v2` is not "
            "in the local cache, so D09's check under both tokenizers is still pending.",
            id="not cached",
        ),
        pytest.param(
            facts(MPNET, texts_compared=1234, texts_counted_differently=0),
            "- **Second model.** The tokenizer of `paraphrase-multilingual-mpnet-base-v2`, from "
            "the local cache, gives the same count as that of "
            "`paraphrase-multilingual-MiniLM-L12-v2` for each of the 1,234 distinct texts that "
            "this profile tokenized (titles, abstracts, their sentences, and the empty text that "
            "gives the special tokens), so every token count here holds for both models (D09).",
            id="same counts",
        ),
        pytest.param(
            facts(MPNET, texts_compared=1234, texts_counted_differently=56),
            "- **Second model.** The tokenizer of `paraphrase-multilingual-mpnet-base-v2`, from "
            "the local cache, gives another count than that of "
            "`paraphrase-multilingual-MiniLM-L12-v2` for 56 of the 1,234 distinct texts that "
            "this profile tokenized (titles, abstracts, their sentences, and the empty text that "
            "gives the special tokens), so the token counts here hold for "
            "`paraphrase-multilingual-MiniLM-L12-v2` only; D09's chunks must fit both tokenizers "
            "(T13).",
            id="other counts",
        ),
        pytest.param(
            facts(MPNET, vocabulary_size=5),
            "- **Second model.** The tokenizer of `paraphrase-multilingual-mpnet-base-v2` in the "
            "local cache has a vocabulary of 5 tokens, while that of "
            "`paraphrase-multilingual-MiniLM-L12-v2` has 1,000, so their token counts were not "
            "compared: that cache may be incomplete, and D09's check under both tokenizers is "
            "still pending.",
            id="other vocabulary",
        ),
        pytest.param(
            {"model": MPNET, "cached": True, "load_error": "JSONDecodeError"},
            "- **Second model.** The tokenizer of `paraphrase-multilingual-mpnet-base-v2` is in "
            "the local cache but failed to load (JSONDecodeError), so its token counts were not "
            "compared with those of `paraphrase-multilingual-MiniLM-L12-v2`: that cache may be "
            "damaged, and D09's check under both tokenizers is still pending.",
            id="failed to load",
        ),
    ],
)
def test_the_second_model_note_says_what_its_tokenizer_was_found_to_do(
    other: Mapping[str, Any], note: str
) -> None:
    summary = script.render_summary(profile_of(THESES), provenance([other]), PROGRAMS)

    assert note in summary.splitlines()
    assert sum(line.startswith("- **Second model.**") for line in summary.splitlines()) == 1


def test_without_another_configured_model_the_report_has_no_second_model_note() -> None:
    summary = script.render_summary(profile_of(THESES), provenance([]), PROGRAMS)

    assert "Second model" not in summary


# A whole run


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


@pytest.fixture
def cli_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a project root that holds a copy of the declared configuration.

    The script then loads no tokenizer, since both configured models get the stand-in
    counter, and asks git nothing.
    """
    (tmp_path / "config").mkdir()
    shutil.copyfile(DEFAULT_CONFIG_PATH, tmp_path / "config" / "default.yaml")
    loader = stand_in_loader({MINILM: count_tokens, MPNET: count_tokens})
    monkeypatch.setattr(script, "load_token_counter", loader)
    monkeypatch.setattr(script, "_git", lambda *arguments: None)
    return tmp_path


def output_dir(project_root: Path) -> Path:
    return project_root / "results" / "eda" / SNAPSHOT_ID


def listing(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def test_main_compares_the_tokenizer_of_the_other_configured_model(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_test_snapshot(cli_root, THESES)
    loader = stand_in_loader({MINILM: count_tokens, MPNET: counts_one_more_for_numbers})
    monkeypatch.setattr(script, "load_token_counter", loader)

    code = script.main(["--project-root", str(cli_root)])

    assert code == 0
    written = json.loads((output_dir(cli_root) / "metadata_profile.json").read_bytes())
    (other,) = written["provenance"]["other_tokenizers"]
    # Every distinct text that the profile tokenized: the empty text, 4 titles, 4 abstracts,
    # and 5 sentences, since each abstract ends with the same one. The titles, the abstracts
    # and their first sentences hold a number, so the other tokenizer counts 12 differently.
    assert other == facts(MPNET, texts_compared=14, texts_counted_differently=12)
    summary = (output_dir(cli_root) / "summary.md").read_text(encoding="utf-8")
    assert "`paraphrase-multilingual-MiniLM-L12-v2` for 12 of the 14 distinct texts" in summary


def test_main_reports_a_second_tokenizer_missing_from_the_cache(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_test_snapshot(cli_root, THESES)
    monkeypatch.setattr(script, "load_token_counter", stand_in_loader({MINILM: count_tokens}))

    code = script.main(["--project-root", str(cli_root)])

    assert code == 0
    written = json.loads((output_dir(cli_root) / "metadata_profile.json").read_bytes())
    assert written["provenance"]["other_tokenizers"] == [NOT_CACHED]


def test_main_reports_a_second_tokenizer_with_another_vocabulary_without_comparing_counts(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_test_snapshot(cli_root, THESES)
    loader = stand_in_loader({MINILM: count_tokens, MPNET: count_tokens}, PARTIAL_CACHE)
    monkeypatch.setattr(script, "load_token_counter", loader)

    code = script.main(["--project-root", str(cli_root)])

    assert code == 0
    written = json.loads((output_dir(cli_root) / "metadata_profile.json").read_bytes())
    assert written["provenance"]["other_tokenizers"] == [facts(MPNET, vocabulary_size=5)]
    summary = (output_dir(cli_root) / "summary.md").read_text(encoding="utf-8")
    assert "has a vocabulary of 5 tokens, while that of" in summary
    assert "holds for both models" not in summary


@pytest.mark.parametrize("error", DAMAGED_CACHE_ERRORS)
def test_main_reports_a_second_tokenizer_that_fails_to_load_and_writes_the_outputs(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    write_test_snapshot(cli_root, THESES)
    monkeypatch.setattr(script, "load_token_counter", failing_loader(MPNET, error))

    code = script.main(["--project-root", str(cli_root)])

    assert code == 0
    written = json.loads((output_dir(cli_root) / "metadata_profile.json").read_bytes())
    assert written["provenance"]["other_tokenizers"] == [failed_to_load(error)]
    summary = (output_dir(cli_root) / "summary.md").read_text(encoding="utf-8")
    assert f"failed to load ({type(error).__name__})" in summary


# A whole run: the privacy gate and the writing, with stand-in outputs

SENSITIVE_TITLE = "Diseño de un secador solar de bajo costo para granos andinos"
SENSITIVE_SENTENCE = "El estudio diseña un secador solar de bajo costo para granos andinos."
OTHER_ITEM_TITLE = "Gestión del mantenimiento preventivo de una flota de camiones mineros"
RECORDS = [
    record(
        1,
        {
            "dc.title": SENSITIVE_TITLE,
            "dc.description.abstract": f"{SENSITIVE_SENTENCE} Se validó con tres prototipos.",
            "dc.contributor.author": "Ficticio Inventado, Ana Lucía",
            "renati.advisor.orcid": "https://orcid.org/0000-0002-1825-0097",
        },
    ),
    record(2),
    record(3, {"renati.type": SUFICIENCIA, "dc.title": OTHER_ITEM_TITLE}),
]
CLEAN = {name: f"aggregates of {name}\n".encode() for name in OUTPUT_NAMES}
"""Stand-in outputs that hold no string of the snapshot."""


def render_with(monkeypatch: pytest.MonkeyPatch, outputs: Mapping[str, bytes]) -> None:
    """Make the script render ``outputs`` instead of the real profile files."""
    monkeypatch.setattr(script, "render_outputs", lambda *arguments: dict(outputs))


def test_main_writes_every_output_once_the_privacy_check_passes(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, RECORDS)
    render_with(monkeypatch, CLEAN)

    code = script.main(["--project-root", str(cli_root)])

    assert code == 0
    assert listing(output_dir(cli_root)) == sorted(OUTPUT_NAMES)
    for name, content in CLEAN.items():
        assert (output_dir(cli_root) / name).read_bytes() == content
    assert "6 outputs searched for" in capsys.readouterr().out


def leak_into(name: str, leak: bytes) -> dict[str, bytes]:
    """Return the stand-in outputs with ``leak`` inside the output ``name``."""
    return {**CLEAN, name: b"\x89PNG\r\n\x1a\n\x00 aggregates " + leak + b" \xff\x00\n"}


def assert_refused(
    project_root: Path, capsys: pytest.CaptureFixture[str], code: int, name: str, leak: str
) -> None:
    errors = capsys.readouterr().err
    assert code == 1
    # Only the leaking output is named. A name also leaks its family-name part, so a leak may
    # count more than once.
    assert re.fullmatch(
        rf"error: {re.escape(name)} holds [1-9][0-9]* string\(s\) taken from the snapshot\n"
        r"error: nothing was written\n",
        errors,
    )
    assert leak.casefold() not in errors.casefold()
    assert not output_dir(project_root).exists()


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_main_writes_nothing_when_any_output_holds_a_title(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    write_test_snapshot(cli_root, RECORDS)
    render_with(monkeypatch, leak_into(name, SENSITIVE_TITLE.encode()))

    code = script.main(["--project-root", str(cli_root)])

    assert_refused(cli_root, capsys, code, name, SENSITIVE_TITLE)


@pytest.mark.parametrize(
    "leak",
    [
        pytest.param(SENSITIVE_SENTENCE.upper(), id="abstract sentence"),
        pytest.param("Ana Lucía Ficticio Inventado", id="person name"),
        pytest.param("0000-0002-1825-0097", id="orcid"),
        pytest.param(OTHER_ITEM_TITLE, id="title of an item that is not a thesis"),
    ],
)
def test_main_writes_nothing_when_an_output_holds_another_string_of_the_snapshot(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    leak: str,
) -> None:
    write_test_snapshot(cli_root, RECORDS)
    render_with(monkeypatch, leak_into("summary.md", leak.encode()))

    code = script.main(["--project-root", str(cli_root)])

    assert_refused(cli_root, capsys, code, "summary.md", leak)


@pytest.mark.parametrize("encoding", ["utf-8", "latin-1"])
def test_main_finds_a_title_inside_binary_bytes_in_either_encoding(
    cli_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    encoding: str,
) -> None:
    # A PNG may carry text in a metadata chunk. The search reads the bytes as UTF-8 and, byte
    # for byte, as Latin-1, so a title is found in either encoding; text that a figure draws
    # becomes pixels, which no string search can read.
    write_test_snapshot(cli_root, RECORDS)
    render_with(monkeypatch, leak_into("top_keywords.png", SENSITIVE_TITLE.encode(encoding)))

    code = script.main(["--project-root", str(cli_root)])

    assert_refused(cli_root, capsys, code, "top_keywords.png", SENSITIVE_TITLE)


def test_a_refused_run_leaves_the_previous_outputs_as_they_were(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_test_snapshot(cli_root, RECORDS)
    output_dir(cli_root).mkdir(parents=True)
    for name in OUTPUT_NAMES:
        (output_dir(cli_root) / name).write_bytes(b"previous\n")
    render_with(monkeypatch, leak_into("summary.md", SENSITIVE_TITLE.encode()))

    assert script.main(["--project-root", str(cli_root)]) == 1

    for name in OUTPUT_NAMES:
        assert (output_dir(cli_root) / name).read_bytes() == b"previous\n"


def test_a_failed_write_leaves_the_previous_outputs_and_no_staged_file(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, RECORDS)
    output_dir(cli_root).mkdir(parents=True)
    for name in OUTPUT_NAMES:
        (output_dir(cli_root) / name).write_bytes(b"previous\n")
    render_with(monkeypatch, CLEAN)
    real_open = Path.open

    def disk_full_at_the_fourth_output(
        self: Path, mode: str = "r", *args: Any, **kwargs: Any
    ) -> IO[Any]:
        # Both a direct write and a staged one open the file through Path.open.
        if "w" in mode and self.name.startswith("abstract_tokens.png"):
            raise OSError(errno.ENOSPC, "No space left on device", str(self))
        return real_open(self, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", disk_full_at_the_fourth_output)

    code = script.main(["--project-root", str(cli_root)])

    assert code == 1
    assert "No space left on device" in capsys.readouterr().err
    for name in OUTPUT_NAMES:
        assert (output_dir(cli_root) / name).read_bytes() == b"previous\n"
    assert listing(output_dir(cli_root)) == sorted(OUTPUT_NAMES)


def test_a_failed_rename_says_that_some_outputs_may_be_new_and_that_a_rerun_fixes_it(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, RECORDS)
    output_dir(cli_root).mkdir(parents=True)
    for name in OUTPUT_NAMES:
        (output_dir(cli_root) / name).write_bytes(b"previous\n")
    render_with(monkeypatch, CLEAN)
    real_replace = os.replace
    renamed: list[str] = []

    def held_open_after_the_first(source: Any, target: Any) -> None:
        # As on Windows while a viewer holds the second output open.
        if renamed:
            raise PermissionError(errno.EACCES, "Access is denied", str(target))
        renamed.append(Path(target).name)
        real_replace(source, target)

    monkeypatch.setattr(os, "replace", held_open_after_the_first)

    code = script.main(["--project-root", str(cli_root)])

    assert code == 1
    errors = capsys.readouterr().err.splitlines()
    assert errors[0].startswith("error: [Errno 13] Access is denied")
    assert errors[1:] == [
        f"error: some outputs in {output_dir(cli_root)} may already have been replaced; run "
        "the script again to write them all"
    ]
    # A rename cannot be undone: the first output is new, the others are the previous ones.
    assert renamed == ["metadata_profile.json"]
    assert (output_dir(cli_root) / "metadata_profile.json").read_bytes() == CLEAN[
        "metadata_profile.json"
    ]
    for name in OUTPUT_NAMES[1:]:
        assert (output_dir(cli_root) / name).read_bytes() == b"previous\n"
    assert listing(output_dir(cli_root)) == sorted(OUTPUT_NAMES)


def test_a_run_leaves_the_other_files_of_the_output_directory_alone(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_test_snapshot(cli_root, RECORDS)
    output_dir(cli_root).mkdir(parents=True)
    (output_dir(cli_root) / "stopword_candidates.json").write_bytes(b"{}\n")
    render_with(monkeypatch, CLEAN)

    assert script.main(["--project-root", str(cli_root)]) == 0

    assert (output_dir(cli_root) / "stopword_candidates.json").read_bytes() == b"{}\n"
    assert listing(output_dir(cli_root)) == sorted([*OUTPUT_NAMES, "stopword_candidates.json"])


def test_a_run_while_another_one_writes_under_results_eda_writes_nothing(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_test_snapshot(cli_root, RECORDS)
    render_with(monkeypatch, CLEAN)
    eda_dir = cli_root.resolve() / "results" / "eda"
    eda_dir.mkdir(parents=True)

    with exclusive_run(eda_dir, RuntimeError("the other run")):
        code = script.main(["--project-root", str(cli_root)])

    assert code == 1
    assert capsys.readouterr().err == (
        f"error: another run is writing the outputs in {eda_dir}; wait until it ends, then "
        "run the script again\nerror: nothing was written\n"
    )
    assert listing(eda_dir) == [LOCK_FILE_NAME]
    # Once the other run ends, a run writes; the lock stays beside the output directory.
    assert script.main(["--project-root", str(cli_root)]) == 0
    assert listing(eda_dir) == sorted([LOCK_FILE_NAME, SNAPSHOT_ID])
    assert listing(output_dir(cli_root)) == sorted(OUTPUT_NAMES)


def git(root: Path, *arguments: str) -> None:
    """Run git in ``root`` as an isolated author, with hooks and signing off."""
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic Author",
            "-c",
            "user.email=author@example.invalid",
            "-c",
            "commit.gpgsign=false",
            *arguments,
        ],
        cwd=root,
        check=True,
        capture_output=True,
    )


def provenance_of_a_run(cli_root: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Run the script and return the provenance that it rendered its outputs from."""
    rendered: dict[str, Any] = {}

    def render(
        profile: MetadataProfile, provenance: dict[str, Any], programs: Mapping[str, str]
    ) -> dict[str, bytes]:
        rendered.update(provenance)
        return dict(CLEAN)

    monkeypatch.setattr(script, "render_outputs", render)
    assert script.main(["--project-root", str(cli_root)]) == 0
    return rendered


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_uncommitted_changes_leave_out_the_outputs_that_the_script_writes(
    cli_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(script, "_git", REAL_GIT)
    write_test_snapshot(cli_root, RECORDS)
    (cli_root / "notes.md").write_text("Notes.\n", encoding="utf-8")
    git(cli_root, "init", "--quiet")
    provenance_of_a_run(cli_root, monkeypatch)  # Writes the outputs that the commit holds.
    git(cli_root, "add", "config", "notes.md", "results")
    git(cli_root, "commit", "--quiet", "--no-verify", "--message", "outputs of an earlier run")
    for name in OUTPUT_NAMES:
        (output_dir(cli_root) / name).write_bytes(b"outputs of a later run, not committed\n")

    # A rerun before the outputs of the last one are committed reads a clean tree.
    assert provenance_of_a_run(cli_root, monkeypatch)["git_uncommitted_changes"] is False

    # Any other tracked change still counts.
    (cli_root / "notes.md").write_text("Notes, edited.\n", encoding="utf-8")
    assert provenance_of_a_run(cli_root, monkeypatch)["git_uncommitted_changes"] is True


def test_searchable_text_reads_the_bytes_as_utf8_and_as_latin1() -> None:
    searchable = script._searchable("Diseño".encode() + b" | " + "Diseño".encode("latin-1"))

    first, second = searchable.split("\n")
    assert "Diseño" in first
    assert "Diseño" in second
