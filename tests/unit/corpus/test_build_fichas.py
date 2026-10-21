"""Behavior of the fichas builder: its rules, its enforced checks, and its command line.

Every record here is synthetic. The uuids are counters, the texts are placeholders, the
people are invented, and the ORCID iDs are the documented examples of the ORCID registry.
Objectives carry the marker ``SYNTHETICOBJECTIVE``, which must never reach the console or a
results file.
"""

import hashlib
import json
import os
import traceback
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
import yaml

from thematic_redundancy.corpus import build_fichas as build_fichas_module
from thematic_redundancy.corpus.anonymize import person_identity, pseudonym
from thematic_redundancy.corpus.build_fichas import (
    BUILD_COMMAND,
    NOTE_INVALID_ADVISOR_ORCID,
    FichasCheckError,
    FichaSources,
    PublishInterrupted,
    build_dataset,
    build_fichas,
    check_fichas,
    count_fichas,
    document_type,
    load_fichas,
    main,
    verify_fichas_outputs,
)
from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.corpus.pdf_manifest import (
    PdfEntry,
    PdfManifest,
    summarize,
    write_pdf_manifest,
)
from thematic_redundancy.corpus.snapshot import (
    ProgramSummary,
    SnapshotManifest,
    SnapshotRecord,
    encode_records,
    write_snapshot,
)
from thematic_redundancy.extraction.locate_objectives import (
    ObjectivesManifest,
    ObjectivesSettings,
)
from thematic_redundancy.extraction.locate_objectives import (
    summarize as summarize_objectives,
)
from thematic_redundancy.extraction.objectives import ObjectivesRecord
from thematic_redundancy.extraction.text_manifest import (
    ExtractionSettings,
    TextEntry,
    TextManifest,
    write_text_manifest,
)
from thematic_redundancy.extraction.text_manifest import (
    summarize as summarize_texts,
)
from thematic_redundancy.shared.config import (
    FichasCountsConfig,
    WrongPdfConfig,
    load_config,
)
from thematic_redundancy.shared.filesystem import write_text_atomically
from thematic_redundancy.shared.storage import ParquetTableStore

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "default.yaml"
DECLARED = load_config(DEFAULT_CONFIG_PATH)
COLLECTIONS = {program.key: program.collection_uuid for program in DECLARED.snapshot.programs}
SNAPSHOT_ID = "20261002T150405Z"
HARVEST_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
SNAPSHOT_DATE = HARVEST_TIME.date()
RUN_TIME = datetime(2026, 10, 7, 9, 0, 0, tzinfo=UTC)
KEY = bytes(range(32))
MARKER = "SYNTHETICOBJECTIVE"

RENATI = "https://purl.org/pe-repo/renati/type#"
TESIS = f"{RENATI}tesis"
OPEN = "https://purl.org/coar/access_right/c_abf2"
EMBARGOED = "https://purl.org/coar/access_right/c_f1cf"
RESTRICTED = "https://purl.org/coar/access_right/c_16ec"
VALID_ORCID = "https://orcid.org/0000-0002-1825-0097"
"""The registry's documented example iD; its check digit is right."""
BAD_CHECK_DIGIT_ORCID = "https://orcid.org/0000-0002-1825-0098"
ADVISOR = "Asesorx Sintético, Docente"
AUTHOR = "Autorx Ficticia, Primera"
SECOND_AUTHOR = "Autorx Inventada, Segunda"
PDF_SHA = "ab" * 32
TWIN = {"dc__title": "Título común", "dc__description__abstract": "Resumen común."}
"""Title and abstract that make records of one thesis, when nothing else differs."""


def item(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def handle(number: int) -> str:
    return f"123456789/{1000 + number}"


def record(number: int, *, program: str = "sistemas", **fields: Any) -> SnapshotRecord:
    """Return a synthetic record; each keyword sets one field's values (a string, a list or
    ``None`` to leave the field out)."""
    values: dict[str, Any] = {
        "renati.type": TESIS,
        "dc.title": f"Título sintético {number}",
        "dc.description.abstract": f"Resumen sintético {number}.",
        "dc.contributor.author": [AUTHOR],
        "dc.contributor.advisor": ADVISOR,
        "dc.date.issued": "2023-05-10",
        "dc.date.accessioned": "2023-06-01T10:00:00Z",
        "dc.rights": OPEN,
        "dc.language.iso": "spa",
        "dc.subject": ["Prueba sintética", "Caso ficticio"],
        "dc.subject.ocde": "https://purl.org/pe-repo/ocde/ford#2.02.04",
    }
    values.update({key.replace("__", "."): value for key, value in fields.items()})
    metadata = {
        key: [{"value": text} for text in ([value] if isinstance(value, str) else value)]
        for key, value in values.items()
        if value is not None
    }
    return SnapshotRecord(
        uuid=item(number),
        handle=handle(number),
        program_key=program,
        collection_uuid=COLLECTIONS[program],
        harvested_at=HARVEST_TIME,
        metadata=metadata,
    )


def pdf_on_disk(number: int, *, sha256: str = PDF_SHA, program: str = "sistemas") -> PdfEntry:
    return PdfEntry(
        program_key=program,
        status="downloaded",
        file=f"{item(number)}.pdf",
        bitstream_uuid=item(900 + number),
        size_bytes=1000,
        checksum_algorithm="MD5",
        checksum="0" * 32,
        md5="0" * 32,
        sha256=sha256,
        checked_at=HARVEST_TIME,
    )


def pdf_restricted(program: str = "sistemas") -> PdfEntry:
    return PdfEntry(
        program_key=program,
        status="restricted",
        reason="rights_restricted",
        checked_at=HARVEST_TIME,
    )


def text_entry(
    number: int, *, ocr_pages: int = 0, text_layer_pages: int = 10, program: str = "sistemas"
) -> TextEntry:
    return TextEntry(
        program_key=program,
        status="ok",
        pdf_sha256=PDF_SHA,
        config_fingerprint="cd" * 32,
        file=f"{item(number)}.parquet",
        output_sha256="ef" * 32,
        pages=ocr_pages + text_layer_pages,
        text_layer_pages=text_layer_pages,
        ocr_pages=ocr_pages,
        ocr_attempted=ocr_pages,
        extracted_at=HARVEST_TIME,
    )


def objectives_row(
    number: int,
    *,
    general: str | None = None,
    specific: str | None = None,
    flags: tuple[str, ...] = (),
    found: bool = True,
    program: str = "sistemas",
) -> ObjectivesRecord:
    if not found:
        general = specific = None
    else:
        general = general or f"Determinar el efecto {MARKER} número {number} en la prueba."
    return ObjectivesRecord(
        item_uuid=item(number),
        handle=handle(number),
        program=program,
        status="extracted" if found else "not_found",
        objective_general=general,
        objectives_specific=specific,
        page_start=5 if found else None,
        page_end=6 if found else None,
        pattern="objetivo_general" if found else None,
        flags=flags,  # type: ignore[arg-type]
        general_chars=len(general or ""),
        specific_chars=len(specific or ""),
        pages=10,
        pages_sha256="12" * 32,
    )


def with_pdf(number: int, **fields: Any) -> tuple[SnapshotRecord, PdfEntry, TextEntry, Any]:
    """Return a thesis with its PDF on disk, its page texts and its extracted objectives."""
    return record(number, **fields), pdf_on_disk(number), text_entry(number), objectives_row(number)


def sources(
    records: Sequence[SnapshotRecord],
    *,
    pdfs: dict[int, PdfEntry] | None = None,
    texts: dict[int, TextEntry] | None = None,
    objectives: Sequence[ObjectivesRecord] = (),
) -> FichaSources:
    return FichaSources(
        snapshot_id=SNAPSHOT_ID,
        snapshot_date=SNAPSHOT_DATE,
        records=tuple(records),
        pdf_entries={item(number): entry for number, entry in (pdfs or {}).items()},
        text_entries={item(number): entry for number, entry in (texts or {}).items()},
        objectives=tuple(objectives),
    )


def config_with(
    *, wrong_pdfs: Sequence[WrongPdfConfig] = (), expected: FichasCountsConfig | None = None
) -> Any:
    fichas = DECLARED.fichas.model_copy(
        update={"wrong_pdfs": tuple(wrong_pdfs), "expected": expected or DECLARED.fichas.expected}
    )
    return DECLARED.model_copy(update={"fichas": fichas})


def build(records: Sequence[SnapshotRecord], **kwargs: Any) -> dict[int, Ficha]:
    """Build the fichas of ``records`` that have no PDF, keyed by record number."""
    wrong = kwargs.pop("wrong_pdfs", ())
    fichas = build_fichas(sources(records, **kwargs), config_with(wrong_pdfs=wrong), KEY)
    return {int(str(ficha.item_uuid)[-12:]): ficha for ficha in fichas}


def build_with_pdfs(*theses: tuple[SnapshotRecord, PdfEntry, TextEntry, Any]) -> dict[int, Ficha]:
    numbers = [int(str(thesis[0].uuid)[-12:]) for thesis in theses]
    return build(
        [thesis[0] for thesis in theses],
        pdfs=dict(zip(numbers, (thesis[1] for thesis in theses), strict=True)),
        texts=dict(zip(numbers, (thesis[2] for thesis in theses), strict=True)),
        objectives=[thesis[3] for thesis in theses],
    )


# Inclusion (D01)


def test_a_record_with_the_exact_thesis_fragment_is_included() -> None:
    ficha = build([record(1)])[1]

    assert (ficha.include, ficha.exclusion_reason) == (True, None)
    assert (ficha.document_type, ficha.renati_type_raw) == ("tesis", TESIS)


@pytest.mark.parametrize(
    ("fragment", "kind"),
    [
        ("trabajoDeSuficienciaProfesional", "trabajo_suficiencia"),
        ("Trabajo de Suficiencia Profesional", "trabajo_suficiencia"),
        ("trabajodesuficienciaprofesional", "trabajo_suficiencia"),
        ("trabajoAcademico", "trabajo_academico"),
        ("Tesis", "other"),
        ("tesisDoctoral", "other"),
    ],
)
def test_any_other_type_is_excluded_with_its_raw_type(fragment: str, kind: str) -> None:
    ficha = build([record(1, renati__type=f"{RENATI}{fragment}")])[1]

    assert (ficha.include, ficha.exclusion_reason) == (False, "non_thesis_type")
    assert (ficha.document_type, ficha.renati_type_raw) == (kind, f"{RENATI}{fragment}")


def test_a_record_without_a_type_is_excluded_as_other() -> None:
    ficha = build([record(1, renati__type=None)])[1]

    assert (ficha.document_type, ficha.renati_type_raw) == ("other", None)
    assert ficha.exclusion_reason == "non_thesis_type"


def test_the_document_type_of_a_thesis_follows_the_configured_fragment() -> None:
    assert document_type(record(1).metadata, "tesis") == "tesis"
    assert document_type(record(1).metadata, "tesisDoctoral") == "other"


# Duplicates (D24)


def test_of_two_identical_theses_the_one_accessioned_first_stays() -> None:
    # The later record has the smaller uuid, so the rule does not fall back to uuid order.
    fichas = build(
        [
            record(1, **TWIN, dc__date__accessioned="2025-06-26T15:30:37Z"),
            record(2, **TWIN, dc__date__accessioned="2025-06-11T14:03:49Z"),
        ]
    )

    assert (fichas[2].include, fichas[2].duplicate_of) == (True, None)
    assert (fichas[1].include, fichas[1].exclusion_reason) == (False, "duplicate")
    assert fichas[1].duplicate_of == item(2)


def test_a_tie_on_the_accession_time_goes_to_the_smaller_uuid() -> None:
    fichas = build([record(2, **TWIN), record(1, **TWIN)])

    assert fichas[1].include
    assert (fichas[2].exclusion_reason, fichas[2].duplicate_of) == ("duplicate", item(1))


def test_every_other_record_of_a_larger_group_names_the_one_that_stays() -> None:
    fichas = build(
        [
            record(1, **TWIN, dc__date__accessioned="2024-01-03T00:00:00Z"),
            record(2, **TWIN, dc__date__accessioned="2024-01-01T00:00:00Z"),
            record(3, **TWIN, dc__date__accessioned="2024-01-02T00:00:00Z"),
        ]
    )

    assert [fichas[n].duplicate_of for n in (1, 2, 3)] == [item(2), None, item(2)]


def test_duplicates_compare_texts_with_whitespace_collapsed_and_case_folded() -> None:
    fichas = build([record(1), record(2, dc__title="  TÍTULO   sintético 1 ")])
    # Both share the title once folded, but each has its own abstract.
    assert fichas[2].include

    fichas = build(
        [
            record(1, dc__description__abstract="Mismo resumen."),
            record(2, dc__title="TÍTULO  sintético 1", dc__description__abstract="mismo  RESUMEN."),
        ]
    )
    assert fichas[2].duplicate_of == item(1)


@pytest.mark.parametrize(
    "difference",
    [
        {"dc__contributor__author": [SECOND_AUTHOR]},
        {"dc__contributor__advisor": "Otrx Asesorx, Ficticio"},
        {"dc__date__issued": "2023-05-11"},
        {"dc__title": "Otro título sintético"},
        {"dc__description__abstract": "Otro resumen sintético."},
    ],
    ids=["author", "advisor", "date", "title", "abstract"],
)
def test_records_that_differ_in_one_compared_field_are_two_theses(
    difference: dict[str, Any],
) -> None:
    fichas = build([record(1, **TWIN), record(2, **{**TWIN, **difference})])

    assert fichas[1].include and fichas[2].include


def test_two_identical_non_theses_are_not_marked_as_duplicates() -> None:
    other_type = {**TWIN, "renati__type": f"{RENATI}trabajoAcademico"}

    fichas = build([record(1, **other_type), record(2, **other_type)])

    assert {fichas[1].exclusion_reason, fichas[2].exclusion_reason} == {"non_thesis_type"}


def test_a_duplicate_group_without_an_accession_time_stops_the_build() -> None:
    with pytest.raises(ValueError, match="no dc.date.accessioned"):
        build([record(1, **TWIN), record(2, **TWIN, dc__date__accessioned=None)])


# Objectives (O05)


def test_a_thesis_with_its_pdf_takes_the_located_objectives() -> None:
    thesis = with_pdf(1)
    row = objectives_row(1, specific=f"Medir {MARKER} A.\nMedir {MARKER} B.")

    ficha = build_with_pdfs((thesis[0], thesis[1], thesis[2], row))[1]

    assert ficha.objectives_status == "extracted"
    assert ficha.objectives == f"{row.objective_general}\n\n{row.objectives_specific}"
    assert ficha.section_source == "title_abstract_objectives"
    assert (ficha.pdf_status, ficha.pdf_sha256) == ("downloaded", PDF_SHA)


def test_a_general_objective_alone_is_the_whole_objectives_text() -> None:
    ficha = build_with_pdfs(with_pdf(1))[1]

    assert ficha.objectives == objectives_row(1).objective_general


def test_objectives_not_found_fall_back_to_title_and_abstract() -> None:
    thesis = with_pdf(1)

    ficha = build_with_pdfs((*thesis[:3], objectives_row(1, found=False)))[1]

    assert (ficha.objectives_status, ficha.objectives) == ("not_found", None)
    assert ficha.section_source == "title_abstract"


def test_a_thesis_without_its_pdf_is_metadata_only() -> None:
    ficha = build([record(1, dc__rights=RESTRICTED)], pdfs={1: pdf_restricted()})[1]

    assert (ficha.objectives_status, ficha.section_source) == ("no_pdf", "title_abstract")
    assert (ficha.pdf_status, ficha.pdf_sha256, ficha.source_format) == ("restricted", None, "none")
    assert ficha.include


def test_a_record_never_looked_at_by_the_downloader_has_no_pdf() -> None:
    ficha = build([record(1, renati__type=f"{RENATI}trabajoAcademico")])[1]

    assert (ficha.pdf_status, ficha.objectives_status) == ("not_attempted", "no_pdf")


@pytest.mark.parametrize(
    ("ocr_pages", "text_layer_pages", "source"),
    [(0, 10, "digital"), (3, 7, "mixed"), (10, 0, "ocr")],
)
def test_the_source_format_tells_how_the_pdf_text_was_read(
    ocr_pages: int, text_layer_pages: int, source: str
) -> None:
    thesis = with_pdf(1)
    pages = text_entry(1, ocr_pages=ocr_pages, text_layer_pages=text_layer_pages)

    assert build_with_pdfs((thesis[0], thesis[1], pages, thesis[3]))[1].source_format == source


def test_the_locator_flags_become_quality_notes() -> None:
    thesis = with_pdf(1)
    row = objectives_row(1, flags=("specific_missing", "ocr_page"))

    notes = build_with_pdfs((*thesis[:3], row))[1].quality_notes

    assert notes == (
        "objectives locator flag: specific_missing",
        "objectives locator flag: ocr_page",
    )


def test_a_wrong_pdf_keeps_its_hash_but_not_its_text() -> None:
    owner, copy = with_pdf(1), with_pdf(2)
    wrong = WrongPdfConfig(item_handle=handle(2), pdf_of_handle=handle(1))

    fichas = build(
        [owner[0], copy[0]],
        pdfs={1: owner[1], 2: copy[1]},
        texts={1: owner[2], 2: copy[2]},
        objectives=[owner[3], copy[3]],
        wrong_pdfs=[wrong],
    )

    ficha = fichas[2]
    assert (ficha.objectives_status, ficha.objectives) == ("no_pdf", None)
    assert (ficha.section_source, ficha.source_format) == ("title_abstract", "none")
    assert (ficha.pdf_status, ficha.pdf_sha256) == ("downloaded", PDF_SHA)
    assert ficha.quality_notes == (f"PDF on disk belongs to another item ({handle(1)})",)
    assert fichas[1].objectives_status == "extracted"


def test_a_declared_wrong_pdf_that_is_not_a_copy_stops_the_build() -> None:
    owner, copy = with_pdf(1), with_pdf(2)

    with pytest.raises(ValueError, match="is not a copy of the PDF of"):
        build(
            [owner[0], copy[0]],
            pdfs={1: owner[1], 2: pdf_on_disk(2, sha256="cd" * 32)},
            texts={1: owner[2], 2: copy[2]},
            objectives=[owner[3], copy[3]],
            wrong_pdfs=[WrongPdfConfig(item_handle=handle(2), pdf_of_handle=handle(1))],
        )


def test_a_declared_wrong_pdf_outside_the_snapshot_stops_the_build() -> None:
    wrong = WrongPdfConfig(item_handle=handle(7), pdf_of_handle=handle(1))

    with pytest.raises(ValueError, match="names a handle that the snapshot lacks"):
        build([record(1)], wrong_pdfs=[wrong])


def test_a_pdf_on_disk_without_an_objectives_row_stops_the_build() -> None:
    thesis = with_pdf(1)

    with pytest.raises(ValueError, match="no row in objectives.parquet"):
        build([thesis[0]], pdfs={1: thesis[1]}, texts={1: thesis[2]})


def test_an_objectives_row_without_a_pdf_on_disk_stops_the_build() -> None:
    with pytest.raises(ValueError, match="whose PDF is not on disk"):
        build([record(1)], objectives=[objectives_row(1)])


def test_a_pdf_on_disk_without_its_page_texts_stops_the_build() -> None:
    thesis = with_pdf(1)

    with pytest.raises(ValueError, match="no extracted page texts"):
        build([thesis[0]], pdfs={1: thesis[1]}, objectives=[thesis[3]])


# People


def test_the_advisor_code_comes_from_a_valid_orcid_whatever_the_name() -> None:
    fichas = build(
        [
            record(1, renati__advisor__orcid=VALID_ORCID),
            record(2, renati__advisor__orcid=VALID_ORCID, dc__contributor__advisor="A. Sintético"),
        ]
    )

    identity = person_identity(VALID_ORCID, None)
    assert identity is not None
    assert fichas[1].advisor_code == fichas[2].advisor_code == pseudonym(identity, KEY, "ADV-")
    assert fichas[1].quality_notes == ()


def test_an_orcid_with_a_wrong_check_digit_falls_back_to_the_name_with_a_note() -> None:
    ficha = build([record(1, renati__advisor__orcid=BAD_CHECK_DIGIT_ORCID)])[1]

    identity = person_identity(None, ADVISOR)
    assert identity is not None
    assert ficha.advisor_code == pseudonym(identity, KEY, "ADV-")
    assert ficha.quality_notes == (NOTE_INVALID_ADVISOR_ORCID,)
    assert "orcid" not in " ".join(ficha.quality_notes).replace("ORCID", "")


def test_an_advisor_without_an_orcid_is_coded_by_name_without_a_note() -> None:
    ficha = build([record(1)])[1]

    identity = person_identity(None, ADVISOR)
    assert identity is not None
    assert (ficha.advisor_code, ficha.quality_notes) == (pseudonym(identity, KEY, "ADV-"), ())


def test_a_thesis_without_an_advisor_has_no_advisor_code() -> None:
    assert build([record(1, dc__contributor__advisor=None)])[1].advisor_code is None


def test_authors_get_one_code_each_in_repository_order() -> None:
    authors = [SECOND_AUTHOR, AUTHOR, " autorx  inventada, SEGUNDA "]

    ficha = build([record(1, dc__contributor__author=authors)])[1]

    expected = tuple(
        pseudonym(identity, KEY, "AUT-")
        for name in (SECOND_AUTHOR, AUTHOR)
        if (identity := person_identity(None, name)) is not None
    )
    assert ficha.author_codes == expected


def test_a_ficha_holds_no_name_of_a_person() -> None:
    fichas = build([record(1, renati__advisor__orcid=BAD_CHECK_DIGIT_ORCID)])

    dumped = fichas[1].model_dump_json().casefold()
    for name in (ADVISOR, AUTHOR, "sintético, docente", "ficticia"):
        assert name.casefold() not in dumped
    assert "1825" not in dumped


# Dates and other fields


@pytest.mark.parametrize(
    ("issued", "year", "after"),
    [
        ("2026-12-04", 2026, True),
        ("2026-10-03", 2026, True),
        (SNAPSHOT_DATE.isoformat(), 2026, False),
        ("2021-01-06", 2021, False),
        ("2027", 2027, True),
        ("sin fecha", None, False),
    ],
)
def test_the_future_dated_flag_compares_the_issue_date_with_the_snapshot_date(
    issued: str, year: int | None, after: bool
) -> None:
    ficha = build([record(1, dc__date__issued=issued)])[1]

    assert (ficha.issue_date_raw, ficha.issue_year, ficha.issued_after_snapshot) == (
        issued,
        year,
        after,
    )


def test_the_metadata_fields_are_kept_as_harvested() -> None:
    ficha = build(
        [
            record(
                1,
                program="minas",
                dc__rights=EMBARGOED,
                dc__date__embargoEnd="2027-03-18",
                dc__subject=[" Prueba sintética ", "Caso ficticio"],
            )
        ]
    )[1]

    assert ficha.doc_code == doc_code_for(item(1))
    assert ficha.handle_url == f"https://hdl.handle.net/{handle(1)}"
    assert (ficha.snapshot_id, ficha.title) == (SNAPSHOT_ID, "Título sintético 1")
    assert ficha.abstract == "Resumen sintético 1."
    assert (ficha.program_key, ficha.program_name) == ("minas", "Ingeniería de Minas")
    assert ficha.keywords == ("Prueba sintética", "Caso ficticio")
    assert (ficha.ocde_codes, ficha.language) == (("2.02.04",), "spa")
    assert (ficha.rights, ficha.embargo_end) == ("embargoed", date(2027, 3, 18))


@pytest.mark.parametrize(
    ("rights", "access"),
    [(OPEN, "open"), (RESTRICTED, "restricted"), ("CC BY 4.0", "unknown"), (None, "unknown")],
)
def test_the_access_rights_follow_the_coar_value(rights: str | None, access: str) -> None:
    assert build([record(1, dc__rights=rights)])[1].rights == access


def test_a_record_that_makes_no_valid_ficha_stops_the_build_naming_it() -> None:
    with pytest.raises(ValueError, match=f"record {item(1)} .* makes no valid ficha"):
        build([record(1, dc__title=None)])


# Enforced checks


def expected_counts(**overrides: int) -> FichasCountsConfig:
    """Return the counts of the small corpus of :func:`small_corpus`, with ``overrides``."""
    counts = {
        "snapshot_id": SNAPSHOT_ID,
        "records": 4,
        "included": 2,
        "duplicates": 1,
        "non_thesis": 1,
        "objectives_extracted": 1,
        "metadata_only": 1,
        **overrides,
    }
    return FichasCountsConfig.model_construct(**counts)  # type: ignore[arg-type]


def small_corpus() -> tuple[Ficha, ...]:
    """Two theses (one with objectives), one duplicate and one non-thesis record."""
    thesis = with_pdf(1)
    records = [
        thesis[0],
        record(2, **TWIN, dc__rights=RESTRICTED),
        record(3, **TWIN, dc__rights=RESTRICTED),
        record(4, renati__type=f"{RENATI}trabajoAcademico"),
    ]
    return build_fichas(
        sources(records, pdfs={1: thesis[1]}, texts={1: thesis[2]}, objectives=[thesis[3]]),
        config_with(),
        KEY,
    )


def test_fichas_that_reach_the_declared_counts_pass_the_checks() -> None:
    fichas = small_corpus()

    assert check_fichas(fichas, expected_counts()) == count_fichas(fichas)
    assert count_fichas(fichas) == {
        "records": 4,
        "included": 2,
        "duplicates": 1,
        "non_thesis": 1,
        "objectives_extracted": 1,
        "metadata_only": 1,
    }


@pytest.mark.parametrize(
    "name",
    ["records", "included", "duplicates", "non_thesis", "objectives_extracted", "metadata_only"],
)
def test_each_count_that_misses_its_declaration_stops_the_build(name: str) -> None:
    fichas = small_corpus()
    wrong = count_fichas(fichas)[name] + 1

    with pytest.raises(FichasCheckError, match=f"{name} is {wrong - 1}, but fichas.expected"):
        check_fichas(fichas, expected_counts(**{name: wrong}))


def test_every_failed_check_is_listed_at_once() -> None:
    with pytest.raises(FichasCheckError, match="fail 2 checks: records is 4.*; included is 2"):
        check_fichas(small_corpus(), expected_counts(included=3, records=5))


def test_a_repeated_document_code_stops_the_build() -> None:
    # These two counter uuids share the first 8 hex digits of their SHA-256 (found by search).
    first = UUID("00000000-0000-4000-8000-000000021083")
    second = UUID("00000000-0000-4000-8000-000000070264")
    assert doc_code_for(first) == doc_code_for(second)
    fichas = build_fichas(
        sources(
            [
                record(21083, dc__title="Uno"),
                record(70264, dc__title="Dos"),
            ]
        ),
        config_with(),
        KEY,
    )

    counts = expected_counts(
        records=2, included=2, duplicates=0, non_thesis=0, objectives_extracted=0, metadata_only=2
    )

    with pytest.raises(FichasCheckError, match="doc_code must be unique; repeated: DOC-27c88a62"):
        check_fichas(fichas, counts)


def test_a_pdf_shared_by_two_included_theses_stops_the_build() -> None:
    first, second = with_pdf(1), with_pdf(2, dc__title="Otro título")
    fichas = build_fichas(
        sources(
            [first[0], second[0]],
            pdfs={1: first[1], 2: second[1]},
            texts={1: first[2], 2: second[2]},
            objectives=[first[3], second[3]],
        ),
        config_with(),
        KEY,
    )
    counts = expected_counts(
        records=2, included=2, duplicates=0, non_thesis=0, objectives_extracted=2, metadata_only=0
    )

    with pytest.raises(FichasCheckError, match="1 PDF serves several included theses"):
        check_fichas(fichas, counts)


# Command line


def write_project(root: Path, *, expected: dict[str, Any] | None = None) -> None:
    """Write a project with the small corpus: config, snapshot, PDF manifest, page-text
    manifest and objectives table, as the earlier steps write them."""
    (root / "config").mkdir(parents=True)
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["fichas"] = {
        "wrong_pdfs": [],
        "expected": expected
        or {
            "snapshot_id": SNAPSHOT_ID,
            "records": 4,
            "included": 2,
            "duplicates": 1,
            "non_thesis": 1,
            "objectives_extracted": 1,
            "metadata_only": 1,
        },
    }
    (root / "config" / "default.yaml").write_text(
        yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8"
    )
    stopwords = root / "config" / "stopwords_domain_es.txt"
    stopwords.write_text("# synthetic list\nobjetivo\ndeterminar\n", encoding="utf-8")
    thesis = with_pdf(1, renati__advisor__orcid=BAD_CHECK_DIGIT_ORCID)
    records = [
        thesis[0],
        record(2, **TWIN, dc__rights=RESTRICTED),
        record(3, **TWIN, dc__rights=RESTRICTED),
        record(4, renati__type=f"{RENATI}trabajoAcademico", dc__date__issued="2026-12-04"),
    ]
    snapshot_dir = root / "data" / "raw" / SNAPSHOT_ID
    content = encode_records(records)
    write_snapshot(
        snapshot_dir,
        content,
        SnapshotManifest(
            snapshot_id=SNAPSHOT_ID,
            harvested_at=HARVEST_TIME,
            base_url="https://repo.example.edu",
            search_url="https://repo.example.edu/server/api/discover/search/objects",
            query_parameters={},
            programs={"sistemas": ProgramSummary(collection_uuid=COLLECTIONS["sistemas"], items=4)},
            total_items=4,
            faculty_community_uuid=UUID("00000000-0000-4000-8000-0000000000f0"),
            faculty_total=4,
            items_by_type={},
            future_dated_items=1,
            unparsed_issue_dates=0,
            dropped_metadata_keys=(),
            metadata_sha256=hashlib.sha256(content).hexdigest(),
        ),
    )
    pdf_entries = {item(1): thesis[1], item(2): pdf_restricted(), item(3): pdf_restricted()}
    (snapshot_dir / "pdfs").mkdir()
    write_pdf_manifest(
        snapshot_dir / "pdfs" / "manifest.json",
        PdfManifest(
            snapshot_id=SNAPSHOT_ID,
            base_url="https://repo.example.edu",
            thesis_type="tesis",
            selection_rule="synthetic",
            updated_at=HARVEST_TIME,
            summary=summarize(pdf_entries),
            items=pdf_entries,
        ),
    )
    interim = root / "data" / "interim" / SNAPSHOT_ID
    (interim / "pages").mkdir(parents=True)
    text_entries = {item(1): thesis[2]}
    settings = ExtractionSettings.from_config(DECLARED)
    write_text_manifest(
        interim / "pages" / "manifest.json",
        TextManifest(
            snapshot_id=SNAPSHOT_ID,
            settings=settings,
            config_fingerprint=settings.fingerprint(),
            updated_at=HARVEST_TIME,
            summary=summarize_texts(text_entries, settings.fingerprint(), text_entries.keys()),
            items=text_entries,
        ),
    )
    table = interim / "objectives.parquet"
    ParquetTableStore().write([thesis[3]], table)
    objectives_settings = ObjectivesSettings.from_config(DECLARED)
    write_text_atomically(
        interim / "objectives_manifest.json",
        ObjectivesManifest(
            snapshot_id=SNAPSHOT_ID,
            settings=objectives_settings,
            settings_fingerprint=objectives_settings.fingerprint(),
            text_settings_fingerprint="cd" * 32,
            created_at=HARVEST_TIME,
            table_file=table.name,
            table_sha256=hashlib.sha256(table.read_bytes()).hexdigest(),
            seconds=0.1,
            summary=summarize_objectives([thesis[3]]),
        ).model_dump_json(indent=2),
    )


def fake_lemmatizer(texts: Sequence[str]) -> list[list[str]]:
    """Stand-in for the full cleaner: the lower-case words of each text, markers dropped."""
    return [
        [word.strip(".,").lower() for word in text.split() if MARKER not in word] for text in texts
    ]


def run(root: Path, *arguments: str) -> int:
    return main(
        ["--project-root", str(root), *arguments],
        clock=lambda: RUN_TIME,
        lemmatize=fake_lemmatizer,
    )


def test_the_command_writes_the_fichas_the_exclusions_and_a_report_of_numbers_only(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)
    report_dir = tmp_path / "results" / "fichas" / SNAPSHOT_ID

    assert run(tmp_path, "--report-out", str(report_dir)) == 0

    interim = tmp_path / "data" / "interim" / SNAPSHOT_ID
    fichas = ParquetTableStore().read(Ficha, interim / "fichas.parquet")
    assert [ficha.include for ficha in fichas] == [True, True, False, False]
    exclusions = json.loads(
        json.dumps([e.model_dump(mode="json") for e in read_exclusions(interim)])
    )
    assert [(row["handle"], row["exclusion_reason"]) for row in exclusions] == [
        (handle(3), "duplicate"),
        (handle(4), "non_thesis_type"),
    ]
    assert exclusions[0]["duplicate_of"] == str(item(2))
    manifest = json.loads((interim / "fichas_manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"]["included"] == 2
    assert (tmp_path / "data" / "interim" / "pseudonym.key").stat().st_size == 32
    report = json.loads((report_dir / "quality_report.json").read_text(encoding="utf-8"))
    assert report["counts"]["records"] == 4
    assert report["stopword_recheck"]["domain_stopword_documents"] == {
        "determinar": 1,
        "objetivo": 0,
    }
    published = "\n".join(
        [
            capsys.readouterr().out,
            (report_dir / "quality_report.json").read_text(encoding="utf-8"),
            (report_dir / "quality_report.md").read_text(encoding="utf-8"),
        ]
    ).casefold()
    for text in (MARKER, "Título sintético", "Título común", "Resumen sintético", "Ficticia"):
        assert text.casefold() not in published


def read_exclusions(interim: Path) -> Any:
    from thematic_redundancy.corpus.build_fichas import Exclusion

    return ParquetTableStore().read(Exclusion, interim / "exclusions.parquet")


def test_counts_that_miss_their_declaration_fail_in_one_line_and_write_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(
        tmp_path,
        expected={
            "snapshot_id": SNAPSHOT_ID,
            "records": 5,
            "included": 3,
            "duplicates": 1,
            "non_thesis": 1,
            "objectives_extracted": 1,
            "metadata_only": 2,
        },
    )

    assert run(tmp_path) == 1

    error = capsys.readouterr().err
    assert error.startswith("error: the fichas of snapshot")
    assert "records is 4, but fichas.expected declares 5" in error
    assert len(error.strip().splitlines()) == 1
    assert not (tmp_path / "data" / "interim" / SNAPSHOT_ID / "fichas.parquet").exists()


def test_a_snapshot_without_declared_counts_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(
        tmp_path,
        expected={
            "snapshot_id": "20261002T224412Z",
            "records": 4,
            "included": 2,
            "duplicates": 1,
            "non_thesis": 1,
            "objectives_extracted": 1,
            "metadata_only": 1,
        },
    )

    assert run(tmp_path) == 1

    assert "declares the counts of snapshot '20261002T224412Z'" in capsys.readouterr().err


def test_an_objectives_table_changed_since_its_manifest_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)
    table = tmp_path / "data" / "interim" / SNAPSHOT_ID / "objectives.parquet"
    ParquetTableStore().write(
        [objectives_row(1, general=f"Otro objetivo {MARKER} distinto.")], table
    )

    assert run(tmp_path) == 1

    assert "changed since" in capsys.readouterr().err


OUTPUT_NAMES = ("fichas.parquet", "exclusions.parquet", "fichas_manifest.json")


def published(interim: Path) -> dict[str, bytes]:
    """Return the bytes of the three fichas outputs, by file name."""
    return {name: (interim / name).read_bytes() for name in OUTPUT_NAMES}


def leftovers(interim: Path) -> list[str]:
    """Return the names in ``interim`` that are neither an input nor an output of the build."""
    known = {"pages", "objectives.parquet", "objectives_manifest.json", *OUTPUT_NAMES}
    return sorted(path.name for path in interim.iterdir() if path.name not in known)


def change_the_key(root: Path) -> None:
    """Replace the pseudonym key, which changes every person code and so the fichas table."""
    (root / "data" / "interim" / "pseudonym.key").write_bytes(bytes(range(32, 64)))


def failing_exclusions(error: BaseException) -> Any:
    """Return a ``ParquetTableStore.write`` that raises ``error`` on the exclusion log."""
    original = ParquetTableStore.write

    def write(self: ParquetTableStore, models: Any, path: Path, *, model_cls: Any = None) -> None:
        if model_cls is not None and model_cls.__name__ == "Exclusion":
            raise error
        original(self, models, path, model_cls=model_cls)

    return write


@pytest.mark.parametrize(
    ("error", "code", "message"),
    [
        (OSError("synthetic disk failure"), 1, "error: synthetic disk failure"),
        (
            KeyboardInterrupt(),
            130,
            "interrupted: nothing half-written was left; run the command again",
        ),
    ],
)
def test_a_build_stopped_while_saving_leaves_the_previous_outputs_as_they_were(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
    code: int,
    message: str,
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path / "data" / "interim" / SNAPSHOT_ID
    before = published(interim)
    change_the_key(tmp_path)
    monkeypatch.setattr(ParquetTableStore, "write", failing_exclusions(error))
    capsys.readouterr()

    assert run(tmp_path) == code

    assert capsys.readouterr().err.strip() == message
    assert published(interim) == before
    assert leftovers(interim) == []


def test_a_build_stopped_before_its_manifest_leaves_outputs_that_the_reader_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path / "data" / "interim" / SNAPSHOT_ID
    change_the_key(tmp_path)

    def failing_write(path: Path, content: str) -> None:
        raise OSError("synthetic disk failure")

    monkeypatch.setattr(build_fichas_module, "write_text_atomically", failing_write)

    assert run(tmp_path) == 1

    # The tables moved into place, but the manifest, the commit point, still describes the
    # previous ones.
    with pytest.raises(ValueError, match="fichas.parquet differs from fichas_manifest.json"):
        load_fichas(interim)
    assert leftovers(interim) == []


NOTHING_HALF_WRITTEN = "interrupted: nothing half-written was left; run the command again"


def test_an_interruption_just_before_the_tables_move_leaves_the_previous_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path / "data" / "interim" / SNAPSHOT_ID
    before = published(interim)
    change_the_key(tmp_path)
    staged_when_interrupted: list[str] = []

    def interrupting_clock() -> datetime:
        # The build reads the clock for its manifest once both tables are staged.
        staged_when_interrupted.extend(leftovers(interim))
        raise KeyboardInterrupt

    capsys.readouterr()

    code = main(["--project-root", str(tmp_path)], clock=interrupting_clock)

    assert code == 130
    assert staged_when_interrupted == ["exclusions.parquet.staged", "fichas.parquet.staged"]
    assert capsys.readouterr().err.strip() == NOTHING_HALF_WRITTEN
    assert published(interim) == before
    assert leftovers(interim) == []


def record_the_moves(monkeypatch: pytest.MonkeyPatch, *, stop_at: int | None = None) -> list[str]:
    """Record the names that staged tables move to, in order.

    Move number ``stop_at`` is interrupted instead of done, as by Ctrl+C.
    """
    original = os.replace
    moved: list[str] = []

    def replace(source: Any, destination: Any) -> None:
        if Path(source).name.endswith(".staged"):
            if len(moved) + 1 == stop_at:
                raise KeyboardInterrupt
            original(source, destination)
            moved.append(Path(destination).name)
        else:
            original(source, destination)

    monkeypatch.setattr(os, "replace", replace)
    return moved


def interrupting_write(path: Path, content: str) -> None:
    raise KeyboardInterrupt


def being_replaced(interim: Path) -> str:
    """Return the message of an interruption once the tables have begun to move."""
    return (
        f"interrupted: the outputs in {interim} were being replaced, so the tables may not "
        "match fichas_manifest.json, and readers refuse them until they do; run the command "
        "again"
    )


def interrupt_the_cleanup(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Interrupt the first removal of a staged table, as a second Ctrl+C would; return the
    name of the file whose removal was interrupted."""
    original = Path.unlink
    interrupted: list[str] = []

    def unlink(self: Path, missing_ok: bool = False) -> None:
        if self.name.endswith(".staged") and not interrupted:
            interrupted.append(self.name)
            raise KeyboardInterrupt
        original(self, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", unlink)
    return interrupted


@pytest.mark.parametrize(
    ("stop_at", "manifest_write_interrupted", "moved_before_the_stop"),
    [(2, False, ["fichas.parquet"]), (None, True, ["fichas.parquet", "exclusions.parquet"])],
    ids=["between-the-moves", "before-the-manifest"],
)
def test_an_interruption_while_the_tables_move_says_that_the_outputs_need_a_new_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    stop_at: int | None,
    manifest_write_interrupted: bool,
    moved_before_the_stop: list[str],
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path.resolve() / "data" / "interim" / SNAPSHOT_ID
    before = published(interim)
    change_the_key(tmp_path)
    moved = record_the_moves(monkeypatch, stop_at=stop_at)
    if manifest_write_interrupted:
        monkeypatch.setattr(build_fichas_module, "write_text_atomically", interrupting_write)
    capsys.readouterr()

    assert run(tmp_path) == 130

    assert moved == moved_before_the_stop
    assert capsys.readouterr().err.strip() == being_replaced(interim)
    after = published(interim)
    assert after["fichas.parquet"] != before["fichas.parquet"]
    assert after["fichas_manifest.json"] == before["fichas_manifest.json"]
    assert leftovers(interim) == []
    with pytest.raises(ValueError, match="fichas.parquet differs from fichas_manifest.json"):
        load_fichas(interim)


def test_a_second_interruption_during_the_cleanup_keeps_the_message_of_the_moves(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path.resolve() / "data" / "interim" / SNAPSHOT_ID
    change_the_key(tmp_path)
    record_the_moves(monkeypatch, stop_at=2)
    interrupted = interrupt_the_cleanup(monkeypatch)
    capsys.readouterr()

    assert run(tmp_path) == 130

    assert interrupted == ["fichas.parquet.staged"]
    assert capsys.readouterr().err.strip() == being_replaced(interim)
    with pytest.raises(ValueError, match="fichas.parquet differs from fichas_manifest.json"):
        load_fichas(interim)
    # The fichas table had moved; the second Ctrl+C stopped the cleanup before the staged
    # exclusions table, whose move the first one interrupted, was removed.
    assert leftovers(interim) == ["exclusions.parquet.staged"]

    # The next run stages that table anew, moves it into place, and leaves nothing behind.
    monkeypatch.undo()
    assert run(tmp_path) == 0
    assert leftovers(interim) == []
    assert load_fichas(interim) == ParquetTableStore().read(Ficha, interim / "fichas.parquet")


def test_an_interruption_during_the_cleanup_of_a_complete_publish_left_nothing_half_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path / "data" / "interim" / SNAPSHOT_ID
    before = published(interim)
    change_the_key(tmp_path)
    interrupted = interrupt_the_cleanup(monkeypatch)
    capsys.readouterr()

    assert run(tmp_path) == 130

    assert interrupted == ["fichas.parquet.staged"]
    assert capsys.readouterr().err.strip() == NOTHING_HALF_WRITTEN
    assert published(interim)["fichas_manifest.json"] != before["fichas_manifest.json"]
    assert load_fichas(interim) == ParquetTableStore().read(Ficha, interim / "fichas.parquet")


def test_an_interruption_while_the_tables_move_stays_a_keyboard_interrupt_for_callers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project(tmp_path)
    record_the_moves(monkeypatch, stop_at=2)
    config = load_config(tmp_path / "config" / "default.yaml")

    with pytest.raises(KeyboardInterrupt) as caught:
        build_dataset(config, tmp_path, key=KEY, clock=lambda: RUN_TIME)

    assert isinstance(caught.value, PublishInterrupted)
    assert caught.value.interim_dir == tmp_path / "data" / "interim" / SNAPSHOT_ID


def test_the_reader_returns_the_fichas_of_outputs_that_match_their_manifest(
    tmp_path: Path,
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path / "data" / "interim" / SNAPSHOT_ID

    manifest = verify_fichas_outputs(interim)

    assert (manifest.snapshot_id, manifest.counts["included"]) == (SNAPSHOT_ID, 2)
    assert load_fichas(interim) == ParquetTableStore().read(Ficha, interim / "fichas.parquet")


def test_the_reader_returns_the_table_it_checked_even_when_it_is_replaced_right_after(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path / "data" / "interim" / SNAPSHOT_ID
    table = interim / "fichas.parquet"
    earlier_table = table.read_bytes()
    change_the_key(tmp_path)
    assert run(tmp_path) == 0
    checked = ParquetTableStore().read(Ficha, table)
    original_read_bytes = Path.read_bytes
    replaced: list[Path] = []

    def read_bytes(self: Path) -> bytes:
        content = original_read_bytes(self)
        if self.name == "fichas.parquet" and not replaced:
            # A concurrent build puts another table in place right after this read.
            self.write_bytes(earlier_table)
            replaced.append(self)
        return content

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    fichas = load_fichas(interim)

    assert replaced == [table]
    assert fichas == checked
    assert ParquetTableStore().read(Ficha, table) != checked


@pytest.mark.parametrize("name", ["fichas.parquet", "exclusions.parquet"])
def test_the_reader_refuses_a_table_that_differs_from_the_manifest(
    tmp_path: Path, name: str
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    path = tmp_path / "data" / "interim" / SNAPSHOT_ID / name
    path.write_bytes(path.read_bytes() + b"\0")

    with pytest.raises(ValueError, match=f"{name} differs from fichas_manifest.json"):
        load_fichas(path.parent)


@pytest.mark.parametrize("name", OUTPUT_NAMES)
def test_the_reader_refuses_outputs_with_a_missing_file(tmp_path: Path, name: str) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    path = tmp_path / "data" / "interim" / SNAPSHOT_ID / name
    path.unlink()

    with pytest.raises(FileNotFoundError, match=name):
        verify_fichas_outputs(path.parent)


LEAK = "SYNTHETICLEAK"
"""Text put into a damaged manifest; no error message may repeat it."""


def truncated(content: bytes) -> bytes:
    return content[: len(content) // 2]


def garbled(content: bytes) -> bytes:
    return b"\x00\xff" + LEAK.encode() + content[2 + len(LEAK) :]


def with_a_bad_snapshot_id(content: bytes) -> bytes:
    """Replace the snapshot id with one that its validator refuses, quoting the value."""
    return content.replace(SNAPSHOT_ID.encode(), f"../{LEAK}".encode())


def with_an_unknown_key(content: bytes) -> bytes:
    manifest = json.loads(content)
    manifest[f"{LEAK} key"] = 1
    return json.dumps(manifest).encode()


def with_a_count_key_on_two_lines(content: bytes) -> bytes:
    manifest = json.loads(content)
    manifest["counts"][f"{LEAK}\nsecond line"] = -5
    return json.dumps(manifest).encode()


@pytest.mark.parametrize(
    ("damage", "issues"),
    [
        (truncated, "json_invalid"),
        (garbled, "json_invalid"),
        (with_a_bad_snapshot_id, "snapshot_id: value_error"),
        (with_an_unknown_key, "<key>: extra_forbidden"),
        (with_a_count_key_on_two_lines, "counts.<key>: greater_than_equal"),
    ],
    ids=["truncated", "garbled", "bad-snapshot-id", "unknown-key", "count-key-on-two-lines"],
)
@pytest.mark.parametrize("reader", [verify_fichas_outputs, load_fichas])
def test_a_damaged_manifest_is_refused_in_one_line_that_points_to_the_build(
    tmp_path: Path, damage: Any, issues: str, reader: Any
) -> None:
    write_project(tmp_path)
    assert run(tmp_path) == 0
    interim = tmp_path / "data" / "interim" / SNAPSHOT_ID
    path = interim / "fichas_manifest.json"
    path.write_bytes(damage(path.read_bytes()))

    with pytest.raises(ValueError) as caught:
        reader(interim)

    # The issues name the manifest's own fields only, never a key or a value of the file.
    assert str(caught.value) == (
        f"fichas_manifest.json in {interim} is damaged ({issues}); run {BUILD_COMMAND} again"
    )
    assert LEAK not in "".join(traceback.format_exception(caught.value))


def two_word_lemmatizer(texts: Sequence[str]) -> list[list[str]]:
    """A lemmatizer whose only lemma has two words, which the report's privacy gate refuses."""
    return [["dos palabras"] for _ in texts]


@pytest.mark.parametrize("failure", ["refused", "failed"])
def test_a_quality_report_that_fails_after_the_build_says_that_the_fichas_were_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], failure: str
) -> None:
    write_project(tmp_path)
    report_dir = tmp_path / "results" / "fichas" / SNAPSHOT_ID
    if failure == "failed":
        # The report directory cannot be made, since a file holds its name.
        report_dir.parent.mkdir(parents=True)
        report_dir.write_bytes(b"not a directory")
    lemmatize = two_word_lemmatizer if failure == "refused" else fake_lemmatizer

    code = main(
        ["--project-root", str(tmp_path), "--report-out", str(report_dir)],
        clock=lambda: RUN_TIME,
        lemmatize=lemmatize,
    )

    assert code == 1
    interim = tmp_path.resolve() / "data" / "interim" / SNAPSHOT_ID
    error = capsys.readouterr().err
    assert error.startswith(
        f"error: the fichas outputs were written to {interim}, but the quality report was not: "
    )
    assert len(error.strip().splitlines()) == 1
    manifest = json.loads((interim / "fichas_manifest.json").read_text(encoding="utf-8"))
    fichas_sha256 = hashlib.sha256((interim / "fichas.parquet").read_bytes()).hexdigest()
    assert manifest["fichas_sha256"] == fichas_sha256
    assert not (report_dir / "quality_report.json").exists()


def interrupting_lemmatizer(texts: Sequence[str]) -> list[list[str]]:
    raise KeyboardInterrupt


def test_an_interruption_during_the_quality_report_says_that_the_fichas_were_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_project(tmp_path)
    report_dir = tmp_path / "results" / "fichas" / SNAPSHOT_ID

    code = main(
        ["--project-root", str(tmp_path), "--report-out", str(report_dir)],
        clock=lambda: RUN_TIME,
        lemmatize=interrupting_lemmatizer,
    )

    assert code == 130
    interim = tmp_path.resolve() / "data" / "interim" / SNAPSHOT_ID
    assert capsys.readouterr().err.strip() == (
        f"interrupted: the fichas outputs were written to {interim}, but the quality report "
        "was not; run the command again"
    )
    assert (interim / "fichas_manifest.json").is_file()
    assert not report_dir.exists()


@pytest.mark.parametrize(
    "content",
    ["snapshot: [not closed\nfichas: {}\n", "fichas: {}\nfichas: {}\n"],
    ids=["bad-syntax", "repeated-key"],
)
def test_an_unreadable_configuration_fails_in_one_line_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str
) -> None:
    write_project(tmp_path)
    (tmp_path / "config" / "default.yaml").write_bytes(content.encode())

    assert run(tmp_path) == 1

    error = capsys.readouterr().err
    assert error.startswith("error: ")
    assert len(error.strip().splitlines()) == 1
    assert not (tmp_path / "data" / "interim" / SNAPSHOT_ID / "fichas.parquet").exists()
