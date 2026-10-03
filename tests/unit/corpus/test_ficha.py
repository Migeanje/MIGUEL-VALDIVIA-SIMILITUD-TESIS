"""The per-thesis record ("ficha"): its fields, its cross-field rules and its document code.

Every value here is synthetic. The uuids are counters, the texts are placeholders, and the
person codes only have the shape of a pseudonym; none was made from a real person.
"""

from datetime import date
from typing import Any, get_args
from uuid import UUID

import pytest
from pydantic import ValidationError

from thematic_redundancy.corpus.ficha import Ficha, FichaPdfStatus, doc_code_for
from thematic_redundancy.corpus.pdf_manifest import PdfStatus

ITEM = UUID("00000000-0000-4000-8000-000000000001")
OTHER_ITEM = UUID("00000000-0000-4000-8000-000000000002")
SHA256 = "0123456789abcdef" * 4
ADVISOR_CODE = "ADV-0123456789"
AUTHOR_CODE = "AUT-abcdef0123"
SECOND_AUTHOR_CODE = "AUT-0123456789"
"""Sorts before :data:`AUTHOR_CODE`, so listing it second shows that order is kept."""
TESIS = "https://purl.org/pe-repo/renati/type#tesis"
SUFICIENCIA = "https://purl.org/pe-repo/renati/type#trabajoDeSuficienciaProfesional"
ACADEMICO = "https://purl.org/pe-repo/renati/type#trabajoAcademico"
FILE_STATUSES = ("downloaded", "already_present")
NO_FILE_STATUSES = tuple(
    status for status in get_args(FichaPdfStatus) if status not in FILE_STATUSES
)


def fields(**overrides: Any) -> dict[str, Any]:
    """Return the fields of a valid, included thesis, with ``overrides`` applied.

    The doc code follows the item uuid unless an override sets it.
    """
    values: dict[str, Any] = {
        "item_uuid": ITEM,
        "handle_url": "https://repo.example.edu/handle/123456789/1001",
        "snapshot_id": "20261002T224412Z",
        "document_type": "tesis",
        "renati_type_raw": TESIS,
        "title": "Título sintético 001",
        "program_key": "sistemas",
        "program_name": "Ingeniería de Sistemas",
        "issue_date_raw": "2023-05-10",
        "issued_after_snapshot": False,
        "rights": "open",
        "include": True,
        **overrides,
    }
    values.setdefault("doc_code", doc_code_for(values["item_uuid"]))
    return values


def excluded(reason: str, **overrides: Any) -> dict[str, Any]:
    """Return the fields of a thesis excluded for ``reason``, with ``overrides`` applied."""
    return fields(include=False, exclusion_reason=reason, **overrides)


def author_codes(count: int) -> tuple[str, ...]:
    """Return ``count`` distinct author codes."""
    return tuple(f"AUT-{number:010x}" for number in range(count))


def refused(values: dict[str, Any], message: str) -> None:
    """Check that ``values`` build no ficha, and that the error says ``message``."""
    with pytest.raises(ValidationError, match=message):
        Ficha(**values)


# Construction


def test_a_complete_included_thesis_keeps_every_field() -> None:
    ficha = Ficha(
        **fields(
            abstract="Resumen sintético: diseño de un sistema de información.",
            objectives="Objetivo sintético: evaluar un prototipo.",
            objectives_status="extracted",
            issue_year=2023,
            keywords=("palabra clave uno", "señal"),
            ocde_codes=("2.02.04", "1.02.01"),
            language="spa",
            embargo_end=date(2024, 6, 30),
            advisor_code=ADVISOR_CODE,
            author_codes=(AUTHOR_CODE, SECOND_AUTHOR_CODE),
            pdf_status="downloaded",
            pdf_sha256=SHA256,
            source_format="digital",
            section_source="title_abstract_objectives",
            quality_notes=("abstract_hard_wrapped",),
        )
    )

    assert (ficha.item_uuid, ficha.doc_code) == (ITEM, doc_code_for(ITEM))
    assert (ficha.document_type, ficha.renati_type_raw) == ("tesis", TESIS)
    assert ficha.objectives == "Objetivo sintético: evaluar un prototipo."
    assert (ficha.issue_year, ficha.embargo_end) == (2023, date(2024, 6, 30))
    assert ficha.keywords == ("palabra clave uno", "señal")
    assert ficha.advisor_code == ADVISOR_CODE
    assert ficha.author_codes == (AUTHOR_CODE, SECOND_AUTHOR_CODE)
    assert (ficha.pdf_status, ficha.pdf_sha256) == ("downloaded", SHA256)
    assert (ficha.include, ficha.exclusion_reason, ficha.duplicate_of) == (True, None, None)


def test_optional_fields_start_empty_and_pipeline_fields_start_pending() -> None:
    ficha = Ficha(**fields())

    assert (ficha.abstract, ficha.objectives, ficha.language, ficha.embargo_end) == (None,) * 4
    assert (ficha.issue_year, ficha.advisor_code, ficha.pdf_sha256) == (None,) * 3
    assert (ficha.keywords, ficha.ocde_codes, ficha.author_codes, ficha.quality_notes) == (
        (),
        (),
        (),
        (),
    )
    assert (ficha.objectives_status, ficha.source_format, ficha.section_source) == (
        ("pending",) * 3
    )
    assert ficha.pdf_status == "not_attempted"


def test_a_ficha_is_immutable_down_to_its_lists() -> None:
    ficha = Ficha(**fields(keywords=["uno", "dos"], quality_notes=["nota"]))

    with pytest.raises(ValidationError, match="frozen"):
        ficha.title = "Otro título sintético"  # type: ignore[misc]
    assert ficha.keywords == ("uno", "dos")
    assert ficha.quality_notes == ("nota",)


def test_a_ficha_refuses_unknown_fields() -> None:
    with pytest.raises(ValidationError, match="author_name") as caught:
        Ficha(**fields(author_name="Autor Sintético, 001"))

    assert [error["type"] for error in caught.value.errors()] == ["extra_forbidden"]


# Document code


def test_the_doc_code_is_a_short_code_derived_from_the_item_uuid() -> None:
    # Pinned values: a change here would rename every document in every report.
    assert doc_code_for(ITEM) == "DOC-11e594f4"
    assert doc_code_for(OTHER_ITEM) == "DOC-e79acd97"
    assert doc_code_for(UUID(str(ITEM).upper())) == doc_code_for(ITEM)


@pytest.mark.parametrize(
    "doc_code",
    [doc_code_for(OTHER_ITEM), "DOC-00000000", "DOC-11E594F4", "11e594f4", ""],
    ids=["other-item", "made-up", "upper-case", "no-prefix", "empty"],
)
def test_a_doc_code_not_derived_from_the_item_uuid_is_refused(doc_code: str) -> None:
    refused(fields(doc_code=doc_code), "doc_code must be 'DOC-11e594f4'")


# Inclusion and exclusion


@pytest.mark.parametrize(
    "values",
    [
        fields(),
        excluded("no_abstract_no_objectives"),
        excluded("restricted_no_text"),
        excluded("other"),
        excluded("duplicate", duplicate_of=OTHER_ITEM),
        excluded(
            "non_thesis_type", document_type="trabajo_suficiencia", renati_type_raw=SUFICIENCIA
        ),
    ],
    ids=["included", "no-text", "restricted", "other", "duplicate", "non-thesis"],
)
def test_an_included_ficha_has_no_reason_and_an_excluded_one_has_one(
    values: dict[str, Any],
) -> None:
    ficha = Ficha(**values)

    assert ficha.include is (ficha.exclusion_reason is None)


def test_an_excluded_ficha_needs_an_exclusion_reason() -> None:
    refused(fields(include=False), "an excluded ficha needs an exclusion_reason")


def test_an_included_ficha_must_not_have_an_exclusion_reason() -> None:
    refused(fields(exclusion_reason="other"), "an included ficha must not have an exclusion_reason")


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (excluded("other", duplicate_of=OTHER_ITEM), "duplicate_of needs exclusion_reason"),
        (fields(duplicate_of=OTHER_ITEM), "duplicate_of needs exclusion_reason"),
        (excluded("duplicate"), "exclusion_reason 'duplicate' needs duplicate_of"),
        (excluded("duplicate", duplicate_of=ITEM), "duplicate_of must name another item"),
    ],
    ids=["other-reason", "included", "no-original", "itself"],
)
def test_duplicate_of_and_the_duplicate_reason_come_together(
    values: dict[str, Any], message: str
) -> None:
    refused(values, message)


# Objectives


@pytest.mark.parametrize("status", ["extracted", "manual"])
def test_objectives_text_comes_with_an_extracted_or_manual_status(status: str) -> None:
    ficha = Ficha(**fields(objectives="Objetivo sintético.", objectives_status=status))

    assert ficha.objectives == "Objetivo sintético."


@pytest.mark.parametrize("status", ["pending", "not_found", "no_pdf"])
def test_without_objectives_text_the_status_tells_why(status: str) -> None:
    assert Ficha(**fields(objectives_status=status)).objectives is None


@pytest.mark.parametrize("status", ["pending", "not_found", "no_pdf"])
def test_objectives_text_is_refused_under_any_other_status(status: str) -> None:
    refused(
        fields(objectives="Objetivo sintético.", objectives_status=status),
        "objectives text needs objectives_status 'extracted' or 'manual'",
    )


@pytest.mark.parametrize("status", ["extracted", "manual"])
def test_an_extracted_or_manual_status_needs_objectives_text(status: str) -> None:
    refused(fields(objectives_status=status), f"objectives_status '{status}' needs objectives text")


# Document type


@pytest.mark.parametrize(
    ("document_type", "renati_type"),
    [
        ("trabajo_suficiencia", SUFICIENCIA),
        ("trabajo_academico", ACADEMICO),
        ("other", "https://purl.org/pe-repo/renati/type#libro"),
        ("other", None),
    ],
    ids=["suficiencia", "academico", "other-type", "no-type"],
)
def test_a_document_that_is_not_a_thesis_is_excluded_as_non_thesis_type(
    document_type: str, renati_type: str | None
) -> None:
    ficha = Ficha(
        **excluded("non_thesis_type", document_type=document_type, renati_type_raw=renati_type)
    )

    assert (ficha.include, ficha.exclusion_reason) == (False, "non_thesis_type")
    assert ficha.renati_type_raw == renati_type


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (
            fields(document_type="trabajo_suficiencia", renati_type_raw=SUFICIENCIA),
            "must be excluded as 'non_thesis_type'",
        ),
        (
            excluded("other", document_type="trabajo_academico", renati_type_raw=ACADEMICO),
            "must be excluded as 'non_thesis_type'",
        ),
        (excluded("non_thesis_type"), "'non_thesis_type' contradicts document_type 'tesis'"),
    ],
    ids=["included-non-thesis", "other-reason", "thesis-as-non-thesis"],
)
def test_only_a_non_thesis_is_excluded_as_non_thesis_type(
    values: dict[str, Any], message: str
) -> None:
    refused(values, message)


# PDF


def test_pdf_statuses_mirror_the_pdf_manifest_plus_not_attempted() -> None:
    assert set(get_args(FichaPdfStatus)) == {*get_args(PdfStatus), "not_attempted"}


@pytest.mark.parametrize("status", FILE_STATUSES)
def test_a_pdf_on_disk_comes_with_its_sha256(status: str) -> None:
    assert Ficha(**fields(pdf_status=status, pdf_sha256=SHA256)).pdf_sha256 == SHA256


@pytest.mark.parametrize("status", FILE_STATUSES)
def test_a_pdf_on_disk_without_its_sha256_is_refused(status: str) -> None:
    refused(fields(pdf_status=status), f"pdf_status '{status}' needs pdf_sha256")


@pytest.mark.parametrize("status", NO_FILE_STATUSES)
def test_a_pdf_status_without_a_file_has_no_sha256(status: str) -> None:
    assert Ficha(**fields(pdf_status=status)).pdf_sha256 is None
    refused(fields(pdf_status=status, pdf_sha256=SHA256), "pdf_sha256 needs pdf_status")


@pytest.mark.parametrize(
    "digest", [SHA256.upper(), SHA256[:-2], "zz" * 32], ids=["upper-case", "short", "not-hex"]
)
def test_a_malformed_sha256_is_refused(digest: str) -> None:
    refused(fields(pdf_status="downloaded", pdf_sha256=digest), "pdf_sha256")


# People


def refused_author_codes(codes: object, error_type: str) -> None:
    """Check that ``codes`` are refused as ``author_codes``, for an error of ``error_type``."""
    with pytest.raises(ValidationError) as caught:
        Ficha(**fields(author_codes=codes))

    errors = caught.value.errors()
    assert {error["type"] for error in errors} == {error_type}
    assert {error["loc"][0] for error in errors} == {"author_codes"}


@pytest.mark.parametrize(
    "value",
    [
        "Asesor Sintético, 001",
        "0000-0000-0000-0001",
        AUTHOR_CODE,
        "ADV-ABCDEF0123",
        "ADV-012345678",
    ],
    ids=["name", "orcid", "author-code", "upper-case", "short"],
)
def test_the_advisor_field_holds_only_an_advisor_code(value: str) -> None:
    refused(fields(advisor_code=value), "advisor_code")


@pytest.mark.parametrize("count", [0, 1, 2, 5], ids=["none", "one", "two", "most"])
def test_a_ficha_lists_up_to_five_author_codes(count: int) -> None:
    assert Ficha(**fields(author_codes=author_codes(count))).author_codes == author_codes(count)


def test_author_codes_keep_the_repository_order() -> None:
    ficha = Ficha(**fields(author_codes=[AUTHOR_CODE, SECOND_AUTHOR_CODE]))

    assert ficha.author_codes == (AUTHOR_CODE, SECOND_AUTHOR_CODE)
    assert ficha.author_codes != tuple(sorted(ficha.author_codes))


def test_author_codes_must_not_repeat_a_code() -> None:
    refused_author_codes((AUTHOR_CODE, SECOND_AUTHOR_CODE, AUTHOR_CODE), "value_error")
    refused(fields(author_codes=(AUTHOR_CODE, AUTHOR_CODE)), "must not repeat a code")


def test_more_than_five_author_codes_are_refused() -> None:
    refused_author_codes(author_codes(6), "too_long")


@pytest.mark.parametrize(
    "codes",
    [
        ("Autor Sintético, 001",),
        (ADVISOR_CODE,),
        ("AUT-ABCDEF0123",),
        ("AUT-abcdef012",),
        ("AUT-abcdef01234",),
        (AUTHOR_CODE, ""),
    ],
    ids=["name", "advisor-code", "upper-case", "short", "long", "blank-second"],
)
def test_author_codes_hold_only_author_codes(codes: tuple[str, ...]) -> None:
    refused_author_codes(codes, "string_pattern_mismatch")


def test_a_bare_string_is_not_taken_for_a_list_of_author_codes() -> None:
    refused_author_codes(AUTHOR_CODE, "tuple_type")


def test_the_single_author_code_field_no_longer_exists() -> None:
    with pytest.raises(ValidationError, match="author_code") as caught:
        Ficha(**fields(author_code=AUTHOR_CODE))

    assert [error["type"] for error in caught.value.errors()] == ["extra_forbidden"]


# Value constraints


@pytest.mark.parametrize("year", [1900, 2023, 2100, None])
def test_the_issue_year_is_optional_and_bounded(year: int | None) -> None:
    assert Ficha(**fields(issue_year=year)).issue_year == year


@pytest.mark.parametrize(
    "year", [1899, 2101, "2023", 2023.0, True], ids=["low", "high", "text", "float", "bool"]
)
def test_an_issue_year_out_of_bounds_or_not_an_integer_is_refused(year: object) -> None:
    refused(fields(issue_year=year), "issue_year")


@pytest.mark.parametrize(
    ("field", "value"),
    [("include", "yes"), ("include", 1), ("issued_after_snapshot", "false")],
)
def test_flags_must_be_real_booleans(field: str, value: object) -> None:
    refused(fields(**{field: value}), field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "   "),
        ("title", ""),
        ("abstract", "\n\t "),
        ("renati_type_raw", " "),
        ("program_name", ""),
        ("issue_date_raw", " "),
        ("keywords", ("uno", " ")),
        ("quality_notes", ("",)),
        ("quality_notes", ("x" * 201,)),
        ("handle_url", "repo.example.edu/handle/123456789/1001"),
        ("handle_url", "ftp://repo.example.edu/handle/123456789/1001"),
        ("handle_url", "https://repo.example.edu/handle/123456789/ 1001"),
        ("program_key", "Sistemas"),
        ("snapshot_id", "../20261002T224412Z"),
        ("ocde_codes", ("https://purl.org/pe-repo/ocde/ford#2.02.04",)),
        ("language", "español"),
        ("rights", "metadata_only"),
        ("exclusion_reason", "unknown"),
    ],
)
def test_malformed_values_are_refused(field: str, value: object) -> None:
    refused(fields(**{field: value}), field)


def test_every_broken_rule_is_reported_at_once() -> None:
    with pytest.raises(ValidationError) as caught:
        Ficha(**fields(include=False, objectives="Objetivo sintético.", doc_code="DOC-00000000"))

    message = str(caught.value)
    assert "doc_code must be" in message
    assert "needs an exclusion_reason" in message
    assert "objectives text needs" in message
