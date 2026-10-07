"""The quality report of the fichas: counts, handles and lemmas only, behind a privacy gate.

Every ficha and record here is synthetic. Texts are placeholders, the people are invented, and
the lemmatizer is a stand-in that splits on spaces, so every expected value can be worked out
by hand.
"""

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.corpus.fichas_quality import (
    duplicate_pairs,
    quality_report,
    render_markdown,
    stopword_recheck,
    write_quality_report,
)
from thematic_redundancy.corpus.snapshot import SnapshotRecord

HARVEST_TIME = datetime(2026, 10, 2, 15, 4, 5, tzinfo=UTC)
COLLECTION = UUID("00000000-0000-4000-8000-0000000000c0")
VALID_ORCID = "https://orcid.org/0000-0002-1825-0097"
BAD_ORCID = "https://orcid.org/0000-0002-1825-0098"
ADVISOR = "Asesorx Sintético, Docente"
TITLE = "Título sintético de una tesis de prueba para el informe"


def item(number: int) -> UUID:
    return UUID(f"00000000-0000-4000-8000-{number:012d}")


def handle(number: int) -> str:
    return f"123456789/{1000 + number}"


def ficha(number: int, **overrides: Any) -> Ficha:
    """Return a valid included ficha with objectives, with ``overrides`` applied."""
    values: dict[str, Any] = {
        "item_uuid": item(number),
        "doc_code": doc_code_for(item(number)),
        "handle_url": f"https://hdl.handle.net/{handle(number)}",
        "snapshot_id": "20261002T150405Z",
        "document_type": "tesis",
        "renati_type_raw": "https://purl.org/pe-repo/renati/type#tesis",
        "title": f"{TITLE} {number}",
        "abstract": "Resumen sintético.",
        "objectives": "Diseñar un prototipo.\n\nEvaluar el prototipo.",
        "objectives_status": "extracted",
        "program_key": "sistemas",
        "program_name": "Ingeniería de Sistemas",
        "issue_date_raw": "2023-05-10",
        "issue_year": 2023,
        "issued_after_snapshot": False,
        "keywords": ("Prueba",),
        "rights": "open",
        "advisor_code": "ADV-0123456789",
        "author_codes": ("AUT-0123456789",),
        "pdf_status": "downloaded",
        "pdf_sha256": f"{number:064x}",
        "source_format": "digital",
        "section_source": "title_abstract_objectives",
        "include": True,
    }
    values.update(overrides)
    return Ficha(**values)


def metadata_only(number: int, **overrides: Any) -> Ficha:
    """Return a valid included ficha without a PDF, with ``overrides`` applied."""
    values: dict[str, Any] = {
        "objectives": None,
        "objectives_status": "no_pdf",
        "pdf_status": "restricted",
        "pdf_sha256": None,
        "source_format": "none",
        "section_source": "title_abstract",
    }
    return ficha(number, **{**values, **overrides})


def record(
    number: int, *, orcid: str | None = None, advisor: str | None = ADVISOR
) -> SnapshotRecord:
    metadata: dict[str, Any] = {"dc.title": [{"value": f"{TITLE} {number}"}]}
    if advisor is not None:
        metadata["dc.contributor.advisor"] = [{"value": advisor}]
    if orcid is not None:
        metadata["renati.advisor.orcid"] = [{"value": orcid}]
    return SnapshotRecord(
        uuid=item(number),
        handle=handle(number),
        program_key="sistemas",
        collection_uuid=COLLECTION,
        harvested_at=HARVEST_TIME,
        metadata=metadata,
    )


def split_words(texts: Sequence[str]) -> list[list[str]]:
    """Stand-in for the full cleaner: lower-case words, with periods dropped."""
    return [text.lower().replace(".", " ").split() for text in texts]


def report_of(fichas: Sequence[Ficha], records: Sequence[SnapshotRecord], **kwargs: Any) -> Any:
    return quality_report(
        fichas,
        records,
        program_keys=["sistemas", "minas"],
        domain_stopwords=kwargs.pop("domain_stopwords", frozenset({"evaluar", "tesis"})),
        lemmatize=split_words,
        provenance={"snapshot_id": "20261002T150405Z"},
        **kwargs,
    )


def corpus() -> tuple[list[Ficha], list[SnapshotRecord]]:
    """Three included theses (one metadata-only, one future-dated) and two exclusions."""
    fichas = [
        ficha(1, quality_notes=("objectives locator flag: ocr_page",)),
        ficha(
            2,
            program_key="minas",
            program_name="Ingeniería de Minas",
            issue_year=2026,
            issue_date_raw="2026-12-04",
            issued_after_snapshot=True,
            objectives="Diseñar un modelo.",
            quality_notes=("advisor ORCID iD is malformed",),
        ),
        metadata_only(
            3, keywords=(), quality_notes=("PDF on disk belongs to another item (123456789/1001)",)
        ),
        ficha(4, include=False, exclusion_reason="duplicate", duplicate_of=item(1)),
        metadata_only(
            5,
            document_type="trabajo_academico",
            include=False,
            exclusion_reason="non_thesis_type",
            pdf_status="not_attempted",
        ),
    ]
    records = [
        record(1, orcid=VALID_ORCID),
        record(2, orcid=BAD_ORCID),
        record(3),
        record(4, orcid=VALID_ORCID),
        record(5, advisor=None),
    ]
    return fichas, records


def test_the_report_counts_the_fichas_by_inclusion_and_by_their_fields() -> None:
    report = report_of(*corpus())

    counts = report["counts"]
    assert counts["records"] == 5
    assert counts["include"] == {"included": 3, "excluded": 2}
    assert counts["exclusion_reason"] == {"duplicate": 1, "non_thesis_type": 1}
    assert counts["included_by_program"] == {"sistemas": 2, "minas": 1}
    assert counts["included_by_year"] == {"2023": 2, "2026": 1}
    assert counts["objectives_status"] == {"extracted": 2, "no_pdf": 1}
    assert counts["section_source"] == {"title_abstract": 1, "title_abstract_objectives": 2}
    assert counts["pdf_status"] == {"downloaded": 2, "restricted": 1}
    assert counts["source_format"] == {"digital": 2, "none": 1}
    assert counts["rights"] == {"open": 3}


def test_the_report_counts_notes_by_type_orcids_dates_and_completeness() -> None:
    report = report_of(*corpus())

    assert report["quality_notes"] == {
        "PDF on disk belongs to another item": 1,
        "advisor ORCID iD is malformed": 1,
        "objectives locator flag: ocr_page": 1,
    }
    assert report["advisor_orcid"] == {
        "theses_with_orcid": 2,
        "valid": 1,
        "invalid": 1,
        "advisor_code_from_orcid": 1,
        "advisor_code_from_name": 2,
        "without_advisor": 0,
        "distinct_advisor_codes": 1,
        "distinct_author_codes": 1,
    }
    assert report["future_dated"] == [handle(2)]
    completeness = report["completeness"]
    assert (completeness["included"], completeness["objectives"]) == (3, 2)
    assert (completeness["keywords"], completeness["embargo_end"]) == (2, 0)
    assert completeness["pdf_sha256"] == 2


def test_text_lengths_give_the_median_and_the_nearest_rank_p95() -> None:
    fichas = [ficha(n, title="x" * (10 * n)) for n in range(1, 21)]

    lengths = report_of(fichas, [record(n) for n in range(1, 21)])["text_lengths"]

    assert lengths["title"] == {"documents": 20, "median": 105.0, "p95": 190}
    assert lengths["objectives"]["documents"] == 20


def test_duplicate_pairs_say_which_record_stays_and_how_the_twins_compare() -> None:
    kept = ficha(2)
    twin = ficha(1, include=False, exclusion_reason="duplicate", duplicate_of=item(2))
    same_pdf = ficha(
        3, include=False, exclusion_reason="duplicate", duplicate_of=item(2), pdf_sha256=f"{2:064x}"
    )

    pairs = duplicate_pairs([kept, twin, same_pdf], [record(1), record(2), record(3)])

    assert pairs == [
        {
            "kept": handle(2),
            "duplicate": handle(1),
            "kept_is_smaller_uuid": False,
            "same_pdf": False,
            "objectives_equal": True,
        },
        {
            "kept": handle(2),
            "duplicate": handle(3),
            "kept_is_smaller_uuid": True,
            "same_pdf": True,
            "objectives_equal": True,
        },
    ]


def test_objectives_of_a_twin_without_text_are_not_compared() -> None:
    twin = metadata_only(1, include=False, exclusion_reason="duplicate", duplicate_of=item(2))

    assert duplicate_pairs([ficha(2), twin], [record(1), record(2)])[0]["objectives_equal"] is None


def test_the_stopword_recheck_counts_documents_and_flags_candidates_only() -> None:
    texts = ["Diseñar un sistema.", "Evaluar el sistema.", "Diseñar el modelo."] * 10
    texts += ["Otra frase."] * 70

    recheck = stopword_recheck(texts, frozenset({"evaluar", "tesis"}), split_words)

    assert recheck["documents"] == 100
    assert recheck["domain_stopword_documents"] == {"evaluar": 10, "tesis": 0}
    assert recheck["domain_stopwords_absent"] == 1
    assert recheck["frequent_unlisted_lemmas"] == {
        "frase": 70,
        "otra": 70,
        "diseñar": 20,
        "el": 20,
        "sistema": 20,
        "modelo": 10,
        "un": 10,
    }
    assert recheck["verb_shaped_candidates"] == ["diseñar"]


def test_a_lemma_just_below_the_threshold_is_left_out() -> None:
    texts = ["raro"] * 4 + ["otro"] * 96

    recheck = stopword_recheck(texts, frozenset(), split_words)

    assert "raro" not in recheck["frequent_unlisted_lemmas"]


def test_the_report_and_its_markdown_hold_numbers_handles_and_lemmas_only(
    tmp_path: Path,
) -> None:
    fichas, records = corpus()
    report = report_of(fichas, records)

    json_path, markdown_path = write_quality_report(tmp_path / "out", report, records)

    published = json_path.read_text(encoding="utf-8") + markdown_path.read_text(encoding="utf-8")
    for text in (TITLE, "Resumen sintético", "Diseñar un prototipo", ADVISOR, "1825"):
        assert text.casefold() not in published.casefold()
    assert json.loads(json_path.read_text(encoding="utf-8")) == report
    assert "| Included theses | 3 |" in markdown_path.read_text(encoding="utf-8")


def test_a_report_that_holds_a_title_is_refused_and_nothing_is_written(tmp_path: Path) -> None:
    fichas, records = corpus()
    report = report_of(fichas, records)
    report["provenance"]["note"] = f"{TITLE} 1"

    with pytest.raises(ValueError, match="1 string.* taken from the snapshot; nothing was written"):
        write_quality_report(tmp_path / "out", report, records)

    assert not (tmp_path / "out").exists()


def test_a_lemma_that_is_not_a_single_word_is_refused(tmp_path: Path) -> None:
    fichas, records = corpus()
    report = report_of(fichas, records)
    report["stopword_recheck"]["frequent_unlisted_lemmas"]["dos palabras"] = 9

    with pytest.raises(ValueError, match="1 lemma.* not a single word"):
        write_quality_report(tmp_path / "out", report, records)


def test_the_markdown_summarizes_the_report() -> None:
    markdown = render_markdown(report_of(*corpus()))

    assert markdown.startswith("# Fichas quality report of snapshot 20261002T150405Z")
    assert f"`{handle(2)}`" in markdown
    assert "diseñar" in markdown
