"""Quality report of a fichas dataset (T12): counts, public handles and lemmas only.

:func:`quality_report` describes the fichas of a snapshot: counts by inclusion, exclusion
reason, program, year, objectives status, section source, PDF status, source format and
rights; quality notes by type; advisor ORCID coverage; the future-dated theses by handle;
field completeness; text lengths; the D24 duplicate pairs; and :func:`stopword_recheck`.

The stopword re-check runs the full cleaner, without domain stopwords, over the objectives
text of the included theses. It reports how many of them hold each domain stopword, and the
lemmas held by at least :data:`MIN_DOCUMENT_RATIO` of them that the list lacks. Lemmas shaped
like a Spanish infinitive are flagged as candidates for the user to review; the list itself
is never changed here.

:func:`write_quality_report` writes ``quality_report.json`` and a short
``quality_report.md``, but only after searching both for every title, abstract, abstract
sentence, person's name and ORCID iD of the snapshot, and checking that each lemma is a
single word. One failure writes nothing.
"""

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import UUID

from thematic_redundancy.corpus.anonymize import normalize_orcid
from thematic_redundancy.corpus.ficha import Ficha
from thematic_redundancy.corpus.profile import (
    ADVISOR_KEY,
    ADVISOR_ORCID_KEY,
    MISSING,
    leaked_strings,
    sensitive_strings,
)
from thematic_redundancy.corpus.snapshot import SnapshotRecord
from thematic_redundancy.extraction.text_manifest import write_json_atomically
from thematic_redundancy.preprocessing.full_cleaner import SPACY_MODEL, clean_full_batch
from thematic_redundancy.preprocessing.stopwords import stopword_candidates

REPORT_JSON_FILE_NAME = "quality_report.json"
REPORT_MARKDOWN_FILE_NAME = "quality_report.md"

MIN_DOCUMENT_RATIO = 0.05
"""Share of the documents that a lemma must reach to be reported, as in the T11 candidates."""

MAX_LEMMA_LENGTH = 40
"""Longest lemma that a report may hold; a longer one is no single lemma."""

LEMMATIZER = f"full cleaner (spaCy {SPACY_MODEL}) without domain stopwords"

Lemmatizer = Callable[[Sequence[str]], list[list[str]]]
"""Turns texts into their lemma tokens, as the full cleaner does."""

COMPLETENESS_FIELDS = (
    "abstract",
    "objectives",
    "issue_year",
    "keywords",
    "ocde_codes",
    "language",
    "embargo_end",
    "advisor_code",
    "author_codes",
    "pdf_sha256",
)
"""Optional ficha fields whose presence the report counts."""

COUNTED_FIELDS = {
    "objectives_status": "Objectives status",
    "section_source": "Section source",
    "pdf_status": "PDF status",
    "source_format": "Source format",
    "rights": "Rights",
}
"""Ficha fields whose values the report counts over the included theses, with their labels."""

_INFINITIVE = re.compile(r"^\w{2,}(?:ar|er|ir)$")
"""A lemma shaped like a Spanish infinitive: likely a verb, perhaps one that reports what the
research did, as ``analizar`` does. Some nouns share the shape (``lugar``)."""

_NOTE_DETAIL = re.compile(r" \([^()]*\)$")
"""Trailing detail of a quality note, such as the handle of another item."""


def spacy_lemmatizer(texts: Sequence[str]) -> list[list[str]]:
    """Return the full cleaner's lemma tokens of each text, without domain stopwords."""
    return clean_full_batch(texts, domain_stopwords=frozenset())


def quality_report(
    fichas: Sequence[Ficha],
    records: Sequence[SnapshotRecord],
    *,
    program_keys: Sequence[str],
    domain_stopwords: frozenset[str],
    lemmatize: Lemmatizer,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the quality report of ``fichas``; ``records`` are the snapshot records that
    they were built from. Every count but the inclusion counts covers the included theses."""
    included = [ficha for ficha in fichas if ficha.include]
    by_uuid = {record.uuid: record for record in records}
    years = Counter(str(ficha.issue_year or MISSING) for ficha in included)
    programs = Counter(ficha.program_key for ficha in included)
    return {
        "provenance": dict(provenance),
        "counts": {
            "records": len(fichas),
            "include": {"included": len(included), "excluded": len(fichas) - len(included)},
            "exclusion_reason": _sorted(
                Counter(str(ficha.exclusion_reason) for ficha in fichas if not ficha.include)
            ),
            "included_by_program": {
                key: programs[key]
                for key in [*program_keys, *sorted(set(programs) - set(program_keys))]
            },
            "included_by_year": dict(sorted(years.items())),
            **{
                field: _sorted(Counter(str(getattr(ficha, field)) for ficha in included))
                for field in COUNTED_FIELDS
            },
        },
        "quality_notes": _ranked(
            Counter(
                _NOTE_DETAIL.sub("", note) for ficha in included for note in ficha.quality_notes
            )
        ),
        "advisor_orcid": _advisor_orcid(included, by_uuid),
        "future_dated": sorted(
            _handle(ficha, by_uuid) for ficha in included if ficha.issued_after_snapshot
        ),
        "completeness": {
            "included": len(included),
            **{
                field: sum(getattr(ficha, field) not in (None, ()) for ficha in included)
                for field in COMPLETENESS_FIELDS
            },
        },
        "text_lengths": {
            field: _lengths(getattr(ficha, field) for ficha in included)
            for field in ("title", "abstract", "objectives")
        },
        "duplicates": duplicate_pairs(fichas, records),
        "stopword_recheck": {
            "lemmatizer": LEMMATIZER,
            **stopword_recheck(
                [ficha.objectives for ficha in included if ficha.objectives],
                domain_stopwords,
                lemmatize,
            ),
        },
    }


def duplicate_pairs(
    fichas: Sequence[Ficha], records: Sequence[SnapshotRecord]
) -> list[dict[str, Any]]:
    """Return one entry per D24 duplicate: the handles of the record that stays and of the
    duplicate, whether the one that stays has the smaller uuid (the placeholder rule of T10),
    whether both have the same PDF, and whether their objectives texts have the same SHA-256
    (``None`` when either has none)."""
    by_uuid = {record.uuid: record for record in records}
    fichas_by_uuid = {ficha.item_uuid: ficha for ficha in fichas}
    pairs = []
    for twin in fichas:
        if twin.duplicate_of is None:
            continue
        kept = fichas_by_uuid[twin.duplicate_of]
        digests = [_digest(ficha.objectives) for ficha in (kept, twin)]
        pairs.append(
            {
                "kept": _handle(kept, by_uuid),
                "duplicate": _handle(twin, by_uuid),
                "kept_is_smaller_uuid": str(kept.item_uuid) < str(twin.item_uuid),
                "same_pdf": kept.pdf_sha256 is not None and kept.pdf_sha256 == twin.pdf_sha256,
                "objectives_equal": None if None in digests else digests[0] == digests[1],
            }
        )
    return pairs


def stopword_recheck(
    texts: Sequence[str],
    domain_stopwords: frozenset[str],
    lemmatize: Lemmatizer,
    *,
    min_ratio: float = MIN_DOCUMENT_RATIO,
) -> dict[str, Any]:
    """Count, over ``texts``, the documents that hold each domain stopword, and list the
    lemmas of at least ``min_ratio`` of them that the list lacks, the most common first."""
    tokens = lemmatize(texts)
    frequencies = Counter(lemma for document in tokens for lemma in set(document))
    unlisted = [
        candidate
        for candidate in stopword_candidates(tokens, min_ratio=min_ratio)
        if candidate.lemma not in domain_stopwords
    ]
    return {
        "documents": len(texts),
        "min_document_ratio": min_ratio,
        "domain_stopwords": len(domain_stopwords),
        "domain_stopwords_absent": sum(frequencies[lemma] == 0 for lemma in domain_stopwords),
        "domain_stopword_documents": {
            lemma: frequencies[lemma]
            for lemma in sorted(domain_stopwords, key=lambda lemma: (-frequencies[lemma], lemma))
        },
        "frequent_unlisted_lemmas": {c.lemma: c.documents for c in unlisted},
        "verb_shaped_candidates": [c.lemma for c in unlisted if _INFINITIVE.match(c.lemma)],
    }


def report_problems(
    contents: Iterable[str], report: Mapping[str, Any], records: Sequence[SnapshotRecord]
) -> list[str]:
    """Return why the report files may not be published, or nothing when they may."""
    problems = []
    sensitive = sensitive_strings(records)
    leaks = {leak for content in contents for leak in leaked_strings(content, sensitive)}
    if leaks:
        problems.append(f"the report holds {len(leaks)} string(s) taken from the snapshot")
    recheck = report.get("stopword_recheck", {})
    lemmas = [
        *recheck.get("domain_stopword_documents", {}),
        *recheck.get("frequent_unlisted_lemmas", {}),
        *recheck.get("verb_shaped_candidates", []),
    ]
    malformed = [
        lemma
        for lemma in lemmas
        if not lemma or len(lemma) > MAX_LEMMA_LENGTH or any(char.isspace() for char in lemma)
    ]
    if malformed:
        problems.append(f"{len(malformed)} lemma(s) of the report are not a single word")
    return problems


def write_quality_report(
    directory: Path, report: Mapping[str, Any], records: Sequence[SnapshotRecord]
) -> tuple[Path, Path]:
    """Write the report as JSON and Markdown into ``directory``, atomically, and return both
    paths, once :func:`report_problems` finds nothing.

    Raises:
        ValueError: if the report may not be published; nothing is written then.
    """
    content = json.dumps(report, ensure_ascii=False, indent=2)
    markdown = render_markdown(report)
    problems = report_problems((content, markdown), report, records)
    if problems:
        raise ValueError(f"{'; '.join(problems)}; nothing was written")
    directory.mkdir(parents=True, exist_ok=True)
    paths = (directory / REPORT_JSON_FILE_NAME, directory / REPORT_MARKDOWN_FILE_NAME)
    # The atomic writer of the JSON manifests takes any text.
    write_json_atomically(paths[0], content)
    write_json_atomically(paths[1], markdown.rstrip("\n"))
    return paths


def render_markdown(report: Mapping[str, Any]) -> str:
    """Return a short Markdown summary of the report."""
    counts = report["counts"]
    reasons = counts["exclusion_reason"]
    statuses = counts["objectives_status"]
    orcid = report["advisor_orcid"]
    completeness = report["completeness"]
    recheck = report["stopword_recheck"]
    listed = recheck["domain_stopword_documents"]
    unlisted = recheck["frequent_unlisted_lemmas"]
    metadata_only = counts["section_source"].get("title_abstract", 0)
    future = ", ".join(f"`{handle}`" for handle in report["future_dated"]) or "none"
    present = {field: n for field, n in completeness.items() if field != "included"}
    lines = [
        f"# Fichas quality report of snapshot {report['provenance'].get('snapshot_id')}",
        "",
        "Counts, public handles and lemmas only. The full figures are in `quality_report.json`;",
        "every count except the first four covers the included theses.",
        "",
        "## At a glance",
        "",
        "| Measure | Value |",
        "|---|---:|",
        f"| Records, one ficha each | {counts['records']} |",
        f"| Included theses | {counts['include']['included']} |",
        *(f"| Excluded as `{reason}` | {count} |" for reason, count in reasons.items()),
        f"| Objectives extracted | {statuses.get('extracted', 0)} |",
        f"| Metadata only (title and abstract) | {metadata_only} |",
        "",
        "## Included theses",
        "",
        "| Field | Theses |",
        "|---|---|",
        f"| Program | {_inline(counts['included_by_program'])} |",
        f"| Issue year | {_inline(counts['included_by_year'])} |",
        *(f"| {label} | {_inline(counts[field])} |" for field, label in COUNTED_FIELDS.items()),
        f"| Quality notes | {_inline(report['quality_notes']) or 'none'} |",
        "",
        "## People and dates",
        "",
        f"- Advisor ORCID iD on {orcid['theses_with_orcid']} theses: {orcid['valid']} valid, "
        f"{orcid['invalid']} malformed or failing the check digit.",
        f"- Advisor codes from the ORCID iD {orcid['advisor_code_from_orcid']}, from the name "
        f"{orcid['advisor_code_from_name']}, none {orcid['without_advisor']}; "
        f"{orcid['distinct_advisor_codes']} distinct advisor codes and "
        f"{orcid['distinct_author_codes']} distinct author codes.",
        f"- Future-dated: {future}.",
        f"- Field completeness: {_inline(present)}.",
        "",
        "| Text | Theses | Median characters | P95 characters |",
        "|---|---:|---:|---:|",
        *(
            f"| {field} | {spread['documents']} | {spread['median']} | {spread['p95']} |"
            for field, spread in report["text_lengths"].items()
        ),
        "",
        "## D24 duplicates",
        "",
        "| Stays | Duplicate | Stays is the smaller uuid | Same PDF | Equal objectives |",
        "|---|---|---|---|---|",
        *(
            f"| `{pair['kept']}` | `{pair['duplicate']}` | {_yes(pair['kept_is_smaller_uuid'])} | "
            f"{_yes(pair['same_pdf'])} | {_yes(pair['objectives_equal'])} |"
            for pair in report["duplicates"]
        ),
        "",
        "## Domain stopwords in the objectives",
        "",
        f"Over the {recheck['documents']} included theses with objectives, by the "
        f"{recheck['lemmatizer']}; a lemma counts once per thesis.",
        "",
        f"- {recheck['domain_stopwords']} domain stopwords; {recheck['domain_stopwords_absent']} "
        "of them occur in no objectives.",
        f"- Most frequent listed: {_inline(dict(list(listed.items())[:15]))}.",
        f"- {len(unlisted)} lemmas not on the list reach "
        f"{recheck['min_document_ratio']:.0%} of the theses. The 25 most frequent: "
        f"{_inline(dict(list(unlisted.items())[:25]))}.",
        f"- Shaped like an infinitive, candidates only for the user's review (the list is "
        f"unchanged): {', '.join(recheck['verb_shaped_candidates']) or 'none'}.",
    ]
    return "\n".join(lines) + "\n"


def _advisor_orcid(
    included: Sequence[Ficha], by_uuid: Mapping[UUID, SnapshotRecord]
) -> dict[str, int]:
    with_orcid = valid = with_name = 0
    for ficha in included:
        metadata = by_uuid[ficha.item_uuid].metadata
        orcids = [entry.get("value") for entry in metadata.get(ADVISOR_ORCID_KEY, ())]
        orcid = next((value for value in orcids if isinstance(value, str) and value.strip()), None)
        with_orcid += orcid is not None
        is_valid = orcid is not None and normalize_orcid(orcid) is not None
        valid += is_valid
        with_name += not is_valid and bool(metadata.get(ADVISOR_KEY))
    return {
        "theses_with_orcid": with_orcid,
        "valid": valid,
        "invalid": with_orcid - valid,
        "advisor_code_from_orcid": valid,
        "advisor_code_from_name": with_name,
        "without_advisor": sum(ficha.advisor_code is None for ficha in included),
        "distinct_advisor_codes": len({f.advisor_code for f in included if f.advisor_code}),
        "distinct_author_codes": len({code for f in included for code in f.author_codes}),
    }


def _lengths(texts: Iterable[str | None]) -> dict[str, Any]:
    """Return the count, median and nearest-rank 95th percentile of the text lengths."""
    lengths = sorted(len(text) for text in texts if text)
    if not lengths:
        return {"documents": 0, "median": None, "p95": None}
    return {
        "documents": len(lengths),
        "median": float(statistics.median(lengths)),
        "p95": lengths[math.ceil(0.95 * len(lengths)) - 1],
    }


def _handle(ficha: Ficha, by_uuid: Mapping[UUID, SnapshotRecord]) -> str:
    record = by_uuid.get(ficha.item_uuid)
    return record.handle if record is not None and record.handle else ficha.handle_url


def _digest(text: str | None) -> str | None:
    return hashlib.sha256(text.encode()).hexdigest() if text is not None else None


def _sorted(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted(counts.items()))


def _ranked(counts: Counter[str]) -> dict[str, int]:
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _inline(counts: Mapping[str, Any]) -> str:
    return ", ".join(f"{key} {value}" for key, value in counts.items())


def _yes(value: bool | None) -> str:
    return "n/a" if value is None else "yes" if value else "no"
