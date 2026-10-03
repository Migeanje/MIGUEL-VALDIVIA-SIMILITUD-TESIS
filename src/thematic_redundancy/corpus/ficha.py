"""Per-thesis record ("ficha"): the fields of Anexo B of the thesis plan, extended.

One :class:`Ficha` describes one item of a snapshot, whether the corpus includes it or not.
An excluded item keeps its ficha with the reason, so the fichas double as the exclusion log.
The dataset builder makes them from the snapshot and the PDF manifest. Text extraction and
the objectives locator later fill in the fields that start as ``pending``.

A ficha holds no person's name. Advisors and authors appear only as the keyed codes of
:mod:`thematic_redundancy.corpus.anonymize`, and the schema refuses any other value there.

Beyond the shape of each field, a ficha checks that its fields agree:

- ``doc_code`` is the code that :func:`doc_code_for` derives from ``item_uuid``;
- an excluded ficha has an ``exclusion_reason``, and an included one has none;
- ``duplicate_of`` is set exactly when the reason is ``duplicate``, and names another item;
- ``objectives`` is set exactly when ``objectives_status`` is ``extracted`` or ``manual``;
- the reason is ``non_thesis_type`` exactly when ``document_type`` is not ``tesis``, because
  the corpus holds theses only;
- ``pdf_sha256`` is set exactly when the PDF is on disk, as in the PDF manifest.
"""

import hashlib
from datetime import date
from typing import Annotated, Literal, Self
from uuid import UUID

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    model_validator,
)

from thematic_redundancy.corpus.anonymize import CODE_HEX_DIGITS
from thematic_redundancy.corpus.pdf_manifest import FILE_STATUSES, PdfStatus
from thematic_redundancy.corpus.snapshot import SnapshotId

DocumentType = Literal["tesis", "trabajo_suficiencia", "trabajo_academico", "other"]
"""Kind of work, from its ``renati.type``. Only ``tesis``, the exact ``#tesis`` fragment,
enters the corpus; ``other`` covers any other type, or none."""

ObjectivesStatus = Literal["pending", "extracted", "not_found", "no_pdf", "manual"]
"""Where the objectives come from, or why there are none:

- ``pending``: not looked for yet;
- ``extracted``: found in the PDF text by the objectives locator;
- ``not_found``: the PDF text holds no objectives that the locator recognizes;
- ``no_pdf``: there is no PDF text to look in;
- ``manual``: entered or corrected by hand.
"""

AccessRights = Literal["open", "embargoed", "restricted", "unknown"]
"""Access to the files, from the COAR access right in ``dc.rights``. ``restricted`` also
covers metadata-only access, and ``unknown`` a missing or unrecognized value."""

FichaPdfStatus = Literal[PdfStatus, "not_attempted"]
"""Status of the thesis PDF in the PDF manifest, or ``not_attempted`` for an item that the
downloader never looked at, such as one that is not a thesis."""

SourceFormat = Literal["pending", "digital", "ocr", "mixed", "none"]
"""How the PDF text was obtained: from its text layer, by OCR, or both; ``none`` when there
is no text, and ``pending`` before extraction."""

SectionSource = Literal[
    "pending", "title_abstract_objectives", "title_abstract", "title_objectives_problem"
]
"""Sections that represent the thesis downstream, or ``pending`` before they are chosen."""

ExclusionReason = Literal[
    "non_thesis_type", "duplicate", "no_abstract_no_objectives", "restricted_no_text", "other"
]
"""Why an item stays out of the corpus:

- ``non_thesis_type``: its ``renati.type`` is not exactly ``tesis``;
- ``duplicate``: it repeats the record named by ``duplicate_of``;
- ``no_abstract_no_objectives``: it has neither an abstract nor objectives;
- ``restricted_no_text``: its files are closed, so there is no text to use;
- ``other``: any other reason, which ``quality_notes`` should explain.
"""

THESIS: DocumentType = "tesis"
OBJECTIVES_TEXT_STATUSES: frozenset[ObjectivesStatus] = frozenset({"extracted", "manual"})
"""Objectives statuses that come with objectives text; the others come without."""

DOC_CODE_PREFIX = "DOC-"
DOC_CODE_HEX_DIGITS = 8
ADVISOR_CODE_PREFIX = "ADV-"
AUTHOR_CODE_PREFIX = "AUT-"
"""Prefixes to pass to :func:`thematic_redundancy.corpus.anonymize.pseudonym`."""

MAX_AUTHOR_CODES = 5
"""Most authors that a ficha lists; no thesis of the first snapshot has more than 2."""

MIN_ISSUE_YEAR = 1900
MAX_ISSUE_YEAR = 2100
QUALITY_NOTE_MAX_LENGTH = 200


def doc_code_for(item_uuid: UUID) -> str:
    """Return the short code of an item for reports, such as ``DOC-11e594f4``.

    It is ``DOC-`` and the first 8 hex digits of the SHA-256 of the uuid's canonical text
    (lower case, with hyphens), so anyone can derive it again. The uuid is public, so the
    code hides nothing; it is only shorter. With 32 bits, ~1,000 items almost never share
    a code, but whoever builds a set of fichas should still check that the codes are unique.
    """
    digest = hashlib.sha256(str(item_uuid).encode("ascii")).hexdigest()
    return f"{DOC_CODE_PREFIX}{digest[:DOC_CODE_HEX_DIGITS]}"


def _require_non_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be blank")
    return value


def _require_distinct(codes: tuple[str, ...]) -> tuple[str, ...]:
    repeated = sorted({code for code in codes if codes.count(code) > 1})
    if repeated:
        raise ValueError(f"must not repeat a code; repeated: {', '.join(repeated)}")
    return codes


def _code_pattern(prefix: str) -> str:
    return rf"^{prefix}[0-9a-f]{{{CODE_HEX_DIGITS}}}$"


NonBlankStr = Annotated[str, AfterValidator(_require_non_blank)]
WebUrl = Annotated[str, Field(pattern=r"^https?://[^\s/]+(?:/\S*)?$")]
ProgramKey = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]*$")]
IssueYear = Annotated[StrictInt, Field(ge=MIN_ISSUE_YEAR, le=MAX_ISSUE_YEAR)]
OcdeCode = Annotated[str, Field(pattern=r"^[0-9]+(?:\.[0-9]+)*$")]
LanguageCode = Annotated[str, Field(pattern=r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{1,8})*$")]
Sha256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
AdvisorCode = Annotated[str, Field(pattern=_code_pattern(ADVISOR_CODE_PREFIX))]
AuthorCode = Annotated[str, Field(pattern=_code_pattern(AUTHOR_CODE_PREFIX))]
AuthorCodes = Annotated[
    tuple[AuthorCode, ...],
    Field(max_length=MAX_AUTHOR_CODES),
    AfterValidator(_require_distinct),
]
QualityNote = Annotated[NonBlankStr, Field(max_length=QUALITY_NOTE_MAX_LENGTH)]


class Ficha(BaseModel):
    """One item of a snapshot, as the corpus sees it: identity, texts, metadata, files and
    whether it is included.

    Lists are tuples, so a ficha is immutable throughout. Texts are kept as harvested or
    extracted; cleaning them is a later step.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    # Identity
    item_uuid: UUID
    """Repository item: the primary key."""
    doc_code: str
    """Short code for reports, derived from ``item_uuid`` by :func:`doc_code_for`."""
    handle_url: WebUrl
    """Public page of the item, preferably its handle URL; every displayed thesis links here."""
    snapshot_id: SnapshotId

    # Type
    document_type: DocumentType
    renati_type_raw: NonBlankStr | None
    """``renati.type`` as harvested, such as ``https://purl.org/pe-repo/renati/type#tesis``,
    or ``None`` when the item has none."""

    # Texts
    title: NonBlankStr
    abstract: NonBlankStr | None = None
    objectives: NonBlankStr | None = None
    """Objectives of the thesis; ``objectives_status`` tells where they come from."""
    objectives_status: ObjectivesStatus = "pending"

    # Program
    program_key: ProgramKey
    program_name: NonBlankStr

    # Dates
    issue_date_raw: NonBlankStr
    """First ``dc.date.issued`` as harvested: a year, a month, a day or a timestamp."""
    issue_year: IssueYear | None = None
    """Year of ``issue_date_raw``, or ``None`` when that is not a date."""
    issued_after_snapshot: StrictBool
    """Whether the issue date lies after the snapshot date; a year or a month counts from
    its first day."""

    # Topic metadata
    keywords: tuple[NonBlankStr, ...] = ()
    ocde_codes: tuple[OcdeCode, ...] = ()
    """OCDE fields of research (FORD codes), such as ``2.02.04``, without the vocabulary URL."""
    language: LanguageCode | None = None
    """``dc.language.iso``, such as ``spa``."""

    # Rights
    rights: AccessRights
    embargo_end: date | None = None
    """``dc.date.embargoEnd``; an item that is open again may still carry one."""

    # People: keyed codes, never names
    advisor_code: AdvisorCode | None = None
    author_codes: AuthorCodes = ()
    """One code per author, without repeats, in the repository's order of authors."""

    # PDF
    pdf_status: FichaPdfStatus = "not_attempted"
    pdf_sha256: Sha256 | None = None
    """SHA-256 of the thesis PDF on disk."""

    # Extraction
    source_format: SourceFormat = "pending"
    section_source: SectionSource = "pending"

    # Inclusion
    include: StrictBool
    exclusion_reason: ExclusionReason | None = None
    duplicate_of: UUID | None = None
    """Item that this record repeats, which stays in the corpus in its place."""

    # Quality
    quality_notes: tuple[QualityNote, ...] = ()
    """Short notes about defects found in the record, such as hard-wrapped abstracts."""

    @model_validator(mode="after")
    def _require_consistent_fields(self) -> Self:
        problems = _consistency_problems(self)
        if problems:
            raise ValueError("; ".join(problems))
        return self


def _consistency_problems(ficha: Ficha) -> list[str]:
    """Return every rule between fields that ``ficha`` breaks, in a readable form."""
    problems = []
    expected_code = doc_code_for(ficha.item_uuid)
    if ficha.doc_code != expected_code:
        problems.append(f"doc_code must be {expected_code!r}, the code derived from item_uuid")

    if not ficha.include and ficha.exclusion_reason is None:
        problems.append("an excluded ficha needs an exclusion_reason")
    if ficha.include and ficha.exclusion_reason is not None:
        problems.append("an included ficha must not have an exclusion_reason")

    is_duplicate = ficha.exclusion_reason == "duplicate"
    if ficha.duplicate_of is not None and not is_duplicate:
        problems.append("duplicate_of needs exclusion_reason 'duplicate'")
    if is_duplicate and ficha.duplicate_of is None:
        problems.append("exclusion_reason 'duplicate' needs duplicate_of")
    if ficha.duplicate_of == ficha.item_uuid:
        problems.append("duplicate_of must name another item than item_uuid")

    with_text = ficha.objectives_status in OBJECTIVES_TEXT_STATUSES
    if ficha.objectives is not None and not with_text:
        problems.append("objectives text needs objectives_status 'extracted' or 'manual'")
    if ficha.objectives is None and with_text:
        problems.append(f"objectives_status {ficha.objectives_status!r} needs objectives text")

    is_non_thesis = ficha.exclusion_reason == "non_thesis_type"
    if ficha.document_type != THESIS and not is_non_thesis:
        problems.append(
            f"document_type {ficha.document_type!r} is not a thesis, so the ficha must be "
            "excluded as 'non_thesis_type'"
        )
    if ficha.document_type == THESIS and is_non_thesis:
        problems.append("exclusion_reason 'non_thesis_type' contradicts document_type 'tesis'")

    on_disk = ficha.pdf_status in FILE_STATUSES
    if on_disk and ficha.pdf_sha256 is None:
        problems.append(f"pdf_status {ficha.pdf_status!r} needs pdf_sha256")
    if not on_disk and ficha.pdf_sha256 is not None:
        problems.append("pdf_sha256 needs pdf_status 'downloaded' or 'already_present'")
    return problems
