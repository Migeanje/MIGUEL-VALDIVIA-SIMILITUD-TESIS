"""Locate the general and the specific objectives of a thesis in its page texts (T10).

Pure rules, with no I/O: :func:`locate_objectives` takes the page texts of one PDF, in page
order, and returns a :class:`LocatedObjectives`. Lines are matched in lower case with their
accents folded, but the objectives come back in their original text, with their whitespace
collapsed, their control characters dropped, and private-use bullet glyphs shown as ``•``.

The text extraction stores one block per PDF line, with a blank line between blocks, so a
paragraph cannot be told from its blocks. The rules therefore work line by line:

1. **Headings.** A line is a heading when, after optional numbering (``1.3.1.``, ``a)``,
   ``II.``) or a bullet, it holds only the heading words and an optional ``:`` or ``.``, or
   the heading words, then ``:``, ``.`` or a dash, then the objective itself:

   - general: ``objetivo general``, ``objetivos generales`` or ``objetivo principal``,
     optionally followed by ``de la investigación`` and the like;
   - specific: ``objetivo(s) específico(s)``;
   - block: ``objetivos``, ``objetivos de la investigación``, ``objetivos del estudio``,
     ``objetivos de la tesis`` or ``objetivos del proyecto``.

   A line that names the general (or the specific) objectives and ends in ``:``, such as
   ``El objetivo general es el siguiente:``, also works as a heading, and so does a bare
   ``General`` line right below a block heading. OCR confusions of ``o``/``0`` and
   ``i``/``l``/``1`` are tolerated in the heading words.
2. **Pages to skip.** A heading line with dotted leaders or a trailing page number is a
   table-of-contents line. Contents pages (a contents title, or many such lines), dedication
   and acknowledgement pages are skipped. Abstract pages (a ``Resumen`` or ``Abstract`` title,
   or a ``Palabras clave:`` line) are used only when the body has no objectives, and the
   result is then flagged ``abstract_only``.
3. **The general objective** is the first general heading in the body, then the first block
   heading, then the first general heading of an abstract. Its text runs from the heading to
   the first line that ends a sentence, or to the next stop, whichever comes first, and it
   may continue across a page break. Leading lines that introduce it and end in ``:`` are
   left out. A heading whose text holds fewer than three words, such as a page number, gives
   way to the next one.
4. **Stops** are another objectives heading; a line of multi-level numbering (``1.4``), or one
   followed by a capital (``1.4. Descripción``); a roman-numeral or ``CAPÍTULO`` heading; and a
   short line that starts, in capitals, with a section keyword such as ``Justificación``,
   ``Hipótesis``, ``Alcance``, ``Delimitación``, ``Limitaciones``, ``Variables``,
   ``Metodología``, ``Marco teórico`` or ``Antecedentes`` (:data:`SECTION_KEYWORDS`).
5. **The specific objectives** start at the first specific heading within
   :data:`SPECIFIC_SEARCH_LINES` lines after the general objective, before any other stop, and
   run to the next stop. Each list item (a bullet, ``1.``, ``a)``, ``OE1``) or sentence starts
   a new line of the result; wrapped lines are joined.
6. **Bounds.** Texts longer than their cap are cut and flagged; a general objective under
   ``min_chars`` characters is flagged as too short.

Raise :data:`LOCATOR_VERSION` whenever a change of these rules alters the results, so the
objectives table records which rules made it.
"""

import re
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, Self, get_args
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator

from thematic_redundancy.corpus.snapshot import Count
from thematic_redundancy.extraction.text_manifest import Sha256
from thematic_redundancy.shared.config import ObjectivesConfig

LOCATOR_VERSION = 1
"""Version of the locator rules, recorded with the objectives table:

- 1: the first rules (T10).
"""

LocatorStatus = Literal["extracted", "not_found"]
"""Outcome of the locator, a subset of the ficha's ``ObjectivesStatus``."""

ObjectivesFlag = Literal[
    "general_too_short",
    "general_too_long",
    "specific_too_long",
    "specific_missing",
    "abstract_only",
    "block_fallback",
    "ocr_page",
]
"""Quality flags of an extraction:

- ``general_too_short``: the general objective has fewer than ``min_chars`` characters;
- ``general_too_long`` / ``specific_too_long``: the text was cut at its cap;
- ``specific_missing``: no specific objectives were found after the general one;
- ``abstract_only``: the objectives come from an abstract page, as the body has none;
- ``block_fallback``: no general heading exists, so the general objective is the first
  sentence of an objectives block;
- ``ocr_page``: a page of the objectives was read by OCR.
"""

OBJECTIVES_FLAGS: tuple[ObjectivesFlag, ...] = get_args(ObjectivesFlag)
"""Every flag, in the order in which a result lists them."""

HeadingPattern = Literal[
    "objetivo_general",
    "objetivo_principal",
    "objetivo_general_intro",
    "general_subheading",
    "objetivos_block",
]
"""Heading that the general objective was found under:

- ``objetivo_general``: ``Objetivo general`` or ``Objetivos generales``;
- ``objetivo_principal``: ``Objetivo principal``;
- ``objetivo_general_intro``: a line naming the general objective and ending in ``:``;
- ``general_subheading``: a bare ``General`` line under an objectives block;
- ``objetivos_block``: the first sentence of an objectives block (``block_fallback``).
"""

SPECIFIC_SEARCH_LINES = 30
"""Lines after the general objective in which a specific heading is looked for."""

SUBHEADING_LINES = 4
"""Lines below an objectives block heading in which a bare ``General`` counts as a heading."""

MIN_WORDS = 3
"""Words that a general objective needs; fewer means the heading was a listing."""

KEYWORD_LINE_MAX_CHARS = 70
"""Longest line that may count as a section-keyword heading."""

SECTION_KEYWORDS = (
    "justificacion",
    "importancia",
    "hipotesis",
    "alcance",
    "alcances",
    "delimitacion",
    "delimitaciones",
    "limitacion",
    "limitaciones",
    "viabilidad",
    "factibilidad",
    "variables",
    "operacionalizacion",
    "metodologia",
    "marco",
    "antecedentes",
    "planteamiento del problema",
    "formulacion del problema",
    "descripcion del problema",
    "preguntas de investigacion",
    "estado del arte",
    "matriz de consistencia",
    "tipo de investigacion",
    "nivel de investigacion",
    "diseno de la investigacion",
    "diseno metodologico",
    "poblacion",
    "muestra",
    "capitulo",
    "introduccion",
    "resultados",
    "conclusiones",
    "recomendaciones",
    "referencias",
    "bibliografia",
    "anexos",
    "glosario",
    "definicion de terminos",
    "palabras clave",
    "keywords",
    "resumen",
    "abstract",
)
"""Section names, accent-folded and in lower case, whose heading stops an objective."""

_ABSTRACT_TITLES = frozenset({"resumen", "abstract", "summary", "resumen ejecutivo"})
_SKIPPED_TITLES = frozenset(
    {"dedicatoria", "dedicatorias", "agradecimiento", "agradecimientos", "epigrafe"}
)
_CONTENTS_TITLE = re.compile(
    r"(?:indice|indice general|contenido|contenidos|tabla de contenidos?|indice de contenidos?"
    r"|sumario)\b"
)
_TITLE_LINES = 4
"""Lines at the top of a page in which its title is looked for."""
_CONTENTS_MIN_LINES = 5
_CONTENTS_MIN_SHARE = 0.4

# Matching runs on lines in lower case with their accents folded; see _fold.
_OBJ = r"[o0]bjet(?:[il1]v|[il1]c|v)[o0]"
"""``objetivo``, with OCR confusions and the typos ``objetico`` and ``objetvo``."""
_SPECIFIC_WORD = r"(?:espec[il1]f[il1]c[o0]s?|secundari[o0]s?)"
_QUALIFIER = r"(?:\s+de\s+(?:la\s+|el\s+)?(?:investigacion|tesis|estudio|proyecto|trabajo))?"
_BLOCK_QUALIFIER = (
    r"(?:\s+de\s+(?:la\s+)?investigacion|\s+del\s+estudio|\s+de\s+la\s+tesis|\s+del\s+proyecto)"
)
_REMARK = r"(?:\s*\([^)]*\))?"
"""A remark in parentheses after a heading, such as a template's ``(Solo va el título)``."""
_LEAD = (
    r"[^\w]*(?:(?:\d{1,2}\s*\.\s*)*\d{1,2}\s*[.)\-]?"
    r"|[ivxl]{1,6}\s*[.)\-]|[a-z]\s*[.)])?[^\w]*"
)
"""Numbering and bullets that may precede a heading."""
_TAIL = r"\s*(?:(?P<sep>[:.\-–—])\s*(?P<rest>.*))?$"
"""Optional separator and the objective that may follow a heading on the same line."""

_GENERAL = re.compile(
    rf"{_LEAD}(?:(?P<general>{_OBJ}s?\s+genera[l1](?:es)?)"
    rf"|(?P<principal>{_OBJ}s?\s+principa[l1](?:es)?)){_QUALIFIER}{_REMARK}{_TAIL}"
)
_SPECIFIC = re.compile(rf"{_LEAD}{_OBJ}s?\s+{_SPECIFIC_WORD}{_QUALIFIER}{_REMARK}{_TAIL}")
_BLOCK = re.compile(
    rf"{_LEAD}(?:{_OBJ}s{_BLOCK_QUALIFIER}?|{_OBJ}{_BLOCK_QUALIFIER}){_REMARK}\s*[:.]?$"
)
"""``Objetivos``, alone or with a qualifier, or ``Objetivo`` with a qualifier only, since a
bare ``Objetivo`` heads many procedures in annexes."""
_BARE_GENERAL = re.compile(rf"{_LEAD}genera[l1](?:es)?\s*[:.]?$")
_BARE_SPECIFIC = re.compile(rf"{_LEAD}{_SPECIFIC_WORD}\s*[:.]?$")
_GENERAL_WORDS = re.compile(rf"{_OBJ}s?\s+(?:genera[l1]|principa[l1])")
_SPECIFIC_WORDS = re.compile(rf"{_OBJ}s?\s+(?:espec[il1]f[il1]c|secundari)")
_INTRO_MAX_CHARS = 160

_PAGE_NUMBER = r"(?:\d{1,3}|(?=[ivx])x{0,3}(?:ix|iv|v?i{0,3}))"
"""A page number: arabic with at most 3 digits, or a valid roman numeral below 40."""
_DOTTED_LEADER = re.compile(r"\.{4,}|(?:\.\s){3,}|…{2,}|_{4,}")
_TRAILING_PAGE = re.compile(rf"[^\W\d_)]\W*\s+{_PAGE_NUMBER}$", re.IGNORECASE)
_BARE_PAGE = re.compile(_PAGE_NUMBER, re.IGNORECASE)
_PAGE_ONLY_REST = re.compile(rf"[\d.\s]*|{_PAGE_NUMBER}", re.IGNORECASE)
_TITLE_NOISE = re.compile(r"[^a-z0-9 ]+")
_KEYWORDS_LINE = re.compile(r"[^\w]*(?:palabras\s+claves?|keywords|key\s+words)\s*:")

_NUMBERING_ONLY = re.compile(r"\d{1,2}(?:\s*\.\s*\d{1,2})+\s*\.?")
_NUMBERING = re.compile(r"\d{1,2}(?:\s*\.\s*\d{1,2})*")
_ITEM_NUMBER_MIN_LEVELS = 3
"""Levels that a numbered line needs, such as ``1.3.2.1``, to count as an item numbered
below its specific heading instead of as the next section."""
_NUMBERED_HEADING = re.compile(r"\d{1,2}(?:\s*\.\s*\d{1,2})+\s*\.?\s*[A-ZÁÉÍÓÚÑ]")
_ROMAN_HEADING = re.compile(r"[IVXL]{1,6}\s*[.)\-–]\s*[A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ ,]{3,}$")
"""A roman-numeral heading in capitals, such as ``II. MARCO TEÓRICO``. One in mixed case
stops only through its keyword, so that list items numbered ``I.`` do not."""
_CHAPTER = re.compile(r"cap[il1]tulo\b")
_KEYWORD = re.compile(rf"{_LEAD}(?P<keyword>{'|'.join(SECTION_KEYWORDS)})\b")

_SENTENCE_END = re.compile(r"[.!?][\"'”’»)\]]*$")
_ITEM_END = re.compile(r"[.;:!?][\"'”’»)\]]*$")
_BULLETS = "•·▪►➢➤✓✔○◦●■□*\\-–—−"
_ITEM_START = re.compile(
    rf"(?:[{_BULLETS}]|\d{{1,2}}\s*[.)]\s|\d{{1,2}}(?:\.\d{{1,2}}){{2,}}\.?\s|[a-z]\s*[.)]\s"
    r"|[ivx]{1,4}\s*[.)]\s|o\.?\s?e\.?\s?\d)",
    re.IGNORECASE,
)
_LEADING_BULLETS = re.compile(rf"^(?:[{_BULLETS}]\s*)+")
_CONTROL = re.compile(r"[\x00-\x08\x0e-\x1b\x7f-\x9f]")
_WORD = re.compile(r"[^\W\d_]{2,}")
_SPLIT_ACCENT = re.compile(r"([´¨˜])\s?([A-Za-zı])")
"""An accent printed before its letter, as some LaTeX PDFs give ``Espec´ıficos``."""
_COMBINING = {"´": "́", "¨": "̈", "˜": "̃"}

_FOLD_TABLE = {
    code: base
    for code in range(0xC0, 0x250)
    if (decomposed := unicodedata.normalize("NFKD", chr(code)))
    and (base := decomposed[0]).isascii()
    and len(decomposed) > 1
} | {ord("ı"): "i"}


class PageLike(Protocol):
    """The fields of a page record that the locator reads, as in ``PageText``."""

    @property
    def page_index(self) -> int: ...

    @property
    def text(self) -> str: ...

    @property
    def source(self) -> str: ...


@dataclass(frozen=True)
class LocatedObjectives:
    """What the locator found in one document.

    ``general`` is set exactly when the status is ``extracted``; ``page_start`` is the page
    of the heading and ``page_end`` the page of the last line used, both as page indices.
    """

    status: LocatorStatus
    general: str | None = None
    specific: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    pattern: HeadingPattern | None = None
    flags: tuple[ObjectivesFlag, ...] = ()


class ObjectivesRecord(BaseModel):
    """One row of the objectives table: the locator's result for one thesis PDF.

    It holds the objectives' text, so the table lives only under the gitignored ``data/``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    item_uuid: UUID
    handle: str | None
    program: str
    status: LocatorStatus
    objective_general: str | None
    objectives_specific: str | None
    page_start: Count | None
    page_end: Count | None
    pattern: HeadingPattern | None
    flags: tuple[ObjectivesFlag, ...]
    general_chars: Count
    specific_chars: Count
    pages: Count
    """Pages of the PDF."""
    pages_sha256: Sha256
    """SHA-256 of the page records file that the locator read."""

    @model_validator(mode="after")
    def _require_consistent_fields(self) -> Self:
        located = (self.objective_general, self.page_start, self.page_end, self.pattern)
        if self.status == "extracted":
            if any(value is None for value in located):
                raise ValueError(
                    "an 'extracted' record needs objective_general, page_start, page_end and "
                    "pattern"
                )
        elif any(value is not None for value in located) or self.objectives_specific is not None:
            raise ValueError("a 'not_found' record holds no objectives, pages or pattern")
        elif self.flags:
            raise ValueError("a 'not_found' record has no flags")
        if self.general_chars != len(self.objective_general or ""):
            raise ValueError("general_chars must be the length of objective_general")
        if self.specific_chars != len(self.objectives_specific or ""):
            raise ValueError("specific_chars must be the length of objectives_specific")
        if (
            self.page_start is not None
            and self.page_end is not None
            and self.page_end < self.page_start
        ):
            raise ValueError("page_end must not come before page_start")
        if len(set(self.flags)) != len(self.flags):
            raise ValueError("a flag may appear only once")
        return self


# The rules


@dataclass(frozen=True)
class _Position:
    page: int
    """Position of the page in the document, counted from 0."""
    line: int
    """Position of the line among the page's non-empty lines."""


@dataclass(frozen=True)
class _Heading:
    position: _Position
    kind: Literal["general", "block"]
    pattern: HeadingPattern
    rest: str
    """Text that follows the heading on its own line, in its original form."""


@dataclass(frozen=True)
class _Span:
    text: str
    last: _Position
    truncated: bool


class _Document:
    """The page texts of one document, split into non-empty, normalized lines on first use."""

    def __init__(self, pages: Sequence[PageLike]) -> None:
        self.pages = pages
        self._lines: dict[int, list[str]] = {}
        self._kinds: dict[int, str] = {}

    def lines(self, page: int) -> list[str]:
        if page not in self._lines:
            self._lines[page] = [
                _normalize(stripped)
                for line in self.pages[page].text.split("\n")
                if (stripped := line.strip())
            ]
        return self._lines[page]

    def after(self, position: _Position) -> Iterator[tuple[_Position, str]]:
        """Yield the lines that follow ``position``, across pages."""
        page, line = position.page, position.line + 1
        while page < len(self.pages):
            lines = self.lines(page)
            while line < len(lines):
                yield _Position(page, line), lines[line]
                line += 1
            page, line = page + 1, 0

    def kind(self, page: int) -> str:
        """Return ``contents``, ``abstract``, ``skipped`` or ``body`` for one page."""
        if page not in self._kinds:
            self._kinds[page] = _page_kind(self.lines(page))
        return self._kinds[page]


def locate_objectives(pages: Sequence[PageLike], settings: ObjectivesConfig) -> LocatedObjectives:
    """Find the general and the specific objectives in the page texts of one document.

    ``pages`` are the page records of one PDF in page order; see the module docstring for
    the rules.
    """
    document = _Document(pages)
    headings = list(_headings(document))
    order = (
        [h for h in headings if h.kind == "general" and document.kind(h.position.page) == "body"]
        + [h for h in headings if h.kind == "block" and document.kind(h.position.page) == "body"]
        + [
            h
            for h in headings
            if h.kind == "general" and document.kind(h.position.page) == "abstract"
        ]
    )
    for heading in order:
        general = _general_text(document, heading, settings.general_max_chars)
        if general is None:
            continue
        specific = _specific_text(document, general.last, settings.specific_max_chars)
        last = specific.last if specific is not None else general.last
        flags: set[ObjectivesFlag] = set()
        if len(general.text) < settings.min_chars:
            flags.add("general_too_short")
        if general.truncated:
            flags.add("general_too_long")
        if specific is None:
            flags.add("specific_missing")
        elif specific.truncated:
            flags.add("specific_too_long")
        if document.kind(heading.position.page) == "abstract":
            flags.add("abstract_only")
        if heading.kind == "block":
            flags.add("block_fallback")
        used = range(heading.position.page, last.page + 1)
        if any(pages[page].source == "ocr" for page in used):
            flags.add("ocr_page")
        return LocatedObjectives(
            status="extracted",
            general=general.text,
            specific=specific.text if specific is not None else None,
            page_start=pages[heading.position.page].page_index,
            page_end=pages[last.page].page_index,
            pattern=heading.pattern,
            flags=tuple(flag for flag in OBJECTIVES_FLAGS if flag in flags),
        )
    return LocatedObjectives(status="not_found")


def _fold(text: str) -> str:
    """Return ``text`` in lower case with its accents folded, keeping one character per
    character, so that positions in it are positions in ``text``. ``text`` is a line that
    :func:`_normalize` gave, already in NFC."""
    folded = text.translate(_FOLD_TABLE).lower()
    if len(folded) != len(text):
        folded = "".join(
            lowered if len(lowered := char.translate(_FOLD_TABLE).lower()) == 1 else char
            for char in text
        )
    return folded


def _headings(document: _Document) -> Iterator[_Heading]:
    """Yield the general and block headings of ``document`` in order, contents lines left out."""
    for page in range(len(document.pages)):
        if "bjet" not in document.pages[page].text.lower():
            continue
        for number, line in enumerate(document.lines(page)):
            folded = _fold(line)
            if "bjet" not in folded or _DOTTED_LEADER.search(folded):
                continue
            position = _Position(page, number)
            if (match := _GENERAL.fullmatch(folded)) is not None:
                rest = _rest(line, match)
                if rest is not None:
                    pattern = "objetivo_general" if match["general"] else "objetivo_principal"
                    yield _Heading(position, "general", pattern, rest)
            elif _is_intro(folded, _GENERAL_WORDS) and not _SPECIFIC_WORDS.search(folded):
                yield _Heading(position, "general", "objetivo_general_intro", "")
            elif _BLOCK.fullmatch(folded):
                yield _Heading(position, "block", "objetivos_block", "")
                yield from _subheadings(document, position)
            elif _SPECIFIC.fullmatch(folded):
                # Some theses head the whole block "Objetivos específicos" by mistake.
                yield from _subheadings(document, position)


def _subheadings(document: _Document, block: _Position) -> Iterator[_Heading]:
    """Yield a bare ``General`` line just below the objectives heading at ``block``."""
    for count, (position, line) in enumerate(document.after(block)):
        if count >= SUBHEADING_LINES:
            return
        if _BARE_GENERAL.fullmatch(_fold(line)):
            yield _Heading(position, "general", "general_subheading", "")
            return


def _rest(line: str, match: re.Match[str]) -> str | None:
    """Return the objective that follows a heading on its line, ``""`` when none does, or
    ``None`` when the line is a contents line that ends in a page number."""
    if match["sep"] is None:
        return ""
    rest = line[match.start("rest") :].strip()
    if rest and _PAGE_ONLY_REST.fullmatch(rest):
        return None
    return rest


def _is_intro(folded: str, words: re.Pattern[str]) -> bool:
    """Tell whether a line names some objectives and ends in a colon, as an introduction."""
    return (
        len(folded) <= _INTRO_MAX_CHARS
        and folded.rstrip().endswith(":")
        and bool(words.search(folded))
    )


def _page_kind(lines: Sequence[str]) -> str:
    """Return ``contents``, ``skipped``, ``abstract`` or ``body`` for a page's lines."""
    titles = [" ".join(_TITLE_NOISE.sub(" ", _fold(line)).split()) for line in lines[:_TITLE_LINES]]
    if any(_CONTENTS_TITLE.fullmatch(title) for title in titles):
        return "contents"
    if any(title in _SKIPPED_TITLES for title in titles):
        return "skipped"
    contents_lines = sum(_is_contents_line(line) for line in lines)
    if contents_lines >= _CONTENTS_MIN_LINES and contents_lines >= _CONTENTS_MIN_SHARE * len(lines):
        return "contents"
    if any(title in _ABSTRACT_TITLES for title in titles) or any(
        _KEYWORDS_LINE.match(_fold(line)) for line in lines
    ):
        return "abstract"
    return "body"


def _is_contents_line(line: str) -> bool:
    """Tell whether a line looks like a contents entry: dotted leaders, a trailing page
    number, or a page number alone."""
    return bool(
        _DOTTED_LEADER.search(line) or _TRAILING_PAGE.search(line) or _BARE_PAGE.fullmatch(line)
    )


def _is_stop(line: str, folded: str, *, below: tuple[int, ...] = ()) -> bool:
    """Tell whether ``line`` starts another section, which ends an objective.

    A line numbered at least :data:`_ITEM_NUMBER_MIN_LEVELS` deep under the number
    ``below`` of a specific heading, such as ``1.3.2.1`` under ``1.3.2``, is an item, so
    its number alone does not stop the list.
    """
    if (
        _GENERAL.fullmatch(folded)
        or _SPECIFIC.fullmatch(folded)
        or _BLOCK.fullmatch(folded)
        or _BARE_SPECIFIC.fullmatch(folded)
        or _is_intro(folded, _GENERAL_WORDS)
        or _is_intro(folded, _SPECIFIC_WORDS)
    ):
        return True
    numbered = _NUMBERING_ONLY.fullmatch(line) or _NUMBERED_HEADING.match(line)
    if (numbered and not _is_numbered_below(line, below)) or (
        _ROMAN_HEADING.match(line) or _CHAPTER.match(folded)
    ):
        return True
    if len(line) > KEYWORD_LINE_MAX_CHARS:
        return False
    keyword = _KEYWORD.match(folded)
    # Folding keeps one character per character, so the keyword starts at the same place in
    # the original line, where a capital tells a heading from a wrapped line of prose.
    return keyword is not None and line[keyword.start("keyword")].isupper()


def _general_text(document: _Document, heading: _Heading, cap: int) -> _Span | None:
    """Return the general objective below ``heading``, or ``None`` when it has no words."""
    parts = [heading.rest] if heading.rest else []
    last = heading.position
    truncated = False
    if not (parts and _SENTENCE_END.search(heading.rest)):
        for position, line in document.after(heading.position):
            folded = _fold(line)
            if not parts and _is_intro(folded, _GENERAL_WORDS):
                continue
            if _is_stop(line, folded):
                break
            parts.append(line)
            last = position
            if _SENTENCE_END.search(line):
                break
            if sum(len(part) for part in parts) > cap:
                break
    text = _LEADING_BULLETS.sub("", _clean(" ".join(parts)))
    if len(_WORD.findall(text)) < MIN_WORDS:
        return None
    if len(text) > cap:
        text, truncated = text[:cap].rstrip(), True
    return _Span(text, last, truncated)


def _specific_text(document: _Document, after: _Position, cap: int) -> _Span | None:
    """Return the specific objectives that follow the general objective ending at ``after``."""
    pending: tuple[int, ...] = ()
    for count, (position, line) in enumerate(document.after(after)):
        if count >= SPECIFIC_SEARCH_LINES:
            return None
        folded = _fold(line)
        rest: str | None = None
        if (match := _SPECIFIC.fullmatch(folded)) is not None:
            rest = _rest(line, match)
        elif _BARE_SPECIFIC.fullmatch(folded) or _is_intro(folded, _SPECIFIC_WORDS):
            rest = ""
        if rest is not None:
            return _items(document, position, rest, cap, _numbering(line) or pending)
        if _NUMBERING_ONLY.fullmatch(line):
            pending = _numbering(line)
            continue  # The number of the next heading, on a line of its own.
        if _is_stop(line, folded):
            return None
        pending = ()
    return None


def _items(
    document: _Document, heading: _Position, rest: str, cap: int, number: tuple[int, ...]
) -> _Span | None:
    """Return the list below a specific heading numbered ``number``, one item or sentence
    per line. Leading lines that introduce the list and end in ``:`` are left out."""
    items = [_clean(rest)] if rest else []
    last = heading
    for position, line in document.after(heading):
        folded = _fold(line)
        if not items and (_is_intro(folded, _SPECIFIC_WORDS) or _is_intro(folded, _GENERAL_WORDS)):
            continue
        if _is_stop(line, folded, below=number):
            break
        text = _clean(line)
        if not text:
            continue
        if not items or _ITEM_START.match(text) or _ITEM_END.search(items[-1]):
            items.append(text)
        else:
            items[-1] = f"{items[-1]} {text}"
        last = position
        if sum(len(item) + 1 for item in items) > cap:
            break
    joined = "\n".join(items)
    if not _WORD.search(joined):
        return None
    if len(joined) > cap:
        return _Span(joined[:cap].rstrip(), last, True)
    return _Span(joined, last, False)


def _numbering(line: str) -> tuple[int, ...]:
    """Return the numbers that a line starts with, such as ``(1, 3, 2)`` for ``1.3.2.``."""
    match = _NUMBERING.match(line)
    if match is None:
        return ()
    return tuple(int(part) for part in match.group().replace(" ", "").split("."))


def _is_numbered_below(line: str, number: tuple[int, ...]) -> bool:
    found = _numbering(line)
    return (
        bool(number)
        and len(found) >= _ITEM_NUMBER_MIN_LEVELS
        and len(found) > len(number)
        and found[: len(number)] == number
    )


def _normalize(line: str) -> str:
    """Return ``line`` in NFC, with accents printed before their letter put back on it."""
    line = unicodedata.normalize("NFC", line)
    if not _SPLIT_ACCENT.search(line):
        return line
    return _SPLIT_ACCENT.sub(_join_accent, line)


def _join_accent(match: re.Match[str]) -> str:
    letter = "i" if match[2] == "ı" else match[2]
    joined = unicodedata.normalize("NFC", letter + _COMBINING[match[1]])
    return joined if len(joined) == 1 else match.group()


def _clean(text: str) -> str:
    """Return ``text`` with control characters dropped, private-use glyphs shown as ``•``,
    and whitespace collapsed."""
    text = _CONTROL.sub("", text)
    text = "".join("•" if unicodedata.category(char) == "Co" else char for char in text)
    return " ".join(text.split())
