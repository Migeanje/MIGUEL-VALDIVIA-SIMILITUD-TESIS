"""Rule-based sentence splitter of the chunker (D27, C1).

:func:`sentence_spans` returns the sentences of a section text as character spans, without the
whitespace around them. It needs no language model, and it runs on the light-cleaned text
(:func:`~thematic_redundancy.preprocessing.light_cleaner.clean_light`), which keeps a line break
only at the end of a sentence or a paragraph:

- **Line breaks.** Every line break ends a sentence, with or without punctuation.
- **Sentence ends.** Within a line, a sentence ends at ``.``, ``!``, ``?`` or ``…``, maybe closed
  by a quote or a bracket, when whitespace follows and then an upper-case letter (maybe after an
  opening quote, bracket, ``¿`` or ``¡``) or an enumerator such as ``2.`` or ``b)``.
- **Protected periods.** A period does not end a sentence after a known abbreviation
  (:data:`ABBREVIATIONS`), such as ``S.A.C.``, ``Dr.`` or ``etc.``; after ``no`` only when a
  number follows, as in ``No. 5``, since ``no`` is also a word that ends a sentence, as in
  ``o no.``; after an initial or a dotted abbreviation, such as ``J.`` or ``e.g.``; or after an
  enumerator that opens the sentence or follows ``:`` or ``;``, such as ``1.``, so that a list
  item keeps its marker. A decimal such as ``3.5`` never ends one, since no whitespace follows
  its period.

Two limits are kept on purpose: a lone letter before a period, as in ``la fase a.``, reads as an
initial, so the sentence goes on; and a period before a lower-case word never ends a sentence.
Either only makes a sentence longer, which the chunker cuts when it does not fit
(:mod:`~thematic_redundancy.representation.chunker`), so no text is lost.
"""

import re

ABBREVIATIONS = frozenset(
    {
        "aa",
        "al",
        "approx",
        "aprox",
        "art",
        "av",
        "cap",
        "cc",
        "cia",
        "cm",
        "cnel",
        "cía",
        "dept",
        "dpto",
        "dr",
        "dra",
        "e.g",
        "e.i.r.l",
        "ed",
        "eds",
        "ee.uu",
        "ej",
        "etc",
        "fig",
        "figs",
        "gral",
        "hr",
        "hrs",
        "i.e",
        "inc",
        "ing",
        "jr",
        "kg",
        "km",
        "lic",
        "mag",
        "max",
        "mg",
        "min",
        "mm",
        "msc",
        "máx",
        "mín",
        "nro",
        "num",
        "nº",
        "núm",
        "p.ej",
        "pag",
        "ph.d",
        "phd",
        "pp",
        "prof",
        "pág",
        "págs",
        "s",
        "s.a",
        "s.a.a",
        "s.a.c",
        "s.r.l",
        "seg",
        "sr",
        "sra",
        "srta",
        "sta",
        "sto",
        "tab",
        "tec",
        "u.s",
        "univ",
        "vol",
        "vs",
    }
)
"""Abbreviations whose period never ends a sentence: lower case, without their last period."""

_NUMBER_ABBREVIATIONS = frozenset({"no"})
"""Abbreviations whose period ends no sentence when a number follows, as in ``No. 5``: ``no``
is also the word that ends ``o no.``, so its period is protected only there."""

_NUMBER_AHEAD = re.compile(r"\s+\d")

_ENUMERATOR = r"(?:\d{1,3}|[a-zA-Z]|[ivxIVX]{1,4})"
"""A list marker before its ``.`` or ``)``: a number, a letter or a Roman numeral."""

_SENTENCE_END = re.compile(
    r"[.!?…][\"'»)\]]*(?=\s+(?:[\"'«(\[¿¡]*[A-ZÁÉÍÓÚÑÜ]|" + _ENUMERATOR + r"[.)]\s))"
)
_ENUMERATOR_ALONE = re.compile(r"\s*" + _ENUMERATOR)
_ENUMERATOR_AFTER_LIST = re.compile(r"[:;]\s*" + _ENUMERATOR + r"$")
_INITIALS = re.compile(r"(?:[a-záéíóúñ]\.)+[a-záéíóúñ]?")
_LINE = re.compile(r"[^\n]+")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return the sentences of ``text`` as ``(start, end)`` spans, in order.

    Each span holds no leading or trailing whitespace and is never empty, so the text between
    two spans, and before the first and after the last, is whitespace only.
    """
    spans: list[tuple[int, int]] = []
    for line in _LINE.finditer(text):
        cursor = line.start()
        for match in _SENTENCE_END.finditer(text, line.start(), line.end()):
            if text[match.start()] == "." and _protected(
                text[cursor : match.start()], text[match.end() : line.end()]
            ):
                continue
            spans.extend(trimmed_span(text, cursor, match.end()))
            cursor = match.end()
        spans.extend(trimmed_span(text, cursor, line.end()))
    return spans


def _protected(before: str, after: str) -> bool:
    """Tell whether the period between ``before``, the text since the last sentence end, and
    ``after``, the rest of the line, is part of an abbreviation, an initial or a list marker
    rather than a sentence end."""
    words = before.split()
    last = words[-1].lower() if words else ""
    return (
        last.rstrip(".") in ABBREVIATIONS
        or (last.rstrip(".") in _NUMBER_ABBREVIATIONS and _NUMBER_AHEAD.match(after) is not None)
        or _ENUMERATOR_ALONE.fullmatch(before) is not None
        or _ENUMERATOR_AFTER_LIST.search(before) is not None
        or _INITIALS.fullmatch(f"{last}.") is not None
    )


def trimmed_span(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Return ``[start, end)`` without its surrounding whitespace, or nothing when it is blank."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return [(start, end)] if start < end else []
