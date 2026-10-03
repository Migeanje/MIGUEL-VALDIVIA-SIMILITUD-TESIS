"""Light cleaning: the text that the embedding models read.

:func:`clean_light` repairs the defects that copying a text out of a PDF leaves, and changes
nothing else. Casing, accents, ñ and punctuation stay as written, so the meaning that an
embedding model reads is kept. It needs no language model.

The abstracts of the corpus are hard-wrapped: 716 of 746 hold line breaks, and 85.7% of those
breaks fall inside a sentence (``results/eda/20261002T224412Z/summary.md``). So a line break
becomes a space, unless it ends a sentence or a paragraph.

:func:`normalize_text` is the flat variant: every run of whitespace, line breaks included,
becomes one space. The metadata profile and the title suffix patterns compare texts with it.
"""

import re
import unicodedata

SOFT_HYPHEN = "\N{SOFT HYPHEN}"
"""Invisible mark of a place where a word may be split at a line end."""

HYPHEN = "-"

_CHARACTER_MAP = {
    # Curly quotes become straight ones. Guillemets and primes keep their own roles.
    **dict.fromkeys(map(ord, "\u201c\u201d\u201e\u201f"), '"'),
    **dict.fromkeys(map(ord, "\u2018\u2019\u201a\u201b"), "'"),
    # Hyphen look-alikes become the ASCII hyphen. The en and em dashes stay: they mark ranges
    # and asides, not words joined by a hyphen.
    **dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2212\ufe63\uff0d"), HYPHEN),
    # Invisible characters without a meaning of their own are dropped.
    **dict.fromkeys(map(ord, "\u200b\u2060\ufeff"), None),
    # Control characters that are neither whitespace nor line breaks become spaces.
    **{code: " " for code in (*range(0x20), 0x7F, *range(0x80, 0xA0)) if not chr(code).isspace()},
}
"""Characters that :func:`clean_light` replaces or drops, applied after NFC."""

_SENTENCE_END = re.compile(r"[.!?…][\"'»)\]]*$")
"""Sentence punctuation at the end of a line, maybe closed by a quote or a bracket."""

_OPENERS = "\"'«([¿¡"
_LIST_MARKERS = ("-", "\u2013", "\u2014", "\u2022", "\u00b7", "\u25aa", "*")
"""Hyphen, en dash, em dash, bullet, middle dot, small square and asterisk."""


def clean_light(text: str) -> str:
    """Return ``text`` ready for an embedding model, with its meaning, casing and accents kept.

    The text is put in NFC, which composes accents and never removes them. Then:

    - **Line breaks.** Every line separator counts as a break, ``\\r\\n`` and ``\\r``
      included. A blank line separates paragraphs, which come out separated by one blank
      line. Inside a paragraph, a break stays, as a sentence boundary, only after ``.``,
      ``!``, ``?`` or ``…`` (maybe closed by a quote or a bracket) when an upper-case letter
      (maybe after an opening quote, bracket, ``¿`` or ``¡``) or a list marker starts the next
      line. Every other break becomes a space. So ``S.A.C.`` followed by ``que`` and ``S/.``
      followed by an amount stay in their sentence.
    - **Hyphenation.** A word split across lines with a hyphen is joined again:
      ``investi-`` and then ``gación`` give ``investigación``. The hyphen is kept, and the
      lines joined without a space, when the next line starts with an upper-case letter or a
      digit, or when a digit comes before the hyphen, as in ``Mecánica-Eléctrica``,
      ``COVID-19`` and ``2020-2021``. A hyphen after a space is a dash and joins nothing. A
      soft hyphen at a line end always joins the word without a mark; elsewhere it is dropped.
      A compound of lower-case words split right at its hyphen, such as ``costo-beneficio``,
      comes out as one word; the abstracts of the corpus hold none.
    - **Whitespace.** Every run of spaces, tabs or other blanks, no-break spaces included,
      becomes one space, and lines lose their leading and trailing blanks.
    - **Characters.** Curly quotes become straight ones, and the hyphen look-alikes (hyphen,
      non-breaking hyphen, figure dash, minus sign) the ASCII hyphen. Guillemets, primes and
      the en and em dashes stay. Zero-width spaces, word joiners and byte-order marks are
      dropped, and other control characters become spaces.

    The result is stable: cleaning it again gives it back unchanged.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\N{PARAGRAPH SEPARATOR}", "\n\n").translate(_CHARACTER_MAP)
    paragraphs: list[list[str]] = [[]]
    for raw_line in text.splitlines():
        line = _clean_line(raw_line)
        if line:
            paragraphs[-1].append(line)
        elif paragraphs[-1]:
            paragraphs.append([])
    return "\n\n".join(_join_lines(lines) for lines in paragraphs if lines)


def normalize_text(text: str) -> str:
    """Return ``text`` in NFC, with every run of whitespace, line breaks included, as one space.

    Accents and ñ are kept: NFC composes them, it never removes them.
    """
    return " ".join(unicodedata.normalize("NFC", text).split())


def _clean_line(line: str) -> str:
    """Return ``line`` with its whitespace collapsed and its soft hyphens dropped.

    A soft hyphen at the end of the line stays, so that :func:`_join_lines` can join the word.
    """
    collapsed = " ".join(line.split())
    words = collapsed.replace(SOFT_HYPHEN, "")
    if not words.strip():
        return ""
    return words.strip() + (SOFT_HYPHEN if collapsed.endswith(SOFT_HYPHEN) else "")


def _join_lines(lines: list[str]) -> str:
    """Join the lines of one paragraph, as :func:`clean_light` describes.

    Each decision reads only the end of the line before and the start of the next one, so
    a long paragraph takes time in proportion to its length.
    """
    pieces = [lines[0]]
    for line in lines[1:]:
        previous = pieces[-1]
        if previous.endswith(SOFT_HYPHEN) or _splits_word(previous, line):
            pieces[-1] = previous[:-1]
            pieces.append(line)
        elif _splits_compound(previous, line):
            pieces.append(line)
        elif _SENTENCE_END.search(previous) and _starts_sentence(line):
            pieces.append(f"\n{line}")
        else:
            pieces.append(f" {line}")
    return "".join(pieces).removesuffix(SOFT_HYPHEN)


def _splits_word(before: str, after: str) -> bool:
    """Tell whether a hyphen at the end of ``before`` splits a word that ``after`` ends."""
    return before.endswith(HYPHEN) and before[-2:-1].isalpha() and after[:1].islower()


def _splits_compound(before: str, after: str) -> bool:
    """Tell whether a hyphen at the end of ``before`` joins two parts of a compound or a code.

    It does when a letter or a digit comes on each side of it; :func:`_splits_word` has
    already taken the split words, a letter before and a lower-case letter after.
    """
    return before.endswith(HYPHEN) and before[-2:-1].isalnum() and after[:1].isalnum()


def _starts_sentence(line: str) -> bool:
    """Tell whether ``line`` opens a sentence: an upper-case letter or a list marker."""
    if line.startswith(_LIST_MARKERS):
        return True
    rest = line.lstrip(_OPENERS)
    return rest[:1].isupper()
