"""Title suffixes: the place and the year that many thesis titles end with.

In snapshot ``20261002T224412Z``, 397 of the 746 theses (53.2%) end their title with a place,
a year or both, such as ``, Arequipa 2023`` (``results/eda/20261002T224412Z/summary.md``).
Decision D10 strips them, so that titles group by topic and not by place or year.

This module is the one definition of those suffixes: the gazetteer of places, the patterns,
:func:`classify_title_suffix`, which the metadata profile counts them with, and
:func:`strip_title_suffix`, which cuts them. It needs no language model.
"""

import re
from typing import NamedTuple

from thematic_redundancy.preprocessing.light_cleaner import normalize_text

TITLE_SUFFIX_PATTERNS = (
    "place_and_year",
    "year_after_separator",
    "year_after_temporal_word",
    "year_after_other_word",
    "place_after_separator",
    "place_after_word",
)
"""How a title may end with a place, a year or both; see :func:`classify_title_suffix`."""
NO_SUFFIX = "none"
YEAR_SUFFIX_PATTERNS = frozenset(TITLE_SUFFIX_PATTERNS[:4])
PLACE_SUFFIX_PATTERNS = frozenset({"place_and_year", "place_after_separator", "place_after_word"})

PLACE_NAMES = (
    "Perú",
    # The 24 departments and the Constitutional Province of Callao, plus the spelling Cuzco.
    "Amazonas",
    "Áncash",
    "Apurímac",
    "Arequipa",
    "Ayacucho",
    "Cajamarca",
    "Callao",
    "Cusco",
    "Cuzco",
    "Huancavelica",
    "Huánuco",
    "Ica",
    "Junín",
    "La Libertad",
    "Lambayeque",
    "Lima",
    "Loreto",
    "Madre de Dios",
    "Moquegua",
    "Pasco",
    "Piura",
    "Puno",
    "San Martín",
    "Tacna",
    "Tumbes",
    "Ucayali",
    # The other provinces of the Arequipa region.
    "Camaná",
    "Caravelí",
    "Castilla",
    "Caylloma",
    "Condesuyos",
    "Islay",
    "La Unión",
    # Districts of the province of Arequipa.
    "Alto Selva Alegre",
    "Cayma",
    "Cerro Colorado",
    "Characato",
    "La Joya",
    "Mariano Melgar",
    "Miraflores",
    "Paucarpata",
    "Sabandía",
    "Sachaca",
    "Socabaya",
    "Tiabaya",
    "Uchumayo",
    "Yanahuara",
    "Yura",
    # Other towns of southern Peru.
    "Chivay",
    "Espinar",
    "Ilo",
    "Juliaca",
    "Majes",
    "Matarani",
    "Mollendo",
)
"""Gazetteer of the place words that a title suffix may hold. A name matches as written or in
capitals, with or without the accents of its vowels; ñ always has to match."""

TEMPORAL_WORDS = frozenset(
    {
        "al",
        "año",
        "años",
        "periodo",
        "periodos",
        "período",
        "períodos",
        "ejercicio",
        "desde",
        "hasta",
        "durante",
    }
)
"""Words that introduce a trailing year, as in ``al 2021`` or ``periodo 2022``."""

_VOWEL_SPELLINGS = {
    "a": "aá",
    "á": "aá",
    "e": "eé",
    "é": "eé",
    "i": "ií",
    "í": "ií",
    "o": "oó",
    "ó": "oó",
    "u": "uúü",
    "ú": "uúü",
    "ü": "uúü",
}


def _spelling_pattern(spelling: str) -> str:
    """Return a pattern for ``spelling`` in which each vowel may carry an accent or not."""
    parts = []
    for char in spelling:
        vowels = _VOWEL_SPELLINGS.get(char.lower())
        if vowels is not None:
            parts.append(f"[{vowels.upper() if char.isupper() else vowels}]")
        elif char.isspace():
            parts.append(r"\s+")
        else:
            parts.append(re.escape(char))
    return "".join(parts)


def _place_pattern(name: str) -> str:
    return f"{_spelling_pattern(name)}|{_spelling_pattern(name.upper())}"


_PLACES_BY_LENGTH = sorted(PLACE_NAMES, key=len, reverse=True)
_ANY_PLACE = "(?:" + "|".join(_place_pattern(name) for name in _PLACES_BY_LENGTH) + ")"
_ONE_PLACE = re.compile(
    "|".join(
        f"(?P<p{index}>{_place_pattern(name)})" for index, name in enumerate(_PLACES_BY_LENGTH)
    )
)
"""One place name; the number in the name of the group that matched indexes the gazetteer."""

_DASHES = r"\-\N{HYPHEN}\N{EN DASH}\N{EM DASH}"
_SEPARATOR = rf"\s*[,({_DASHES}]\s*"
"""A comma, an opening parenthesis, a hyphen or a dash, with any spaces around it."""
_GAP = rf"(?:{_SEPARATOR}|\s+)"
_PLACES = rf"{_ANY_PLACE}(?:{_GAP}{_ANY_PLACE})*"
_YEAR = rf"(?:19|20)[0-9]{{2}}(?:\s*[{_DASHES}/]\s*(?:(?:19|20)[0-9]{{2}}|[0-9]{{2}}))?"
"""A year from 1900 to 2099, or a range such as ``2020-2021`` or ``2021-22``."""
_END = r"[\s.)]*$"

TRAILING_YEAR = re.compile(
    rf"(?:(?<!\w)(?P<places>{_PLACES}))?"
    rf"(?<![0-9])(?P<separator>{_SEPARATOR}|\s+)"
    rf"(?P<year>{_YEAR}){_END}"
)
"""A year at the end of a title, maybe closed by a period or a parenthesis.

The year follows a separator (see :data:`_SEPARATOR`) or a space, but never a digit, so
``ISO 9001:2015`` and ``ISO 9001-2015`` do not count. One or more places may come right
before it, joined by separators or spaces: ``, Arequipa 2023``, ``- Arequipa, Perú 2021``,
``en Arequipa – 2024``."""

TRAILING_PLACE = re.compile(rf"(?:(?P<separator>{_SEPARATOR})|\s+|^)(?P<places>{_PLACES}){_END}")
"""One or more places at the end of a title, after a separator or a word."""

_WORD_PUNCTUATION = ".,;:()\"'«»“”"
"""Punctuation stripped from a word before it is compared with a word list."""


class TitleSuffix(NamedTuple):
    """Pattern of a title's ending, and the places in it, spelled as :data:`PLACE_NAMES`."""

    pattern: str
    places: tuple[str, ...]


class _SuffixMatch(NamedTuple):
    """A suffix found at the end of a text: its kind, where it starts, and what it holds."""

    suffix: TitleSuffix
    start: int
    has_year: bool
    has_place: bool


def classify_title_suffix(title: str) -> TitleSuffix:
    """Tell how a title ends: with places and a year, a year, places, or neither.

    The patterns, tried in this order, are:

    - ``place_and_year``: places right before a trailing year, as in ``, Arequipa 2023``;
    - ``year_after_separator``: a year after a comma, a dash or a parenthesis, as in ``, 2025``;
    - ``year_after_temporal_word``: a year after one of :data:`TEMPORAL_WORDS`;
    - ``year_after_other_word``: a year after any other word;
    - ``place_after_separator``: trailing places after a separator, as in ``, Moquegua``;
    - ``place_after_word``: trailing places after a word, as in ``de Arequipa``;
    - ``none``.

    A place is a name of :data:`PLACE_NAMES`, so a place word missing from it is not seen,
    and a place at the end of an organization's name counts as a place.
    """
    found = _find_suffix(normalize_text(title))
    return TitleSuffix(NO_SUFFIX, ()) if found is None else found.suffix


def _find_suffix(text: str) -> _SuffixMatch | None:
    """Return the suffix at the end of ``text``, already normalized, or ``None``."""
    match = TRAILING_YEAR.search(text)
    if match is not None:
        if match["places"]:
            suffix = TitleSuffix("place_and_year", _place_names(match["places"]))
            return _SuffixMatch(suffix, match.start(), has_year=True, has_place=True)
        if match["separator"].strip():
            pattern = "year_after_separator"
        else:
            previous = text[: match.start()].split()
            temporal = bool(previous) and _word_key(previous[-1]) in TEMPORAL_WORDS
            pattern = "year_after_temporal_word" if temporal else "year_after_other_word"
        return _SuffixMatch(TitleSuffix(pattern, ()), match.start(), has_year=True, has_place=False)
    match = TRAILING_PLACE.search(text)
    if match is not None:
        pattern = "place_after_separator" if match["separator"] else "place_after_word"
        suffix = TitleSuffix(pattern, _place_names(match["places"]))
        return _SuffixMatch(suffix, match.start(), has_year=False, has_place=True)
    return None


def _word_key(word: str) -> str:
    """Return ``word`` without surrounding punctuation and with its case folded."""
    return word.strip(_WORD_PUNCTUATION).casefold()


def _place_names(text: str) -> tuple[str, ...]:
    """Return the gazetteer names of the places in ``text``, in order and without repeats."""
    names = []
    for match in _ONE_PLACE.finditer(text):
        if match.lastgroup is not None:
            names.append(_PLACES_BY_LENGTH[int(match.lastgroup[1:])])
    return tuple(dict.fromkeys(names))


# Stripping

MIN_TITLE_WORDS = 3
"""Fewest words that a title may keep once its suffix is cut. The shortest title of the corpus
has 7 words, so only a title that is little more than its suffix keeps it."""

CONNECTOR_WORDS = frozenset(
    {
        "a",
        "al",
        "con",
        "de",
        "del",
        "e",
        "el",
        "en",
        "entre",
        "la",
        "las",
        "los",
        "para",
        "por",
        "y",
    }
)
"""Function words that a cut can leave hanging at the end of a title, as in ``... de``."""

PLACE_KIND_WORDS = frozenset(
    {"ciudad", "departamento", "distrito", "norte", "provincia", "region", "región", "sur"}
)
"""Words that name the kind of place before a place name, as in ``la ciudad de Arequipa`` or
``el sur del Perú``. They hang at the end of a title once the place is cut."""

# The same limit as QUALITY_NOTE_MAX_LENGTH in corpus/ficha.py, written again because corpus
# imports preprocessing and not the other way round. A unit test keeps the two equal.
NOTE_MAX_LENGTH = 200
"""Longest :attr:`StrippedTitle.quality_note`: the longest quality note that a ficha takes."""

_CUT_PUNCTUATION = " ,;:([-\N{HYPHEN}\N{EN DASH}\N{EM DASH}"
"""Separators and opening brackets that a cut can leave at the end of a title."""


class StrippedTitle(NamedTuple):
    """A title with its place and year suffix cut, and what was cut, for the quality notes."""

    title: str
    """The title without its suffix, or the whole title when none was cut. Its whitespace is
    collapsed, as :func:`normalize_text` does."""
    suffix: str
    """The suffix found at the end of the title, as written there, with the separators and
    connector words that it leaves hanging: ``, Arequipa 2023``, ``en la ciudad de Arequipa``.
    Empty when the title has no suffix."""
    pattern: str
    """How the title ends, as :func:`classify_title_suffix` tells: a pattern of
    :data:`TITLE_SUFFIX_PATTERNS`, or ``none``."""
    places: tuple[str, ...]
    """The places named in :attr:`suffix`, spelled as :data:`PLACE_NAMES`."""
    stripped: bool
    """Whether :attr:`suffix` was cut from :attr:`title`."""

    @property
    def refused(self) -> bool:
        """Whether a suffix was found but kept, because cutting it would leave too little."""
        return bool(self.suffix) and not self.stripped

    @property
    def quality_note(self) -> str | None:
        """A note on the suffix for the quality notes of a ficha, or ``None`` without one.

        It quotes the suffix, shortened with ``…`` when the note would exceed
        :data:`NOTE_MAX_LENGTH` characters.
        """
        if not self.suffix:
            return None
        if self.stripped:
            prefix = f"title suffix removed ({self.pattern}): "
        else:
            prefix = (
                f"title suffix kept, as cutting it would leave fewer than {MIN_TITLE_WORDS} "
                f"words ({self.pattern}): "
            )
        room = NOTE_MAX_LENGTH - len(prefix) - 2
        quoted = self.suffix if len(self.suffix) <= room else f"{self.suffix[: room - 1]}…"
        return f'{prefix}"{quoted}"'


def strip_title_suffix(title: str) -> StrippedTitle:
    """Cut the trailing places and years from ``title``, and tell what was cut.

    The title is first normalized as :func:`normalize_text` does. A suffix is what
    :func:`classify_title_suffix` finds: places and a year, a year, or places, as in
    ``, Arequipa 2023``, ``Arequipa - 2025``, ``, 2026.``, ``Pasco 2026`` or
    ``en Arequipa, 2023``. Each cut also takes what it leaves hanging at the new end:
    separators, opening brackets and :data:`CONNECTOR_WORDS`; :data:`TEMPORAL_WORDS` after a
    year, as in ``en el año 2024``; and :data:`PLACE_KIND_WORDS` after a place, as in
    ``en la ciudad de Arequipa``. Cuts repeat while a suffix is left, so
    ``en Arequipa en el año 2023`` goes whole.

    A title is never emptied: a cut that would leave fewer than :data:`MIN_TITLE_WORDS` words
    is not made. When even the first cut is refused, the whole title comes back,
    :attr:`StrippedTitle.stripped` is false, and :attr:`StrippedTitle.suffix` names what was
    kept, so that the caller can report it.
    """
    text = normalize_text(title)
    first = _find_suffix(text)
    if first is None:
        return StrippedTitle(text, "", NO_SUFFIX, (), stripped=False)
    head = text
    places: list[str] = []
    found: _SuffixMatch | None = first
    while found is not None:
        shorter = _without_hanging_words(head[: found.start], found)
        if _word_count(shorter) < MIN_TITLE_WORDS:
            break
        head = shorter
        places[:0] = found.suffix.places
        found = _find_suffix(head)
    if head == text:
        kept = text[len(_without_hanging_words(text[: first.start], first)) :].strip()
        return StrippedTitle(text, kept, first.suffix.pattern, first.suffix.places, stripped=False)
    return StrippedTitle(
        head,
        text[len(head) :].strip(),
        first.suffix.pattern,
        tuple(dict.fromkeys(places)),
        stripped=True,
    )


def _without_hanging_words(head: str, cut: _SuffixMatch) -> str:
    """Drop the separators and words that ``cut`` leaves hanging at the end of ``head``."""
    while True:
        head = head.rstrip(_CUT_PUNCTUATION)
        words = head.split()
        if not words:
            return ""
        key = _word_key(words[-1])
        if not (
            key in CONNECTOR_WORDS
            or (cut.has_year and key in TEMPORAL_WORDS)
            or (cut.has_place and key in PLACE_KIND_WORDS)
        ):
            return head
        head = head[: -len(words[-1])]


def _word_count(text: str) -> int:
    """Count the words of ``text`` that hold a letter or a digit."""
    return sum(1 for word in text.split() if any(char.isalnum() for char in word))
