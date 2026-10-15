"""Sentence-aligned chunks of a section text that fit every tokenizer (D09, D27).

This is the pure core of the chunker: no model and no file. A chunk is a span of the section
text that the embedding models read whole, so it must hold at most ``max_tokens`` tokens,
special tokens included, under every tokenizer:

- **C5, token counters.** The tokenizers are reached through the :class:`TokenCounter` port.
  :class:`MaxTokenCounter` counts with a sequence of counters and keeps the largest count, so a
  chunk that fits it fits each of them. Tests inject fake counters; the Hugging Face adapter is
  :mod:`~thematic_redundancy.representation.hf_tokenizers`.
- **C1, units.** A section is cut into sentences
  (:func:`~thematic_redundancy.representation.sentences.sentence_spans`).
- **C2, over-long sentences.** A sentence that does not fit is cut after each ``;`` or ``:``
  followed by whitespace; a piece that still does not fit, after each ``,`` followed by
  whitespace; and a piece that still does not fit, into the longest word-aligned windows that
  fit, cut inside a word only when that word alone does not fit. Nothing overlaps and nothing
  is dropped, and every piece goes back to the packer as a unit of its own.
- **C7, greedy packing.** Units are joined in order while the joined span fits; a unit that
  would not fit starts the next chunk.
- **C3, the title.** A title is one unit, never split into sentences, so a title that fits is
  one chunk; one that does not falls back to the cuts of C2.

Each chunk records how it ends (:data:`CUT_KINDS`). The text between two chunks, and around
them, is whitespace only, so the chunks hold the whole section.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal, Protocol

from thematic_redundancy.representation.sentences import sentence_spans, trimmed_span

CutKind = Literal["sentence", "clause", "comma", "window"]

CUT_KINDS: tuple[CutKind, ...] = ("sentence", "clause", "comma", "window")
"""How a chunk ends: at the end of a sentence or of the section, or at a cut inside an
over-long sentence, after ``;`` or ``:`` (clause), after ``,`` (comma), or at a window."""

_CUTS: tuple[tuple[CutKind, re.Pattern[str]], ...] = (
    ("clause", re.compile(r"[;:](?=\s)")),
    ("comma", re.compile(r",(?=\s)")),
)
"""The cuts of an over-long sentence, tried in this order before the windows."""

_WORD = re.compile(r"\S+")


class TokenCounter(Protocol):
    """Port to a tokenizer: how many tokens a model reads for a text."""

    def count(self, text: str) -> int:
        """Return the number of tokens of ``text``, special tokens included, untruncated."""
        ...


@dataclass(frozen=True)
class MaxTokenCounter:
    """Counts with every one of ``counters`` and returns the largest count."""

    counters: Sequence[TokenCounter]

    def __post_init__(self) -> None:
        if not self.counters:
            raise ValueError("a MaxTokenCounter needs at least one counter")

    def count(self, text: str) -> int:
        return max(counter.count(text) for counter in self.counters)


@dataclass(frozen=True)
class Chunk:
    """A span ``[start, end)`` of a section text, without whitespace at either end."""

    start: int
    end: int
    n_tokens: int
    """Tokens of the span under the counter that made it, special tokens included."""
    cut_kind: CutKind


def chunk_section(text: str | None, counter: TokenCounter, max_tokens: int) -> list[Chunk]:
    """Return the chunks of the section ``text``, in order; none for a missing or blank one.

    Raises:
        ValueError: if ``max_tokens`` leaves no room for a single character under ``counter``.
    """
    if text is None or not text.strip():
        return []
    return pack_units(text, section_units(text, counter, max_tokens), counter, max_tokens)


def chunk_title(text: str | None, counter: TokenCounter, max_tokens: int) -> list[Chunk]:
    """Return the chunks of the title ``text``: one when it fits, as a title always has so far,
    and otherwise the pieces of an over-long sentence, packed.

    Raises:
        ValueError: if ``max_tokens`` leaves no room for a single character under ``counter``.
    """
    if text is None or not text.strip():
        return []
    units = section_units(text, counter, max_tokens, sentences=False)
    return pack_units(text, units, counter, max_tokens)


def section_units(
    text: str, counter: TokenCounter, max_tokens: int, *, sentences: bool = True
) -> list[Chunk]:
    """Return the units that :func:`pack_units` joins: the sentences of ``text``, or the whole
    text when ``sentences`` is false, with each one that does not fit cut as C2 says.

    Each unit fits ``max_tokens`` and records the kind of its end.

    Raises:
        ValueError: if ``max_tokens`` leaves no room for a single character under ``counter``.
    """
    spans = sentence_spans(text) if sentences else trimmed_span(text, 0, len(text))
    units: list[Chunk] = []
    for start, end in spans:
        units.extend(_fitting_pieces(text, start, end, "sentence", 0, counter, max_tokens))
    return units


def pack_units(
    text: str, units: Sequence[Chunk], counter: TokenCounter, max_tokens: int
) -> list[Chunk]:
    """Join ``units`` greedily, in order, while the joined span of ``text`` fits ``max_tokens``.

    A chunk takes the end kind of its last unit, and no chunk could take the next unit and
    still fit.

    Raises:
        ValueError: if a unit does not fit on its own.
    """
    chunks: list[Chunk] = []
    current: Chunk | None = None
    for unit in units:
        if unit.n_tokens > max_tokens:
            raise ValueError(f"a unit of {unit.n_tokens} tokens exceeds {max_tokens}")
        if current is not None:
            n_tokens = counter.count(text[current.start : unit.end])
            if n_tokens <= max_tokens:
                current = Chunk(current.start, unit.end, n_tokens, unit.cut_kind)
                continue
            chunks.append(current)
        current = unit
    if current is not None:
        chunks.append(current)
    return chunks


def _fitting_pieces(
    text: str,
    start: int,
    end: int,
    end_kind: CutKind,
    level: int,
    counter: TokenCounter,
    max_tokens: int,
) -> list[Chunk]:
    """Return ``[start, end)`` as one unit when it fits, or cut by the cuts from ``level`` on.

    The last piece keeps ``end_kind``; every other piece ends with the kind of its cut.
    """
    n_tokens = counter.count(text[start:end])
    if n_tokens <= max_tokens:
        return [Chunk(start, end, n_tokens, end_kind)]
    if level == len(_CUTS):
        return _windows(text, start, end, end_kind, counter, max_tokens)
    kind, pattern = _CUTS[level]
    bounds = [start, *(match.end() for match in pattern.finditer(text, start, end)), end]
    pieces = [piece for a, b in pairwise(bounds) for piece in trimmed_span(text, a, b)]
    units: list[Chunk] = []
    for index, (piece_start, piece_end) in enumerate(pieces):
        piece_kind = end_kind if index == len(pieces) - 1 else kind
        units.extend(
            _fitting_pieces(
                text, piece_start, piece_end, piece_kind, level + 1, counter, max_tokens
            )
        )
    return units


def _windows(
    text: str, start: int, end: int, end_kind: CutKind, counter: TokenCounter, max_tokens: int
) -> list[Chunk]:
    """Cut ``[start, end)`` into the longest windows that fit, each ending at a word end, or
    inside a word that does not fit on its own."""
    windows: list[Chunk] = []
    cursor = start
    while cursor < end:
        word_ends = [match.end() for match in _WORD.finditer(text, cursor, end)]
        found = _longest_fit(text, cursor, word_ends, counter, max_tokens)
        if found is None:
            found = _longest_fit(text, cursor, range(cursor + 1, word_ends[0]), counter, max_tokens)
        if found is None:
            raise ValueError(
                f"max_tokens ({max_tokens}) leaves no room for a single character under the "
                "token counter"
            )
        stop, n_tokens = found
        windows.append(Chunk(cursor, stop, n_tokens, "window"))
        cursor = stop
        while cursor < end and text[cursor].isspace():
            cursor += 1
    return [*windows[:-1], replace(windows[-1], cut_kind=end_kind)]


def _longest_fit(
    text: str, start: int, stops: Sequence[int], counter: TokenCounter, max_tokens: int
) -> tuple[int, int] | None:
    """Return the largest of the increasing ``stops`` whose span from ``start`` fits, with its
    count, by bisection; ``None`` when none that was tried fits.

    Bisection assumes that a longer span never has fewer tokens. Where that fails, the stop
    returned may not be the largest, but it was counted and it fits.
    """
    found = None
    low, high = 0, len(stops) - 1
    while low <= high:
        middle = (low + high) // 2
        n_tokens = counter.count(text[start : stops[middle]])
        if n_tokens <= max_tokens:
            found = (stops[middle], n_tokens)
            low = middle + 1
        else:
            high = middle - 1
    return found
