"""Sentence-aligned chunks under several token counters (D27: C2, C3, C5, C7, C8).

No model is loaded: two fake counters stand in for the tokenizers, and they disagree on
purpose. One counts words and the other characters, both with two special tokens, so a long
word is cheap for the first and dear for the second. Texts are synthetic, generated from a
seeded ``random.Random``, so every failure is reproducible from its seed.
"""

import random
import re
from dataclasses import dataclass
from itertools import pairwise

import pytest

from thematic_redundancy.representation.chunker import (
    Chunk,
    MaxTokenCounter,
    TokenCounter,
    chunk_section,
    chunk_title,
    pack_units,
    section_units,
)
from thematic_redundancy.representation.sentences import sentence_spans


@dataclass(frozen=True)
class WordCounter:
    """Two special tokens plus one per whitespace-separated word."""

    def count(self, text: str) -> int:
        return len(text.split()) + 2


@dataclass(frozen=True)
class CharCounter:
    """Two special tokens plus one per four characters."""

    def count(self, text: str) -> int:
        return len(text) // 4 + 2


COUNTERS: tuple[TokenCounter, ...] = (WordCounter(), CharCounter())
BOTH = MaxTokenCounter(COUNTERS)

WORDS = (
    "proceso",
    "costo",
    "planta",
    "mejora",
    "sistema",
    "datos",
    "control",
    "calidad",
    "modelo",
    "red",
    "análisis",
    "diseño",
    "eficiencia",
    "producción",
    "gestión",
    "año",
    "señal",
    "niño",
)


def random_sentence(rng: random.Random) -> str:
    """A sentence of a few words, or of many, with commas, clause marks and, rarely, a run of
    letters without spaces longer than any limit."""
    size = rng.choice((rng.randint(1, 8), rng.randint(9, 30), rng.randint(30, 80)))
    words = []
    for index in range(size):
        word = rng.choice(WORDS)
        word = word.capitalize() if index == 0 else word
        draw = rng.random()
        if draw < 0.02:
            word = "x" * rng.randint(30, 320)
        elif draw < 0.06:
            word += rng.choice((";", ":"))
        elif draw < 0.16:
            word += ","
        words.append(word)
    opener = rng.choice(("", "", "", "¿", "1. ", "b) "))
    spaces = rng.choice((" ", " ", "  ", "\t"))
    return opener + spaces.join(words) + rng.choice((".", ".", "?", "!", "...", ""))


def random_text(rng: random.Random) -> str:
    paragraphs = []
    for _ in range(rng.randint(1, 3)):
        lines = [
            " ".join(random_sentence(rng) for _ in range(rng.randint(1, 5)))
            for _ in range(rng.randint(1, 3))
        ]
        paragraphs.append("\n".join(lines))
    return rng.choice(("", " ", "\n")) + "\n\n".join(paragraphs) + rng.choice(("", " \n"))


def assert_valid_chunks(text: str, chunks: list[Chunk], max_tokens: int) -> None:
    """Every chunk fits each counter on its own, and the chunks lose no text."""
    assert chunks
    bounds = [0, *(bound for chunk in chunks for bound in (chunk.start, chunk.end)), len(text)]
    assert bounds == sorted(bounds)
    assert all(chunk.start < chunk.end for chunk in chunks)
    # Ordered spans with only whitespace around and between them hold the whole text.
    assert all(
        text[start:end].strip() == "" for start, end in zip(bounds[::2], bounds[1::2], strict=True)
    )
    for chunk in chunks:
        piece = text[chunk.start : chunk.end]
        assert piece == piece.strip()
        assert all(counter.count(piece) <= max_tokens for counter in COUNTERS)
        assert chunk.n_tokens == max(counter.count(piece) for counter in COUNTERS)


def assert_sentence_aligned(text: str, chunks: list[Chunk], max_tokens: int) -> None:
    """A chunk ends at a sentence end, or inside an over-long sentence at a cut of its kind."""
    assert chunks
    spans = sentence_spans(text)
    starts = {start for start, _ in spans}
    ends = {end for _, end in spans}
    for chunk in chunks:
        for bound, aligned in ((chunk.start, starts), (chunk.end, ends)):
            if bound in aligned:
                continue
            host = next(((start, end) for start, end in spans if start < bound < end), None)
            assert host is not None
            assert BOTH.count(text[host[0] : host[1]]) > max_tokens
        if chunk.cut_kind == "sentence":
            assert chunk.end in ends
        elif chunk.cut_kind == "clause":
            assert text[chunk.end - 1] in ";:"
        elif chunk.cut_kind == "comma":
            assert text[chunk.end - 1] == ","
        else:
            assert chunk.cut_kind == "window"
            if not text[chunk.end].isspace():
                # A cut inside a word only when the word alone does not fit.
                word = next(match for match in re.finditer(r"\S+", text) if match.end() > chunk.end)
                assert BOTH.count(word.group()) > max_tokens


SEEDS = range(60)


@pytest.mark.parametrize("seed", SEEDS)
def test_every_chunk_fits_both_counters_and_no_text_is_lost(seed: int) -> None:
    rng = random.Random(seed)
    text = random_text(rng)
    max_tokens = rng.randint(12, 40)

    chunks = chunk_section(text, BOTH, max_tokens)

    assert_valid_chunks(text, chunks, max_tokens)


@pytest.mark.parametrize("seed", SEEDS)
def test_chunks_follow_sentences_and_cut_only_inside_an_over_long_one(seed: int) -> None:
    rng = random.Random(seed)
    text = random_text(rng)
    max_tokens = rng.randint(12, 40)

    assert_sentence_aligned(text, chunk_section(text, BOTH, max_tokens), max_tokens)


@pytest.mark.parametrize("seed", SEEDS)
def test_greedy_packing_leaves_no_room_for_the_next_unit(seed: int) -> None:
    rng = random.Random(seed)
    text = random_text(rng)
    max_tokens = rng.randint(12, 40)

    units = section_units(text, BOTH, max_tokens)
    chunks = pack_units(text, units, BOTH, max_tokens)

    assert chunks == chunk_section(text, BOTH, max_tokens)
    assert len(units) >= len(chunks) >= 1
    assert all(unit.n_tokens <= max_tokens for unit in units)
    for chunk, following in pairwise(chunks):
        first = next(unit for unit in units if unit.start == following.start)
        assert BOTH.count(text[chunk.start : first.end]) > max_tokens


@pytest.mark.parametrize("seed", range(20))
def test_chunking_its_own_chunk_gives_that_chunk_back(seed: int) -> None:
    rng = random.Random(seed)
    text = random_text(rng)
    max_tokens = rng.randint(12, 40)

    chunks = chunk_section(text, BOTH, max_tokens)

    assert chunks
    for chunk in chunks:
        piece = text[chunk.start : chunk.end]
        assert chunk_section(piece, BOTH, max_tokens) == [
            Chunk(0, len(piece), chunk.n_tokens, "sentence")
        ]


def test_the_same_texts_give_the_same_chunks_in_any_order() -> None:
    rng = random.Random(20261012)
    texts = [random_text(rng) for _ in range(12)]

    forward = [chunk_section(text, BOTH, 20) for text in texts]
    backward = [chunk_section(text, BOTH, 20) for text in reversed(texts)][::-1]
    fresh = [chunk_section(text, MaxTokenCounter(COUNTERS[::-1]), 20) for text in texts]

    assert all(forward)
    assert forward == backward == fresh


# Over-long sentences: the cut order (C2)


def chunk_texts(text: str, counter: TokenCounter, max_tokens: int) -> list[tuple[str, str]]:
    return [
        (text[chunk.start : chunk.end], chunk.cut_kind)
        for chunk in chunk_section(text, counter, max_tokens)
    ]


def test_an_over_long_sentence_is_cut_after_a_semicolon_or_colon_first() -> None:
    # 8 words do not fit 6; each clause does, so its comma is never cut.
    text = "Uno dos tres cuatro; cinco seis, siete ocho."

    assert chunk_texts(text, WordCounter(), 8) == [
        ("Uno dos tres cuatro;", "clause"),
        ("cinco seis, siete ocho.", "sentence"),
    ]


def test_a_clause_that_does_not_fit_is_cut_after_its_commas_and_the_pieces_repack() -> None:
    text = "Uno dos; tres cuatro, cinco seis siete, ocho nueve diez."

    assert chunk_texts(text, WordCounter(), 8) == [
        ("Uno dos; tres cuatro,", "comma"),
        ("cinco seis siete, ocho nueve diez.", "sentence"),
    ]


def test_a_sentence_without_clause_marks_or_commas_is_cut_into_word_windows() -> None:
    text = "Uno dos tres cuatro cinco seis siete ocho nueve."

    assert chunk_texts(text, WordCounter(), 8) == [
        ("Uno dos tres cuatro cinco seis", "window"),
        ("siete ocho nueve.", "sentence"),
    ]


def test_a_word_longer_than_the_limit_is_cut_inside_and_nothing_is_lost() -> None:
    # 19 characters is the most that 6 tokens hold under the character counter.
    run = "x" * 45
    text = f"ab {run} cd."

    assert chunk_texts(text, CharCounter(), 6) == [
        ("ab", "window"),
        ("x" * 19, "window"),
        ("x" * 19, "window"),
        ("x" * 7 + " cd.", "sentence"),
    ]


def test_a_run_without_spaces_longer_than_300_characters_still_ends_in_fitting_pieces() -> None:
    text = "y" * 320

    chunks = chunk_section(text, BOTH, 6)

    assert_valid_chunks(text, chunks, 6)
    assert "".join(text[chunk.start : chunk.end] for chunk in chunks) == text


@pytest.mark.parametrize("text", [None, "", "   ", " \n\t\n "])
def test_a_missing_or_blank_section_has_no_chunk(text: str | None) -> None:
    assert chunk_section(text, BOTH, 16) == []
    assert chunk_title(text, BOTH, 16) == []


def test_a_limit_that_holds_no_character_is_refused() -> None:
    # The word counter gives a single character 3 tokens, more than 2.
    with pytest.raises(ValueError, match="no room"):
        chunk_section("Uno.", WordCounter(), 2)


# Title (C3) and counters (C5)


@pytest.mark.parametrize("seed", range(10))
def test_a_title_within_the_limit_is_one_chunk_even_with_sentences_inside(seed: int) -> None:
    rng = random.Random(seed)
    words = [rng.choice(WORDS) for _ in range(rng.randint(2, 126))]
    for index in rng.sample(range(1, len(words)), k=min(3, len(words) - 1)):
        words[index - 1] += "."
        words[index] = words[index].capitalize()
    title = " ".join(words)

    assert chunk_title(f" {title} ", WordCounter(), 128) == [
        Chunk(1, 1 + len(title), WordCounter().count(title), "sentence")
    ]


def test_a_title_over_the_limit_falls_back_to_the_cuts_of_an_over_long_sentence() -> None:
    title = "Mejora del proceso, caso de una planta; un modelo de control de la calidad"

    chunks = chunk_title(title, WordCounter(), 10)

    assert [title[chunk.start : chunk.end] for chunk in chunks] == [
        "Mejora del proceso, caso de una planta;",
        "un modelo de control de la calidad",
    ]
    assert [chunk.cut_kind for chunk in chunks] == ["clause", "sentence"]


def test_the_counter_of_several_counters_takes_the_largest_count() -> None:
    assert BOTH.count("a b c d") == 6
    assert BOTH.count("x" * 40) == 12
    assert BOTH.count("") == 2


def test_a_counter_of_no_counters_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MaxTokenCounter(())
