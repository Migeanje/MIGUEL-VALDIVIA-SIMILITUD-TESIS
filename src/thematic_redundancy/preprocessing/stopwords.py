"""Stopwords of the full cleaner: spaCy's Spanish list and the curated domain list.

The domain list holds the genre and boilerplate lemmas of the corpus, which carry no topic,
such as ``tesis`` or ``arequipa``. It lives in a text file, ``config/stopwords_domain_es.txt``,
built from document frequencies by ``experiments/domain_stopwords.py`` and then reviewed by
hand. Nothing here imports spaCy until :func:`spacy_stopwords` is called.
"""

import functools
import unicodedata
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

DOMAIN_STOPWORDS_FILE = Path("config") / "stopwords_domain_es.txt"
"""Location of the curated domain stopword list, relative to the project root."""

COMMENT_PREFIX = "#"


class StopwordCandidate(BaseModel):
    """A lemma, and the number and share of the documents that hold it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lemma: str
    documents: int
    ratio: float
    """``documents`` over all the documents, rounded to four decimals."""


def stopword_candidates(
    documents: Iterable[Iterable[str]], *, min_ratio: float
) -> list[StopwordCandidate]:
    """Return the lemmas held by at least ``min_ratio`` of ``documents``, the most common first.

    Each document is the lemma tokens of one text, such as :func:`clean_full_batch` gives;
    a lemma counts once per document that holds it. Ties rank in lemma order.

    Raises:
        ValueError: if ``min_ratio`` is not above 0 and at most 1.
    """
    if not 0 < min_ratio <= 1:
        raise ValueError(f"min_ratio must be above 0 and at most 1, got {min_ratio}")
    frequencies: Counter[str] = Counter()
    total = 0
    for tokens in documents:
        frequencies.update(set(tokens))
        total += 1
    ranked = sorted(frequencies.items(), key=lambda item: (-item[1], item[0]))
    return [
        StopwordCandidate(lemma=lemma, documents=count, ratio=round(count / total, 4))
        for lemma, count in ranked
        if count / total >= min_ratio
    ]


def load_domain_stopwords(path: Path) -> frozenset[str]:
    """Read a domain stopword list: one lemma per line.

    A line that starts with ``#`` is a comment, and blank lines are skipped. Each lemma is put
    in NFC and in lower case, so that it matches the tokens of the full cleaner; accents and ñ
    are kept.

    Raises:
        OSError: if the file cannot be read.
        ValueError: if a line holds more than one word.
    """
    lemmas = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        entry = line.strip()
        if not entry or entry.startswith(COMMENT_PREFIX):
            continue
        if len(entry.split()) > 1:
            raise ValueError(f"{path}, line {number}: a stopword must be a single lemma")
        lemmas.add(normalize_stopword(entry))
    return frozenset(lemmas)


def normalize_stopword(word: str) -> str:
    """Return ``word`` in NFC and in lower case, the form in which tokens are compared."""
    return unicodedata.normalize("NFC", word).lower()


@functools.cache
def spacy_stopwords() -> frozenset[str]:
    """Return spaCy's Spanish stopwords, in lower case."""
    from spacy.lang.es.stop_words import STOP_WORDS

    return frozenset(normalize_stopword(word) for word in STOP_WORDS)
