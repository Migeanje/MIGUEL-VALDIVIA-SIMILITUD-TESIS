"""Full cleaning: the lemma tokens that TF-IDF, c-TF-IDF and NPMI coherence read (D10, D12).

:func:`clean_full_batch` takes a text through these steps:

1. :func:`~thematic_redundancy.preprocessing.light_cleaner.clean_light`;
2. lower case;
3. tokenization and lemmatization by the spaCy model :data:`SPACY_MODEL`, except for the
   few words of :data:`KEPT_FORMS` that it gets wrong;
4. removal of spaCy's Spanish stopwords, the domain stopwords, punctuation, numbers and other
   tokens without a letter, tokens shorter than :data:`MIN_TOKEN_LENGTH` characters, and web
   or email addresses;
5. removal of the legal-entity suffixes of company names (:data:`LEGAL_ENTITY_SUFFIXES`).

Accents and ñ are kept throughout. The model loads on first use, once per process, without
the components that lemmas do not need, so importing this module stays cheap.
"""

import functools
import unicodedata
from collections.abc import Collection, Iterable, Iterator, Sequence
from typing import TYPE_CHECKING

from thematic_redundancy.preprocessing.light_cleaner import clean_light
from thematic_redundancy.preprocessing.stopwords import normalize_stopword, spacy_stopwords

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

SPACY_MODEL = "es_core_news_md"
"""Spanish spaCy pipeline, installed as a project dependency (version 3.8.0)."""

EXCLUDED_COMPONENTS = ("parser", "ner", "senter")
"""Components that the lemmas do not need, left out when the model loads. The lemmatizer works
from the part of speech that the morphologizer tags."""

DEFAULT_BATCH_SIZE = 64
"""Texts that spaCy processes together."""

MIN_TOKEN_LENGTH = 2

KEPT_FORMS = frozenset({"lean"})
"""Words kept as written instead of lemmatized. The Spanish lemmatizer reads the English
``lean`` of Lean Manufacturing, Lean Service or Lean Office as a form of ``leer``: it did so
for 61 of the 172 occurrences in the abstracts of snapshot ``20261002T224412Z``."""

LEGAL_ENTITY_SUFFIXES = frozenset({"sa", "saa", "sac", "scrl", "srl", "eirl"})
"""Legal forms of Peruvian companies, written without dots or spaces: S.A., S.A.A., S.A.C.,
S.C.R.L., S.R.L. and E.I.R.L. A spelling with dots or capitals matches as well. When spaces
split one, as in ``S. A. C.``, its letters fall below :data:`MIN_TOKEN_LENGTH` instead."""

_DROPPED_MARKS = str.maketrans(dict.fromkeys(".'\u2019\u00b4`"))
"""Periods and apostrophes, which a token loses: ``s.a.c.`` gives ``sac`` and ``5´s`` gives
``5s``. The acute accent here is the spacing mark that some texts use as an apostrophe; the
accents on letters are other characters, and they stay."""


@functools.cache
def spanish_pipeline() -> "Language":
    """Return the spaCy pipeline, loaded on the first call and shared afterwards.

    Raises:
        OSError: if the model :data:`SPACY_MODEL` is not installed.
    """
    import spacy

    return spacy.load(SPACY_MODEL, exclude=list(EXCLUDED_COMPONENTS))


def clean_full(text: str, *, domain_stopwords: Collection[str]) -> list[str]:
    """Return the lemma tokens of ``text``; see :func:`clean_full_batch`."""
    return clean_full_batch([text], domain_stopwords=domain_stopwords)[0]


def clean_full_batch(
    texts: Iterable[str],
    *,
    domain_stopwords: Collection[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[list[str]]:
    """Return the lemma tokens of each text, in order, processing the texts in batches.

    Each text is light-cleaned, put in lower case, and tokenized and lemmatized by spaCy. A
    token is dropped when it is punctuation, a space, or a web or email address, or when its
    lower-case form is a spaCy stopword or one of ``domain_stopwords``. The lemma of each
    remaining token (its written form for :data:`KEPT_FORMS`), which may expand an attached
    pronoun into a second word, then gives one token per word, after it loses its periods,
    its apostrophes, and any symbols around it. That token is dropped when it is shorter
    than :data:`MIN_TOKEN_LENGTH`, holds no letter (numbers among them), is a stopword of
    either list, or is one of :data:`LEGAL_ENTITY_SUFFIXES`.

    ``domain_stopwords`` are compared in NFC and in lower case; pass an empty set for none.

    Raises:
        ValueError: if ``batch_size`` is below 1.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be at least 1, got {batch_size}")
    stopwords = spacy_stopwords()
    domain = frozenset(normalize_stopword(word) for word in domain_stopwords)
    prepared = (clean_light(text).lower() for text in texts)
    docs = spanish_pipeline().pipe(prepared, batch_size=batch_size)
    return [list(_tokens(doc, stopwords, domain)) for doc in docs]


def join_tokens(tokens: Sequence[str]) -> str:
    """Join tokens with single spaces, as the vectorizers read a cleaned document."""
    return " ".join(tokens)


def _tokens(doc: "Doc", stopwords: frozenset[str], domain: frozenset[str]) -> Iterator[str]:
    for token in doc:
        if token.is_punct or token.is_space or token.like_url or token.like_email:
            continue
        form = unicodedata.normalize("NFC", token.lower_)
        if form in stopwords or form in domain:
            continue
        lemma = form if form in KEPT_FORMS else token.lemma_.lower()
        for part in lemma.split():
            word = _bare_word(part)
            if (
                len(word) >= MIN_TOKEN_LENGTH
                and any(char.isalpha() for char in word)
                and word not in stopwords
                and word not in domain
                and word not in LEGAL_ENTITY_SUFFIXES
            ):
                yield word


def _bare_word(lemma: str) -> str:
    """Return ``lemma`` in NFC without periods, apostrophes, or symbols at either end."""
    word = unicodedata.normalize("NFC", lemma).translate(_DROPPED_MARKS)
    start, end = 0, len(word)
    while start < end and not word[start].isalnum():
        start += 1
    while end > start and not word[end - 1].isalnum():
        end -= 1
    return word[start:end]
