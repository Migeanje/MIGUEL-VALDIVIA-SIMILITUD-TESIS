"""Aggregate profile of a snapshot's metadata, for the exploratory analysis of the corpus.

Every function here is pure: it reads :class:`SnapshotRecord` objects and returns counts and
distribution statistics. No result holds a title, an abstract, a person's name or an ORCID,
so a profile can be published with the results. A profile does hold keywords, OCDE codes and
place names, which are short public terms.

Token counts come from an injected :data:`TokenCounter`, so this module needs no model. The
experiment script passes the tokenizer of the embedding model. :func:`sensitive_strings` and
:func:`leaked_strings` let a caller check its outputs before it writes them.
"""

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import date

import numpy as np
from pydantic import BaseModel, ConfigDict, JsonValue

from thematic_redundancy.corpus.snapshot import SnapshotRecord
from thematic_redundancy.corpus.thesis_files import access_restriction, is_thesis
from thematic_redundancy.preprocessing.light_cleaner import normalize_text
from thematic_redundancy.preprocessing.title_suffix import (
    NO_SUFFIX,
    PLACE_SUFFIX_PATTERNS,
    TITLE_SUFFIX_PATTERNS,
    YEAR_SUFFIX_PATTERNS,
    classify_title_suffix,
)

# The title suffix patterns and the place list live in the preprocessing package, which cuts
# the suffixes that this profile counts. These names stay importable from here as well.
from thematic_redundancy.preprocessing.title_suffix import PLACE_NAMES as PLACE_NAMES
from thematic_redundancy.preprocessing.title_suffix import TEMPORAL_WORDS as TEMPORAL_WORDS
from thematic_redundancy.preprocessing.title_suffix import TRAILING_PLACE as TRAILING_PLACE
from thematic_redundancy.preprocessing.title_suffix import TRAILING_YEAR as TRAILING_YEAR
from thematic_redundancy.preprocessing.title_suffix import TitleSuffix as TitleSuffix

TYPE_KEY = "renati.type"
ISSUED_KEY = "dc.date.issued"
TITLE_KEY = "dc.title"
ABSTRACT_KEY = "dc.description.abstract"
KEYWORD_KEY = "dc.subject"
OCDE_KEY = "dc.subject.ocde"
LANGUAGE_KEY = "dc.language.iso"
ADVISOR_KEY = "dc.contributor.advisor"
ADVISOR_ORCID_KEY = "renati.advisor.orcid"
RIGHTS_KEY = "dc.rights"
EMBARGO_END_KEY = "dc.date.embargoEnd"
PERSON_KEYS = ("dc.contributor.author", ADVISOR_KEY, "renati.juror")
"""Fields that hold people's names: authors, advisors and jurors."""

MISSING = "(none)"
"""Key under which a count gathers the items without a value."""

TokenCounter = Callable[[str], int]
"""Number of tokens that a model sees for one text, its special tokens included."""

Metadata = Mapping[str, Sequence[Mapping[str, JsonValue]]]


class _ProfileModel(BaseModel):
    """Base of the profile models: immutable and closed to unknown keys."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Distribution(_ProfileModel):
    """Size, mean and percentiles of a sample, rounded to three decimals.

    The percentiles interpolate linearly between the two closest ranks, numpy's default.
    """

    count: int
    mean: float
    min: float
    p25: float
    median: float
    p75: float
    p95: float
    max: float


class Histogram(_ProfileModel):
    """Values per bin of ``bin_width``, keyed by the lower edge of each bin.

    The bins run from the lowest non-empty one to the highest, empty ones included.
    """

    bin_width: int
    counts: dict[int, int]


class LengthProfile(_ProfileModel):
    """Lengths of a set of texts, measured after their whitespace is collapsed."""

    texts: int
    characters: Distribution
    words: Distribution
    """Runs of characters between whitespace."""
    tokens: Distribution
    """Tokens as the model sees them, special tokens included."""
    token_histogram: Histogram
    token_limit: int
    above_limit: int
    """Texts with more tokens than ``token_limit``, which the model would truncate."""
    share_above_limit: float


class LineBreakProfile(_ProfileModel):
    """Hard line breaks inside texts, as left by copying them from a PDF.

    A run of breaks with only whitespace between them counts once.
    """

    texts_with_breaks: int
    breaks: int
    inside_sentence: int
    """Breaks whose line does not end with sentence punctuation."""
    after_hyphen: int
    """Breaks right after a hyphen, which may split a word in two."""


class ChunkProfile(_ProfileModel):
    """Chunks of at most ``token_limit`` tokens that each abstract would need.

    ``sentence_aligned`` packs whole sentences, as :func:`pack_sentences` does, and
    ``lower_bound`` packs tokens perfectly, ignoring sentence boundaries. Both map a number
    of chunks to the abstracts that need it.
    """

    token_limit: int
    special_tokens: int
    """Special tokens that every chunk carries, so each holds ``token_limit`` minus these."""
    sentence_aligned: dict[int, int]
    sentence_aligned_stats: Distribution | None
    lower_bound: dict[int, int]
    sentence_sums_differing: int
    """Abstracts whose sentences, tokenized one at a time, do not add up to the whole abstract
    tokenized at once. The estimate assumes that they do."""


class AbstractProfile(_ProfileModel):
    """Presence, lengths, chunk needs and text defects of the abstracts (first value each)."""

    present: int
    missing: int
    lengths: LengthProfile | None
    chunks: ChunkProfile
    line_breaks: LineBreakProfile
    with_replacement_character: int
    """Abstracts holding U+FFFD, the mark of a character lost to a wrong encoding."""
    with_keywords_section: int
    """Abstracts that mention ``Palabras clave``, so they likely embed their keywords."""


class TitleProfile(_ProfileModel):
    """Lengths of the titles (first value each) and the place or year suffixes they end with."""

    present: int
    missing: int
    lengths: LengthProfile | None
    suffix_patterns: dict[str, int]
    """Titles per pattern of :data:`TITLE_SUFFIX_PATTERNS`, and ``none``."""
    with_suffix: int
    with_year_suffix: int
    with_place_suffix: int
    share_with_suffix: float
    places: dict[str, int]
    """Titles per place named in their suffix."""


class KeywordCount(_ProfileModel):
    """A normalized keyword and the number of theses that list it."""

    keyword: str
    theses: int


class KeywordProfile(_ProfileModel):
    """Keywords (``dc.subject``), counted once per thesis after :func:`normalize_keyword`."""

    values: int
    """Keyword values as harvested, before normalization."""
    per_thesis: Distribution | None
    per_thesis_counts: dict[int, int]
    """Theses per number of distinct keywords."""
    theses_without_keywords: int
    distinct: int
    used_by_one_thesis: int
    top: list[KeywordCount]
    top_by_program: dict[str, list[KeywordCount]]
    values_with_comma: int
    """Values with a comma, which may pack several keywords into one."""
    values_with_trailing_period: int


class OcdeProfile(_ProfileModel):
    """OCDE fields of research (``dc.subject.ocde``), as FORD codes such as ``2.11.04``."""

    theses_with_code: int
    theses_without_code: int
    distinct: int
    by_code: dict[str, int]
    by_field: dict[str, int]
    """Theses per major field, named as in :data:`OCDE_FIELDS`."""
    by_program: dict[str, dict[str, int]]


class AdvisorProfile(_ProfileModel):
    """Counts about advisors that never name one.

    Advisors are told apart by their name, compared without case or spacing differences but
    with its accents, so two spellings of one person count twice. The ORCID counts show how
    often that happens.
    """

    theses_with_advisor: int
    theses_without_advisor: int
    theses_with_several_advisors: int
    distinct_advisors: int
    theses_per_advisor: Distribution | None
    advisors_by_theses: dict[str, int]
    """Advisors per band of supervised theses, such as ``2-5``."""
    share_of_theses_with_top_advisors: float
    """Share of the theses with an advisor that one of the :data:`TOP_ADVISORS` advisors with
    the most theses supervised."""
    theses_with_advisor_orcid: int
    distinct_advisor_orcids: int
    orcids_with_several_names: int
    """ORCIDs found with more than one advisor name, counting only the theses with exactly
    one advisor and one ORCID."""


class RightsProfile(_ProfileModel):
    """Access rights (``dc.rights``) and embargo end dates, judged at the snapshot date."""

    by_access: dict[str, int]
    """Theses per access class of :data:`ACCESS_CLASSES`."""
    embargoed_end_passed: int
    """Embargoed theses whose end date lies before the snapshot date."""
    embargoed_end_pending: int
    embargoed_end_missing: int
    not_embargoed_with_end: int
    """Theses with an embargo end date that are not embargoed, such as reopened ones."""


class CalendarProfile(_ProfileModel):
    """Items per program and issue year, from the first day of a partial ``dc.date.issued``."""

    by_program: dict[str, int]
    """Every item per program, those without a readable date included."""
    by_year: dict[int, int]
    by_program_and_year: dict[str, dict[int, int]]
    future_dated: int
    """Items issued after the snapshot date."""
    unparsed_dates: int
    earliest: date | None
    latest: date | None


class OtherItemsProfile(_ProfileModel):
    """The items that are not theses, reported apart from the profile of the theses."""

    items: int
    by_type: dict[str, int]
    """Items per ``renati.type`` fragment, as the repository spells it."""
    by_program: dict[str, int]
    by_year: dict[int, int]
    future_dated: int


class DuplicateProfile(_ProfileModel):
    """Theses that repeat both the title and the abstract of another one.

    The texts are compared with collapsed whitespace and folded case, but with their accents,
    so a group is most likely one thesis deposited more than once.
    """

    groups: int
    records: int
    extra_records: int
    """Records beyond the first of each group: the theses that a census counts twice."""


class MetadataProfile(_ProfileModel):
    """Profile of one snapshot: the theses in detail, and the other items in summary."""

    snapshot_id: str
    snapshot_date: date
    """UTC date of the harvest; future dates and lapsed embargoes are judged against it."""
    thesis_type: str
    items: int
    theses: int
    calendar: CalendarProfile
    abstracts: AbstractProfile
    titles: TitleProfile
    keywords: KeywordProfile
    ocde: OcdeProfile
    languages: dict[str, int]
    advisors: AdvisorProfile
    rights: RightsProfile
    duplicates: DuplicateProfile
    other_items: OtherItemsProfile


def profile_metadata(
    records: Sequence[SnapshotRecord],
    *,
    snapshot_id: str,
    snapshot_date: date,
    thesis_type: str,
    program_keys: Sequence[str],
    count_tokens: TokenCounter,
    token_limit: int,
    top_keywords: int = 30,
    top_keywords_per_program: int = 10,
) -> MetadataProfile:
    """Profile the theses of ``records`` and summarize the other items apart.

    A thesis is an item with the exact ``renati.type`` fragment ``thesis_type``. Programs
    follow ``program_keys``, and any other program found comes after them.

    Raises:
        ValueError: if ``token_limit`` leaves no room beside the special tokens.
    """
    theses = [record for record in records if is_thesis(record.metadata, thesis_type)]
    others = [record for record in records if not is_thesis(record.metadata, thesis_type)]
    return MetadataProfile(
        snapshot_id=snapshot_id,
        snapshot_date=snapshot_date,
        thesis_type=thesis_type,
        items=len(records),
        theses=len(theses),
        calendar=calendar_profile(theses, program_keys, snapshot_date),
        abstracts=abstract_profile(theses, count_tokens, token_limit),
        titles=title_profile(theses, count_tokens, token_limit),
        keywords=keyword_profile(
            theses, program_keys, top=top_keywords, top_per_program=top_keywords_per_program
        ),
        ocde=ocde_profile(theses, program_keys),
        languages=language_profile(theses),
        advisors=advisor_profile(theses),
        rights=rights_profile(theses, snapshot_date),
        duplicates=duplicate_profile(theses),
        other_items=other_items_profile(others, program_keys, snapshot_date),
    )


# Text and statistics helpers


def describe(values: Iterable[float]) -> Distribution | None:
    """Return the :class:`Distribution` of ``values``, or ``None`` when there are none."""
    sample = [float(value) for value in values]
    return _summarize(sample) if sample else None


def histogram(values: Iterable[int], bin_width: int) -> Histogram:
    """Count ``values`` per bin of ``bin_width``: bin ``e`` holds ``e <= value < e + bin_width``.

    Raises:
        ValueError: if ``bin_width`` is below 1.
    """
    if bin_width < 1:
        raise ValueError(f"bin_width must be at least 1, got {bin_width}")
    counts = Counter(value // bin_width * bin_width for value in values)
    if not counts:
        return Histogram(bin_width=bin_width, counts={})
    edges = range(min(counts), max(counts) + 1, bin_width)
    return Histogram(bin_width=bin_width, counts={edge: counts[edge] for edge in edges})


def _summarize(sample: Sequence[float]) -> Distribution:
    data = np.asarray(sample, dtype=float)
    p25, median, p75, p95 = np.percentile(data, [25, 50, 75, 95], method="linear")
    return Distribution(
        count=int(data.size),
        mean=_round(data.mean()),
        min=_round(data.min()),
        p25=_round(p25),
        median=_round(median),
        p75=_round(p75),
        p95=_round(p95),
        max=_round(data.max()),
    )


def _round(value: float) -> float:
    return round(float(value), 3)


def _share(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def _ranked(counts: Mapping[str, int]) -> dict[str, int]:
    """Return ``counts`` from the largest count down, with ties in key order."""
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _values(metadata: Metadata, key: str) -> list[str]:
    """Return the non-blank text values of the metadata field ``key``, stripped, in order."""
    return [
        text.strip()
        for entry in metadata.get(key, ())
        if isinstance(text := entry.get("value"), str) and text.strip()
    ]


def _program_order(records: Iterable[SnapshotRecord], program_keys: Sequence[str]) -> list[str]:
    """Return ``program_keys``, then any other program of ``records`` in key order."""
    found = {record.program_key for record in records}
    return [*program_keys, *sorted(found - set(program_keys))]


# Dates and programs

_PARTIAL_DATE = re.compile(r"([0-9]{4})(?:-([0-9]{2})(?:-([0-9]{2}))?)?(?=$|[T ])")
"""Start of a metadata date: a year, a month or a day, maybe followed by a time."""


def issue_day(value: str) -> date | None:
    """Return the first day that a metadata date can mean, or ``None`` if it is not a date.

    The value may be a year (``2026``), a month (``2026-12``), a day, or a timestamp. A year
    or a month counts from its first day, the rule by which the harvest manifest counts
    future-dated items, so only a date that surely lies ahead counts as future.
    """
    match = _PARTIAL_DATE.match(value.strip())
    if match is None:
        return None
    year, month, day = (int(part) if part else 1 for part in match.groups())
    try:
        return date(year, month, day)
    except ValueError:  # Such as month 13.
        return None


def calendar_profile(
    records: Sequence[SnapshotRecord], program_keys: Sequence[str], snapshot_date: date
) -> CalendarProfile:
    """Count ``records`` per program and issue year, and those issued after ``snapshot_date``.

    Only the first ``dc.date.issued`` value counts. The years run without gaps from the
    first to the last one found.
    """
    programs = _program_order(records, program_keys)
    by_program = dict.fromkeys(programs, 0)
    dated: list[tuple[str, date]] = []
    for record in records:
        by_program[record.program_key] += 1
        day = _first_day(record.metadata, ISSUED_KEY)
        if day is not None:
            dated.append((record.program_key, day))
    years = _year_span(day.year for _, day in dated)
    grid = {program: dict.fromkeys(years, 0) for program in programs}
    for program, day in dated:
        grid[program][day.year] += 1
    days = [day for _, day in dated]
    return CalendarProfile(
        by_program=by_program,
        by_year={year: sum(grid[program][year] for program in programs) for year in years},
        by_program_and_year=grid,
        future_dated=sum(day > snapshot_date for day in days),
        unparsed_dates=len(records) - len(dated),
        earliest=min(days, default=None),
        latest=max(days, default=None),
    )


def other_items_profile(
    others: Sequence[SnapshotRecord], program_keys: Sequence[str], snapshot_date: date
) -> OtherItemsProfile:
    """Summarize the items that are not theses: their types, programs and issue years."""
    calendar = calendar_profile(others, program_keys, snapshot_date)
    by_type: Counter[str] = Counter()
    for record in others:
        by_type.update(
            {_fragment(value) for value in _values(record.metadata, TYPE_KEY)} or {MISSING}
        )
    return OtherItemsProfile(
        items=len(others),
        by_type=_ranked(by_type),
        by_program=calendar.by_program,
        by_year=calendar.by_year,
        future_dated=calendar.future_dated,
    )


def _first_day(metadata: Metadata, key: str) -> date | None:
    values = _values(metadata, key)
    return issue_day(values[0]) if values else None


def _year_span(years: Iterable[int]) -> list[int]:
    found = set(years)
    return list(range(min(found), max(found) + 1)) if found else []


def _fragment(value: str) -> str:
    """Return the part of a URI after ``#``, or the whole value when it has no ``#``."""
    return value.rpartition("#")[2] if "#" in value else value


# Abstracts, sentences and chunks

_SENTENCE_BREAK = re.compile(
    r"(?:(?<=[.!?…])|(?<=[.!?…][\"'”’»)\]]))\s+(?=[\"'“‘«(\[¿¡]*[A-ZÁÉÍÓÚÜÑ0-9])"
)
"""Whitespace after sentence punctuation, maybe closed by a quote or a bracket, and before a
capital letter or a digit, maybe opened by a quote, a bracket, ``¿`` or ``¡``."""

_LINE_BREAK = re.compile(r"\s*[\r\n]\s*")
_SENTENCE_END = re.compile(r"[.!?…][\"'”’»)\]]*$")
_HYPHENS = ("-", "\N{HYPHEN}")
_KEYWORDS_SECTION = re.compile(r"\bpalabras\s+clave\b", re.IGNORECASE)
REPLACEMENT_CHARACTER = "\N{REPLACEMENT CHARACTER}"


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences, after joining its hard-wrapped lines.

    This is an approximation. A sentence ends at ``.``, ``!``, ``?`` or ``…`` (maybe closed
    by a quote or a bracket) followed by whitespace and a capital letter or a digit. So
    ``3.5`` and ``kg. luego`` never split, while an abbreviation before a capital, such as
    ``S.A.C. La``, does. A line break never ends a sentence on its own, because the
    abstracts are hard-wrapped.
    """
    flat = normalize_text(text)
    return _SENTENCE_BREAK.split(flat) if flat else []


def pack_sentences(sentence_tokens: Iterable[int], budget: int) -> int:
    """Return how many chunks of ``budget`` tokens hold the sentences, packed greedily in order.

    A sentence joins the open chunk when it fits, and otherwise opens a new one. A sentence
    longer than ``budget`` starts a chunk of its own and is cut into ``budget``-sized pieces,
    as a word-level fallback would; its last piece may share a chunk with what follows.

    Raises:
        ValueError: if ``budget`` is below 1.
    """
    if budget < 1:
        raise ValueError(f"budget must be at least 1 token, got {budget}")
    chunks = 0
    room = 0  # Tokens still free in the open chunk.
    for tokens in sentence_tokens:
        if tokens <= room:
            room -= tokens
        elif tokens <= budget:
            chunks += 1
            room = budget - tokens
        else:
            pieces = math.ceil(tokens / budget)
            chunks += pieces
            room = pieces * budget - tokens
    return chunks


def abstract_profile(
    theses: Sequence[SnapshotRecord],
    count_tokens: TokenCounter,
    token_limit: int,
    *,
    bin_width: int = 16,
) -> AbstractProfile:
    """Measure the abstracts and estimate the chunks of ``token_limit`` tokens they need.

    The special tokens of a chunk are those that ``count_tokens`` gives for an empty text.
    Each sentence of :func:`split_sentences` is tokenized alone, without its special tokens.

    Raises:
        ValueError: if ``token_limit`` leaves no room beside the special tokens.
    """
    special = count_tokens("")
    budget = token_limit - special
    if budget < 1:
        raise ValueError(
            f"token_limit ({token_limit}) must exceed the {special} special tokens of a chunk"
        )
    abstracts = [
        values[0] for record in theses if (values := _values(record.metadata, ABSTRACT_KEY))
    ]
    flat = [normalize_text(abstract) for abstract in abstracts]
    tokens = [count_tokens(text) for text in flat]
    aligned: list[int] = []
    lower: list[int] = []
    differing = 0
    for abstract, whole in zip(abstracts, tokens, strict=True):
        sentence_tokens = [
            count_tokens(sentence) - special for sentence in split_sentences(abstract)
        ]
        aligned.append(pack_sentences(sentence_tokens, budget))
        lower.append(math.ceil((whole - special) / budget))
        differing += sum(sentence_tokens) != whole - special
    breaks = [_line_breaks(abstract) for abstract in abstracts]
    return AbstractProfile(
        present=len(abstracts),
        missing=len(theses) - len(abstracts),
        lengths=_length_profile(flat, tokens, token_limit, bin_width),
        chunks=ChunkProfile(
            token_limit=token_limit,
            special_tokens=special,
            sentence_aligned=dict(sorted(Counter(aligned).items())),
            sentence_aligned_stats=describe(aligned),
            lower_bound=dict(sorted(Counter(lower).items())),
            sentence_sums_differing=differing,
        ),
        line_breaks=LineBreakProfile(
            texts_with_breaks=sum(1 for found, _, _ in breaks if found),
            breaks=sum(found for found, _, _ in breaks),
            inside_sentence=sum(inside for _, inside, _ in breaks),
            after_hyphen=sum(hyphen for _, _, hyphen in breaks),
        ),
        with_replacement_character=sum(REPLACEMENT_CHARACTER in text for text in abstracts),
        with_keywords_section=sum(bool(_KEYWORDS_SECTION.search(text)) for text in abstracts),
    )


def _line_breaks(text: str) -> tuple[int, int, int]:
    """Return the line breaks of ``text``, those inside a sentence, and those after a hyphen."""
    lines = _LINE_BREAK.split(text.strip())
    broken = lines[:-1]
    return (
        len(broken),
        sum(1 for line in broken if not _SENTENCE_END.search(line)),
        sum(1 for line in broken if line.endswith(_HYPHENS)),
    )


def _length_profile(
    texts: Sequence[str], tokens: Sequence[int], token_limit: int, bin_width: int
) -> LengthProfile | None:
    if not texts:
        return None
    above = sum(count > token_limit for count in tokens)
    return LengthProfile(
        texts=len(texts),
        characters=_summarize([len(text) for text in texts]),
        words=_summarize([len(text.split()) for text in texts]),
        tokens=_summarize(tokens),
        token_histogram=histogram(tokens, bin_width),
        token_limit=token_limit,
        above_limit=above,
        share_above_limit=_share(above, len(texts)),
    )


# Titles


def title_profile(
    theses: Sequence[SnapshotRecord],
    count_tokens: TokenCounter,
    token_limit: int,
    *,
    bin_width: int = 8,
) -> TitleProfile:
    """Measure the titles and count the place and year suffixes that they end with."""
    titles = [
        normalize_text(values[0])
        for record in theses
        if (values := _values(record.metadata, TITLE_KEY))
    ]
    suffixes = [classify_title_suffix(title) for title in titles]
    patterns = Counter(suffix.pattern for suffix in suffixes)
    places = Counter(place for suffix in suffixes for place in suffix.places)
    with_suffix = len(titles) - patterns[NO_SUFFIX]
    return TitleProfile(
        present=len(titles),
        missing=len(theses) - len(titles),
        lengths=_length_profile(
            titles, [count_tokens(title) for title in titles], token_limit, bin_width
        ),
        suffix_patterns={
            pattern: patterns[pattern] for pattern in (*TITLE_SUFFIX_PATTERNS, NO_SUFFIX)
        },
        with_suffix=with_suffix,
        with_year_suffix=sum(patterns[pattern] for pattern in YEAR_SUFFIX_PATTERNS),
        with_place_suffix=sum(patterns[pattern] for pattern in PLACE_SUFFIX_PATTERNS),
        share_with_suffix=_share(with_suffix, len(titles)),
        places=_ranked(places),
    )


# Keywords, OCDE fields and languages


def normalize_keyword(value: str) -> str:
    """Return a keyword ready for counting.

    Its whitespace is collapsed, trailing periods are dropped, and its letter case is
    folded. Accents and ñ are kept, so ``Producción`` and ``Produccion`` stay apart.
    """
    return normalize_text(value).rstrip(". ").casefold()


def keyword_profile(
    theses: Sequence[SnapshotRecord],
    program_keys: Sequence[str],
    *,
    top: int = 30,
    top_per_program: int = 10,
) -> KeywordProfile:
    """Count the keywords per thesis and rank them by the number of theses that list them.

    Ties rank in keyword order, so the ranking is reproducible.
    """
    overall: Counter[str] = Counter()
    by_program: dict[str, Counter[str]] = {
        program: Counter() for program in _program_order(theses, program_keys)
    }
    sizes = []
    values = with_comma = with_period = 0
    for record in theses:
        raw = _values(record.metadata, KEYWORD_KEY)
        values += len(raw)
        with_comma += sum("," in value for value in raw)
        with_period += sum(value.endswith(".") for value in raw)
        keywords = {keyword for value in raw if (keyword := normalize_keyword(value))}
        sizes.append(len(keywords))
        overall.update(keywords)
        by_program[record.program_key].update(keywords)
    return KeywordProfile(
        values=values,
        per_thesis=describe(sizes),
        per_thesis_counts=dict(sorted(Counter(sizes).items())),
        theses_without_keywords=sizes.count(0),
        distinct=len(overall),
        used_by_one_thesis=sum(1 for count in overall.values() if count == 1),
        top=_top_keywords(overall, top),
        top_by_program={
            program: _top_keywords(counts, top_per_program)
            for program, counts in by_program.items()
        },
        values_with_comma=with_comma,
        values_with_trailing_period=with_period,
    )


def _top_keywords(counts: Mapping[str, int], size: int) -> list[KeywordCount]:
    ranked = list(_ranked(counts).items())[:size]
    return [KeywordCount(keyword=keyword, theses=theses) for keyword, theses in ranked]


OCDE_FIELDS = {
    "1": "Natural sciences",
    "2": "Engineering and technology",
    "3": "Medical and health sciences",
    "4": "Agricultural and veterinary sciences",
    "5": "Social sciences",
    "6": "Humanities and the arts",
}
"""Major fields of research and development (FORD) of the OECD Frascati Manual 2015."""


def ocde_code(value: str) -> str:
    """Return the FORD code of a ``dc.subject.ocde`` value, such as ``2.11.04``."""
    return _fragment(value.strip())


def ocde_profile(theses: Sequence[SnapshotRecord], program_keys: Sequence[str]) -> OcdeProfile:
    """Count the theses per OCDE code, per major field, and per code within each program."""
    by_code: Counter[str] = Counter()
    by_program: dict[str, Counter[str]] = {
        program: Counter() for program in _program_order(theses, program_keys)
    }
    with_code = 0
    for record in theses:
        codes = {ocde_code(value) for value in _values(record.metadata, OCDE_KEY)}
        with_code += bool(codes)
        by_code.update(codes)
        by_program[record.program_key].update(codes)
    by_field: Counter[str] = Counter()
    for code, count in by_code.items():
        major = code.split(".", 1)[0]
        by_field[f"{major} {OCDE_FIELDS.get(major, 'unknown field')}"] += count
    return OcdeProfile(
        theses_with_code=with_code,
        theses_without_code=len(theses) - with_code,
        distinct=len(by_code),
        by_code=_ranked(by_code),
        by_field=_ranked(by_field),
        by_program={program: _ranked(counts) for program, counts in by_program.items()},
    )


def language_profile(theses: Iterable[SnapshotRecord]) -> dict[str, int]:
    """Count the theses per ``dc.language.iso`` value; one with several counts under each."""
    counts: Counter[str] = Counter()
    for record in theses:
        counts.update(set(_values(record.metadata, LANGUAGE_KEY)) or {MISSING})
    return _ranked(counts)


# Advisors

TOP_ADVISORS = 10
_ADVISOR_BANDS = ((1, 1), (2, 5), (6, 10), (11, 20), (21, None))
_ORCID_ID = re.compile(r"[0-9]{4}-[0-9]{4}-[0-9]{4}-[0-9]{3}[0-9X]", re.IGNORECASE)


def advisor_profile(theses: Sequence[SnapshotRecord]) -> AdvisorProfile:
    """Count advisors and the theses that each one supervised, without naming anyone."""
    theses_per_advisor: Counter[str] = Counter()
    advisors_per_thesis: list[set[str]] = []
    names_per_orcid: defaultdict[str, set[str]] = defaultdict(set)
    orcids: set[str] = set()
    with_orcid = 0
    for record in theses:
        advisors = {_person_key(name) for name in _values(record.metadata, ADVISOR_KEY)}
        advisors_per_thesis.append(advisors)
        theses_per_advisor.update(advisors)
        thesis_orcids = {_orcid_key(value) for value in _values(record.metadata, ADVISOR_ORCID_KEY)}
        with_orcid += bool(thesis_orcids)
        orcids.update(thesis_orcids)
        if len(advisors) == 1 and len(thesis_orcids) == 1:
            names_per_orcid[next(iter(thesis_orcids))].update(advisors)
    with_advisor = sum(1 for advisors in advisors_per_thesis if advisors)
    top = set(list(_ranked(theses_per_advisor))[:TOP_ADVISORS])
    return AdvisorProfile(
        theses_with_advisor=with_advisor,
        theses_without_advisor=len(theses) - with_advisor,
        theses_with_several_advisors=sum(
            1 for advisors in advisors_per_thesis if len(advisors) > 1
        ),
        distinct_advisors=len(theses_per_advisor),
        theses_per_advisor=describe(theses_per_advisor.values()),
        advisors_by_theses=_bands(theses_per_advisor.values()),
        share_of_theses_with_top_advisors=_share(
            sum(1 for advisors in advisors_per_thesis if advisors & top), with_advisor
        ),
        theses_with_advisor_orcid=with_orcid,
        distinct_advisor_orcids=len(orcids),
        orcids_with_several_names=sum(1 for names in names_per_orcid.values() if len(names) > 1),
    )


def _person_key(name: str) -> str:
    """Return a name ready for counting: spacing evened out around commas, case folded."""
    return re.sub(r"\s*,\s*", ", ", normalize_text(name)).casefold()


def _orcid_key(value: str) -> str:
    """Return the bare ORCID iD of a value, which may be an ``orcid.org`` URL."""
    match = _ORCID_ID.search(value)
    return match.group(0).upper() if match else value.strip().casefold()


def _bands(counts: Iterable[int]) -> dict[str, int]:
    labels = [
        str(low) if low == high else f"{low}+" if high is None else f"{low}-{high}"
        for low, high in _ADVISOR_BANDS
    ]
    bands = dict.fromkeys(labels, 0)
    for count in counts:
        for label, (low, high) in zip(labels, _ADVISOR_BANDS, strict=True):
            if count >= low and (high is None or count <= high):
                bands[label] += 1
                break
    return bands


# Rights

ACCESS_CLASSES = ("open", "embargoed", "restricted", "metadata_only", "other", MISSING)
"""Access classes of a thesis, from its COAR access right (``dc.rights``)."""

_COAR_ACCESS_RIGHTS = "purl.org/coar/access_right/"
_OPEN_ACCESS = "c_abf2"
_RESTRICTED_CLASSES = {"c_f1cf": "embargoed", "c_16ec": "restricted", "c_14cb": "metadata_only"}


def access_class(metadata: Metadata) -> str:
    """Return the access class of an item, one of :data:`ACCESS_CLASSES`.

    A COAR access right that closes the files wins over open access, as it does when the
    PDF downloader decides what to skip. A value that is no COAR access right is ``other``.
    """
    restriction = access_restriction(metadata)
    if restriction is not None:
        return _RESTRICTED_CLASSES.get(restriction, "restricted")
    rights = _values(metadata, RIGHTS_KEY)
    if not rights:
        return MISSING
    for value in rights:
        prefix, _, code = value.rpartition("/")
        if f"{prefix}/".endswith(_COAR_ACCESS_RIGHTS) and code == _OPEN_ACCESS:
            return "open"
    return "other"


def rights_profile(theses: Iterable[SnapshotRecord], snapshot_date: date) -> RightsProfile:
    """Count the theses per access class, and tell lapsed embargoes from pending ones.

    An embargo has lapsed when its first ``dc.date.embargoEnd`` lies before
    ``snapshot_date``; one that ends on that day is still pending.
    """
    by_access = dict.fromkeys(ACCESS_CLASSES, 0)
    passed = pending = missing = not_embargoed_with_end = 0
    for record in theses:
        access = access_class(record.metadata)
        by_access[access] += 1
        end = _first_day(record.metadata, EMBARGO_END_KEY)
        if access == "embargoed":
            if end is None:
                missing += 1
            elif end < snapshot_date:
                passed += 1
            else:
                pending += 1
        elif end is not None:
            not_embargoed_with_end += 1
    return RightsProfile(
        by_access=by_access,
        embargoed_end_passed=passed,
        embargoed_end_pending=pending,
        embargoed_end_missing=missing,
        not_embargoed_with_end=not_embargoed_with_end,
    )


# Duplicates


def duplicate_profile(theses: Iterable[SnapshotRecord]) -> DuplicateProfile:
    """Count the groups of theses that share both their title and their abstract.

    Only the first value of each field counts, and a thesis without either is left out.
    """
    copies: Counter[tuple[str, str]] = Counter()
    for record in theses:
        titles = _values(record.metadata, TITLE_KEY)
        abstracts = _values(record.metadata, ABSTRACT_KEY)
        if titles and abstracts:
            copies[
                (normalize_text(titles[0]).casefold(), normalize_text(abstracts[0]).casefold())
            ] += 1
    sizes = [count for count in copies.values() if count > 1]
    return DuplicateProfile(
        groups=len(sizes), records=sum(sizes), extra_records=sum(sizes) - len(sizes)
    )


# Privacy guard


def sensitive_strings(
    records: Iterable[SnapshotRecord], *, min_sentence_length: int = 40
) -> frozenset[str]:
    """Return the strings of ``records`` that a published output must never hold, case folded.

    They are every title and abstract, both as stored and with collapsed whitespace; every
    abstract sentence of at least ``min_sentence_length`` characters; every author, advisor
    and juror name, as written, as given names then family names, and its family-name part
    when that has two words or more; and every ORCID value.
    """
    found: set[str] = set()
    for record in records:
        metadata = record.metadata
        for text in (*_values(metadata, TITLE_KEY), *_values(metadata, ABSTRACT_KEY)):
            found.update((text, normalize_text(text)))
        for abstract in _values(metadata, ABSTRACT_KEY):
            found.update(
                sentence
                for sentence in split_sentences(abstract)
                if len(sentence) >= min_sentence_length
            )
        for key in PERSON_KEYS:
            for name in _values(metadata, key):
                found.update(_name_forms(name))
        for key in metadata:
            if key.casefold().endswith(".orcid"):
                for value in _values(metadata, key):
                    found.update((value, _orcid_key(value)))
    return frozenset(folded for text in found if (folded := _fold(text)))


def leaked_strings(text: str, sensitive: Iterable[str]) -> list[str]:
    """Return the strings of ``sensitive`` found in ``text``, ignoring letter case and spacing."""
    folded = _fold(text)
    flat = " ".join(folded.split())
    return sorted(
        {
            needle
            for item in sensitive
            if (needle := _fold(item)) and (needle in folded or needle in flat)
        }
    )


def _name_forms(name: str) -> set[str]:
    """Return the ways in which a name written ``Family, Given`` may appear in a text."""
    flat = normalize_text(name)
    family, comma, given = (part.strip() for part in flat.partition(","))
    forms = {name, flat}
    if comma and given and family:
        forms.add(f"{given} {family}")
    if comma and len(family.split()) >= 2:
        forms.add(family)
    return forms


def _fold(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().strip()
