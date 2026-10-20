"""Analysis of the expert questionnaire (T32; plan §3.8.2, PDF p. 82, printed p. 73).

The questionnaire has four dimensions of four affirmative items each, answered on a five-point
Likert scale (Anexo A, PDF pp. 117-118, printed pp. 108-109). The analysis is descriptive:

- per item: the frequency of each scale point, the missing answers counted apart, and the
  median and quartiles of the answers given;
- per dimension: the same over all its answers pooled, the median and quartiles of each complete
  respondent's mean score, and **Cronbach's alpha** with its **alpha-if-item-deleted** table.

Cronbach's alpha is ``k / (k - 1) * (1 - sum of item variances / variance of the total score)``,
over ``k`` items, with the sample variance (``ddof = 1``) in both places. It uses the respondents
who answered every item of the dimension (listwise deletion), and the alpha-if-item-deleted
table uses those same respondents. Its undefined cases follow the package's convention, ``None``
with a reason: ``"fewer_than_two_items"``, ``"fewer_than_two_respondents"`` (complete ones) and
``"zero_total_variance"``. A negative alpha is a value, not an undefined case.

Reverse-coded items are an explicit argument. Their answers are recoded as ``low + high - a``
before every statistic, the frequencies included. The median is numpy's (the mean of the two
middle answers for an even count); the quartiles use the numpy ``quantile`` method given.

:func:`answer_matrix` and :func:`item_dimensions` check a table of answers on a bounded integer
scale and its items' dimensions; :mod:`.aiken` reuses them for the judges' ratings.
"""

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

QUARTILE_METHODS: tuple[str, ...] = (
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
)
"""The methods of ``numpy.quantile`` that Hyndman and Fan (1996) define."""

AlphaUndefined = Literal[
    "fewer_than_two_items", "fewer_than_two_respondents", "zero_total_variance"
]


@dataclass(frozen=True)
class Quartiles:
    """The median and quartiles of some values."""

    count: int
    median: float | None
    first_quartile: float | None
    third_quartile: float | None
    interquartile_range: float | None
    undefined_reason: Literal["no_values"] | None


@dataclass(frozen=True)
class CronbachAlpha:
    """Cronbach's alpha of a set of items over the respondents who answered them all."""

    items: int
    respondents: int
    """The complete respondents used."""
    excluded_respondents: int
    """Respondents left out because they missed an item."""
    alpha: float | None
    undefined_reason: AlphaUndefined | None


@dataclass(frozen=True)
class AlphaIfItemDeleted:
    """Cronbach's alpha of a dimension without one of its items."""

    item: str
    alpha: float | None
    undefined_reason: AlphaUndefined | None


@dataclass(frozen=True)
class ItemSummary:
    """The answers to one item, after reverse coding when it applies."""

    item: str
    dimension: str
    reverse_coded: bool
    missing: int
    frequencies: tuple[tuple[int, int], ...]
    """``(scale point, answers)`` for every point from ``low`` to ``high``, zeros included."""
    answers: Quartiles


@dataclass(frozen=True)
class DimensionSummary:
    """The answers to the items of one dimension."""

    dimension: str
    items: tuple[str, ...]
    missing: int
    frequencies: tuple[tuple[int, int], ...]
    """Pooled over the dimension's items and respondents, for every scale point."""
    answers: Quartiles
    """Over every answer to the dimension's items, pooled."""
    respondent_scores: Quartiles
    """Over each complete respondent's mean answer to the dimension's items."""
    alpha: CronbachAlpha
    alpha_if_item_deleted: tuple[AlphaIfItemDeleted, ...]


@dataclass(frozen=True)
class LikertAnalysis:
    """The questionnaire analysis, items and dimensions in the order of ``dimensions``."""

    low: int
    high: int
    quartile_method: str
    respondents: int
    reverse_coded: tuple[str, ...]
    items: tuple[ItemSummary, ...]
    dimensions: tuple[DimensionSummary, ...]


def answer_matrix(
    answers: Mapping[str, Sequence[int | None]], *, low: int, high: int
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return the item ids and their answers as an ``items × respondents`` float array, with
    ``NaN`` for a missing answer.

    Raises:
        ValueError: if the scale is not ``low < high`` integers, there is no item, an item id is
            not a non-empty string, an answer is neither an integer of the scale nor ``None``,
            or the items have different numbers of answers.
    """
    if not (_is_integer(low) and _is_integer(high)) or low >= high:
        raise ValueError(f"the scale must be integers low < high, not {low!r} and {high!r}")
    items = tuple(answers)
    if not items:
        raise ValueError("there must be at least one item")
    rows: list[list[float]] = []
    for item in items:
        if not isinstance(item, str) or not item:
            raise ValueError(f"an item id must be a non-empty string, not {item!r}")
        row = [_answer(item, value, low, high) for value in answers[item]]
        if rows and len(row) != len(rows[0]):
            raise ValueError(
                f"item {item!r} has {len(row)} answers but item {items[0]!r} has {len(rows[0])}"
            )
        rows.append(row)
    return items, np.array(rows, dtype=np.float64).reshape(len(items), len(rows[0]))


def item_dimensions(
    dimensions: Mapping[str, Sequence[str]], items: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    """Return each dimension's items, after checking that every item is in exactly one
    dimension and that every dimension has an item.

    Raises:
        ValueError: otherwise.
    """
    known = set(items)
    owner: dict[str, str] = {}
    grouped: dict[str, tuple[str, ...]] = {}
    for dimension, listed in dimensions.items():
        if isinstance(listed, str):
            raise ValueError(
                f"dimension {dimension!r} must be a sequence of item ids, not a string"
            )
        members = tuple(listed)
        if not members:
            raise ValueError(f"dimension {dimension!r} has no item")
        for item in members:
            if item not in known:
                raise ValueError(f"dimension {dimension!r} lists {item!r}, which has no answers")
            if item in owner:
                raise ValueError(
                    f"item {item!r} is in dimensions {owner[item]!r} and {dimension!r}"
                )
            owner[item] = dimension
        grouped[dimension] = members
    for item in items:
        if item not in owner:
            raise ValueError(f"item {item!r} is in no dimension")
    return grouped


def cronbach_alpha(scores: np.ndarray | Sequence[Sequence[float]]) -> CronbachAlpha:
    """Return Cronbach's alpha of a ``respondents × items`` matrix, ``NaN`` marking a missing
    answer; a respondent with one is left out (listwise deletion).

    Raises:
        ValueError: if the scores are not a 2-D matrix of finite values or ``NaN``.
    """
    matrix = np.asarray(scores, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"the scores must be a 2-D respondents × items matrix, not of shape {matrix.shape}"
        )
    if np.isinf(matrix).any():
        raise ValueError("the scores must be finite, or NaN for a missing answer")
    complete = ~np.isnan(matrix).any(axis=1)
    return _alpha(matrix[complete], excluded=int((~complete).sum()))


def likert_analysis(
    responses: Mapping[str, Sequence[int | None]],
    dimensions: Mapping[str, Sequence[str]],
    *,
    low: int,
    high: int,
    reverse_coded: Collection[str],
    quartile_method: str,
) -> LikertAnalysis:
    """Return the per-item and per-dimension analysis of the questionnaire's answers.

    Args:
        responses: each item's answers, one per respondent, in the same respondent order for
            every item; ``None`` is a missing answer.
        dimensions: each dimension's items; every item is in exactly one dimension.
        low: the lowest point of the scale.
        high: the highest point of the scale.
        reverse_coded: the items whose answers are recoded as ``low + high - a``; may be empty.
        quartile_method: one of :data:`QUARTILE_METHODS`.

    Raises:
        ValueError: if an argument is invalid (see :func:`answer_matrix` and
            :func:`item_dimensions`), a reverse-coded item has no answers, or the quartile
            method is unknown.
    """
    items, matrix = answer_matrix(responses, low=low, high=high)
    grouped = item_dimensions(dimensions, items)
    if quartile_method not in QUARTILE_METHODS:
        raise ValueError(
            f"quartile_method must be one of {QUARTILE_METHODS}, not {quartile_method!r}"
        )
    if isinstance(reverse_coded, str):
        raise ValueError(
            f"reverse_coded must be a collection of item ids, not a string: {reverse_coded!r}"
        )
    reverse = set(reverse_coded)
    unknown = sorted(str(item) for item in reverse - set(items))
    if unknown:
        raise ValueError(f"the reverse-coded item {unknown[0]!r} has no answers")
    answers = {
        item: low + high - row if item in reverse else row
        for item, row in zip(items, matrix, strict=True)
    }
    summaries = [
        ItemSummary(
            item=item,
            dimension=dimension,
            reverse_coded=item in reverse,
            missing=_missing(answers[item]),
            frequencies=_frequencies(answers[item], low, high),
            answers=_quartiles(answers[item], quartile_method),
        )
        for dimension, members in grouped.items()
        for item in members
    ]
    return LikertAnalysis(
        low=low,
        high=high,
        quartile_method=quartile_method,
        respondents=matrix.shape[1],
        reverse_coded=tuple(summary.item for summary in summaries if summary.reverse_coded),
        items=tuple(summaries),
        dimensions=tuple(
            _dimension(dimension, members, answers, low, high, quartile_method)
            for dimension, members in grouped.items()
        ),
    )


def _dimension(
    dimension: str,
    members: tuple[str, ...],
    answers: dict[str, np.ndarray],
    low: int,
    high: int,
    quartile_method: str,
) -> DimensionSummary:
    """Summarize one dimension; ``answers`` are already reverse-coded."""
    block = np.column_stack([answers[item] for item in members])
    complete = ~np.isnan(block).any(axis=1)
    kept, excluded = block[complete], int((~complete).sum())
    deleted = [
        _alpha(np.delete(kept, column, axis=1), excluded=excluded) for column in range(len(members))
    ]
    return DimensionSummary(
        dimension=dimension,
        items=members,
        missing=_missing(block),
        frequencies=_frequencies(block, low, high),
        answers=_quartiles(block.ravel(), quartile_method),
        respondent_scores=_quartiles(kept.mean(axis=1), quartile_method),
        alpha=_alpha(kept, excluded=excluded),
        alpha_if_item_deleted=tuple(
            AlphaIfItemDeleted(item, alpha.alpha, alpha.undefined_reason)
            for item, alpha in zip(members, deleted, strict=True)
        ),
    )


def _alpha(complete: np.ndarray, *, excluded: int) -> CronbachAlpha:
    """Return Cronbach's alpha of a ``respondents × items`` matrix with no missing answer."""
    respondents, items = complete.shape
    result = CronbachAlpha(items, respondents, excluded, None, None)
    if items < 2:
        return replace(result, undefined_reason="fewer_than_two_items")
    if respondents < 2:
        return replace(result, undefined_reason="fewer_than_two_respondents")
    total_variance = complete.sum(axis=1).var(ddof=1)
    if total_variance == 0:
        return replace(result, undefined_reason="zero_total_variance")
    item_variances = complete.var(axis=0, ddof=1).sum()
    return replace(result, alpha=float(items / (items - 1) * (1 - item_variances / total_variance)))


def _quartiles(values: np.ndarray, method: str) -> Quartiles:
    given = values[~np.isnan(values)]
    if given.size == 0:
        return Quartiles(0, None, None, None, None, "no_values")
    first, third = np.quantile(given, [0.25, 0.75], method=method)
    return Quartiles(
        count=int(given.size),
        median=float(np.median(given)),
        first_quartile=float(first),
        third_quartile=float(third),
        interquartile_range=float(third - first),
        undefined_reason=None,
    )


def _frequencies(values: np.ndarray, low: int, high: int) -> tuple[tuple[int, int], ...]:
    return tuple((point, int(np.count_nonzero(values == point))) for point in range(low, high + 1))


def _missing(values: np.ndarray) -> int:
    return int(np.isnan(values).sum())


def _answer(item: str, value: object, low: int, high: int) -> float:
    if value is None:
        return np.nan
    if not _is_integer(value):
        raise ValueError(f"item {item!r} has the answer {value!r}, which is not an integer or None")
    if not low <= value <= high:
        raise ValueError(f"item {item!r} has the answer {value}, outside the scale {low} to {high}")
    return float(value)


def _is_integer(value: object) -> bool:
    return isinstance(value, int | np.integer) and not isinstance(value, bool | np.bool_)
