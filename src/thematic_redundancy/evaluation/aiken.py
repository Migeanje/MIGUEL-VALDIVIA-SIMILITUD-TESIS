"""Content validity of the questionnaire items: Aiken's V (T30; plan §3.8.2, PDF p. 82,
printed p. 73).

Before the questionnaire is applied, a group of judges rates each item, for example on its
clarity, relevance or sufficiency, on a scale from ``low`` to ``high`` with ``c`` points. For an
item rated by ``n`` judges, Aiken's (1985) coefficient is

    V = S / (n (c - 1)),   S = sum of (r - low) over the judges' ratings r,

from 0 (every judge gave ``low``) to 1 (every judge gave ``high``). A missing rating is left out,
so ``n`` counts the ratings given.

Its **score confidence interval** (Penfield and Giacobbi, 2004) is Wilson's score interval for
the proportion ``V`` over ``N = n (c - 1)`` trials:

    (2 N V + z^2 -/+ z sqrt(4 N V (1 - V) + z^2)) / (2 (N + z^2)),

with ``z`` the standard normal quantile of ``(1 + confidence) / 2``. The interval always
holds ``V`` and lies in ``[0, 1]``; the bounds are clipped to keep both true under rounding,
so that ``V = 1`` gives an upper bound of exactly 1 and ``V = 0`` a lower bound of exactly 0.

An item is accepted when its ``V`` (``criterion="v"``) or the lower bound of its interval
(``criterion="ci_lower"``) is at least ``cutoff``; the criterion, the cut-off and the confidence
level are explicit arguments with no default. An item with no rating has no ``V``
(``"no_ratings"``), and a dimension with such an item has no mean (``"undefined_items"``).
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np
from scipy.stats import norm

from thematic_redundancy.evaluation.likert import answer_matrix, item_dimensions

AikenCriterion = Literal["v", "ci_lower"]
AIKEN_CRITERIA: tuple[AikenCriterion, ...] = ("v", "ci_lower")


@dataclass(frozen=True)
class AikenItem:
    """Aiken's V of one item, with its score interval."""

    item: str
    dimension: str
    judges: int
    """The ratings given."""
    missing: int
    distance_sum: int
    """``S``, the sum of ``rating - low`` over the ratings given."""
    v: float | None
    ci_lower: float | None
    ci_upper: float | None
    accepted: bool | None
    """Whether the criterion's value is at least the cut-off; ``None`` when ``v`` is."""
    undefined_reason: Literal["no_ratings"] | None


@dataclass(frozen=True)
class AikenDimension:
    """A summary of the V of a dimension's items."""

    dimension: str
    items: tuple[str, ...]
    mean_v: float | None
    minimum_v: float | None
    accepted_items: tuple[str, ...]
    rejected_items: tuple[str, ...]
    undefined_items: tuple[str, ...]
    undefined_reason: Literal["undefined_items"] | None


@dataclass(frozen=True)
class AikenValidity:
    """Aiken's V of every item, in the order of ``dimensions``, and per dimension."""

    low: int
    high: int
    judges: int
    """The ratings per item, missing ones included."""
    confidence: float
    z: float
    criterion: AikenCriterion
    cutoff: float
    items: tuple[AikenItem, ...]
    dimensions: tuple[AikenDimension, ...]


def aiken_v(ratings: Sequence[int | None], *, low: int, high: int) -> float | None:
    """Return Aiken's V of one item's ratings, ``None`` marking a missing rating; ``None`` when
    no rating was given.

    Raises:
        ValueError: if the scale is not ``low < high`` integers, or a rating is neither an
            integer of the scale nor ``None``.
    """
    _, matrix = answer_matrix({"ratings": ratings}, low=low, high=high)
    _, judges, distance_sum = _distances(matrix[0], low)
    return distance_sum / (judges * (high - low)) if judges else None


def aiken_score_interval(
    v: float, *, judges: int, categories: int, confidence: float
) -> tuple[float, float]:
    """Return the score confidence interval of Aiken's ``v`` from ``judges`` ratings on a scale
    of ``categories`` points.

    Raises:
        ValueError: if ``v`` is outside ``[0, 1]``, ``judges`` is below 1, ``categories`` is
            below 2, or ``confidence`` is outside ``(0, 1)``.
    """
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"v must be in [0, 1], not {v!r}")
    if isinstance(judges, bool) or not isinstance(judges, int) or judges < 1:
        raise ValueError(f"judges must be an integer of at least 1, not {judges!r}")
    if isinstance(categories, bool) or not isinstance(categories, int) or categories < 2:
        raise ValueError(f"categories must be an integer of at least 2, not {categories!r}")
    z, v = _z(confidence), float(v)
    trials = judges * (categories - 1)
    centre = 2 * trials * v + z**2
    margin = z * math.sqrt(4 * trials * v * (1 - v) + z**2)
    denominator = 2 * (trials + z**2)
    lower, upper = (centre - margin) / denominator, (centre + margin) / denominator
    return min(v, max(0.0, lower)), max(v, min(1.0, upper))


def aiken_validity(
    ratings: Mapping[str, Sequence[int | None]],
    dimensions: Mapping[str, Sequence[str]],
    *,
    low: int,
    high: int,
    confidence: float,
    criterion: AikenCriterion,
    cutoff: float,
) -> AikenValidity:
    """Return Aiken's V of each item, its score interval and its acceptance, and a summary per
    dimension.

    Args:
        ratings: each item's ratings, one per judge, in the same judge order for every item;
            ``None`` is a missing rating.
        dimensions: each dimension's items; every item is in exactly one dimension.
        low: the lowest point of the rating scale.
        high: the highest point of the rating scale.
        confidence: the level of the score interval, in ``(0, 1)``.
        criterion: ``"v"`` or ``"ci_lower"``, the value compared with the cut-off.
        cutoff: the acceptance cut-off, in ``[0, 1]``.

    Raises:
        ValueError: if an argument is invalid (see :func:`.likert.answer_matrix` and
            :func:`.likert.item_dimensions`).
    """
    items, matrix = answer_matrix(ratings, low=low, high=high)
    grouped = item_dimensions(dimensions, items)
    z = _z(confidence)
    if criterion not in AIKEN_CRITERIA:
        raise ValueError(f"criterion must be one of {AIKEN_CRITERIA}, not {criterion!r}")
    if isinstance(cutoff, bool) or not isinstance(cutoff, int | float) or not 0 <= cutoff <= 1:
        raise ValueError(f"cutoff must be a number in [0, 1], not {cutoff!r}")
    rows = dict(zip(items, matrix, strict=True))
    results = {
        item: _item(item, dimension, rows[item], low, high, confidence, criterion, cutoff)
        for dimension, members in grouped.items()
        for item in members
    }
    return AikenValidity(
        low=low,
        high=high,
        judges=matrix.shape[1],
        confidence=confidence,
        z=z,
        criterion=criterion,
        cutoff=cutoff,
        items=tuple(results.values()),
        dimensions=tuple(
            _dimension(dimension, [results[item] for item in members])
            for dimension, members in grouped.items()
        ),
    )


def _item(
    item: str,
    dimension: str,
    row: np.ndarray,
    low: int,
    high: int,
    confidence: float,
    criterion: AikenCriterion,
    cutoff: float,
) -> AikenItem:
    missing, judges, distance_sum = _distances(row, low)
    result = AikenItem(
        item=item,
        dimension=dimension,
        judges=judges,
        missing=missing,
        distance_sum=distance_sum,
        v=None,
        ci_lower=None,
        ci_upper=None,
        accepted=None,
        undefined_reason="no_ratings",
    )
    if not judges:
        return result
    v = distance_sum / (judges * (high - low))
    lower, upper = aiken_score_interval(
        v, judges=judges, categories=high - low + 1, confidence=confidence
    )
    return replace(
        result,
        v=v,
        ci_lower=lower,
        ci_upper=upper,
        accepted=(v if criterion == "v" else lower) >= cutoff,
        undefined_reason=None,
    )


def _dimension(dimension: str, items: list[AikenItem]) -> AikenDimension:
    values = [item.v for item in items if item.v is not None]
    undefined = tuple(item.item for item in items if item.v is None)
    return AikenDimension(
        dimension=dimension,
        items=tuple(item.item for item in items),
        mean_v=None if undefined else float(np.mean(values)),
        minimum_v=None if undefined else min(values),
        accepted_items=tuple(item.item for item in items if item.accepted is True),
        rejected_items=tuple(item.item for item in items if item.accepted is False),
        undefined_items=undefined,
        undefined_reason="undefined_items" if undefined else None,
    )


def _distances(row: np.ndarray, low: int) -> tuple[int, int, int]:
    """Return the missing ratings, the ratings given, and ``S``, the sum of their distances
    from ``low``, as integers."""
    given = row[~np.isnan(row)]
    return int(row.size - given.size), int(given.size), int(round(float((given - low).sum())))


def _z(confidence: float) -> float:
    """Return the standard normal quantile of ``(1 + confidence) / 2``."""
    if isinstance(confidence, bool) or not isinstance(confidence, float) or not 0 < confidence < 1:
        raise ValueError(f"confidence must be a float in (0, 1), not {confidence!r}")
    return float(norm.ppf((1 + confidence) / 2))
