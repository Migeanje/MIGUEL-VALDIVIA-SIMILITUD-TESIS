"""Agreement between the two annotators of each labeled pair, and their consensus (T21, D15).

- **Cohen's kappa** (Cohen, 1960) over nominal labels: ``(p_o - p_e) / (1 - p_e)``, where
  ``p_o`` is the observed share of agreement and ``p_e`` the agreement expected by chance from
  the two annotators' own marginal distributions.
- **Weighted kappa** (Cohen, 1968) for an ordered scale, with disagreement weights
  ``|i - j| / (k - 1)`` (``"linear"``) or ``((i - j) / (k - 1)) ** 2`` (``"quadratic"``) between
  the positions ``i`` and ``j`` of two categories in ``categories``, ``k`` of them. Then
  ``p_o`` and ``p_e`` are one minus the observed and the expected mean disagreement weight, and
  ``kappa = 1 - D_o / D_e``, which is scikit-learn's ``cohen_kappa_score`` with the same
  ``weights``. With ``"none"`` every disagreement weighs 1, which is the unweighted kappa.

The categories are an explicit, ordered argument, so the contingency table always has the same
shape and an unused category is an empty row and column. Cohen's kappa compares the same two
annotators over every item; when the annotators rotate across the pairs, the caller decides how
to apply it (for example per pair of annotators).

Undefined values follow the package's convention, ``None`` with a reason: ``"no_items"`` for no
labeled item, and ``"expected_agreement_is_one"`` when both annotators gave every item one same
category, so that no disagreement could be expected by chance (``p_e = 1``, kappa is 0 / 0).

The **consensus** keeps the label of each pair on which both annotators agree and lists every
other pair for adjudication, with both labels. It never breaks a tie on its own, since the plan
defines no automatic rule (T21; R3 plans the re-labeling of disputed pairs).
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

Label = str | int
"""A category of the labeling scale: a string or an integer, never a boolean."""

KappaWeights = Literal["none", "linear", "quadratic"]
KAPPA_WEIGHTS: tuple[KappaWeights, ...] = ("none", "linear", "quadratic")

KappaUndefined = Literal["no_items", "expected_agreement_is_one"]


@dataclass(frozen=True)
class CohenKappa:
    """Cohen's kappa of two annotators, with its parts."""

    categories: tuple[Label, ...]
    weights: KappaWeights
    items: int
    confusion: tuple[tuple[int, ...], ...]
    """Counts of items, ``categories × categories``: the row is the first annotator's label and
    the column the second's."""
    observed_agreement: float | None
    """``p_o``, weighted when ``weights`` is not ``"none"``; ``None`` when there is no item."""
    expected_agreement: float | None
    """``p_e``, the chance agreement from the two marginal distributions; ``None`` when there is
    no item."""
    kappa: float | None
    undefined_reason: KappaUndefined | None


@dataclass(frozen=True)
class AgreedLabel:
    """A pair on which both annotators gave the same label."""

    pair_id: str
    label: Label


@dataclass(frozen=True)
class Disagreement:
    """A pair whose two labels differ, so it needs adjudication."""

    pair_id: str
    first: Label
    second: Label


@dataclass(frozen=True)
class Consensus:
    """The agreed labels, and the pairs left for adjudication, both in the input order; each
    label is given as its category in ``categories``."""

    agreed: tuple[AgreedLabel, ...]
    needs_adjudication: tuple[Disagreement, ...]


def cohen_kappa(
    first: Sequence[Label],
    second: Sequence[Label],
    *,
    categories: Sequence[Label],
    weights: KappaWeights,
) -> CohenKappa:
    """Return Cohen's kappa of two annotators who labeled the same items in the same order.

    Args:
        first: the first annotator's label of each item.
        second: the second annotator's label of each item.
        categories: every category of the scale, at least two, without repeats; for a weighted
            kappa, in the order of the scale.
        weights: ``"none"`` for nominal labels, ``"linear"`` or ``"quadratic"`` for an ordered
            scale.

    Raises:
        ValueError: if ``weights`` is unknown, the categories are invalid, a label is not one of
            them, or the two annotators labeled a different number of items.
    """
    if weights not in KAPPA_WEIGHTS:
        raise ValueError(f"weights must be one of {KAPPA_WEIGHTS}, not {weights!r}")
    scale = _check_categories(categories)
    if len(first) != len(second):
        raise ValueError(
            f"the first annotator has {len(first)} labels but the second annotator has "
            f"{len(second)}"
        )
    rows, columns = _positions(first, scale), _positions(second, scale)
    size = len(scale)
    confusion = np.zeros((size, size), dtype=np.int64)
    np.add.at(confusion, (rows, columns), 1)
    result = CohenKappa(
        categories=tuple(scale),
        weights=weights,
        items=len(rows),
        confusion=tuple(tuple(int(count) for count in row) for row in confusion),
        observed_agreement=None,
        expected_agreement=None,
        kappa=None,
        undefined_reason="no_items",
    )
    if not rows:
        return result

    disagreement = _disagreement_weights(size, weights)
    # The chance table is the product of the first annotator's row margins and the second
    # annotator's column margins, in counts: ``items²`` times the expected proportions.
    chance = np.outer(confusion.sum(axis=1), confusion.sum(axis=0))
    observed = float((disagreement * confusion).sum()) / len(rows)
    expected = float((disagreement * chance).sum()) / len(rows) ** 2
    # Every weight off the diagonal is positive, so no disagreement is expected by chance
    # exactly when both annotators gave one same category; checked on integer counts.
    no_chance_disagreement = not chance[~np.eye(size, dtype=bool)].any()
    return replace(
        result,
        observed_agreement=1.0 - observed,
        expected_agreement=1.0 - expected,
        kappa=None if no_chance_disagreement else 1.0 - observed / expected,
        undefined_reason="expected_agreement_is_one" if no_chance_disagreement else None,
    )


def consensus_labels(
    pair_ids: Sequence[str],
    first: Sequence[Label],
    second: Sequence[Label],
    *,
    categories: Sequence[Label],
) -> Consensus:
    """Return the label of each pair on which both annotators agree, and the other pairs for
    adjudication.

    Raises:
        ValueError: if a pair id is empty, not a string or repeated, the categories are invalid,
            a label is not one of them, or the three sequences differ in length.
    """
    scale = _check_categories(categories)
    if not len(pair_ids) == len(first) == len(second):
        raise ValueError(
            f"there are {len(pair_ids)} pair ids but {len(first)} labels from the first "
            f"annotator and {len(second)} from the second"
        )
    seen: set[str] = set()
    for pair_id in pair_ids:
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError(f"a pair id must be a non-empty string, not {pair_id!r}")
        if pair_id in seen:
            raise ValueError(f"the pair id {pair_id!r} is repeated")
        seen.add(pair_id)
    labels = tuple(scale)
    agreed, disputed = [], []
    for pair_id, one, other in zip(
        pair_ids, _positions(first, scale), _positions(second, scale), strict=True
    ):
        if one == other:
            agreed.append(AgreedLabel(pair_id, labels[one]))
        else:
            disputed.append(Disagreement(pair_id, labels[one], labels[other]))
    return Consensus(agreed=tuple(agreed), needs_adjudication=tuple(disputed))


def _disagreement_weights(size: int, weights: KappaWeights) -> np.ndarray:
    """Return the ``size × size`` disagreement weights, 0 on the diagonal and at most 1."""
    positions = np.arange(size)
    distance = np.abs(np.subtract.outer(positions, positions)) / (size - 1)
    if weights == "linear":
        return distance
    if weights == "quadratic":
        return distance**2
    return (distance > 0).astype(np.float64)


def _check_categories(categories: Sequence[Label]) -> dict[Label, int]:
    """Return each category's position, after checking that there are at least two, that
    none repeats and that none is a boolean."""
    scale: dict[Label, int] = {}
    for category in categories:
        if isinstance(category, bool | np.bool_) or not isinstance(
            category, str | int | np.integer
        ):
            raise ValueError(
                f"a category must be a string or a non-boolean integer, not {category!r}"
            )
        if category in scale:
            raise ValueError(f"the category {category!r} repeats")
        scale[category] = len(scale)
    if len(scale) < 2:
        raise ValueError(f"there must be at least two categories, not {len(scale)}")
    return scale


def _positions(labels: Sequence[Label], scale: dict[Label, int]) -> list[int]:
    """Return the position of each label in the scale."""
    positions = []
    for label in labels:
        position = None if isinstance(label, bool | np.bool_) else scale.get(label)
        if position is None:
            raise ValueError(f"the label {label!r} is not one of the categories {tuple(scale)}")
        positions.append(position)
    return positions
