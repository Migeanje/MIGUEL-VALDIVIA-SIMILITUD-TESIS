"""Known true duplicates among document pairs (T22, O09).

Some pairs of included theses are known to be one thesis deposited twice (O09: a re-deposit in
the repository, kept by D24's exact-match rule). They stay in the labeled pairs and in the
search, and the results flag them. A pair is flagged when it joins the two documents of a
listed duplicate, in either order. The list comes from the configuration
(``redundancy.known_duplicates``) or the caller; no thesis text is involved.
"""

from collections.abc import Iterable, Sequence


def known_duplicate_flags(
    pairs: Sequence[tuple[str, str]], known: Iterable[tuple[str, str]]
) -> tuple[bool, ...]:
    """Return, for each of ``pairs``, whether it is one of the ``known`` duplicates.

    Raises:
        ValueError: if a pair, given or known, does not join two different documents.
    """
    listed = {_unordered(pair) for pair in known}
    return tuple(_unordered(pair) in listed for pair in pairs)


def _unordered(pair: tuple[str, str]) -> frozenset[str]:
    if len(pair) != 2:
        raise ValueError(f"a pair must hold two document codes, not {pair!r}")
    if pair[0] == pair[1]:
        raise ValueError(f"a pair must join two different documents, not {pair!r}")
    return frozenset(pair)
