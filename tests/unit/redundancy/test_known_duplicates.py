"""Known true duplicates among document pairs (T22, O09): a pair is flagged when it joins the
two documents of a listed duplicate, in either order. The codes are made up; the real pair is
read from the configuration."""

import pytest

from thematic_redundancy.redundancy.known_duplicates import known_duplicate_flags

KNOWN = [("DOC-aaaaaaaa", "DOC-bbbbbbbb")]


def test_a_listed_pair_is_flagged_in_either_order() -> None:
    pairs = [
        ("DOC-aaaaaaaa", "DOC-bbbbbbbb"),
        ("DOC-bbbbbbbb", "DOC-aaaaaaaa"),
        ("DOC-aaaaaaaa", "DOC-cccccccc"),
        ("DOC-cccccccc", "DOC-bbbbbbbb"),
    ]

    assert known_duplicate_flags(pairs, KNOWN) == (True, True, False, False)


def test_a_listed_pair_given_in_reverse_order_still_flags() -> None:
    reverse = [("DOC-bbbbbbbb", "DOC-aaaaaaaa")]

    assert known_duplicate_flags([("DOC-aaaaaaaa", "DOC-bbbbbbbb")], reverse) == (True,)


def test_several_listed_pairs_are_all_found() -> None:
    known = [*KNOWN, ("DOC-cccccccc", "DOC-dddddddd")]
    pairs = [("DOC-dddddddd", "DOC-cccccccc"), ("DOC-aaaaaaaa", "DOC-dddddddd")]

    assert known_duplicate_flags(pairs, known) == (True, False)


def test_no_listed_pair_flags_nothing() -> None:
    assert known_duplicate_flags([("DOC-aaaaaaaa", "DOC-bbbbbbbb")], []) == (False,)
    assert known_duplicate_flags([], KNOWN) == ()


@pytest.mark.parametrize(
    ("pairs", "known", "message"),
    [
        ([("DOC-aaaaaaaa", "DOC-aaaaaaaa")], KNOWN, "different"),
        ([("DOC-aaaaaaaa",)], KNOWN, "two"),
        ([("DOC-aaaaaaaa", "DOC-bbbbbbbb")], [("DOC-cccccccc", "DOC-cccccccc")], "different"),
        ([("DOC-aaaaaaaa", "DOC-bbbbbbbb")], [("DOC-a", "DOC-b", "DOC-c")], "two"),
    ],
)
def test_a_pair_must_join_two_different_documents(pairs, known, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        known_duplicate_flags(pairs, known)
