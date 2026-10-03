"""Rules that pick the thesis file of an item and tell which items to fetch it for.

Every file name here is synthetic. Real names follow the patterns observed on the UCSM
repository: the thesis, a similarity report ending in ``.RT.pdf``, and a publication
authorization form whose name starts with ``Autorización``.
"""

from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest

from thematic_redundancy.corpus.repository import Bitstream, Bundle, Checksum
from thematic_redundancy.corpus.thesis_files import (
    access_restriction,
    is_thesis,
    thesis_candidates,
)

TESIS = "https://purl.org/pe-repo/renati/type#tesis"
SUFICIENCIA = "https://purl.org/pe-repo/renati/type#trabajoDeSuficienciaProfesional"
OPEN = "https://purl.org/coar/access_right/c_abf2"
EMBARGOED = "https://purl.org/coar/access_right/c_f1cf"
RESTRICTED = "https://purl.org/coar/access_right/c_16ec"
METADATA_ONLY = "https://purl.org/coar/access_right/c_14cb"

THESIS = "70.0001.SYNTHETICNAME.pdf"
REPORT = "70.0001.SYNTHETICNAME.RT.pdf"
AUTHORIZATION = "Autorización_70.0001.SYNTHETICNAME.pdf"


def bundle(name: str, file_names: Sequence[str]) -> Bundle:
    """Return a bundle whose files carry the given names, in that order."""
    files = tuple(
        Bitstream(
            uuid=UUID(f"00000000-0000-4000-8000-{sum(map(ord, name)):04d}0000{number:04d}"),
            name=file_name,
            size_bytes=1_000 + number,
            checksum=Checksum(algorithm="MD5", value=f"{number:032x}"),
        )
        for number, file_name in enumerate(file_names)
    )
    return Bundle(name=name, bitstreams=files)


def candidate_names(*bundles: Bundle) -> list[str]:
    return [file.name for file in thesis_candidates(bundles)]


def entry(value: str) -> dict[str, Any]:
    return {"value": value, "language": None, "authority": None, "confidence": -1, "place": 0}


def test_the_thesis_is_the_pdf_left_after_the_similarity_report_and_the_authorization() -> None:
    bundles = (
        bundle("ORIGINAL", [THESIS, REPORT, AUTHORIZATION]),
        bundle("LICENSE", ["license.txt"]),
        bundle("TEXT", [f"{THESIS}.txt"]),
        bundle("THUMBNAIL", [f"{THESIS}.jpg"]),
    )

    [thesis] = thesis_candidates(bundles)

    assert thesis == bundles[0].bitstreams[0]


@pytest.mark.parametrize(
    "report", ["70.0001.X.RT.pdf", "70.0001.X.rt.pdf", "70.0001.X.Rt.PDF", "  70.0001.X.RT.pdf  "]
)
def test_a_similarity_report_is_never_a_candidate_in_any_letter_case(report: str) -> None:
    assert candidate_names(bundle("ORIGINAL", [THESIS, report])) == [THESIS]


@pytest.mark.parametrize("name", ["70.0001.X_RT.pdf", "Smart.pdf", "70.0001.RTX.pdf"])
def test_only_a_name_ending_in_dot_rt_dot_pdf_counts_as_a_similarity_report(name: str) -> None:
    assert candidate_names(bundle("ORIGINAL", [name])) == [name]


@pytest.mark.parametrize(
    "authorization",
    [
        "Autorización_70.0001.X.pdf",
        "AUTORIZACIÓN_70.0001.X.pdf",
        "autorizacion_70.0001.X.pdf",
        "Autorización_70.0001.X.pdf",  # The accent as a combining character.
        "Áutorización de publicación.PDF",
        " Autorizacion.pdf",
    ],
    ids=["accented", "upper-case", "unaccented", "decomposed-accent", "accented-initial", "padded"],
)
def test_an_authorization_form_is_never_a_candidate_whatever_its_accents_and_case(
    authorization: str,
) -> None:
    assert candidate_names(bundle("ORIGINAL", [authorization, THESIS])) == [THESIS]


def test_a_name_that_only_contains_autoriz_later_on_stays_a_candidate() -> None:
    name = "70.0001.Autorizacion.pdf"

    assert candidate_names(bundle("ORIGINAL", [name])) == [name]


def test_only_pdf_files_of_the_original_bundle_are_candidates() -> None:
    original = bundle("ORIGINAL", ["Datos.xlsx", "Planos.zip", "70.0001.X.pdf.jpg", "TESIS.PDF"])
    others = (bundle("ANEXOS", ["Anexo.pdf"]), bundle("TEXT", ["TESIS.pdf"]))

    assert candidate_names(original, *others) == ["TESIS.PDF"]


@pytest.mark.parametrize(
    "bundles",
    [
        (bundle("ORIGINAL", [REPORT, AUTHORIZATION]), bundle("TEXT", ["x.pdf.txt"])),
        (bundle("ORIGINAL", []),),
        (bundle("LICENSE", ["license.txt"]), bundle("THUMBNAIL", ["x.pdf.jpg"])),
        (),
    ],
    ids=["only-excluded-files", "empty-original", "no-original-bundle", "no-bundles"],
)
def test_no_candidate_remains_when_the_original_bundle_holds_no_thesis(
    bundles: tuple[Bundle, ...],
) -> None:
    assert thesis_candidates(bundles) == ()


def test_several_candidates_are_all_returned_in_repository_order() -> None:
    original = bundle("ORIGINAL", ["Tomo_2.pdf", REPORT, "Tomo_1.pdf"])

    assert candidate_names(original) == ["Tomo_2.pdf", "Tomo_1.pdf"]


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        ([TESIS], True),
        ([SUFICIENCIA, TESIS], True),
        ([SUFICIENCIA], False),
        (["https://purl.org/pe-repo/renati/type#Tesis"], False),
        (["https://purl.org/pe-repo/renati/type#tesisDoctoral"], False),
        (["tesis"], False),
        ([], False),
    ],
    ids=[
        "tesis",
        "tesis-among-others",
        "other-type",
        "other-case",
        "longer",
        "no-fragment",
        "none",
    ],
)
def test_an_item_is_a_thesis_only_with_the_exact_type_fragment(
    types: list[str], expected: bool
) -> None:
    metadata = {"renati.type": [entry(value) for value in types]} if types else {}

    assert is_thesis(metadata, "tesis") is expected


@pytest.mark.parametrize(
    ("rights", "expected"),
    [
        ([OPEN], None),
        ([EMBARGOED], "c_f1cf"),
        ([RESTRICTED], "c_16ec"),
        ([METADATA_ONLY], "c_14cb"),
        (["http://purl.org/coar/access_right/c_f1cf"], "c_f1cf"),
        ([OPEN, RESTRICTED], "c_16ec"),
        ([], None),
    ],
    ids=["open", "embargoed", "restricted", "metadata-only", "http", "mixed", "missing"],
)
def test_access_restriction_names_the_coar_access_right_that_closes_the_file(
    rights: list[str], expected: str | None
) -> None:
    metadata = {"dc.rights": [entry(value) for value in rights]} if rights else {}

    assert access_restriction(metadata) == expected
