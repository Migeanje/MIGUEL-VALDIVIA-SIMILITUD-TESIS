"""Rules that choose which items to fetch a thesis file for, and which of their files it is.

They are pure functions of the snapshot metadata and of an item's bundles, so they make no
request. On the UCSM repository the ``ORIGINAL`` bundle of a thesis may hold three PDFs:

- the thesis itself;
- a similarity report, whose name ends in ``.RT.pdf``;
- a publication authorization form, whose name starts with ``Autorización``.

Only the thesis is wanted. :func:`thesis_candidates` drops the other two and returns what
remains, so that the caller can tell one thesis from none or from an ambiguous choice.
"""

import unicodedata
from collections.abc import Iterable, Mapping, Sequence

from pydantic import JsonValue

from thematic_redundancy.corpus.repository import Bitstream, Bundle

ORIGINAL_BUNDLE = "ORIGINAL"
"""Bundle that holds the files deposited with an item; the others hold derived files."""

PDF_SUFFIX = ".pdf"
SIMILARITY_REPORT_SUFFIX = ".rt.pdf"
"""Name ending of a similarity report, compared after case folding."""
AUTHORIZATION_PREFIX = "autoriz"
"""Name start of a publication authorization form, compared without accents or case."""

SELECTION_RULE = (
    f"the only PDF of the {ORIGINAL_BUNDLE} bundle whose name neither ends in '.RT.pdf' nor "
    "starts with 'Autoriz' (ignoring letter case and accents); none or several is recorded "
    "without a download"
)
"""Plain statement of the rule, recorded for provenance with the downloaded files."""

TYPE_KEY = "renati.type"
RIGHTS_KEY = "dc.rights"

RESTRICTED_ACCESS_RIGHTS = {
    "c_f1cf": "embargoed access",
    "c_16ec": "restricted access",
    "c_14cb": "metadata only access",
}
"""COAR access rights under which anonymous users may not read an item's files, by code."""

_COAR_ACCESS_RIGHTS = "purl.org/coar/access_right/"

Metadata = Mapping[str, Sequence[Mapping[str, JsonValue]]]


def thesis_candidates(bundles: Iterable[Bundle]) -> tuple[Bitstream, ...]:
    """Return the files that may be the thesis, in repository order.

    A candidate is a PDF of the ``ORIGINAL`` bundle that is neither a similarity report nor
    an authorization form. Names are judged without surrounding spaces, and PDFs are told by
    their ``.pdf`` ending in any letter case.
    """
    return tuple(
        bitstream
        for bundle in bundles
        if bundle.name == ORIGINAL_BUNDLE
        for bitstream in bundle.bitstreams
        if _is_candidate(bitstream.name)
    )


def is_thesis(metadata: Metadata, thesis_type: str) -> bool:
    """Tell whether one ``renati.type`` value has exactly ``thesis_type`` as its fragment.

    For example, ``https://purl.org/pe-repo/renati/type#tesis`` is a thesis for ``tesis``,
    while ``#Tesis``, ``#tesisDoctoral`` and a value without ``#`` are not.
    """
    for value in _text_values(metadata, TYPE_KEY):
        _, hash_sign, fragment = value.rpartition("#")
        if hash_sign and fragment == thesis_type:
            return True
    return False


def access_restriction(metadata: Metadata) -> str | None:
    """Return the COAR code of the first ``dc.rights`` value that closes the item's files.

    The codes are those of :data:`RESTRICTED_ACCESS_RIGHTS`, such as ``c_f1cf`` for an
    embargo. An item without such a value, open access (``c_abf2``) included, gives ``None``.
    """
    for value in _text_values(metadata, RIGHTS_KEY):
        prefix, _, code = value.rpartition("/")
        if f"{prefix}/".endswith(_COAR_ACCESS_RIGHTS) and code in RESTRICTED_ACCESS_RIGHTS:
            return code
    return None


def _is_candidate(name: str) -> bool:
    folded = _fold(name.strip())
    return (
        folded.endswith(PDF_SUFFIX)
        and not folded.endswith(SIMILARITY_REPORT_SUFFIX)
        and not folded.startswith(AUTHORIZATION_PREFIX)
    )


def _fold(text: str) -> str:
    """Return ``text`` case folded and without accents: ``AUTORIZACIÓN`` gives ``autorizacion``."""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char)).casefold()


def _text_values(metadata: Metadata, key: str) -> list[str]:
    """Return the text values of the metadata field ``key``, without surrounding spaces."""
    return [
        text.strip()
        for entry in metadata.get(key, ())
        if isinstance(text := entry.get("value"), str)
    ]
