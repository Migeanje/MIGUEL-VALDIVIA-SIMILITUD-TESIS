"""Import the labeled set's workbooks (T20c, D31, D34 (9)): each annotator's labels, their
agreement, the adjudication of the disputed pairs, and the consensus labels for T22.

Three steps of :mod:`.pair_workbooks`' command, run after its ``main`` step:

- ``import-main`` reads every annotator's workbook back (:func:`read_workbooks`) and refuses them
  all, listing every problem of every workbook, unless each holds its pairs unchanged and a
  label on every pair: an incomplete workbook is refused (T21 N4), and so is a file that is not a
  readable xlsx workbook, ``zlib.error`` and ``EOFError`` included (T10c). It stores one row per
  pair and annotator (``labels/labels.parquet``: the annotator, the pair, the 3-level label and
  the binary one) and the Fleiss' kappa inputs (``labels/fleiss_inputs.parquet``), and reports
  the agreement. With no disputed pair, it also writes the consensus labels.
- ``adjudication`` gives each disputed pair to an annotator who did not label it
  (:func:`assign_adjudicators`) and makes one short workbook per adjudicator, which does not show
  the earlier labels, with the same blind codes, sides, layout and name guard.
- ``import-adjudication`` reads those workbooks back as ``import-main`` reads the annotators',
  stores the third labels (``labels/adjudication_labels.parquet``) and writes the consensus
  labels (:mod:`.consensus`).

**Agreement (D31).** On the first two labels of every pair only (D34 (9)), with
:mod:`~thematic_redundancy.evaluation.agreement`: Fleiss' kappa over all the labeled pairs on the
binary labels, the one checked against D15's target of :data:`KAPPA_TARGET`; Fleiss' kappa on the
3 levels, a complement; and Cohen's kappa per pair of annotators over the pairs they share,
binary and 3 levels, unweighted, a diagnostic. The tracked manifest holds each with its parts,
and how many pairs got each combination of two labels, from which Fleiss' kappa can be computed
again; no pair id or blind code.

**Disputes.** A pair is disputed when its two labels differ on the 3 levels, not only on the
binary label, since the consensus needs a level for every pair; T22 uses the binary label, so
a pair disputed on the levels alone (``Relacionado`` against ``No relacionado``) changes no
binary label. Its final level is the majority of its three labels, or their median on the
scale when all three differ (:func:`~.consensus.final_level`).

**Adjudicators (D34 (9)).** With 3 annotators, the annotator outside the pair's two; with 4, one
of the two others, rotated: the disputed pairs, sorted by pair id and shuffled with
``random.Random("<seed>:adjudication")``, go one by one to the candidate with fewer pairs so far,
the lower number on a tie.
"""

import random
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import openpyxl
from pydantic import BaseModel, ConfigDict, model_validator

from thematic_redundancy.evaluation.agreement import (
    CohenKappa,
    Consensus,
    FleissKappa,
    cohen_kappa,
    consensus_labels,
    fleiss_kappa,
)
from thematic_redundancy.labeling.consensus import (
    ADJUDICATION,
    AGREEMENT,
    CONSENSUS_FILE_NAME,
    LABELS_DIR_NAME,
    LEVELS,
    ConsensusLabel,
    Level,
    binary_label,
    final_level,
)
from thematic_redundancy.labeling.draw_pairs import PairRow
from thematic_redundancy.labeling.objectives_check import _NameRedactor
from thematic_redundancy.labeling.pair_workbooks import (
    _COMPARED,
    _JUDGE,
    ASSIGNMENT_FILE_NAME,
    MAIN_LAYOUT_FILE_NAME,
    MAIN_MANIFEST_FILE_NAME,
    RUN_COMMAND,
    AssignmentRow,
    LayoutRow,
    Places,
    Prepared,
    Run,
    Step,
    Table,
    WorkbookInputs,
    WorkbooksError,
    _guard_line,
    _manifest_at,
    _read_recorded,
    _scale_lines,
    _sha256,
    _shared_lines,
    annotator_file_name,
    layout_rows,
    pair_views,
    pilot_labels,
    pilot_records,
)
from thematic_redundancy.labeling.pairs import LABELED
from thematic_redundancy.labeling.workbook import (
    BINARY,
    EXAMPLES_SHEET,
    LABELING_SHEET,
    MAIN_LAYOUT,
    PILOT_LAYOUT,
    LabelingWorkbookError,
    ReadLabel,
    SheetContent,
    build_workbook,
    guard_workbooks,
    label_text,
    read_labels,
    screen_notes,
)

KAPPA_TARGET = 0.6
"""D15's target for the agreement, checked as Fleiss' kappa on the binary labels (D31 (a))."""
BINARY_CATEGORIES = (1, 0)
"""The binary labels, as the kappas count them: redundant, then not."""
LABELS_FILE_NAME = "labels.parquet"
FLEISS_INPUTS_FILE_NAME = "fleiss_inputs.parquet"
ADJUDICATION_LABELS_FILE_NAME = "adjudication_labels.parquet"
ADJUDICATION_LAYOUT_FILE_NAME = "adjudication_layout.parquet"
MAIN_LABELS_MANIFEST_FILE_NAME = "main_labels_manifest.json"
ADJUDICATION_MANIFEST_FILE_NAME = "adjudication_manifest.json"
ADJUDICATION_LABELS_MANIFEST_FILE_NAME = "adjudication_labels_manifest.json"


def in_labels(name: str) -> str:
    """Return ``name``'s path under the records directory: in its ``labels/`` folder."""
    return f"{LABELS_DIR_NAME}/{name}"


def adjudication_workbook(adjudicator: int) -> str:
    """Return the key of an adjudication workbook in the layout records and the order's seed."""
    return f"adjudication-{adjudicator}"


ADJUDICATION_FILE_PATTERN = "desempate_anotador_*.xlsx"


def adjudication_file_name(adjudicator: int) -> str:
    """Return the file name of an adjudication workbook, such as ``desempate_anotador_3.xlsx``."""
    return ADJUDICATION_FILE_PATTERN.replace("*", str(adjudicator))


# --- Records --------------------------------------------------------------------------------


class _Row(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)


class AnnotatorLabelRow(_Row):
    """One row of the gitignored ``labels/labels.parquet`` (or ``adjudication_labels.parquet``):
    one annotator's label of one pair."""

    pair_id: str
    code: str
    annotator: int
    label_3: Level
    """The level, as the workbook offers it."""
    label: int
    """T22's binary label: 1 for ``Redundante`` only (D31 (b))."""
    note: str | None

    @model_validator(mode="after")
    def _label_follows_the_level(self) -> Self:
        if self.label != binary_label(self.label_3):
            raise ValueError("label must be 1 when label_3 is Redundante and 0 otherwise")
        return self


class FleissInputRow(_Row):
    """One row of the gitignored ``labels/fleiss_inputs.parquet``: how many of a pair's first two
    labels fell in each category, the input of Fleiss' kappa."""

    pair_id: str
    annotator_a: int
    annotator_b: int
    redundant: int
    related: int
    unrelated: int
    binary_1: int
    binary_0: int


# --- Reading the returned workbooks -----------------------------------------------------------


def read_workbooks(
    places: Places,
    layout: Sequence[LayoutRow],
    file_name: Callable[[int], str],
    *,
    redact: Callable[[str], str],
    guard: frozenset[str],
) -> dict[int, tuple[ReadLabel, ...]]:
    """Read the labels of each workbook of ``layout``, by its annotator, from the workbook named
    ``file_name(annotator)`` in the workbooks' folder; each note goes through ``redact`` and the
    name guard of ``guard`` (:func:`~.workbook.screen_notes`).

    Raises:
        LabelingWorkbookError: listing every problem of every workbook, each after its file
            name: a missing or unreadable file, a changed header, a pair lost, repeated or
            added, a key column edited, a label blank (an incomplete workbook) or not one of
            the scale, or a note with a name the redactor leaves (never echoed).
    """
    problems: list[str] = []
    found: dict[int, tuple[ReadLabel, ...]] = {}
    for annotator in sorted({row.annotator for row in layout if row.annotator is not None}):
        name = file_name(annotator)
        path = places.workbooks / name
        expected = {row.code: row.n for row in layout if row.annotator == annotator}
        try:
            read = read_labels(path, MAIN_LAYOUT, expected)
        except LabelingWorkbookError as error:
            problems += [f"{name}: {problem}" for problem in error.problems]
        except FileNotFoundError:
            problems.append(f"{name} is missing; save the returned workbook as {path}")
        except ValueError as error:  # not a readable xlsx workbook
            problems.append(" ".join(str(error).split()))
        else:
            screened, named = screen_notes(read, redact, guard, column=MAIN_LAYOUT.note_column)
            found[annotator] = screened
            problems += [f"{name}: {problem}" for problem in named]
    if problems:
        raise LabelingWorkbookError(problems)
    return found


def label_rows(
    labels: Mapping[int, Sequence[ReadLabel]], layout: Sequence[LayoutRow]
) -> tuple[AnnotatorLabelRow, ...]:
    """Return one row per label read, by annotator then in workbook order."""
    pair_of = {(row.annotator, row.code): row.pair_id for row in layout}
    return tuple(
        AnnotatorLabelRow(
            pair_id=pair_of[(annotator, label.code)],
            code=label.code,
            annotator=annotator,
            label_3=label_text(label.level),  # type: ignore[arg-type]
            label=BINARY[label.level],
            note=label.note,
        )
        for annotator, read in sorted(labels.items())
        for label in read
    )


# --- Agreement --------------------------------------------------------------------------------


@dataclass(frozen=True)
class Agreement:
    """The first two labels of every labeled pair and what they agree on."""

    pairs: tuple[AssignmentRow, ...]
    """The labeled pairs with their two annotators, in the assignment's order."""
    first: tuple[AnnotatorLabelRow, ...]
    """Each pair's label by its first annotator (``annotator_a``)."""
    second: tuple[AnnotatorLabelRow, ...]
    fleiss_binary: FleissKappa
    fleiss_levels: FleissKappa
    cohen: tuple[tuple[tuple[int, int], CohenKappa, CohenKappa], ...]
    """For each pair of annotators: the pair, Cohen's kappa on the binary labels and on the 3
    levels, over the pairs they share."""
    consensus: Consensus
    """The pairs agreed on the 3 levels, and the disputed ones."""

    @property
    def disputed(self) -> tuple[AssignmentRow, ...]:
        disputed = {d.pair_id for d in self.consensus.needs_adjudication}
        return tuple(pair for pair in self.pairs if pair.pair_id in disputed)


def measure_agreement(
    labels: Sequence[AnnotatorLabelRow], assignment: Sequence[AssignmentRow]
) -> Agreement:
    """Return the agreement of the two annotators of each pair of ``assignment``.

    Raises:
        ValueError: if a pair lacks the label of one of its two annotators, or has another.
    """
    by_key = {(row.annotator, row.pair_id): row for row in labels}
    expected = {(a, p.pair_id) for p in assignment for a in (p.annotator_a, p.annotator_b)}
    if set(by_key) != expected or len(by_key) != len(labels):
        raise ValueError("the labels are not those of each pair's two annotators")
    first = tuple(by_key[(p.annotator_a, p.pair_id)] for p in assignment)
    second = tuple(by_key[(p.annotator_b, p.pair_id)] for p in assignment)
    cohen = []
    for group in sorted({(p.annotator_a, p.annotator_b) for p in assignment}):
        mine = [i for i, p in enumerate(assignment) if (p.annotator_a, p.annotator_b) == group]
        cohen.append(
            (
                group,
                cohen_kappa(
                    [first[i].label for i in mine],
                    [second[i].label for i in mine],
                    categories=BINARY_CATEGORIES,
                    weights="none",
                ),
                cohen_kappa(
                    [first[i].label_3 for i in mine],
                    [second[i].label_3 for i in mine],
                    categories=LEVELS,
                    weights="none",
                ),
            )
        )
    return Agreement(
        pairs=tuple(assignment),
        first=first,
        second=second,
        fleiss_binary=fleiss_kappa(
            [(a.label, b.label) for a, b in zip(first, second, strict=True)],
            categories=BINARY_CATEGORIES,
        ),
        fleiss_levels=fleiss_kappa(
            [(a.label_3, b.label_3) for a, b in zip(first, second, strict=True)],
            categories=LEVELS,
        ),
        cohen=tuple(cohen),
        consensus=consensus_labels(
            [p.pair_id for p in assignment],
            [row.label_3 for row in first],
            [row.label_3 for row in second],
            categories=LEVELS,
        ),
    )


def fleiss_inputs(agreement: Agreement) -> tuple[FleissInputRow, ...]:
    """Return each pair's counts of its first two labels per category."""
    rows = []
    for pair, a, b in zip(agreement.pairs, agreement.first, agreement.second, strict=True):
        levels = Counter((a.label_3, b.label_3))
        binary = Counter((a.label, b.label))
        rows.append(
            FleissInputRow(
                pair_id=pair.pair_id,
                annotator_a=pair.annotator_a,
                annotator_b=pair.annotator_b,
                redundant=levels[LEVELS[0]],
                related=levels[LEVELS[1]],
                unrelated=levels[LEVELS[2]],
                binary_1=binary[1],
                binary_0=binary[0],
            )
        )
    return tuple(rows)


def agreement_details(agreement: Agreement) -> dict[str, Any]:
    """Return the agreement as the tracked manifest holds it: counts and kappas, no pair id."""
    binary = agreement.fleiss_binary.kappa

    def combination(one: Any, other: Any, order: Sequence[Any]) -> str:
        return "+".join(str(value) for value in sorted((one, other), key=list(order).index))

    levels = Counter(
        combination(a.label_3, b.label_3, LEVELS)
        for a, b in zip(agreement.first, agreement.second, strict=True)
    )
    binaries = Counter(
        combination(a.label, b.label, BINARY_CATEGORIES)
        for a, b in zip(agreement.first, agreement.second, strict=True)
    )
    by_annotator: dict[str, Counter[str]] = {}
    for row in (*agreement.first, *agreement.second):
        by_annotator.setdefault(str(row.annotator), Counter())[row.label_3] += 1
    return {
        "kappa_target": KAPPA_TARGET,
        "meets_target": None if binary is None else binary >= KAPPA_TARGET,
        "fleiss_binary": _fleiss_record(agreement.fleiss_binary),
        "fleiss_levels": _fleiss_record(agreement.fleiss_levels),
        "cohen": [
            {
                "annotators": list(group),
                "pairs": on_binary.items,
                "binary": _cohen_record(on_binary),
                "levels": _cohen_record(on_levels),
            }
            for group, on_binary, on_levels in agreement.cohen
        ],
        "combinations": {
            "binary": {key: binaries[key] for key in sorted(binaries)},
            "levels": {key: levels[key] for key in sorted(levels)},
        },
        "levels_by_annotator": {
            annotator: {level: counts[level] for level in LEVELS}
            for annotator, counts in sorted(by_annotator.items(), key=lambda entry: int(entry[0]))
        },
    }


def _fleiss_record(kappa: FleissKappa) -> dict[str, Any]:
    return {
        "items": kappa.items,
        "raters_per_item": kappa.raters_per_item,
        "categories": list(kappa.categories),
        "category_shares": None if kappa.category_shares is None else list(kappa.category_shares),
        "observed_agreement": kappa.observed_agreement,
        "expected_agreement": kappa.expected_agreement,
        "kappa": kappa.kappa,
        "undefined_reason": kappa.undefined_reason,
    }


def _cohen_record(kappa: CohenKappa) -> dict[str, Any]:
    return {
        "categories": list(kappa.categories),
        "confusion": [list(row) for row in kappa.confusion],
        "observed_agreement": kappa.observed_agreement,
        "expected_agreement": kappa.expected_agreement,
        "kappa": kappa.kappa,
        "undefined_reason": kappa.undefined_reason,
    }


def _kappa_text(kappa: float | None, reason: str | None) -> str:
    return f"{kappa:.3f}" if kappa is not None else f"undefined ({reason})"


def agreement_lines(agreement: Agreement) -> list[str]:
    """Describe the agreement in counts and kappas, without a pair id."""
    binary = agreement.fleiss_binary
    verdict = (
        "undefined"
        if binary.kappa is None
        else ("met" if binary.kappa >= KAPPA_TARGET else "not met")
    )
    disputed = len(agreement.consensus.needs_adjudication)
    on_binary = sum(
        a.label != b.label for a, b in zip(agreement.first, agreement.second, strict=True)
    )
    return [
        f"Fleiss' kappa on the binary labels: {_kappa_text(binary.kappa, binary.undefined_reason)} "
        f"over {binary.items} pairs (target {KAPPA_TARGET}: {verdict})",
        "Fleiss' kappa on the 3 levels: "
        + _kappa_text(agreement.fleiss_levels.kappa, agreement.fleiss_levels.undefined_reason),
        *(
            f"  Cohen's kappa, annotators {a} and {b}, {on_bin.items} pairs: binary "
            f"{_kappa_text(on_bin.kappa, on_bin.undefined_reason)}, 3 levels "
            f"{_kappa_text(on_lev.kappa, on_lev.undefined_reason)}"
            for (a, b), on_bin, on_lev in agreement.cohen
        ),
        f"Agreed on the level: {len(agreement.consensus.agreed)} pairs; disputed: {disputed} "
        f"({on_binary} of them on the binary label too)",
    ]


# --- Adjudication ---------------------------------------------------------------------------


def assign_adjudicators(
    disputed: Sequence[AssignmentRow], annotators: int, *, seed: int
) -> dict[str, int]:
    """Give each disputed pair to an annotator who did not label it, as the module docstring
    describes; return the adjudicator of each pair id.

    Raises:
        ValueError: if a pair has no annotator left to adjudicate it.
    """
    order = sorted(pair.pair_id for pair in disputed)
    random.Random(f"{seed}:adjudication").shuffle(order)
    by_id = {pair.pair_id: pair for pair in disputed}
    load: Counter[int] = Counter()
    chosen: dict[str, int] = {}
    for identifier in order:
        pair = by_id[identifier]
        candidates = [
            k for k in range(1, annotators + 1) if k not in (pair.annotator_a, pair.annotator_b)
        ]
        if not candidates:
            raise ValueError("a disputed pair has no annotator left to adjudicate it")
        adjudicator = min(candidates, key=lambda k: (load[k], k))
        load[adjudicator] += 1
        chosen[identifier] = adjudicator
    return chosen


def adjudication_guide(adjudicator: int, pairs: int, examples: int) -> list[tuple[str, str]]:
    """Return the lines of an adjudication workbook's guide."""
    counted = f"{pairs} par{'es' if pairs != 1 else ''}"
    return [
        (f"Etiquetado de pares de tesis: desempate, anotador {adjudicator}", ""),
        (
            "Propósito",
            f"Este libro reúne {counted} en los que los dos anotadores que los etiquetaron "
            "eligieron niveles distintos. Usted no etiquetó estos pares. Su etiqueta y las dos "
            "anteriores deciden por mayoría la etiqueta final de cada par.",
        ),
        _COMPARED,
        *_scale_lines(),
        (
            "Ejemplos",
            f"La hoja «{EXAMPLES_SHEET}» muestra los {examples} pares del piloto con la etiqueta "
            "que se acordó en la reunión de calibración y su justificación. Úselos como "
            "referencia.",
        ),
        ("Cómo etiquetar", ""),
        ("1.", f"Vaya a la hoja «{LABELING_SHEET}». Cada par tiene un número y un código."),
        _JUDGE,
        (
            "3.",
            "Elija una etiqueta en «Etiqueta» con la lista desplegable. «Notas (opcional)» "
            "admite cualquier comentario.",
        ),
        (
            "4.",
            "Etiquete todos los pares: la importación rechaza un libro con pares sin etiqueta. "
            "Guarde el archivo con el mismo nombre y en la misma carpeta.",
        ),
        (
            "Trabajo individual",
            "El libro no muestra las etiquetas anteriores. Etiquete cada par por su cuenta, sin "
            "consultarlas ni hablar de estos pares con los otros anotadores.",
        ),
        *_shared_lines(),
    ]


# --- The steps --------------------------------------------------------------------------------


@dataclass(frozen=True)
class MainRecords:
    """The records of the main workbooks, each checked against their manifest."""

    annotators: int
    layout: tuple[LayoutRow, ...]
    assignment: tuple[AssignmentRow, ...]


def main_records(places: Places) -> MainRecords:
    """Return the main workbooks' layout and assignment, checked against their manifest.

    Raises:
        WorkbooksError: if the main workbooks were never made.
        ValueError: if a record differs from the manifest.
    """
    path = places.results / MAIN_MANIFEST_FILE_NAME
    if not path.is_file():
        raise WorkbooksError(
            f"the annotators' workbooks of snapshot {places.snapshot_id} were never made ({path} "
            f"is missing); run {RUN_COMMAND} main first"
        )
    manifest = _manifest_at(path)
    return MainRecords(
        annotators=manifest.counts["annotators"],
        layout=_read_recorded(LayoutRow, places.records, MAIN_LAYOUT_FILE_NAME, manifest, path),
        assignment=_read_recorded(
            AssignmentRow, places.records, ASSIGNMENT_FILE_NAME, manifest, path
        ),
    )


def main_labels(places: Places) -> tuple[AnnotatorLabelRow, ...]:
    """Return the annotators' labels, checked against their manifest.

    Raises:
        WorkbooksError: if the annotators' workbooks were never imported.
        ValueError: if the record differs from the manifest.
    """
    path = places.results / MAIN_LABELS_MANIFEST_FILE_NAME
    if not path.is_file():
        raise WorkbooksError(
            f"the annotators' labels of snapshot {places.snapshot_id} are not imported ({path} "
            f"is missing); run {RUN_COMMAND} import-main first"
        )
    manifest = _manifest_at(path)
    return _read_recorded(
        AnnotatorLabelRow, places.records, in_labels(LABELS_FILE_NAME), manifest, path
    )


def consensus_rows(
    inputs: WorkbookInputs,
    agreement: Agreement,
    finals: Mapping[str, str] | None = None,
) -> tuple[ConsensusLabel, ...]:
    """Return the consensus label of every pair of the labeled set, in draw order: the agreed
    level, or the final level of ``finals`` for a disputed pair.

    Raises:
        ValueError: if a labeled pair has no final label, or a final label is of another pair.
    """
    agreed = {label.pair_id: str(label.label) for label in agreement.consensus.agreed}
    finals = dict(finals or {})
    labeled = [pair.pair_id for pair in inputs.pairs if pair.phase == LABELED]
    if set(agreed) | set(finals) != set(labeled) or set(agreed) & set(finals):
        raise ValueError(
            "the consensus would not hold exactly one label for every pair of the labeled set"
        )
    rows = []
    for identifier in labeled:
        level = agreed.get(identifier) or finals[identifier]
        rows.append(
            ConsensusLabel(
                pair_id=identifier,
                label_3=level,  # type: ignore[arg-type]
                label=binary_label(level),
                source=AGREEMENT if identifier in agreed else ADJUDICATION,
            )
        )
    return tuple(rows)


def _consensus_lines(rows: Sequence[ConsensusLabel]) -> list[str]:
    levels = Counter(row.label_3 for row in rows)
    sources = Counter(row.source for row in rows)
    return [
        f"Consensus labels: {len(rows)} pairs ("
        + ", ".join(f"{level} {levels[level]}" for level in LEVELS)
        + f"); {sources[AGREEMENT]} agreed, {sources[ADJUDICATION]} adjudicated; "
        f"{sum(row.label for row in rows)} redundant for T22",
    ]


def prepare_import_main(run: Run, inputs: WorkbookInputs) -> Prepared:
    """Read every annotator's workbook, measure the agreement, and, with no disputed pair, make
    the consensus labels."""
    places = run.places
    records = main_records(places)
    _, codes = pilot_records(places)
    redactor = _NameRedactor(inputs.names)
    read = read_workbooks(
        places, records.layout, annotator_file_name, redact=redactor, guard=inputs.guard
    )
    labels = label_rows(read, records.layout)
    agreement = measure_agreement(labels, records.assignment)
    tables: list[Table] = [
        (in_labels(LABELS_FILE_NAME), labels, AnnotatorLabelRow),
        (in_labels(FLEISS_INPUTS_FILE_NAME), fleiss_inputs(agreement), FleissInputRow),
    ]
    disputed = len(agreement.consensus.needs_adjudication)
    lines = agreement_lines(agreement)
    consensus: tuple[ConsensusLabel, ...] = ()
    if not disputed:
        consensus = consensus_rows(inputs, agreement)
        tables.append((in_labels(CONSENSUS_FILE_NAME), consensus, ConsensusLabel))
        lines += _consensus_lines(consensus)
    else:
        lines.append(f"Next: {RUN_COMMAND} adjudication, for the {disputed} disputed pairs")
    return Prepared(
        counts={
            "annotators": records.annotators,
            "pairs": len(records.assignment),
            "labels": len(labels),
            "agreed": len(agreement.consensus.agreed),
            "disputed": disputed,
            "consensus_rows": len(consensus),
            "redacted_notes": redactor.redacted,
        },
        details=agreement_details(agreement),
        tables=tables,
        workbooks={},
        codes=codes,
        lines=lines,
        written=f"the gitignored {in_labels(LABELS_FILE_NAME)}, "
        f"{in_labels(FLEISS_INPUTS_FILE_NAME)}"
        + (f" and {in_labels(CONSENSUS_FILE_NAME)}" if consensus else ""),
        inputs={
            MAIN_MANIFEST_FILE_NAME: _sha256(places.results / MAIN_MANIFEST_FILE_NAME),
            **{
                annotator_file_name(k): _sha256(places.workbooks / annotator_file_name(k))
                for k in sorted({row.annotator for row in labels})
            },
        },
    )


@dataclass(frozen=True)
class AdjudicationBuild:
    """The adjudication workbooks and their layout, ready to write."""

    layout: tuple[LayoutRow, ...]
    workbooks: dict[str, openpyxl.Workbook]
    redacted: int
    texts_checked: int


def build_adjudication(
    run: Run,
    inputs: WorkbookInputs,
    adjudicators: Mapping[str, int],
    codes: Mapping[str, str],
) -> AdjudicationBuild:
    """Lay out and build one workbook per adjudicator, with the pilot's examples, and pass them
    through the name guard.

    Raises:
        NameGuardError: if a workbook holds a person's name or an ORCID iD of the snapshot.
    """
    places, seed = run.places, run.config.labeling.seed
    by_id: dict[str, PairRow] = {pair.pair_id: pair for pair in inputs.pairs}
    redactor = _NameRedactor(inputs.names)
    pilot_layout, _ = pilot_records(places)
    agreed = {row.code: row for row in pilot_labels(places)}
    examples = pair_views(pilot_layout, inputs.fichas, redactor, agreed)
    layout: list[LayoutRow] = []
    workbooks: dict[str, openpyxl.Workbook] = {}
    for adjudicator in sorted(set(adjudicators.values())):
        mine = [by_id[i] for i, k in adjudicators.items() if k == adjudicator]
        rows = layout_rows(adjudication_workbook(adjudicator), adjudicator, mine, codes, seed=seed)
        layout += rows
        workbooks[adjudication_file_name(adjudicator)] = build_workbook(
            adjudication_guide(adjudicator, len(rows), len(examples)),
            [
                SheetContent(EXAMPLES_SHEET, PILOT_LAYOUT, examples, editable=False),
                SheetContent(
                    LABELING_SHEET,
                    MAIN_LAYOUT,
                    pair_views(rows, inputs.fichas, redactor),
                    editable=True,
                ),
            ],
        )
    checked = guard_workbooks(workbooks, inputs.guard)
    return AdjudicationBuild(tuple(layout), workbooks, redactor.redacted, checked)


def prepare_adjudication(run: Run, inputs: WorkbookInputs) -> Prepared:
    """Give each disputed pair its adjudicator and build the adjudication workbooks.

    Raises:
        WorkbooksError: if no pair is disputed.
    """
    places = run.places
    records = main_records(places)
    agreement = measure_agreement(main_labels(places), records.assignment)
    if not agreement.disputed:
        raise WorkbooksError(
            "no pair is disputed, so there is nothing to adjudicate: import-main wrote the "
            "consensus labels"
        )
    _, codes = pilot_records(places)
    adjudicators = assign_adjudicators(
        agreement.disputed, records.annotators, seed=run.config.labeling.seed
    )
    built = build_adjudication(run, inputs, adjudicators, codes)
    load = Counter(adjudicators.values())
    on_binary = sum(
        a.label != b.label
        for p, a, b in zip(agreement.pairs, agreement.first, agreement.second, strict=True)
        if p.pair_id in adjudicators
    )
    lines = [
        f"Adjudication workbooks: {len(adjudicators)} disputed pairs ({on_binary} on the binary "
        f"label too), each for an annotator who did not label it",
        "  pairs per adjudicator: " + ", ".join(f"{k} {load[k]}" for k in sorted(load)),
        f"  names redacted in {built.redacted} cells",
        _guard_line(built.texts_checked, inputs),
    ]
    return Prepared(
        counts={
            "disputed": len(adjudicators),
            "disputed_binary": on_binary,
            "adjudicators": len(load),
            "redacted_cells": built.redacted,
            "guard_strings": len(inputs.guard),
            "texts_checked": built.texts_checked,
        },
        details={"pairs_per_adjudicator": {str(k): load[k] for k in sorted(load)}},
        tables=[(ADJUDICATION_LAYOUT_FILE_NAME, built.layout, LayoutRow)],
        workbooks=built.workbooks,
        codes=codes,
        lines=lines,
        written=f"{len(built.workbooks)} adjudication workbooks under "
        f"{places.relative(places.workbooks)} and the gitignored {ADJUDICATION_LAYOUT_FILE_NAME}",
        inputs={
            MAIN_LABELS_MANIFEST_FILE_NAME: _sha256(
                places.results / MAIN_LABELS_MANIFEST_FILE_NAME
            ),
        },
    )


def prepare_import_adjudication(run: Run, inputs: WorkbookInputs) -> Prepared:
    """Read the adjudication workbooks and make the consensus labels of every labeled pair."""
    places = run.places
    path = places.results / ADJUDICATION_MANIFEST_FILE_NAME
    manifest = _manifest_at(path)
    layout = _read_recorded(
        LayoutRow, places.records, ADJUDICATION_LAYOUT_FILE_NAME, manifest, path
    )
    records = main_records(places)
    agreement = measure_agreement(main_labels(places), records.assignment)
    if {row.pair_id for row in layout} != {pair.pair_id for pair in agreement.disputed}:
        raise WorkbooksError("the adjudication workbooks' pairs are not the disputed pairs")
    _, codes = pilot_records(places)
    redactor = _NameRedactor(inputs.names)
    read = read_workbooks(
        places, layout, adjudication_file_name, redact=redactor, guard=inputs.guard
    )
    third = label_rows(read, layout)
    by_pair = {row.pair_id: row for row in third}
    finals = {}
    without_majority = 0
    for pair, a, b in zip(agreement.pairs, agreement.first, agreement.second, strict=True):
        if pair.pair_id in by_pair:
            levels = [a.label_3, b.label_3, by_pair[pair.pair_id].label_3]
            without_majority += len(set(levels)) == 3
            finals[pair.pair_id] = final_level(levels)
    consensus = consensus_rows(inputs, agreement, finals)
    levels = Counter(row.label_3 for row in consensus if row.source == ADJUDICATION)
    return Prepared(
        counts={
            "adjudicated": len(third),
            "without_majority": without_majority,
            "redacted_notes": redactor.redacted,
            "consensus_rows": len(consensus),
            "redundant": sum(row.label for row in consensus),
        },
        details={
            "adjudicated_levels": {level: levels[level] for level in LEVELS},
            "consensus_levels": {
                level: sum(row.label_3 == level for row in consensus) for level in LEVELS
            },
        },
        tables=[
            (in_labels(ADJUDICATION_LABELS_FILE_NAME), third, AnnotatorLabelRow),
            (in_labels(CONSENSUS_FILE_NAME), consensus, ConsensusLabel),
        ],
        workbooks={},
        codes=codes,
        lines=[
            f"Adjudication labels: {len(third)} pairs; {without_majority} with three different "
            "levels, settled by their median",
            *_consensus_lines(consensus),
        ],
        written=f"the gitignored {in_labels(ADJUDICATION_LABELS_FILE_NAME)} and "
        f"{in_labels(CONSENSUS_FILE_NAME)}",
        inputs={
            ADJUDICATION_MANIFEST_FILE_NAME: _sha256(path),
            MAIN_LABELS_MANIFEST_FILE_NAME: _sha256(
                places.results / MAIN_LABELS_MANIFEST_FILE_NAME
            ),
            **{
                adjudication_file_name(k): _sha256(places.workbooks / adjudication_file_name(k))
                for k in sorted({row.annotator for row in third})
            },
        },
    )


def _waits_for(manifest_name: str, what: str, step: str) -> Callable[[Run], None]:
    def wait(run: Run) -> None:
        if not (run.places.results / manifest_name).is_file():
            raise WorkbooksError(
                f"{what} of snapshot {run.places.snapshot_id} are missing ({manifest_name}); "
                f"run {RUN_COMMAND} {step} first"
            )

    return wait


def _adjudication_outputs(run: Run) -> tuple[Path, ...]:
    places = run.places
    return (
        places.records / ADJUDICATION_LAYOUT_FILE_NAME,
        *sorted(places.workbooks.glob(ADJUDICATION_FILE_PATTERN)),
        places.results / ADJUDICATION_MANIFEST_FILE_NAME,
    )


IMPORT_STEPS = {
    "import-main": Step(
        name="import-main",
        manifest_name=MAIN_LABELS_MANIFEST_FILE_NAME,
        outputs=lambda run: (
            run.places.records / in_labels(LABELS_FILE_NAME),
            run.places.records / in_labels(FLEISS_INPUTS_FILE_NAME),
            run.places.records / in_labels(CONSENSUS_FILE_NAME),
            run.places.results / MAIN_LABELS_MANIFEST_FILE_NAME,
        ),
        exists="the annotators' labels of snapshot {snapshot} are already imported ({path})",
        prepare=prepare_import_main,
        wait=_waits_for(MAIN_MANIFEST_FILE_NAME, "the annotators' workbooks", "main"),
    ),
    "adjudication": Step(
        name="adjudication",
        manifest_name=ADJUDICATION_MANIFEST_FILE_NAME,
        outputs=_adjudication_outputs,
        exists="the adjudication workbooks of snapshot {snapshot} already exist ({path}) and may "
        "hold labels",
        prepare=prepare_adjudication,
        wait=_waits_for(MAIN_LABELS_MANIFEST_FILE_NAME, "the annotators' labels", "import-main"),
    ),
    "import-adjudication": Step(
        name="import-adjudication",
        manifest_name=ADJUDICATION_LABELS_MANIFEST_FILE_NAME,
        outputs=lambda run: (
            run.places.records / in_labels(ADJUDICATION_LABELS_FILE_NAME),
            run.places.records / in_labels(CONSENSUS_FILE_NAME),
            run.places.results / ADJUDICATION_LABELS_MANIFEST_FILE_NAME,
        ),
        exists="the adjudication labels of snapshot {snapshot} are already imported ({path})",
        prepare=prepare_import_adjudication,
        wait=_waits_for(
            ADJUDICATION_MANIFEST_FILE_NAME, "the adjudication workbooks", "adjudication"
        ),
    ),
}
"""The importer's steps (T20c), in the order they run, after the workbooks' steps."""
