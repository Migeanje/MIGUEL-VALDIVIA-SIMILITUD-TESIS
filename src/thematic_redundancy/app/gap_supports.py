"""T24a's stable gaps as T24's :class:`~thematic_redundancy.gaps.stability.GapSupport` objects,
rebuilt from the saved records for T26's alternative lines (T28), with no Streamlit import.

T26's :func:`~thematic_redundancy.recommendation.alternatives.alternative_lines` reads the shown
map's stable gaps as ``GapSupport`` objects, which the gap run computes but does not save. They
are computed here from what it saved, with T24's own code:

- each gap of every run is rebuilt from its row of ``gaps.parquet`` and its theses in
  ``members.parquet``, the inverse of :func:`~thematic_redundancy.gaps.results_store.gap_rows`
  and :func:`~thematic_redundancy.gaps.results_store.member_rows`, with the values that the gap
  run's manifest records (``settings``) and each run's topic count from T19's ``topics.parquet``;
- the stability of the reference run's gaps is computed again by
  :func:`~thematic_redundancy.gaps.stability.gap_support` over the runs in the manifest's order.

The result is accepted only when every rebuilt gap gives exactly its saved ``reason``, and every
gap of the reference run gives its saved support and stability; the stable ones are returned in
the order of the gaps page's stable gaps. Otherwise the records are refused as inconsistent.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence

from thematic_redundancy.app.data import GAPS_COMMAND, ArtifactError, GapRecords, SavedGapRun
from thematic_redundancy.gaps.results_store import GapRow
from thematic_redundancy.gaps.rules import (
    Gap,
    GapInputError,
    InterstitialPair,
    SmallCoherentTopic,
    TemporalDecline,
    TransferableTopic,
)
from thematic_redundancy.gaps.stability import GapSupport, gap_support
from thematic_redundancy.topic_map.experiment import TopicRow


def rebuild_gap(
    row: GapRow,
    parts: Mapping[str, frozenset[str]],
    settings: Mapping[str, int | float],
    topics_in_map: int,
) -> Gap:
    """Return the gap that ``row`` records, with its theses by part (``topic``, or
    ``first_topic``, ``second_topic`` and ``bridge`` for rule (b)), the gap run's ``settings``
    and the number of topics of the row's run.

    Raises:
        KeyError: if a setting is missing.
        TypeError: if a column that the row's rule fills is empty.
    """
    programs = tuple(zip(row.programs, row.program_counts, strict=True))
    if row.rule == "a":
        return SmallCoherentTopic(
            topic=row.topics[0],
            size=row.theses,
            size_threshold=_number(row.size_cut),
            topics_in_map=topics_in_map,
            npmi=_number(row.npmi),
            npmi_floor=_number(row.npmi_floor),
            members=parts.get("topic", frozenset()),
            program_counts=programs,
        )
    if row.rule == "b":
        return InterstitialPair(
            first_topic=row.topics[0],
            second_topic=row.topics[1],
            centroid_cosine=_number(row.centroid_cosine),
            neighbor_rank=int(_number(row.neighbor_rank)),
            nearest_topics=int(settings["nearest_topics"]),
            bridge_documents=int(_number(row.bridge_theses)),
            bridge_max_docs=int(settings["bridge_max_docs"]),
            bridges=parts.get("bridge", frozenset()),
            first_members=parts.get("first_topic", frozenset()),
            second_members=parts.get("second_topic", frozenset()),
        )
    if row.rule == "c":
        return TransferableTopic(
            topic=row.topics[0],
            source=str(row.source),
            target=str(row.target),
            affinity=int(_number(row.affinity)),
            min_affinity=int(settings["min_affinity"]),
            source_count=int(_number(row.source_count)),
            source_theses=int(_number(row.source_theses)),
            source_share=_number(row.source_share),
            target_count=int(_number(row.target_count)),
            target_theses=int(_number(row.target_theses)),
            target_share=_number(row.target_share),
            present_min_count=int(settings["present_min_count"]),
            present_min_share=settings["present_min_share"],
            rare_max_share=settings["rare_max_share"],
            members=parts.get("topic", frozenset()),
        )
    first, split, last = (
        int(settings[name])
        for name in ("temporal_first_year", "temporal_split_year", "temporal_last_year")
    )
    return TemporalDecline(
        topic=row.topics[0],
        early_years=(first, split - 1),
        late_years=(split, last),
        early_count=int(_number(row.early_count)),
        early_theses=int(_number(row.early_theses)),
        early_share=_number(row.early_share),
        late_count=int(_number(row.late_count)),
        late_theses=int(_number(row.late_theses)),
        late_share=_number(row.late_share),
        decline=_number(row.decline),
        min_count=int(settings["temporal_min_count"]),
        min_decline=settings["min_decline"],
        members=parts.get("topic", frozenset()),
        program_counts=programs,
    )


def stable_supports(
    saved: SavedGapRun, topic_rows: Sequence[TopicRow], records: GapRecords
) -> tuple[GapSupport, ...]:
    """Return the stable gaps of the reference run, rebuilt and assessed again by T24's
    stability filter, in the order of ``records.gaps`` (see the module).

    Raises:
        ArtifactError: if the records cannot be rebuilt into the gaps, reasons, supports and
            stable gaps that they hold.
    """
    manifest = saved.manifest
    settings = manifest.settings
    topics_per_run: dict[str, int] = defaultdict(int)
    for topic in topic_rows:
        topics_per_run[topic.run_id] += 1
    parts: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for member in saved.members:
        parts[member.gap_id][member.part].add(member.doc_code)
    by_run: dict[str, list[tuple[GapRow, Gap]]] = {run: [] for run in manifest.runs}
    try:
        for row in saved.rows:
            gap = rebuild_gap(
                row,
                {part: frozenset(codes) for part, codes in parts[row.gap_id].items()},
                settings,
                topics_per_run[row.run_id],
            )
            if gap.reason != row.reason or row.run_id not in by_run:
                raise ValueError("a rebuilt gap differs from its row")
            by_run[row.run_id].append((row, gap))
        reference = manifest.runs.index(manifest.reference_run)
        supports = gap_support(
            [[gap for _, gap in by_run[run]] for run in manifest.runs],
            reference=reference,
            min_runs=int(settings["stability_min_runs"]),
            min_overlap=settings["stability_min_overlap"],
        )
    except (KeyError, TypeError, ValueError, GapInputError) as error:
        raise _inconsistent(saved, type(error).__name__) from None
    reference_rows = [row for row, _ in by_run[manifest.reference_run]]
    agree = all(
        row.reference and row.support == support.support and row.stable == support.stable
        for row, support in zip(reference_rows, supports, strict=True)
    )
    stable = tuple(support for support in supports if support.stable)
    wanted = [gap.row.gap_id for gap in records.gaps]
    kept = [
        row.gap_id for row, support in zip(reference_rows, supports, strict=True) if support.stable
    ]
    if not agree or kept != wanted:
        raise _inconsistent(saved, "the stability filter gives other supports than the saved ones")
    return stable


def _number(value: float | int | None) -> float:
    if value is None:
        raise TypeError("a column of the row's rule is empty")
    return value


def _inconsistent(saved: SavedGapRun, detail: str) -> ArtifactError:
    path = saved.gaps_path
    return ArtifactError(
        "inconsistent", path.name, path.parent, command=GAPS_COMMAND, detail=detail
    )
