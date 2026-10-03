"""Profile the metadata of a corpus snapshot: the exploratory analysis of task T07.

Run it from the project root, offline::

    uv run python experiments/eda_metadata.py [--snapshot-id ID]

It reads ``data/raw/<snapshot_id>/`` (by default the snapshot harvested last) through
:func:`read_snapshot`, so a snapshot changed after its harvest is refused. Tokens are counted
with the tokenizer of the MiniLM embedding model, loaded from the local Hugging Face cache
only. The outputs go to ``results/eda/<snapshot_id>/``:

- ``metadata_profile.json``: the whole profile, with its provenance;
- ``summary.md``: tables, observations, and notes on the method;
- ``program_year.png``, ``abstract_tokens.png``, ``title_tokens.png``, ``top_keywords.png``.

Every output holds aggregates only. Before anything is written, each output is searched for
every title, abstract, long abstract sentence, person's name and ORCID of the snapshot, and a
single hit stops the run with nothing written.
"""

import argparse
import hashlib
import io
import json
import re
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml
from matplotlib import rc_context
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from transformers import AutoTokenizer
from transformers.utils import logging as transformers_logging

from thematic_redundancy.corpus.profile import (
    NO_SUFFIX,
    PLACE_NAMES,
    TITLE_SUFFIX_PATTERNS,
    Distribution,
    LengthProfile,
    MetadataProfile,
    TokenCounter,
    leaked_strings,
    profile_metadata,
    sensitive_strings,
)
from thematic_redundancy.corpus.snapshot import (
    RAW_DIR_NAME,
    SnapshotManifest,
    check_snapshot_id,
    latest_snapshot_id,
    read_snapshot,
)
from thematic_redundancy.shared.config import load_config

CONFIG_PATH = Path("config") / "default.yaml"
SCRIPT_PATH = "experiments/eda_metadata.py"
OUTPUT_DIR_NAME = "eda"
"""Directory under ``paths.results_dir`` that holds one directory per profiled snapshot."""

TOKENIZER_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
"""Embedding model whose tokenizer counts the tokens; it must be a configured model."""

SUFFIX_SHAPES = {
    "place_and_year": "`…, Arequipa 2023`, `… - Arequipa, Perú 2021`",
    "year_after_separator": "`…, 2025`, `… - 2019`, `… (2021)`",
    "year_after_temporal_word": "`… al 2021`, `… periodo 2022`",
    "year_after_other_word": "`… productivos 2024`",
    "place_after_separator": "`…, Moquegua`",
    "place_after_word": "`… de Arequipa`",
    NO_SUFFIX: "",
}
"""Synthetic examples of each title suffix pattern, for the report."""

_NUMBER = re.compile(r"-?[0-9][0-9,]*(?:\.[0-9]+)?%?")
"""A number as the report writes it, such as ``1,234``, ``12.5`` or ``45.6%``."""

# Figure style: the reference palette of the data-visualization guide, light mode.
DPI = 192
PX = DPI / 96
"""Device pixels per CSS pixel: figures are drawn at twice the density of a 96 dpi screen."""
HAIRLINE = 72 / 96
"""One CSS pixel, in points."""
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
"""Categorical colors, assigned in this order and never cycled."""
STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "text.color": INK,
    "axes.facecolor": SURFACE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.labelsize": 9,
    "figure.facecolor": SURFACE,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
}


def main(argv: Sequence[str] | None = None) -> int:
    """Profile the snapshot, check the outputs for leaks, write them, and return the exit code."""
    arguments = _parse_arguments(argv)
    project_root = arguments.project_root.resolve()
    config_path = project_root / CONFIG_PATH
    try:
        config = load_config(config_path)
        if TOKENIZER_MODEL not in config.embedding.models:
            raise ValueError(f"{TOKENIZER_MODEL} is not one of the configured embedding.models")
        paths = config.paths.resolve_against(project_root)
        raw_dir = paths.data_dir / RAW_DIR_NAME
        if arguments.snapshot_id is None:
            snapshot_id = latest_snapshot_id(raw_dir)
        else:
            snapshot_id = check_snapshot_id(arguments.snapshot_id)
        manifest, records = read_snapshot(raw_dir / snapshot_id)
        count_tokens, tokenizer = load_token_counter(TOKENIZER_MODEL)
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    programs = {program.key: program.name for program in config.snapshot.programs}
    profile = profile_metadata(
        records,
        snapshot_id=snapshot_id,
        snapshot_date=manifest.harvested_at.date(),
        thesis_type=config.snapshot.thesis_type,
        program_keys=list(programs),
        count_tokens=count_tokens,
        token_limit=config.chunking.max_tokens,
    )
    future_dated = profile.calendar.future_dated + profile.other_items.future_dated
    if future_dated != manifest.future_dated_items:
        print(
            f"error: the profile finds {future_dated} future-dated items, but the snapshot "
            f"manifest records {manifest.future_dated_items}; the two date rules disagree",
            file=sys.stderr,
        )
        return 1
    provenance = _provenance(project_root, config_path, manifest, tokenizer)
    outputs = render_outputs(profile, provenance, programs)
    sensitive = sensitive_strings(records)
    leaks = {
        name: leaked_strings(_searchable(content), sensitive) for name, content in outputs.items()
    }
    if any(leaks.values()):
        for name, found in leaks.items():
            if found:
                print(
                    f"error: {name} holds {len(found)} string(s) taken from the snapshot",
                    file=sys.stderr,
                )
        print("error: nothing was written", file=sys.stderr)
        return 1
    output_dir = paths.results_dir / OUTPUT_DIR_NAME / snapshot_id
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, content in outputs.items():
        (output_dir / name).write_bytes(content)
    print(f"Profiled snapshot {snapshot_id}: {profile.items} items, {profile.theses} theses")
    print(
        f"Privacy check: {len(outputs)} outputs searched for {len(sensitive)} strings taken "
        "from the snapshot (titles, abstracts and their sentences, names, ORCIDs): 0 hits"
    )
    print(f"Wrote {output_dir}:")
    for name, content in outputs.items():
        print(f"  {name:<24} {len(content):>9,} bytes")
    return 0


def _parse_arguments(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=f"python {SCRIPT_PATH}",
        description=(
            "Profile the metadata of a corpus snapshot into aggregate tables and figures "
            "under results/eda/<snapshot_id>/."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(),
        help=f"project root that holds {CONFIG_PATH.as_posix()} (default: current directory)",
    )
    parser.add_argument(
        "--snapshot-id", help="snapshot to profile (default: the one harvested last)"
    )
    return parser.parse_args(argv)


def load_token_counter(model_id: str) -> tuple[TokenCounter, dict[str, Any]]:
    """Return a counter of the tokens that ``model_id`` sees, special tokens included.

    The tokenizer comes from the local Hugging Face cache only, so nothing is downloaded.
    Texts are counted whole, without truncation.

    Raises:
        OSError: if the tokenizer is not in the local cache.
    """
    # Counting a text longer than the model's limit is the point here, so silence the warning.
    transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

    def count_tokens(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    description = {
        "model": model_id,
        "class": type(tokenizer).__name__,
        "vocabulary_size": tokenizer.vocab_size,
        "special_tokens_per_sequence": count_tokens(""),
    }
    return count_tokens, description


def _provenance(
    project_root: Path, config_path: Path, manifest: SnapshotManifest, tokenizer: dict[str, Any]
) -> dict[str, Any]:
    return {
        "script": SCRIPT_PATH,
        "snapshot_id": manifest.snapshot_id,
        "harvested_at": manifest.harvested_at.isoformat().replace("+00:00", "Z"),
        "metadata_sha256": manifest.metadata_sha256,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "git_commit": _git(project_root, "rev-parse", "HEAD"),
        "git_uncommitted_changes": bool(
            _git(project_root, "status", "--porcelain", "--untracked-files=no")
        ),
        "tokenizer": tokenizer,
    }


def _git(project_root: Path, *arguments: str) -> str | None:
    """Return the output of a git command, or ``None`` when git is unavailable or fails."""
    try:
        run = subprocess.run(
            ["git", *arguments], cwd=project_root, capture_output=True, text=True, check=False
        )
    except OSError:
        return None
    return run.stdout.strip() if run.returncode == 0 else None


def _searchable(content: bytes) -> str:
    """Return ``content`` as text to search: as UTF-8, and byte for byte as Latin-1."""
    return f"{content.decode('utf-8', errors='ignore')}\n{content.decode('latin-1')}"


# Outputs


def render_outputs(
    profile: MetadataProfile, provenance: Mapping[str, Any], programs: Mapping[str, str]
) -> dict[str, bytes]:
    """Return every output file, by name, without writing any."""
    with rc_context(STYLE):
        figures = {
            "program_year.png": plot_program_year(profile),
            "abstract_tokens.png": plot_token_histogram(
                profile.abstracts.lengths,
                title="Abstract length in MiniLM tokens",
                noun="abstracts",
            ),
            "title_tokens.png": plot_token_histogram(
                profile.titles.lengths, title="Title length in MiniLM tokens", noun="titles"
            ),
            "top_keywords.png": plot_top_keywords(profile),
        }
    document = {"provenance": provenance, "profile": profile.model_dump(mode="json")}
    return {
        "metadata_profile.json": f"{json.dumps(document, ensure_ascii=False, indent=2)}\n".encode(),
        "summary.md": render_summary(profile, provenance, programs).encode(),
        **figures,
    }


# Report


def render_summary(
    profile: MetadataProfile, provenance: Mapping[str, Any], programs: Mapping[str, str]
) -> str:
    """Return the Markdown report: provenance, observations, one section per profile part."""
    tokenizer = provenance["tokenizer"]
    commit = provenance["git_commit"] or "unknown"
    changes = " with uncommitted changes" if provenance["git_uncommitted_changes"] else ""
    limit = profile.abstracts.chunks.token_limit
    lines = [
        f"# Metadata profile of snapshot {profile.snapshot_id}",
        "",
        "Aggregates only: no title, abstract, author, advisor or juror name appears in this "
        "report, in `metadata_profile.json`, or in the figures.",
        "",
        f"- **Source:** `data/raw/{profile.snapshot_id}/metadata.jsonl`, harvested "
        f"{provenance['harvested_at']} (SHA-256 `{provenance['metadata_sha256'][:12]}…`).",
        f"- **Code:** `{SCRIPT_PATH}` at git `{commit[:7]}`{changes}; configuration SHA-256 "
        f"`{provenance['config_sha256'][:12]}…`.",
        f"- **Tokens:** the tokenizer of `{tokenizer['model']}`, special tokens included "
        f"({tokenizer['special_tokens_per_sequence']} per sequence); the model reads at most "
        f"{limit} tokens.",
        "",
        "## Observations",
        "",
        *(f"- {observation}" for observation in observations(profile, programs)),
        "",
        *_corpus_section(profile),
        *_calendar_section(profile, programs),
        *_abstract_section(profile),
        *_title_section(profile),
        *_keyword_section(profile),
        *_ocde_section(profile),
        *_advisor_section(profile),
        *_rights_section(profile),
        *_method_section(profile),
        "## Figures",
        "",
        "![Theses per issue year, by program](program_year.png)",
        "",
        "![Abstract length in MiniLM tokens](abstract_tokens.png)",
        "",
        "![Title length in MiniLM tokens](title_tokens.png)",
        "",
        "![Most frequent keywords](top_keywords.png)",
    ]
    return "\n".join(lines) + "\n"


def observations(profile: MetadataProfile, programs: Mapping[str, str]) -> list[str]:
    """Return the findings that matter for chunking, cleaning and topic modeling."""
    found = []
    abstracts, titles = profile.abstracts.lengths, profile.titles.lengths
    chunks = profile.abstracts.chunks
    if abstracts is not None and titles is not None and chunks.sentence_aligned_stats:
        verdict = (
            "Abstracts need chunking (D09); titles do not."
            if abstracts.share_above_limit > 0.5 and titles.above_limit == 0
            else "Chunking (D09) matters for both abstracts and titles."
        )
        found.append(
            f"**{verdict}** {abstracts.above_limit} of {abstracts.texts} abstracts "
            f"({_percent(abstracts.share_above_limit)}) exceed the {abstracts.token_limit}-token "
            f"window (median {_number(abstracts.tokens.median)} tokens, P95 "
            f"{_number(abstracts.tokens.p95)}). Sentence-aligned packing needs a median of "
            f"{_number(chunks.sentence_aligned_stats.median)} chunks per abstract, up to "
            f"{_number(chunks.sentence_aligned_stats.max)}. The longest title has "
            f"{_number(titles.tokens.max)} tokens."
        )
    breaks = profile.abstracts.line_breaks
    suffixes = profile.titles
    found.append(
        f"**Both texts need cleaning before use.** {suffixes.with_suffix} titles "
        f"({_percent(suffixes.share_with_suffix)}) end with a place and/or year suffix (D10); "
        f"{suffixes.places.get('Arequipa', 0)} of them name Arequipa. "
        f"{breaks.texts_with_breaks} of {profile.abstracts.present} abstracts are hard-wrapped, "
        f"and {_share_text(breaks.inside_sentence, breaks.breaks)} of their {breaks.breaks:,} "
        "line breaks fall inside a sentence, so the light cleaner must join lines before "
        "sentences are split (T11, T13)."
    )
    keywords = profile.keywords
    if keywords.per_thesis is not None and keywords.top:
        single = _share_text(keywords.used_by_one_thesis, keywords.distinct)
        found.append(
            f"**Keywords are sparse.** A thesis lists a median of "
            f"{_number(keywords.per_thesis.median)} keywords; {keywords.distinct:,} distinct "
            f"keywords remain after case folding, {single} of them used by one thesis only, "
            f"and the most frequent one appears in {keywords.top[0].theses} theses. They can "
            "label topics, but they are too sparse to define them."
        )
    by_program = {key: count for key, count in profile.calendar.by_program.items() if count}
    largest = max(by_program, key=by_program.__getitem__)
    smallest = min(by_program, key=by_program.__getitem__)
    by_year = profile.calendar.by_year
    busiest = max(by_year, key=by_year.__getitem__)
    found.append(
        f"**The programs are unbalanced.** {programs.get(largest, largest)} holds "
        f"{_share_text(by_program[largest], profile.theses)} of the theses and "
        f"{programs.get(smallest, smallest)} {_share_text(by_program[smallest], profile.theses)}, "
        f"so topic sizes and gap rules need per-program normalization (R5). The busiest year "
        f"is {busiest} ({by_year[busiest]} theses)."
    )
    dominant = _dominant_codes(profile)
    if dominant:
        shares = [share for _, _, share in dominant.values()]
        verdict = (
            "mostly restate the program" if min(shares) >= 0.5 else "vary within some programs"
        )
        found.append(
            f"**OCDE codes {verdict}.** Within each program, its most frequent code covers "
            f"{_percent(min(shares))} to {_percent(max(shares))} of its theses; "
            f"{profile.ocde.distinct} distinct codes occur in all."
        )
    rights, duplicates = profile.rights, profile.duplicates
    duplicate_note = (
        f"Groups of theses that share their title and abstract: {duplicates.groups}, holding "
        f"{duplicates.records} records, so the census has "
        f"{profile.theses - duplicates.extra_records} distinct theses and T12 must keep one "
        "record of each. "
        if duplicates.groups
        else ""
    )
    found.append(
        f"**{'Duplicates, dates' if duplicates.groups else 'Dates'} and rights need care.** "
        f"{duplicate_note}Theses issued after the snapshot date "
        f"({profile.snapshot_date.isoformat()}): {profile.calendar.future_dated}; T06 and T27 "
        f"must handle them. Embargoed theses: {rights.by_access['embargoed']}, of which "
        f"{rights.embargoed_end_passed} have an end date already passed; restricted theses: "
        f"{rights.by_access['restricted']}."
    )
    return found


def _corpus_section(profile: MetadataProfile) -> list[str]:
    others = profile.other_items
    return [
        "## Corpus",
        "",
        f"{profile.items} items: {profile.theses} theses (exact `renati.type` fragment "
        f"`#{profile.thesis_type}`), profiled below, and {others.items} other items, "
        "reported apart.",
        "",
        "Groups of theses that repeat both the title and the abstract of another: "
        f"{profile.duplicates.groups}, holding {profile.duplicates.records} records. Counted "
        f"once each, the theses number {profile.theses - profile.duplicates.extra_records}. "
        "The profile below still counts every record.",
        "",
        *_table(["Other item type", "Items"], others.by_type.items()),
        "",
        "Their programs: "
        + ", ".join(f"`{key}` {count}" for key, count in others.by_program.items() if count)
        + f". Issued after the snapshot date: {others.future_dated}.",
        "",
    ]


def _calendar_section(profile: MetadataProfile, programs: Mapping[str, str]) -> list[str]:
    calendar = profile.calendar
    years = list(calendar.by_year)
    rows = [
        [f"{programs.get(key, key)} (`{key}`)", *counts.values(), calendar.by_program[key]]
        for key, counts in calendar.by_program_and_year.items()
    ]
    rows.append(["**Total**", *calendar.by_year.values(), profile.theses])
    return [
        "## Theses by program and issue year",
        "",
        *_table(["Program", *map(str, years), "Total"], rows),
        "",
        f"- Issue dates run from {_date(calendar.earliest)} to {_date(calendar.latest)}. "
        f"Theses without a readable `dc.date.issued`: {calendar.unparsed_dates}.",
        f"- Theses issued after the snapshot date ({profile.snapshot_date.isoformat()}): "
        f"{calendar.future_dated}. The year {years[-1] if years else '-'} is partial.",
        "",
    ]


def _abstract_section(profile: MetadataProfile) -> list[str]:
    abstracts = profile.abstracts
    lengths = abstracts.lengths
    if lengths is None:
        return ["## Abstracts", "", "No thesis has an abstract.", ""]
    chunks = abstracts.chunks
    counts = sorted(set(chunks.sentence_aligned) | set(chunks.lower_bound))
    breaks = abstracts.line_breaks
    return [
        "## Abstracts",
        "",
        f"Theses with an abstract: {abstracts.present}; without one: {abstracts.missing}.",
        "",
        *_length_table(lengths),
        "",
        f"Abstracts above {lengths.token_limit} tokens: {lengths.above_limit} "
        f"({_percent(lengths.share_above_limit)}).",
        "",
        f"Chunks of {chunks.token_limit} tokens that each abstract needs:",
        "",
        *_table(
            ["Chunks", "Abstracts, sentence-aligned", "Abstracts, lower bound"],
            [
                [count, chunks.sentence_aligned.get(count, 0), chunks.lower_bound.get(count, 0)]
                for count in counts
            ],
        ),
        "",
        "Text defects that the cleaner must handle:",
        "",
        f"- Hard-wrapped abstracts: {breaks.texts_with_breaks}, with {breaks.breaks:,} line "
        f"breaks in all: {breaks.inside_sentence:,} inside a sentence and "
        f"{breaks.after_hyphen} right after a hyphen.",
        "- Abstracts holding U+FFFD, a character lost to a wrong encoding: "
        f"{abstracts.with_replacement_character}.",
        f"- Abstracts that embed a `Palabras clave` list: {abstracts.with_keywords_section}.",
        "",
    ]


def _title_section(profile: MetadataProfile) -> list[str]:
    titles = profile.titles
    lengths = titles.lengths
    if lengths is None:
        return ["## Titles", "", "No thesis has a title.", ""]
    patterns = [*TITLE_SUFFIX_PATTERNS, NO_SUFFIX]
    places = list(titles.places.items())
    place_rows: list[Sequence[Any]] = places[:10]
    if len(places) > 10:
        place_rows.append([f"{len(places) - 10} other places", sum(n for _, n in places[10:])])
    return [
        "## Titles",
        "",
        *_length_table(lengths),
        "",
        f"Titles above {lengths.token_limit} tokens: {lengths.above_limit}.",
        "",
        f"Titles that end with a suffix: {titles.with_suffix} "
        f"({_percent(titles.share_with_suffix)}); with a year: {titles.with_year_suffix}; with "
        f"a place: {titles.with_place_suffix} (a title with both counts in each).",
        "",
        *_table(
            ["Suffix pattern", "Example shape (synthetic)", "Titles", "Share"],
            [
                [
                    f"`{pattern}`",
                    SUFFIX_SHAPES[pattern],
                    titles.suffix_patterns[pattern],
                    _share_text(titles.suffix_patterns[pattern], titles.present),
                ]
                for pattern in patterns
            ],
        ),
        "",
        *_table(["Place in the suffix", "Titles"], place_rows),
        "",
    ]


def _keyword_section(profile: MetadataProfile) -> list[str]:
    keywords = profile.keywords
    per_thesis = keywords.per_thesis
    ranked = [[rank, entry.keyword, entry.theses] for rank, entry in enumerate(keywords.top, 1)]
    by_program = keywords.top_by_program
    depth = max((len(top) for top in by_program.values()), default=0)
    program_rows = [
        [
            rank + 1,
            *(
                f"{top[rank].keyword} ({top[rank].theses})" if rank < len(top) else ""
                for top in by_program.values()
            ),
        ]
        for rank in range(depth)
    ]
    summary = (
        f"A thesis lists {_number(per_thesis.min)} to {_number(per_thesis.max)} keywords "
        f"(median {_number(per_thesis.median)}, mean {per_thesis.mean:.1f})"
        if per_thesis is not None
        else "No thesis lists keywords"
    )
    return [
        "## Keywords",
        "",
        f"{summary}. Theses without keywords: {keywords.theses_without_keywords}. The "
        f"{keywords.values:,} keyword values give {keywords.distinct:,} distinct keywords "
        f"after normalization, {keywords.used_by_one_thesis:,} of them listed by one thesis "
        f"only. Values with a comma, which may pack several keywords: "
        f"{keywords.values_with_comma}; values that end with a period: "
        f"{keywords.values_with_trailing_period}.",
        "",
        *_table(["Rank", "Keyword", "Theses"], ranked),
        "",
        "Most frequent keywords per program (theses in parentheses):",
        "",
        *_table(["Rank", *by_program], program_rows),
        "",
    ]


def _ocde_section(profile: MetadataProfile) -> list[str]:
    ocde = profile.ocde
    top_codes = list(ocde.by_code.items())[:10]
    rest = sum(list(ocde.by_code.values())[10:])
    rows = [
        [f"`{code}`", count, _share_text(count, ocde.theses_with_code)] for code, count in top_codes
    ]
    if rest:
        rows.append(
            [
                f"{len(ocde.by_code) - 10} other codes",
                rest,
                _share_text(rest, ocde.theses_with_code),
            ]
        )
    dominant = _dominant_codes(profile)
    return [
        "## OCDE fields and language",
        "",
        f"{ocde.theses_with_code} theses carry an OCDE code (`dc.subject.ocde`), "
        f"{ocde.theses_without_code} do not; {ocde.distinct} distinct codes occur.",
        "",
        *_table(["OCDE code", "Theses", "Share"], rows),
        "",
        *_table(["Major field", "Theses"], ocde.by_field.items()),
        "",
        *_table(
            ["Program", "Most frequent code", "Theses", "Share of the program"],
            [
                [key, f"`{code}`", count, _percent(share)]
                for key, (code, count, share) in dominant.items()
            ],
        ),
        "",
        *_table(["Language (`dc.language.iso`)", "Theses"], profile.languages.items()),
        "",
    ]


def _advisor_section(profile: MetadataProfile) -> list[str]:
    advisors = profile.advisors
    per_advisor = advisors.theses_per_advisor
    stats = (
        f"An advisor supervised {_number(per_advisor.min)} to {_number(per_advisor.max)} "
        f"theses (median {_number(per_advisor.median)}, mean {per_advisor.mean:.1f}, P95 "
        f"{_number(per_advisor.p95)})."
        if per_advisor is not None
        else "No thesis names an advisor."
    )
    return [
        "## Advisors",
        "",
        "Counts only; no advisor is named.",
        "",
        f"- Distinct advisor names: {advisors.distinct_advisors}, over "
        f"{advisors.theses_with_advisor} theses. Theses without an advisor: "
        f"{advisors.theses_without_advisor}; with several: "
        f"{advisors.theses_with_several_advisors}.",
        f"- {stats} The 10 advisors with the most theses supervised "
        f"{_percent(advisors.share_of_theses_with_top_advisors)} of them.",
        f"- Theses with an advisor ORCID: {advisors.theses_with_advisor_orcid}, with "
        f"{advisors.distinct_advisor_orcids} distinct ORCIDs. ORCIDs found with more than one "
        f"advisor name: {advisors.orcids_with_several_names}, so advisor names need "
        "reconciliation before any per-advisor feature.",
        "",
        *_table(["Theses supervised", "Advisors"], advisors.advisors_by_theses.items()),
        "",
    ]


def _rights_section(profile: MetadataProfile) -> list[str]:
    rights = profile.rights
    return [
        "## Rights and embargoes",
        "",
        *_table(
            ["Access (`dc.rights`)", "Theses"],
            [(access, count) for access, count in rights.by_access.items() if count],
        ),
        "",
        f"- Embargoed theses: {rights.embargoed_end_passed} with an end date already passed, "
        f"{rights.embargoed_end_pending} still pending, {rights.embargoed_end_missing} without "
        "an end date.",
        "- Theses with an embargo end date that are no longer embargoed: "
        f"{rights.not_embargoed_with_end}.",
        "",
    ]


def _method_section(profile: MetadataProfile) -> list[str]:
    chunks = profile.abstracts.chunks
    budget = chunks.token_limit - chunks.special_tokens
    return [
        "## Method notes",
        "",
        "- **Texts.** Titles and abstracts are the first value of their field, with every run "
        "of whitespace, line breaks included, collapsed into one space. Characters are counted "
        "in NFC, and words are runs between spaces.",
        f"- **Chunk estimate.** Sentences end at `.`, `!`, `?` or `…` (maybe closed by a quote "
        "or a bracket) when whitespace and a capital letter or a digit follow; a line break "
        "alone never ends one. Each sentence is tokenized alone, without special tokens, and "
        f"the sentences are packed greedily into chunks of {budget} tokens plus "
        f"{chunks.special_tokens} special tokens. A sentence longer than a chunk starts its own "
        "and is cut into full pieces. The lower bound packs tokens perfectly. This is an "
        "approximation: an abbreviation before a capital (such as `S.A.C.`) splits a sentence, "
        "which can only lower the estimate, while a missed sentence end can raise it. In "
        f"{chunks.sentence_sums_differing} abstracts the sentence token counts do not add up to "
        "the whole abstract.",
        "- **Second model.** The tokenizer of `paraphrase-multilingual-mpnet-base-v2` is not in "
        "the local cache, so D09's check under both tokenizers is still pending.",
        "- **Title suffixes.** `TRAILING_YEAR` and `TRAILING_PLACE` in "
        "`src/thematic_redundancy/corpus/profile.py` define the patterns. A year is 19xx or "
        "20xx, or a range, after a comma, a dash, a parenthesis or a space, and never after a "
        "digit (so `ISO 9001:2015` does not count). Places come from a gazetteer of "
        f"{len(PLACE_NAMES)} names: Peru, its departments, the provinces of the Arequipa region, "
        "the districts of the province of Arequipa, and a few southern towns. Place words "
        "missing from it are not seen, and a place that ends an organization's name counts.",
        "- **Keywords.** In NFC, with whitespace collapsed, trailing periods dropped, and case "
        "folded; accents and ñ are kept. Each keyword counts once per thesis.",
        "- **Dates.** The first `dc.date.issued` value; a year or a month counts from its first "
        "day, as in the snapshot manifest, and is compared with the UTC snapshot date.",
        "- **Advisors.** Names are compared with case and spacing folded but accents kept. "
        "ORCIDs are compared only on theses with exactly one advisor and one ORCID.",
        "- **Duplicates.** Theses whose title and abstract both match, with whitespace "
        "collapsed and case folded but accents kept.",
        "",
    ]


def _dominant_codes(profile: MetadataProfile) -> dict[str, tuple[str, int, float]]:
    """Return, per program with OCDE codes, its most frequent code, its theses and its share."""
    dominant = {}
    for key, codes in profile.ocde.by_program.items():
        theses = profile.calendar.by_program.get(key, 0)
        if codes and theses:
            code, count = next(iter(codes.items()))
            dominant[key] = (code, count, count / theses)
    return dominant


def _length_table(lengths: LengthProfile) -> list[str]:
    rows = [
        ["Characters", *_quantiles(lengths.characters)],
        ["Words", *_quantiles(lengths.words)],
        ["MiniLM tokens", *_quantiles(lengths.tokens)],
    ]
    return _table(["Measure", "Min", "P25", "Median", "P75", "P95", "Max"], rows)


def _quantiles(stats: Distribution) -> list[str]:
    values = (stats.min, stats.p25, stats.median, stats.p75, stats.p95, stats.max)
    return [_number(value) for value in values]


def _table(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    """Return a Markdown table; a column whose filled cells are all numbers aligns right."""
    cells = [[str(cell) for cell in row] for row in rows]
    columns = [[row[index] for row in cells if row[index]] for index in range(len(header))]
    numeric = [bool(column) and all(map(_NUMBER.fullmatch, column)) for column in columns]
    return [
        f"| {' | '.join(header)} |",
        "|" + "|".join("---:" if right else "---" for right in numeric) + "|",
        *(f"| {' | '.join(row)} |" for row in cells),
    ]


def _number(value: float) -> str:
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.1f}"


def _percent(share: float) -> str:
    return f"{100 * share:.1f}%"


def _share_text(part: int, whole: int) -> str:
    return _percent(part / whole) if whole else "n/a"


def _date(day: Any) -> str:
    return day.isoformat() if day is not None else "-"


# Figures


def plot_program_year(profile: MetadataProfile) -> bytes:
    """Draw the theses per issue year as columns stacked by program."""
    calendar = profile.calendar
    years = list(calendar.by_year)
    programs = [key for key, count in calendar.by_program.items() if count]
    colors = {key: SERIES[index] for index, key in enumerate(calendar.by_program)}
    figure, axes = _new_figure(8.0, 4.6, left=0.7, right=0.3, top=1.25, bottom=0.45)
    axes.set_xlim(-0.6, len(years) - 0.4)
    axes.set_ylim(0, max(calendar.by_year.values(), default=1) * 1.14)
    per_pixel_x, per_pixel_y = _data_per_pixel(axes)
    width = 24 * PX * per_pixel_x
    gap = 2 * PX * per_pixel_y
    for index, year in enumerate(years):
        segments = [
            (key, calendar.by_program_and_year[key][year])
            for key in programs
            if calendar.by_program_and_year[key][year]
        ]
        bottom = 0.0
        for position, (key, count) in enumerate(segments):
            lift = min(gap, count / 2) if position else 0.0
            _bar(
                axes,
                (index - width / 2, bottom + lift),
                (width, count - lift),
                colors[key],
                rounded=position == len(segments) - 1,
            )
            bottom += count
        axes.text(
            index,
            bottom + 6 * PX * per_pixel_y,
            f"{calendar.by_year[year]}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    axes.set_xticks(range(len(years)), [str(year) for year in years])
    axes.set_ylabel("Theses")
    _style_axes(axes, grid="y")
    _heading(
        figure,
        "Theses per issue year, by program",
        f"{profile.theses} theses of snapshot {profile.snapshot_id}; "
        f"{years[-1] if years else ''} is partial (harvested {profile.snapshot_date.isoformat()})",
    )
    figure.legend(
        handles=[Patch(facecolor=colors[key], label=key) for key in programs],
        loc="upper left",
        bbox_to_anchor=(0.25 / 8.0, 1 - 0.8 / 4.6),
        ncol=len(programs),
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
        handlelength=0.9,
        handleheight=0.9,
        handletextpad=0.5,
        columnspacing=1.6,
        borderaxespad=0,
        borderpad=0,
    )
    return _png(figure)


def plot_token_histogram(lengths: LengthProfile | None, *, title: str, noun: str) -> bytes:
    """Draw the token counts of a set of texts as a histogram, with the model's limit."""
    figure, axes = _new_figure(8.0, 4.4, left=0.7, right=0.3, top=0.95, bottom=0.65)
    if lengths is None:
        _heading(figure, title, f"No {noun}")
        return _png(figure)
    counts = lengths.token_histogram.counts
    bin_width = lengths.token_histogram.bin_width
    limit = lengths.token_limit
    right = max(max(counts) + 2 * bin_width, limit + 2 * bin_width)
    top = max(counts.values()) * 1.3
    axes.set_xlim(0, right)
    axes.set_ylim(0, top)
    per_pixel_x, per_pixel_y = _data_per_pixel(axes)
    gap = 2 * PX * per_pixel_x
    for edge, count in counts.items():
        if count:
            _bar(axes, (edge + gap / 2, 0.0), (bin_width - gap, count), SERIES[0], rounded=True)
    axes.axvline(limit, color=INK, linewidth=1.5 * HAIRLINE)
    share = _percent(lengths.share_above_limit)
    note = f"{limit}-token limit: {lengths.above_limit} of {lengths.texts} {noun} ({share}) above"
    # When most texts exceed the limit, its line sits near the left edge, so the note runs
    # rightward; otherwise the line sits near the right edge, and the note runs leftward.
    side = 1 if lengths.share_above_limit > 0.5 else -1
    axes.text(
        limit + side * 6 * PX * per_pixel_x,
        top * 0.97,
        note,
        ha="left" if side > 0 else "right",
        va="top",
        fontsize=9,
    )
    axes.set_xlabel("Tokens, special tokens included")
    axes.set_ylabel(noun.capitalize())
    _style_axes(axes, grid="y")
    _heading(
        figure,
        title,
        f"{lengths.texts} {noun}; bins of {bin_width} tokens; median "
        f"{_number(lengths.tokens.median)}, P95 {_number(lengths.tokens.p95)}, max "
        f"{_number(lengths.tokens.max)} tokens",
    )
    return _png(figure)


def plot_top_keywords(profile: MetadataProfile) -> bytes:
    """Draw the most frequent keywords as horizontal bars, the most frequent on top."""
    top = profile.keywords.top
    height = 1.3 + 0.24 * max(len(top), 1)
    label_width = _text_width([entry.keyword for entry in top], fontsize=9)
    left = label_width + 0.35
    figure, axes = _new_figure(8.0, height, left=left, right=0.4, top=0.95, bottom=0.55)
    if not top:
        _heading(figure, "Most frequent keywords", "No thesis lists keywords")
        return _png(figure)
    axes.set_xlim(0, max(entry.theses for entry in top) * 1.05)
    axes.set_ylim(len(top) - 0.5, -0.5)
    per_pixel_x, per_pixel_y = _data_per_pixel(axes)
    thickness = min(24 * PX * abs(per_pixel_y), 0.62)
    for index, entry in enumerate(top):
        _bar(
            axes,
            (0.0, index - thickness / 2),
            (entry.theses, thickness),
            SERIES[0],
            rounded=True,
            horizontal=True,
        )
    axes.set_yticks(range(len(top)), [entry.keyword for entry in top])
    axes.set_xlabel("Theses that list the keyword")
    _style_axes(axes, grid="x")
    _heading(
        figure,
        "Most frequent keywords",
        f"Top {len(top)} of {profile.keywords.distinct:,} distinct keywords (dc.subject), "
        "case folded with accents kept",
    )
    return _png(figure)


def _new_figure(
    width: float, height: float, *, left: float, right: float, top: float, bottom: float
) -> tuple[Figure, Axes]:
    """Return a figure, in inches, with one axes inside fixed margins, also in inches.

    Margins are fixed rather than computed at save time, so the axes geometry is known
    before drawing, which the rounded bar ends need.
    """
    figure = Figure(figsize=(width, height), dpi=DPI)
    FigureCanvasAgg(figure)
    axes = figure.add_axes(
        (left / width, bottom / height, 1 - (left + right) / width, 1 - (top + bottom) / height)
    )
    return figure, axes


def _heading(figure: Figure, title: str, subtitle: str) -> None:
    """Write the title and the subtitle at the top left of the figure."""
    width, height = figure.get_size_inches()
    x = 0.25 / width
    figure.text(x, 1 - 0.18 / height, title, ha="left", va="top", fontsize=13, weight="semibold")
    figure.text(
        x, 1 - 0.48 / height, subtitle, ha="left", va="top", fontsize=9, color=INK_SECONDARY
    )


def _style_axes(axes: Axes, *, grid: str) -> None:
    """Keep only the baseline spine and hairline gridlines along ``grid``."""
    baseline = "left" if grid == "x" else "bottom"
    for side, spine in axes.spines.items():
        spine.set_visible(side == baseline)
        spine.set_color(BASELINE)
        spine.set_linewidth(HAIRLINE)
    axes.tick_params(length=0, pad=6, labelsize=9)
    axes.grid(True, axis=grid, color=GRIDLINE, linewidth=HAIRLINE)
    axes.set_axisbelow(True)


def _data_per_pixel(axes: Axes) -> tuple[float, float]:
    """Return the data units in one device pixel along x and along y, for the current limits."""
    box = axes.get_window_extent()
    (x_low, x_high), (y_low, y_high) = axes.get_xlim(), axes.get_ylim()
    return (x_high - x_low) / box.width, (y_high - y_low) / box.height


def _bar(
    axes: Axes,
    corner: tuple[float, float],
    size: tuple[float, float],
    color: str,
    *,
    rounded: bool,
    horizontal: bool = False,
) -> None:
    """Draw a bar from its lower-left ``corner``, square at the baseline.

    A rounded bar gets a 4 px radius on the corners of its data end: the top of a column, or
    the right of a horizontal bar. The radius shrinks to fit a small bar.
    """
    (x, y), (width, height) = corner, size
    if width <= 0 or height <= 0:
        return
    per_pixel_x, per_pixel_y = (abs(value) for value in _data_per_pixel(axes))
    across = height / per_pixel_y if horizontal else width / per_pixel_x
    along = width / per_pixel_x if horizontal else height / per_pixel_y
    radius = min(4 * PX, across / 2, along / 2) if rounded else 0.0
    style = {"facecolor": color, "edgecolor": "none", "linewidth": 0}
    if radius < 1:
        axes.add_patch(Rectangle((x, y), width, height, **style))
        return
    if horizontal:
        axes.add_patch(Rectangle((x, y), width - radius * per_pixel_x, height, **style))
    else:
        axes.add_patch(Rectangle((x, y), width, height - radius * per_pixel_y, **style))
    axes.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"round,pad=0,rounding_size={radius * per_pixel_x}",
            mutation_aspect=per_pixel_y / per_pixel_x,
            **style,
        )
    )


def _text_width(texts: Sequence[str], *, fontsize: float) -> float:
    """Return the width, in inches, of the widest of ``texts`` set at ``fontsize``."""
    if not texts:
        return 0.0
    figure = Figure(dpi=DPI)
    renderer = FigureCanvasAgg(figure).get_renderer()
    widths = [
        figure.text(0, 0, text, fontsize=fontsize).get_window_extent(renderer).width
        for text in texts
    ]
    return max(widths) / DPI


def _png(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=DPI, metadata={"Software": None})
    return buffer.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())
