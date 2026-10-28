"""The gaps page (T28): the stable gaps of T24a's reference run (D32, D33).

It lists the gaps that the stability filter keeps, filtered by rule, by program and by rule
(c)'s high-confidence group (affinity 2), each with its rule, topics and their terms, a
description of its counts and shares, its programs and years, its affinity group and its
stability; then the theses of the chosen gap, each linked to its repository item. Streamlit
runs this script as a page of ``main.py``.
"""

import streamlit as st

from thematic_redundancy.app import tables, texts, ui
from thematic_redundancy.app.data import filter_gaps, members_of


def render() -> None:
    """Show the gaps page."""
    st.title(texts.GAPS_TITLE)
    ui.provisional_notice(ui.experiment())
    records = ui.gap_records()
    st.caption(texts.GAPS_INTRO.format(min_runs=records.min_runs, runs=records.runs))

    names = records.program_names
    filters = st.columns([3, 3, 2], vertical_alignment="bottom")
    rules = filters[0].multiselect(
        texts.GAPS_RULES_FILTER,
        list(texts.RULE_LABELS),
        format_func=lambda rule: texts.RULE_LABELS[rule],
        placeholder=texts.GAPS_RULES_PLACEHOLDER,
        key="gap_rules",
    )
    programs = filters[1].multiselect(
        texts.GAPS_PROGRAMS_FILTER,
        list(names),
        format_func=lambda program: names[program],
        placeholder=texts.GAPS_PROGRAMS_PLACEHOLDER,
        help=texts.GAPS_PROGRAMS_HELP,
        key="gap_programs",
    )
    high_affinity = filters[2].toggle(
        texts.GAPS_HIGH_AFFINITY_TOGGLE, help=texts.GAPS_HIGH_AFFINITY_HELP, key="gap_high_affinity"
    )
    shown = filter_gaps(
        records.gaps, rules=rules, programs=programs, high_affinity_only=high_affinity
    )
    st.caption(texts.GAPS_SHOWN.format(count=len(shown), total=len(records.gaps)))
    if not shown:
        st.info(texts.NO_GAPS)
        return
    st.dataframe(
        tables.gap_table(shown, records),
        hide_index=True,
        column_config={"Descripción": st.column_config.TextColumn(width="large")},
    )

    st.subheader(texts.GAPS_MEMBERS_HEADER)
    by_id = {gap.row.gap_id: gap for gap in shown}
    chosen = st.selectbox(
        texts.GAPS_MEMBER_SELECT,
        list(by_id),
        format_func=lambda gap_id: texts.gap_title(by_id[gap_id], names),
        key="gap_selected",
    )
    st.dataframe(
        tables.member_table(members_of(records, chosen)),
        hide_index=True,
        column_config=ui.link_column(),
    )


if __name__ == "__main__":
    render()
