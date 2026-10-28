"""The query page (T28): one thesis proposal, compared with the saved artifacts.

The user types a proposal's title, abstract and objectives in a form. On submission it is
represented as the corpus is (:mod:`thematic_redundancy.app.query`), and the page shows its
assigned topic with O06's flag and the topic's saturation (T23), its most similar theses with
their cosine and no redundancy verdict, which waits for the calibration of τ (T22), its
antecedents by MMR, and its alternative lines: the neighbor topics and the nearby stable gaps
(T26). The proposal lives only in this run of the page: it is not cached, written or logged.
Streamlit runs this script as a page of ``main.py``.
"""

import streamlit as st

from thematic_redundancy.app import tables, texts, ui
from thematic_redundancy.app.query import Proposal, answer_query, represent

# Form limits, above the corpus's longest title and abstract. A comment, not a bare string:
# Streamlit's magic would render a bare string literal on the page.
MAX_TITLE_CHARACTERS = 500
MAX_TEXT_CHARACTERS = 6000


def render() -> None:
    """Show the query page."""
    st.title(texts.QUERY_TITLE)
    ui.provisional_notice(ui.experiment())
    st.caption(texts.QUERY_INTRO)
    with st.form("query_form", enter_to_submit=False):
        title = st.text_input(
            texts.QUERY_TITLE_FIELD, max_chars=MAX_TITLE_CHARACTERS, key="query_title"
        )
        abstract = st.text_area(
            texts.QUERY_ABSTRACT_FIELD, max_chars=MAX_TEXT_CHARACTERS, key="query_abstract"
        )
        objectives = st.text_area(
            texts.QUERY_OBJECTIVES_FIELD, max_chars=MAX_TEXT_CHARACTERS, key="query_objectives"
        )
        st.caption(texts.QUERY_FORM_NOTE)
        submitted = st.form_submit_button(texts.QUERY_SUBMIT, key="query_submit", type="primary")
    if not submitted:
        st.info(texts.QUERY_WAITING)
        return
    proposal = Proposal(title=title or "", abstract=abstract or "", objectives=objectives or "")
    missing = proposal.missing()
    if missing:
        for problem in missing:
            st.error(texts.MISSING_FIELDS[problem])
        return
    index = ui.query_index()
    encoder = ui.proposal_encoder()
    try:
        vector = represent(proposal, encoder)
    except ValueError:
        # Never logged: the message could hold the proposal's text.
        st.error(texts.QUERY_NOT_REPRESENTED)
        return
    answer = answer_query(vector, index)
    theses, records = ui.theses(), ui.gap_records()

    st.subheader(texts.QUERY_TOPIC_HEADER)
    st.markdown(texts.assignment_text(answer.lines))
    st.caption(texts.assigned_terms_text(answer.lines))
    if answer.atypical.atypical:
        st.warning(texts.atypical_text(answer.atypical), icon=":material/report:")
    else:
        st.success(texts.atypical_text(answer.atypical), icon=":material/check_circle:")
    st.markdown(texts.saturation_text(answer.saturation, index.saturation))

    st.subheader(texts.QUERY_SIMILAR_HEADER)
    st.info(texts.PENDING_CALIBRATION, icon=":material/pending:")
    st.caption(texts.SIMILAR_CAPTION.format(count=len(answer.similar)))
    st.dataframe(
        tables.similar_table(answer, index, theses),
        hide_index=True,
        column_config=ui.link_column(),
    )

    st.subheader(texts.QUERY_ANTECEDENTS_HEADER)
    st.caption(
        texts.ANTECEDENTS_CAPTION.format(
            count=len(answer.antecedents.antecedents), mmr=answer.antecedents.mmr_lambda
        )
    )
    st.dataframe(
        tables.antecedent_table(answer, index, theses),
        hide_index=True,
        column_config=ui.link_column(),
    )

    st.subheader(texts.QUERY_LINES_HEADER)
    st.markdown(f"**{texts.NEIGHBORS_HEADER}**")
    if answer.lines.neighbors:
        st.caption(texts.NEIGHBORS_CAPTION.format(count=len(answer.lines.neighbors)))
        st.dataframe(tables.neighbor_table(answer), hide_index=True)
    else:
        st.caption(texts.NO_NEIGHBORS)
    st.markdown(f"**{texts.NEARBY_GAPS_HEADER}**")
    if answer.lines.gaps:
        st.caption(texts.NEARBY_GAPS_CAPTION.format(count=len(answer.lines.gaps)))
        st.dataframe(
            tables.nearby_gap_table(answer, index, records),
            hide_index=True,
            column_config={"Descripción": st.column_config.TextColumn(width="large")},
        )
    else:
        st.caption(texts.NO_NEARBY_GAPS)


if __name__ == "__main__":
    render()
