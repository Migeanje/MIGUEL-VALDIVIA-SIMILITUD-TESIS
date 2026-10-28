"""The entry point of the Streamlit app (T28), from the project root::

    uv run streamlit run src/thematic_redundancy/app/main.py [-- [--project-root PATH]
        [--snapshot-id ID]]

``--project-root`` is the folder that holds ``config/default.yaml`` (default: the current
directory), and ``--snapshot-id`` the corpus snapshot (default: the one harvested last). The
project's ``.streamlit/config.toml`` turns Streamlit's usage statistics off and runs the server
headless on this machine only.

The app shows the saved artifacts of the snapshot, in Spanish (O01), on the pages of
:data:`~thematic_redundancy.app.registry.PAGES`: the topic map and the research gaps. It
computes nothing new, writes no file, keeps no query and shows no author or advisor.
"""

import streamlit as st

from thematic_redundancy.app import texts, ui
from thematic_redundancy.app.registry import PAGES


def main() -> None:
    """Build the navigation from the page registry, show the sidebar and run the chosen page."""
    st.set_page_config(page_title=texts.APP_TITLE, page_icon=":material/hub:", layout="wide")
    navigation = st.navigation(
        [
            st.Page(
                str(page.path),
                title=page.title,
                url_path=page.url_path,
                icon=page.icon,
                default=index == 0,
            )
            for index, page in enumerate(PAGES)
        ]
    )
    ui.sidebar()
    navigation.run()


if __name__ == "__main__":
    main()
