"""The Streamlit app (T28): the topic map and the research gaps of a corpus snapshot, in Spanish
for UCSM students and advisors (O01).

Start it from the project root with ``uv run streamlit run src/thematic_redundancy/app/main.py``
(see :mod:`.main`). The app reads the artifacts that T19, T24a and the fichas builder saved,
checked against their manifests, through the data layer (:mod:`.data`), and builds its copy
(:mod:`.texts`), tables (:mod:`.tables`) and map figure (:mod:`.charts`) without Streamlit; only
the entry point, the page scripts and :mod:`.ui` import it. The pages are listed in
:mod:`.registry`. The app computes nothing new, writes no file, keeps no query, shows no author
or advisor, and sends no usage statistics (``.streamlit/config.toml``).
"""
