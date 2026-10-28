"""The app's pages (T28), in the order of its navigation, with no Streamlit import.

Each page is a script beside the entry point, ``main.py``, which builds ``st.navigation`` from
:data:`PAGES`; the first page opens by default. A new view, such as the query view of the next
unit, adds its script and one :class:`PageSpec` here, and reads its data through the data layer
(:mod:`thematic_redundancy.app.data`) and its cached loaders (:mod:`thematic_redundancy.app.ui`).
"""

from dataclasses import dataclass
from pathlib import Path

from thematic_redundancy.app import texts

APP_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = "main.py"


@dataclass(frozen=True)
class PageSpec:
    """One page: its script beside ``main.py``, its Spanish title, its URL path and its icon."""

    script: str
    title: str
    url_path: str
    icon: str

    @property
    def path(self) -> Path:
        return APP_DIR / self.script


PAGES = (
    PageSpec("map_view.py", texts.MAP_TITLE, "mapa", ":material/scatter_plot:"),
    PageSpec("query_view.py", texts.QUERY_TITLE, "consulta", ":material/search:"),
    PageSpec("gaps_view.py", texts.GAPS_TITLE, "vacios", ":material/travel_explore:"),
)
