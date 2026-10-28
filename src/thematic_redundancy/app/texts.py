"""Every text that the app shows (T28), in neutral, professional Spanish (O01).

Code, identifiers and the server's log stay in English; the app's users, UCSM students and
advisors, read only what this module builds. Each builder is a pure function of plain values,
so the copy is tested without Streamlit. No text names a person: the app shows theses by their
title, program, year and repository link only.
"""

from collections.abc import Mapping, Sequence

from thematic_redundancy.app.data import (
    HIGH_AFFINITY,
    NOISE_TOPIC,
    ArtifactError,
    SettingsError,
    StableGap,
    Winner,
)
from thematic_redundancy.app.query import ModelError, ProposalProblem
from thematic_redundancy.corpus.snapshot import RAW_DIR_NAME
from thematic_redundancy.recommendation.alternatives import AlternativeLines
from thematic_redundancy.redundancy.saturation import SaturationQuartiles, TopicSaturation
from thematic_redundancy.topic_map.assignment import AtypicalCheck

APP_TITLE = "Redundancia temática y vacíos de investigación"
APP_CAPTION = (
    "Tesis sustentadas de cinco carreras de la FCIFF de la Universidad Católica de Santa María, "
    "2021–2026."
)
PRIVACY_NOTE = (
    "La aplicación no muestra autores ni asesores y no guarda lo que se escribe o se selecciona. "
    "Cada tesis enlaza a su registro en el repositorio institucional."
)
RELOAD_BUTTON = "Volver a cargar los resultados guardados"
LOADING = "Cargando los resultados guardados…"

MAP_TITLE = "Mapa temático"
MAP_INTRO = (
    "Cada punto es una tesis, ubicada según la similitud de su título, resumen y objetivos. "
    "Las tesis cercanas tratan temas parecidos; los ejes no tienen unidades."
)
MAP_TOPICS_FILTER = "Temas"
MAP_TOPICS_PLACEHOLDER = "Todos los temas"
MAP_PROGRAMS_FILTER = "Carreras"
MAP_PROGRAMS_PLACEHOLDER = "Todas las carreras"
MAP_NOISE_TOGGLE = "Mostrar las tesis sin tema asignado"
MAP_THESES_METRIC = "Tesis en el mapa"
MAP_TOPICS_METRIC = "Temas"
MAP_NOISE_METRIC = "Tesis sin tema asignado"
MAP_SHOWN = "Tesis mostradas con los filtros actuales: {count}."
MAP_EMPTY = "Ninguna tesis cumple los filtros elegidos."
MAP_TOPICS_HEADER = "Temas y sus términos principales"
MAP_TOPICS_CAPTION = (
    "Los términos son los de mayor peso c-TF-IDF de cada tema. Las tesis sin tema asignado son "
    "las que el agrupamiento (HDBSCAN) no ubicó con suficiente densidad en ningún tema."
)
MAP_THESES_HEADER = "Tesis mostradas"

GAPS_TITLE = "Vacíos de investigación"
GAPS_INTRO = (
    "Vacíos estables: se repiten en al menos {min_runs} de las {runs} corridas del "
    "agrupamiento con distintas semillas. Son candidatos para que asesores y estudiantes los "
    "evalúen, no conclusiones."
)
GAPS_RULES_FILTER = "Reglas"
GAPS_RULES_PLACEHOLDER = "Todas las reglas"
GAPS_PROGRAMS_FILTER = "Carreras"
GAPS_PROGRAMS_PLACEHOLDER = "Todas las carreras"
GAPS_HIGH_AFFINITY_TOGGLE = "Solo alta confianza (afinidad 2)"
GAPS_HIGH_AFFINITY_HELP = (
    "Muestra solo el grupo de alta confianza de la regla (c): los vacíos entre carreras con "
    "afinidad 2, la más alta de la matriz de afinidad entre carreras."
)
GAPS_PROGRAMS_HELP = (
    "Una carrera coincide con un vacío cuando sus tesis están en los temas del vacío o, en la "
    "regla (c), cuando es la carrera de origen o la de destino."
)
GAPS_SHOWN = "Vacíos mostrados con los filtros actuales: {count} de {total}."
NO_GAPS = "Ningún vacío estable cumple los filtros elegidos."
GAPS_MEMBERS_HEADER = "Tesis del vacío"
GAPS_MEMBER_SELECT = "Vacío"

NOISE_LABEL = "Sin tema asignado"
RULE_LABELS = {
    "a": "(a) Tema pequeño y coherente",
    "b": "(b) Región intersticial entre temas vecinos",
    "c": "(c) Tema transferible a una carrera afín",
    "d": "(d) Tema en declive",
}
PART_LABELS = {
    "topic": "Tema del vacío",
    "first_topic": "Primer tema",
    "second_topic": "Segundo tema",
    "bridge": "Tesis puente",
}
LINK_TEXT = "Abrir en el repositorio"
NO_VALUE = "—"


def topic_label(topic: int, terms: Sequence[str], *, count: int = 3) -> str:
    """Return a topic's name: its number and its first ``count`` terms, such as
    ``Tema 3: diseño, estructura, molino``; the noise is :data:`NOISE_LABEL`."""
    if topic == NOISE_TOPIC:
        return NOISE_LABEL
    return f"Tema {topic}: {', '.join(terms[:count])}"


def algorithm_text(algorithm: str, parameter: int) -> str:
    """Return the clustering algorithm and its parameter, such as ``K-means, K = 12``."""
    if algorithm == "kmeans":
        return f"K-means, K = {parameter}"
    if algorithm == "hdbscan":
        return f"HDBSCAN, tamaño mínimo de grupo {parameter}"
    return f"{algorithm} ({parameter})"


def provisional_notice(winner: Winner) -> str:
    """Return the notice that the results wait for the labeled-set evaluation (T22)."""
    model = winner.representation.rsplit("/", 1)[-1]
    return (
        f"Resultados provisionales. La configuración del mapa ({winner.configuration}: modelo "
        f"{model}, {algorithm_text(winner.algorithm, winner.parameter)}) se confirmará con la "
        "evaluación del conjunto de pares etiquetados. Los temas, el mapa y los vacíos pueden "
        "cambiar."
    )


def error_message(error: ArtifactError) -> str:
    """Return what the user reads when a saved artifact cannot be used: the file, the problem
    and the command that writes it again."""
    if error.artifact == RAW_DIR_NAME and error.problem == "missing":
        return (
            f"No hay ninguna instantánea del corpus en {error.directory / error.artifact}. "
            "Indique una con --snapshot-id."
        )
    name = f"«{error.artifact}»"
    if error.problem == "missing":
        opening = f"Falta el archivo {name} en {error.directory}."
    elif error.problem == "invalid":
        opening = (
            f"El archivo {name} no se puede usar: está dañado o no coincide con el SHA-256 que "
            "registra su manifiesto, por ejemplo tras una ejecución interrumpida o una edición "
            "posterior."
        )
    elif error.problem == "inconsistent":
        opening = (
            f"El archivo {name} no concuerda con los demás resultados guardados de la instantánea."
        )
    else:
        opening = (
            f"El archivo {name} se generó a partir de otra versión de sus entradas, que ya no "
            "son las guardadas."
        )
    if error.command is None:
        return f"{opening} Revise la carpeta del proyecto indicada con --project-root."
    return f"{opening} Vuelva a generarlo con: {error.command}"


def settings_message(error: SettingsError) -> str:
    """Return what the user reads when the app's arguments are not valid."""
    return (
        "Los argumentos de la aplicación no son válidos. Se admiten --project-root (la carpeta "
        "del proyecto) y --snapshot-id (la instantánea del corpus), después de -- en "
        "streamlit run."
    )


def percent(share: float) -> str:
    """Return ``share`` as a percentage with one decimal, such as ``66.7 %``."""
    return f"{100 * share:.1f} %"


def gap_description(
    gap: StableGap,
    *,
    program_names: Mapping[str, str],
    windows: tuple[tuple[int, int], tuple[int, int]],
) -> str:
    """Return one sentence on why the gap's rule found it, with its counts and shares."""
    row = gap.row
    if row.rule == "a":
        return (
            f"Tema pequeño y coherente: {row.theses} tesis (el corte del cuartil inferior de "
            f"tamaño es {row.size_cut:g}) y una coherencia NPMI de {row.npmi:.3f} (el mínimo es "
            f"{row.npmi_floor:.3f})."
        )
    if row.rule == "b":
        return (
            f"Temas vecinos (coseno entre centroides {row.centroid_cosine:.3f}) con "
            f"{row.bridge_theses} tesis puente entre ellos."
        )
    if row.rule == "c":
        source = program_names.get(row.source or "", row.source)
        target = program_names.get(row.target or "", row.target)
        return (
            f"Presente en {source}: {row.source_count} de {row.source_theses} tesis "
            f"({percent(row.source_share)}). Ausente en {target}: {row.target_count} de "
            f"{row.target_theses} tesis ({percent(row.target_share)})."
        )
    (early_first, early_last), (late_first, late_last) = windows
    return (
        f"Su participación bajó de {percent(row.early_share)} ({row.early_count} de "
        f"{row.early_theses} tesis, {early_first}–{early_last}) a {percent(row.late_share)} "
        f"({row.late_count} de {row.late_theses} tesis, {late_first}–{late_last}): una caída "
        f"de {100 * row.decline:.1f} puntos porcentuales."
    )


def gap_topics_text(gap: StableGap) -> str:
    """Return the names of the gap's topics, joined by a middle dot."""
    return " · ".join(
        topic_label(topic, terms)
        for topic, terms in zip(gap.row.topics, gap.topic_terms, strict=True)
    )


def gap_title(gap: StableGap, program_names: Mapping[str, str]) -> str:
    """Return a short name for the gap, for the selector of its theses."""
    row = gap.row
    title = f"{gap.number}. {RULE_LABELS.get(row.rule, row.rule)} · {gap_topics_text(gap)}"
    if row.rule == "c":
        source = program_names.get(row.source or "", row.source)
        target = program_names.get(row.target or "", row.target)
        title += f" · {source} → {target}"
    return title


def program_counts_text(
    programs: Sequence[str], counts: Sequence[int], program_names: Mapping[str, str]
) -> str:
    """Return each program's name with its count of theses, such as
    ``Ingeniería Industrial (29); Ingeniería de Sistemas (1)``."""
    return "; ".join(
        f"{program_names.get(program, program)} ({count})"
        for program, count in zip(programs, counts, strict=True)
    )


def year_counts_text(years: Sequence[int], counts: Sequence[int]) -> str:
    """Return each year with its count of theses, such as ``2021 (3), 2022 (4)``."""
    return ", ".join(f"{year} ({count})" for year, count in zip(years, counts, strict=True))


def affinity_text(gap: StableGap) -> str:
    """Return rule (c)'s affinity group of the gap, or :data:`NO_VALUE` for another rule."""
    affinity = gap.row.affinity
    if gap.row.rule != "c" or affinity is None:
        return NO_VALUE
    if affinity >= HIGH_AFFINITY:
        return f"Alta confianza (afinidad {affinity})"
    return f"Afinidad {affinity}"


def support_text(gap: StableGap, runs: int) -> str:
    """Return in how many of the seed runs the gap is present."""
    return f"En {gap.row.support} de {runs} corridas"


# --- The query view ---------------------------------------------------------------------------

QUERY_TITLE = "Consulta de una propuesta"
QUERY_INTRO = (
    "Escriba una propuesta de tesis para ubicarla en el mapa temático: el tema al que se "
    "acerca, las tesis más parecidas, los antecedentes que conviene revisar y las líneas "
    "alternativas cercanas, con los vacíos estables próximos."
)
QUERY_TITLE_FIELD = "Título de la propuesta (obligatorio)"
QUERY_ABSTRACT_FIELD = "Resumen"
QUERY_OBJECTIVES_FIELD = "Objetivos (general y específicos)"
QUERY_FORM_NOTE = (
    "Escriba el título y, al menos, el resumen o los objetivos, las mismas partes con las que se "
    "representan las tesis del corpus. Lo que escriba se usa solo para esta consulta: no se "
    "guarda en ningún archivo ni registro y se pierde al cerrar la sesión."
)
QUERY_SUBMIT = "Analizar la propuesta"
QUERY_WAITING = "Complete el formulario y pulse «Analizar la propuesta» para ver los resultados."
MISSING_FIELDS: dict[ProposalProblem, str] = {
    "title": "Escriba el título de la propuesta.",
    "body": "Escriba el resumen, los objetivos o ambos.",
}
QUERY_NOT_REPRESENTED = (
    "No se pudo representar la propuesta: revise que el título, el resumen o los objetivos "
    "contengan texto."
)
LOADING_MODEL = "Cargando el modelo de lenguaje desde la caché local…"
QUERY_TOPIC_HEADER = "Tema asignado"
QUERY_SIMILAR_HEADER = "Tesis más parecidas"
QUERY_ANTECEDENTS_HEADER = "Antecedentes recomendados"
QUERY_LINES_HEADER = "Líneas alternativas"
PENDING_CALIBRATION = (
    "Pendiente de calibración: la aplicación todavía no indica si una tesis es redundante con "
    "la propuesta. El umbral de similitud que separa una tesis redundante de una solo parecida "
    "se calibrará con el conjunto de pares etiquetados por los asesores. Mientras tanto, la "
    "similitud del coseno (de −1 a 1; cuanto más alta, más parecida) solo ordena las tesis."
)
SIMILAR_CAPTION = "Las {count} tesis del corpus más parecidas a la propuesta, de mayor a menor."
ANTECEDENTS_CAPTION = (
    "Las {count} tesis elegidas por relevancia marginal máxima (MMR, λ = {mmr:g}): cada una se "
    "parece a la propuesta y se diferencia de las ya elegidas, para cubrir más enfoques que la "
    "lista anterior."
)
NEIGHBORS_HEADER = "Temas vecinos"
NEIGHBORS_CAPTION = "Los {count} temas más cercanos a la propuesta, sin contar el tema asignado."
NEARBY_GAPS_HEADER = "Vacíos estables cercanos"
NEARBY_GAPS_CAPTION = (
    "Los {count} vacíos de la página «Vacíos de investigación» más cercanos a la propuesta, "
    "medidos desde el centro de su tema o, para un par de temas, desde el punto medio entre ambos."
)
NO_NEIGHBORS = "El mapa no tiene otros temas."
NO_NEARBY_GAPS = "No hay vacíos estables en el mapa."
DUPLICATE_NOTE = "Tesis depositada dos veces en el repositorio"


def assignment_text(lines: AlternativeLines) -> str:
    """Return the proposal's assigned topic, its similarity and its runner-up."""
    assignment = lines.assignment
    text = (
        f"**{topic_label(assignment.topic, lines.assigned_terms)}** ({lines.assigned_size} "
        f"tesis). Similitud con el centro del tema: {assignment.similarity:.3f}."
    )
    if assignment.runner_up is not None:
        text += (
            f" Segundo tema más cercano: Tema {assignment.runner_up} "
            f"({assignment.runner_up_similarity:.3f})."
        )
    return text


def assigned_terms_text(lines: AlternativeLines) -> str:
    """Return every top term of the assigned topic."""
    return f"Términos principales del tema: {', '.join(lines.assigned_terms)}."


def atypical_text(check: AtypicalCheck) -> str:
    """Return O06's flag in plain words: typical or atypical for its topic, with both values."""
    rule = (
        f"el umbral de {check.threshold:.3f}: el 95 % de las tesis del mapa tiene una similitud "
        "con el centro de su propio tema igual o mayor (el percentil 5)"
    )
    if check.atypical:
        return (
            f"Propuesta atípica para su tema: su similitud con el centro del tema "
            f"({check.similarity:.3f}) está por debajo de {rule}. La propuesta se aleja de este "
            "tema más que casi todas las tesis del mapa del suyo, así que el tema asignado es "
            "solo una referencia débil."
        )
    return (
        f"Propuesta típica para su tema: su similitud con el centro del tema "
        f"({check.similarity:.3f}) alcanza {rule}."
    )


def saturation_text(entry: TopicSaturation, quartiles: SaturationQuartiles) -> str:
    """Return the assigned topic's saturation quartile in plain words."""
    if entry.quartile is None or quartiles.cuts is None:
        return (
            "La saturación del tema no está definida: todos los temas del mapa tienen el mismo "
            "número de tesis."
        )
    first, second, third = (f"{cut:g}" for cut in quartiles.cuts)
    return (
        f"Saturación del tema: cuartil {entry.level} de 4 por número de tesis ({entry.size} "
        f"tesis; los cortes entre cuartiles son {first}, {second} y {third} tesis). Q1 reúne los "
        "temas con menos tesis y Q4 los más trabajados."
    )


def model_message(error: ModelError) -> str:
    """Return what the user reads when the model cannot represent a proposal."""
    if error.problem == "missing":
        return (
            f"El modelo «{error.model_id}» no está en la caché local de modelos de este equipo "
            "(Hugging Face). La aplicación no descarga nada: copie el modelo a la caché local o "
            "descárguelo una vez con conexión a Internet."
        )
    return (
        f"El modelo «{error.model_id}» de la caché local no es la versión que representó las "
        "tesis del corpus, así que sus resultados no serían comparables. Instale la versión "
        "registrada en los manifiestos de los embeddings y de los fragmentos."
    )
