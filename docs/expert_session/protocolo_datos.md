# Protocolo de datos de la validación con expertos

> **Estado: borrador.** Ningún comando lee todavía estos archivos. La lectura de la sección
> "Análisis" es la propuesta; la unidad que cierre T30 y T32 la convierte en un comando probado.

Cómo se recogen, codifican, guardan y analizan las valoraciones de los jueces y las respuestas del
panel, de modo que ningún nombre llegue a un archivo digital ni al repositorio del proyecto.

## Flujo

| # | Paso | Entrada | Salida | Tarea |
|:-:|---|---|---|---|
| 1 | Validación de contenido | [Formulario de jueces](formulario_jueces.md) en papel | `jueces/*.csv`; ítems reformulados; cuestionario final | T30 |
| 2 | Sesión con el panel | [Guion](guion_sesion.md), [consentimiento](consentimiento_informado.md), [muestras](muestras_anonimizadas.md) | Sobre A (consentimientos, lista de códigos) y sobre B (cuestionarios) | T31 |
| 3 | Transcripción | Sobre B | `cuestionario/respuestas.csv` y `abiertas.csv` | T32 |
| 4 | Análisis | Los CSV | Tablas agregadas y la comprobación del criterio (c) | T32 |
| 5 | Informe | Tablas agregadas | Resultados en la tesis | T32 |

## Códigos

| Quién | Código | Cómo se asigna |
|---|---|---|
| Juez | `J01`, `J02`, … | Por sorteo, al entregarle el formulario |
| Experto del panel | `E01` a `E08` | Por sorteo de tarjetas en el bloque 2 del guion |
| Caso de una muestra | `A01`, `B01`, `C01`, `D01`, … | Al azar, al generar las muestras |

- El código no indica el perfil, la carrera ni el orden de llegada.
- Los archivos de respuestas no registran el perfil (especialista en inteligencia artificial o
  asesor de tesis). Con dos o tres personas por perfil, un resultado por perfil permitiría
  atribuir respuestas (§3.10). La composición del panel se informa con conteos tomados de la lista
  de códigos.

**Lista de códigos.** Se llena a mano, se guarda en el sobre A y **nunca se digitaliza**:

| Código | Nombre | Rol (juez o experto) | Perfil | Sesión | Consentimiento firmado | Retiro (fecha) |
|---|---|---|---|---|:-:|---|
| | | | | | ☐ | |

## Qué se guarda y dónde

| Material | Soporte | Lugar | ¿Lleva nombres? | ¿Entra en git? |
|---|---|---|:-:|:-:|
| Consentimientos firmados y lista de códigos | Papel | Sobre A, bajo custodia del investigador | Sí | No |
| Formularios de jueces y cuestionarios respondidos | Papel | Sobre B, separado del sobre A | No | No |
| Valoraciones y respuestas transcritas | CSV | `data/expert_session/<snapshot_id>/jueces/` y `.../cuestionario/` | No | No |
| Bitácoras de las sesiones | Texto | `data/expert_session/<snapshot_id>/bitacora/` | No | No |
| Muestras y su clave de casos | Archivos | `data/expert_session/<snapshot_id>/muestras/` | No | No |
| Resultados agregados | Tablas e informe | [propuesto: `results/expert_session/<snapshot_id>/`; lo fija la unidad del comando] | No | Sí, después de pasar `check_privacy` |

`data/` está en `.gitignore`, así que nada bajo esa carpeta llega al repositorio. El snapshot en
uso es `20261002T224412Z`:

```text
data/expert_session/20261002T224412Z/
├── jueces/
│   ├── claridad.csv
│   ├── pertinencia.csv
│   ├── suficiencia.csv
│   └── observaciones.csv
├── cuestionario/
│   ├── respuestas.csv
│   └── abiertas.csv
├── bitacora/
└── muestras/
```

Los formatos de cada CSV están en la parte B del [cuestionario](cuestionario.md) y del
[formulario de jueces](formulario_jueces.md); las plantillas, en [plantillas/](plantillas/).

## Transcripción

1. Transcriba cada hoja dos veces, en dos archivos: por ejemplo, `respuestas.csv` y
   `respuestas_verificacion.csv`.
2. Compare ambos con `differences` (sección "Análisis"). Corrija cada diferencia contra el papel y
   conserve solo el archivo corregido.
3. Una marca ambigua (dos casillas, ilegible) queda como celda vacía y se anota en la bitácora.
4. Las respuestas abiertas se copian literalmente, con `[nombre omitido]` en lugar de cualquier
   nombre de persona.

## Retiro de un participante

1. El participante indica su código o su nombre. Si da su nombre, busque su código en la lista.
2. Borre la fila de ese código en cada CSV donde aparezca y destruya sus hojas en papel.
3. Anote la fecha en la columna "Retiro" de la lista de códigos.
4. Repita el análisis. El informe dice cuántos participantes se retiraron, sin más detalle.

El retiro se acepta hasta la fecha límite del [consentimiento](consentimiento_informado.md)
(sección 4).

## Análisis

Funciones ya construidas y probadas con datos sintéticos: `likert_analysis` (T32) y
`aiken_validity` (T30), en `src/thematic_redundancy/evaluation/`. Sus argumentos:

| Análisis | Argumento | Valor | Origen |
|---|---|---|---|
| Cuestionario | `low`, `high` | `1`, `5` | Anexo A |
| Cuestionario | `reverse_coded` | `()` | Nota N11: todos los ítems son afirmativos |
| Cuestionario | `quartile_method` | `"linear"` | Nota N13, propuesta por confirmar |
| Jueces | `low`, `high` | Los extremos de la escala acordada | Nota N6, **abierta** |
| Jueces | `confidence`, `criterion`, `cutoff` | `0.95`, `"ci_lower"`, `0.50` | D31 (d) |
| Criterio de éxito (c) | `respondent_scores.median` de cada dimensión ≥ 4 | — | D31 (c) |

Lectura propuesta, desde la raíz del proyecto (`uv run python <archivo>.py`):

```python
"""Read the expert-session sheets and run the T30 and T32 analyses (draft, not a command)."""

import csv
from pathlib import Path

from thematic_redundancy.evaluation.aiken import aiken_validity
from thematic_redundancy.evaluation.likert import likert_analysis

BASE = Path("data/expert_session/20261002T224412Z")
DIMENSIONS = {
    "D1": ("I01", "I02", "I03", "I04"),
    "D2": ("I05", "I06", "I07", "I08"),
    "D3": ("I09", "I10", "I11", "I12"),
    "D4": ("I13", "I14", "I15", "I16"),
}
SUFFICIENCY = {dimension: (dimension,) for dimension in DIMENSIONS}
JUDGES_LOW, JUDGES_HIGH = None, None  # N6: set the judges' scale before running


def read_sheet(path: Path) -> dict[str, list[int | None]]:
    """Return each column but `codigo` in row order, a blank cell as None."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = [name for name in reader.fieldnames or () if name != "codigo"]
    codes = [row["codigo"] for row in rows]
    if len(set(codes)) != len(codes):
        raise ValueError(f"{path}: a code is repeated")
    return {
        column: [int(row[column]) if (row[column] or "").strip() else None for row in rows]
        for column in columns
    }


def differences(first: Path, second: Path) -> list[tuple[int, str]]:
    """Return the (line, column) cells where two transcriptions of a sheet differ."""
    with (
        first.open(encoding="utf-8-sig", newline="") as a,
        second.open(encoding="utf-8-sig", newline="") as b,
    ):
        rows_a, rows_b = list(csv.DictReader(a)), list(csv.DictReader(b))
    if len(rows_a) != len(rows_b):
        raise ValueError("the transcriptions have different numbers of rows")
    return [
        (line, column)
        for line, (row_a, row_b) in enumerate(zip(rows_a, rows_b), start=2)
        for column in row_a
        if row_a[column] != row_b.get(column)
    ]


answers = likert_analysis(
    read_sheet(BASE / "cuestionario" / "respuestas.csv"),
    DIMENSIONS,
    low=1,
    high=5,
    reverse_coded=(),
    quartile_method="linear",
)
for dimension in answers.dimensions:
    median = dimension.respondent_scores.median
    meets = median is not None and median >= 4
    print(dimension.dimension, median, meets, dimension.alpha.alpha)

for name, dimensions in (
    ("claridad", DIMENSIONS),
    ("pertinencia", DIMENSIONS),
    ("suficiencia", SUFFICIENCY),
):
    validity = aiken_validity(
        read_sheet(BASE / "jueces" / f"{name}.csv"),
        dimensions,
        low=JUDGES_LOW,
        high=JUDGES_HIGH,
        confidence=0.95,
        criterion="ci_lower",
        cutoff=0.50,
    )
    for item in validity.items:
        print(name, item.item, item.v, item.ci_lower, item.ci_upper, item.accepted)
```

Las funciones rechazan con un `ValueError` explícito una respuesta fuera de la escala, una columna
de ítem que falte o sobre en las dimensiones y una escala no fijada; `read_sheet` rechaza además
un código repetido y `int` rechaza una celda que no sea un número entero.

### Qué se informa

- Por ítem y por dimensión: medianas, cuartiles y distribución de frecuencias (§3.8.2), con las
  respuestas faltantes contadas aparte.
- Criterio de éxito (c), por dimensión: la mediana, sobre los expertos, de la media de cada
  experto en los cuatro ítems de la dimensión, contando solo a quienes respondieron los cuatro
  (D31 (c)); la mediana de todas las respuestas juntas y las medianas por ítem, como complemento.
- El alfa de Cronbach por dimensión, "como referencia de consistencia interna del instrumento, con
  las reservas interpretativas propias del tamaño reducido del panel" (§3.8.2; nota N14).
- Por criterio de los jueces: la V de Aiken de cada ítem con ambos límites de su intervalo de
  puntaje al 95 % y si cumple el valor de aceptación (D31 (d)).
- La comprobación de los criterios de éxito (a), (b) y (d) necesita además los resultados de T19
  y T22 (T32).

### Valor mínimo de V que cumple el valor de aceptación

Con `criterion="ci_lower"`, `cutoff=0.50` y `confidence=0.95` (D31 (d)), un ítem se acepta cuando
`S = Σ (r − low)` alcanza el mínimo de esta tabla. Calculado con `aiken_score_interval`:

| Jueces | Escala de 1 a 4: S mínimo / máximo | V mínimo | Escala de 1 a 5: S mínimo / máximo | V mínimo |
|:-:|:-:|:-:|:-:|:-:|
| 3 | 8 / 9 | 0.889 | 10 / 12 | 0.833 |
| 4 | 10 / 12 | 0.833 | 12 / 16 | 0.750 |
| 5 | 12 / 15 | 0.800 | 15 / 20 | 0.750 |
| 6 | 14 / 18 | 0.778 | 17 / 24 | 0.708 |
| 7 | 15 / 21 | 0.714 | 20 / 28 | 0.714 |
| 8 | 17 / 24 | 0.708 | 22 / 32 | 0.688 |

Por ejemplo, con 3 jueces y la escala de 1 a 4, un ítem se acepta solo si dos jueces dan 4 y uno
da 3, o si los tres dan 4. El mínimo no siempre baja al sumar un juez (7 frente a 6 en la escala de
1 a 5), porque S es entero. La tabla sirve para elegir el número de jueces (U6) y la escala (N6);
no los decide.

## Qué se publica

- Solo resultados agregados. Ninguna tabla por participante.
- Las respuestas abiertas se resumen; una cita textual solo con la autorización de la sección 9
  del consentimiento y sin detalles que identifiquen a quien la escribió.
- Las frecuencias por ítem se publican, como pide §3.8.2: con códigos que no se vinculan a nombres
  fuera del sobre A, no atribuyen respuestas a personas identificables.
- Todo resultado versionado pasa antes la guarda de privacidad en código (`check_privacy` de
  `gaps/results_store.py`), como los registros de T20a y T24a.

## Conservación y destrucción

Los plazos son los de la sección 7 del [consentimiento](consentimiento_informado.md):
[completar]. Al vencer el plazo de la lista de códigos, se destruye la lista y, desde entonces,
las respuestas transcritas quedan anónimas.
