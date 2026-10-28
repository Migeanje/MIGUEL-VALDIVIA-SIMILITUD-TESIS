# Formulario de jueces: validez de contenido (V de Aiken)

> **Estado: borrador.** La escala de valoración de los jueces **no está decidida** (nota N6,
> abierta): el formulario la trata como un parámetro que se fija con los jueces antes de
> entregarlo. Complete cada campo `[completar: …]` y retire esta nota antes de usarlo.

Antes de aplicar el cuestionario al panel, un grupo de jueces verifica "que los ítems sean
claros, pertinentes y suficientes para cada dimensión"; los ítems que no alcanzan el valor de
aceptación se reformulan (plan de tesis, §3.8.2, PDF p. 82). Este documento tiene dos partes:

- **Parte A, para imprimir:** el formulario que completa cada juez.
- **Parte B, para el investigador:** las hojas de valoraciones y su correspondencia con
  `aiken.py` (T30).

---

## Parte A. Formulario para el juez (para imprimir)

**Código del juez:** ____________ **Fecha:** ____________

No escriba su nombre en esta hoja.

### Qué se le pide

Usted revisará el borrador de un cuestionario de 16 afirmaciones, agrupadas en cuatro
dimensiones, que responderá un panel de docentes después de conocer un prototipo de análisis de
los trabajos de titulación del repositorio institucional. No se le pide responder el
cuestionario, sino valorar cada afirmación como instrumento de medición:

| Criterio | Qué valora | Dónde se anota |
|---|---|---|
| **Claridad** | La afirmación se comprende con facilidad y su redacción es precisa, sin ambigüedad. | Cada ítem |
| **Pertinencia** | La afirmación corresponde a la dimensión en la que está ubicada y contribuye a evaluarla. | Cada ítem |
| **Suficiencia** | Los ítems de la dimensión, en conjunto, bastan para evaluarla. | Cada dimensión |

### Escala de valoración

Anote en cada casilla un número entero de **[completar: valor mínimo]** a
**[completar: valor máximo]**, donde:

| Valor | Significado |
|---|---|
| [completar: valor mínimo] | [completar] |
| [completar: valores intermedios, uno por fila] | [completar] |
| [completar: valor máximo] | [completar] |

Si no puede valorar un criterio, deje la casilla en blanco. En "Observaciones" puede proponer
otra redacción o explicar su valoración.

### Dimensiones del cuestionario

Las definiciones son las del plan de tesis (§3.8.2):

| Dimensión | Qué evalúa |
|---|---|
| D1. Coherencia temática | Si los grupos generados agrupan documentos con afinidad real y si sus términos representativos los describen adecuadamente. |
| D2. Pertinencia de los vacíos de investigación | Si las áreas señaladas como poco exploradas constituyen oportunidades genuinas de investigación. |
| D3. Calidad de las recomendaciones | La relevancia de los antecedentes sugeridos y la utilidad de las líneas alternativas. |
| D4. Utilidad general y usabilidad | La percepción global sobre el valor del sistema y la claridad de su interfaz. |

La escala con la que el panel responderá cada afirmación es: 1 = Totalmente en desacuerdo;
2 = En desacuerdo; 3 = Ni de acuerdo ni en desacuerdo; 4 = De acuerdo; 5 = Totalmente de acuerdo.

### D1. Coherencia temática

| Ítem | Afirmación | Claridad | Pertinencia | Observaciones |
|---|---|:-:|:-:|---|
| 1 | Los documentos agrupados en cada grupo temático comparten efectivamente afinidad de tema. | | | |
| 2 | Los términos representativos de cada grupo describen adecuadamente su contenido. | | | |
| 3 | Los grupos reflejan líneas de investigación reconocibles de mi carrera o facultad. | | | |
| 4 | La cantidad y granularidad de los grupos es apropiada para el repositorio analizado. | | | |

**Suficiencia de D1:** ________ **Observaciones:** ______________________________________

### D2. Pertinencia de los vacíos de investigación

| Ítem | Afirmación | Claridad | Pertinencia | Observaciones |
|---|---|:-:|:-:|---|
| 5 | Las áreas señaladas como poco exploradas corresponden a temas con escasa investigación previa en la facultad. | | | |
| 6 | Los vacíos identificados constituyen oportunidades genuinas de investigación para tesistas. | | | |
| 7 | La presentación de los vacíos es comprensible e interpretable. | | | |
| 8 | Los vacíos detectados aportan información que no obtendría fácilmente con una revisión manual. | | | |

**Suficiencia de D2:** ________ **Observaciones:** ______________________________________

### D3. Calidad de las recomendaciones

| Ítem | Afirmación | Claridad | Pertinencia | Observaciones |
|---|---|:-:|:-:|---|
| 9 | Los antecedentes sugeridos para una propuesta son relevantes para esa propuesta. | | | |
| 10 | Las líneas alternativas propuestas son útiles para reorientar o enriquecer una propuesta. | | | |
| 11 | Las detecciones de redundancia temática señalan trabajos previos verdaderamente afines. | | | |
| 12 | Las recomendaciones orientarían adecuadamente a un tesista real. | | | |

**Suficiencia de D3:** ________ **Observaciones:** ______________________________________

### D4. Utilidad general y usabilidad

| Ítem | Afirmación | Claridad | Pertinencia | Observaciones |
|---|---|:-:|:-:|---|
| 13 | El sistema sería útil como herramienta de apoyo en la elaboración de propuestas de titulación. | | | |
| 14 | El sistema sería útil como apoyo para asesores en la evaluación de propuestas. | | | |
| 15 | La interfaz del prototipo es clara y fácil de usar. | | | |
| 16 | Recomendaría la adopción institucional de una herramienta como esta. | | | |

**Suficiencia de D4:** ________ **Observaciones:** ______________________________________

### Preguntas abiertas

El cuestionario cierra con dos preguntas abiertas, que no se valoran con la escala:
"17. ¿Qué fortalezas y limitaciones observa en los resultados presentados?" y
"18. ¿Qué mejoras sugeriría para el sistema?". Si tiene observaciones sobre ellas, anótelas aquí:

______________________________________________________________________________

### Observaciones generales

______________________________________________________________________________

______________________________________________________________________________

Gracias por su participación.

---

## Parte B. Hojas de valoraciones (para el investigador)

No se imprime para los jueces. Las valoraciones en papel se transcriben en archivos CSV que viven
solo en la carpeta `data/` del proyecto, excluida de git (ver [protocolo_datos.md](protocolo_datos.md)).

### Notas sobre la parte A

- El plan de tesis solo nombra los criterios ("claros, pertinentes y suficientes"); sus
  definiciones en la tabla "Qué se le pide" son una propuesta de este borrador.
- Las definiciones de las dimensiones y los 16 ítems son los del plan de tesis (§3.8.2 y
  Anexo A), sin cambios.
- La escala se fija con los jueces antes de entregar el formulario y es la misma para los tres
  criterios (nota N6). El número de jueces depende de U6; la tabla del valor mínimo de V del
  [protocolo](protocolo_datos.md) muestra lo que exige cada combinación.
- Cada juez firma antes el [consentimiento informado](consentimiento_informado.md), con el rol
  "Juez de validez de contenido".
- Se propone que los jueces no integren después el panel, para que cada experto conozca el
  cuestionario recién al responderlo. Si U6 no lo permite, se anota en la bitácora de la sesión.

### Archivos

| Archivo | Plantilla | Columnas |
|---|---|---|
| `claridad.csv` | [plantillas/jueces_claridad.csv](plantillas/jueces_claridad.csv) | `codigo`, `I01` a `I16` |
| `pertinencia.csv` | [plantillas/jueces_pertinencia.csv](plantillas/jueces_pertinencia.csv) | `codigo`, `I01` a `I16` |
| `suficiencia.csv` | [plantillas/jueces_suficiencia.csv](plantillas/jueces_suficiencia.csv) | `codigo`, `D1` a `D4` |
| `observaciones.csv` | [plantillas/jueces_observaciones.csv](plantillas/jueces_observaciones.csv) | `codigo`, `elemento`, `observacion` |

| Regla | Detalle |
|---|---|
| Formato | CSV en UTF-8, separado por comas, con una fila de encabezado. |
| Filas | Una por juez en cada archivo de valoraciones. `codigo` es el código del juez (`J01`, `J02`, …) y no se repite. |
| Celda | El número entero anotado por el juez, dentro de la escala fijada (nota N6). |
| Valoración faltante | Celda vacía. Se excluye del cálculo de ese ítem, de modo que `n` cuenta las valoraciones dadas (nota N8). |
| Observaciones | Una fila por observación. `elemento` es `I01` a `I16`, `D1` a `D4` (suficiencia), `P17`, `P18` o `GENERAL`. |

### Correspondencia con `aiken_validity` (T30)

`aiken_validity(ratings, dimensions, *, low, high, confidence, criterion, cutoff)` se ejecuta una
vez por criterio:

| Argumento | Claridad y pertinencia | Suficiencia |
|---|---|---|
| `ratings` | `{"I01": [...], ..., "I16": [...]}`: la columna de cada ítem leída de arriba abajo, con `None` en cada celda vacía | `{"D1": [...], ..., "D4": [...]}` |
| `dimensions` | `{"D1": ("I01", …, "I04"), "D2": ("I05", …, "I08"), "D3": ("I09", …, "I12"), "D4": ("I13", …, "I16")}` | `{"D1": ("D1",), "D2": ("D2",), "D3": ("D3",), "D4": ("D4",)}` |
| `low`, `high` | Los extremos de la escala fijada (nota N6) | Los mismos |
| `confidence`, `criterion`, `cutoff` | `0.95`, `"ci_lower"`, `0.50` (D31 (d)) | Los mismos |

La suficiencia se trata como un "ítem" por dimensión, como propone la nota N6. Un ítem cumple el
valor de aceptación cuando el límite inferior de su intervalo de puntaje al 95 % es al menos 0.50
(D31 (d)); el resultado informa V y ambos límites. La tabla del valor mínimo de V que lo cumple,
según el número de jueces y la escala, está en [protocolo_datos.md](protocolo_datos.md).
