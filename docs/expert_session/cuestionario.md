# Cuestionario de validación con expertos

> **Estado: borrador pendiente de validación por jueces (V de Aiken, T30).**
> La parte A transcribe sin cambios el Anexo A del plan de tesis (PDF pp. 117–118, impresas
> 108–109). Los ítems pueden cambiar después de la validación de contenido: solo se aplica al
> panel la versión que resulte de ella.

Este documento tiene dos partes:

- **Parte A, para imprimir:** el cuestionario que responde cada experto del panel.
- **Parte B, para el investigador:** la hoja de respuestas en la que se transcriben las
  respuestas y su correspondencia con el análisis (`likert.py`, T32).

---

## Parte A. Cuestionario (para imprimir)

**Código del participante:** ____________ **Fecha:** ____________

No escriba su nombre en esta hoja.

**Instrucciones:** valore su grado de acuerdo con cada afirmación empleando la siguiente escala:
1 = Totalmente en desacuerdo; 2 = En desacuerdo; 3 = Ni de acuerdo ni en desacuerdo; 4 = De
acuerdo; 5 = Totalmente de acuerdo. El instrumento será sometido a validación de contenido
mediante el coeficiente V de Aiken previo a su aplicación.

Marque una sola casilla por afirmación. Si prefiere no responder alguna, déjela en blanco.

### Dimensión 1: Coherencia temática

| N.° | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| 1 | Los documentos agrupados en cada grupo temático comparten efectivamente afinidad de tema. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 2 | Los términos representativos de cada grupo describen adecuadamente su contenido. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 3 | Los grupos reflejan líneas de investigación reconocibles de mi carrera o facultad. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 4 | La cantidad y granularidad de los grupos es apropiada para el repositorio analizado. | ☐ | ☐ | ☐ | ☐ | ☐ |

### Dimensión 2: Pertinencia de los vacíos de investigación

| N.° | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| 5 | Las áreas señaladas como poco exploradas corresponden a temas con escasa investigación previa en la facultad. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 6 | Los vacíos identificados constituyen oportunidades genuinas de investigación para tesistas. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 7 | La presentación de los vacíos es comprensible e interpretable. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 8 | Los vacíos detectados aportan información que no obtendría fácilmente con una revisión manual. | ☐ | ☐ | ☐ | ☐ | ☐ |

### Dimensión 3: Calidad de las recomendaciones

| N.° | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| 9 | Los antecedentes sugeridos para una propuesta son relevantes para esa propuesta. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 10 | Las líneas alternativas propuestas son útiles para reorientar o enriquecer una propuesta. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 11 | Las detecciones de redundancia temática señalan trabajos previos verdaderamente afines. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 12 | Las recomendaciones orientarían adecuadamente a un tesista real. | ☐ | ☐ | ☐ | ☐ | ☐ |

### Dimensión 4: Utilidad general y usabilidad

| N.° | Afirmación | 1 | 2 | 3 | 4 | 5 |
|---|---|:-:|:-:|:-:|:-:|:-:|
| 13 | El sistema sería útil como herramienta de apoyo en la elaboración de propuestas de titulación. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 14 | El sistema sería útil como apoyo para asesores en la evaluación de propuestas. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 15 | La interfaz del prototipo es clara y fácil de usar. | ☐ | ☐ | ☐ | ☐ | ☐ |
| 16 | Recomendaría la adopción institucional de una herramienta como esta. | ☐ | ☐ | ☐ | ☐ | ☐ |

### Preguntas abiertas

17. ¿Qué fortalezas y limitaciones observa en los resultados presentados?

    ______________________________________________________________________________

    ______________________________________________________________________________

    ______________________________________________________________________________

18. ¿Qué mejoras sugeriría para el sistema?

    ______________________________________________________________________________

    ______________________________________________________________________________

    ______________________________________________________________________________

Gracias por su participación.

---

## Parte B. Hoja de respuestas (para el investigador)

No se imprime para el panel. Las respuestas en papel se transcriben en dos archivos CSV que
viven solo en la carpeta `data/` del proyecto, excluida de git (ver
[protocolo_datos.md](protocolo_datos.md)).

### Notas sobre la parte A

- La última frase de las instrucciones ("El instrumento será sometido a validación…") es del
  borrador. Se retira de la versión que se aplique al panel, una vez validada.
- La línea "Marque una sola casilla…" y el campo de código se agregaron para la aplicación; no
  cambian ningún ítem.
- Los 16 ítems son afirmativos (§3.8.2, "cuatro ítems afirmativos"), así que ninguno se recodifica
  (nota N11).

### Archivo de respuestas cerradas: `respuestas.csv`

Plantilla: [plantillas/respuestas_cuestionario.csv](plantillas/respuestas_cuestionario.csv).

| Regla | Detalle |
|---|---|
| Formato | CSV en UTF-8, separado por comas, con una fila de encabezado. En Excel: "CSV UTF-8 (delimitado por comas)". |
| Columnas | `codigo`, luego `I01` a `I16`, en ese orden. Ninguna otra columna: ni nombre, ni perfil, ni fecha. |
| Filas | Una por participante, en cualquier orden. `codigo` es el código de su tarjeta (`E01`, `E02`, …) y no se repite. |
| Celda | El número entero de 1 a 5 de la casilla marcada en el ítem de esa columna. |
| Respuesta faltante | Celda vacía. También queda vacía si hay dos casillas marcadas o la marca es ilegible; el caso se anota en la bitácora de la sesión. |
| Participante retirado | Se borra su fila completa (ver el protocolo). |

Ejemplo con datos ficticios (tres participantes; `E02` dejó en blanco el ítem 3):

```csv
codigo,I01,I02,I03,I04,I05,I06,I07,I08,I09,I10,I11,I12,I13,I14,I15,I16
E01,4,4,3,4,5,4,4,3,4,4,3,4,5,4,4,4
E02,5,4,,4,4,4,5,4,3,4,4,4,4,5,4,5
E03,3,4,4,3,4,3,4,4,4,3,4,3,4,4,5,4
```

### Archivo de respuestas abiertas: `abiertas.csv`

Plantilla: [plantillas/respuestas_abiertas.csv](plantillas/respuestas_abiertas.csv). Columnas
`codigo`, `P17` y `P18`; cada celda copia el texto escrito por el participante, entre comillas
dobles si contiene comas o saltos de línea. Si el texto menciona a una persona, se reemplaza su
nombre por `[nombre omitido]` al transcribir. Este archivo no lo lee `likert.py`: alimenta la
síntesis cualitativa del tercer momento (§3.8.3).

### Correspondencia con `likert_analysis` (T32)

`likert_analysis(responses, dimensions, *, low, high, reverse_coded, quartile_method)` recibe las
respuestas por ítem, no por participante:

| Argumento | Valor |
|---|---|
| `responses` | `{"I01": [...], ..., "I16": [...]}`: cada lista es la columna del ítem leída de arriba abajo, un valor por fila, con `None` en cada celda vacía. Todas las listas siguen el mismo orden de filas. La columna `codigo` no entra. |
| `dimensions` | `{"D1": ("I01", "I02", "I03", "I04"), "D2": ("I05", …, "I08"), "D3": ("I09", …, "I12"), "D4": ("I13", …, "I16")}` |
| `low`, `high` | `1` y `5` |
| `reverse_coded` | `()` (nota N11) |
| `quartile_method` | `"linear"` (propuesta de la nota N13, por confirmar) |

Las dimensiones siguen el Anexo A: D1 Coherencia temática, D2 Pertinencia de los vacíos de
investigación, D3 Calidad de las recomendaciones y D4 Utilidad general y usabilidad. El criterio
de éxito (c) se lee en `respondent_scores.median` de cada dimensión (D31 (c)). La lectura de los
archivos y la llamada están en [protocolo_datos.md](protocolo_datos.md).
