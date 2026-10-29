# Muestras anonimizadas: especificación

> **Cómo se generan.** Las muestras las genera el comando
> `uv run python -m thematic_redundancy.evaluation.run_expert_samples`, que lee los registros
> guardados del mapa temático (T19), de la partición temporal con su muestra para expertos (T27,
> O04) y de la corrida de vacíos (T24a), reemplaza cada marcador `{{…}}` con datos, aplica la
> guarda de nombres y escribe las muestras, su clave y su composición en
> `data/expert_session/<snapshot_id>/muestras/`. No corre mientras no exista la partición
> temporal; con `--dry-run` calcula y verifica todo sin escribir nada. Mientras la configuración
> ganadora sea provisional, cada muestra lo indica, y cuando T22 la confirme se generan de nuevo.
> Los campos `[completar: …]` los llena el investigador a mano.

Las muestras son la "muestra representativa de sus resultados sobre el corpus institucional" que
se presenta a cada experto (plan de tesis, §3.8.3), "con los datos personales anonimizados
conforme a la sección 3.10". Se usan en los bloques 4 a 8 del [guion](guion_sesion.md).

## Qué se muestra

| Muestra | Fuente | Qué muestra cada caso | Ítems que apoya | Tamaño por defecto |
|---|---|---|---|---|
| **A. Asignaciones de la partición temporal** | T27: mapa de 2021–2024 y tesis de 2025–2026 (O04) | Una tesis de 2025–2026 y el grupo del mapa de 2021–2024 al que el sistema la asigna, con los términos del grupo, tres tesis del grupo, la marca de propuesta atípica (O06) y el cuartil de saturación del grupo (T23) | 1, 3, 12 | 10 casos, hasta 2 por carrera |
| **B. Consultas simuladas** | T27 con la búsqueda de T22 y el recomendador de T26, como los muestra la vista de consulta (T28) | Una tesis de 2025–2026 como propuesta; las tesis de 2021–2024 más similares, con su posición y su similitud coseno, **sin veredicto de redundancia** mientras τ no exista; los antecedentes sugeridos y las líneas alternativas | 9, 10, 11, 12 | 5 casos, 1 por carrera, tomados de la muestra A; 5 trabajos por ranking |
| **C. Grupos temáticos** | El mapa de la configuración ganadora (T19, tras T22) | Un grupo: su tamaño, su distribución por carrera, sus términos representativos y cinco de sus tesis | 1, 2, 3, 4 | 6 grupos |
| **D. Vacíos** | Los vacíos estables de la corrida de referencia (T24a) | Un vacío: su tipo en palabras simples, sus carreras y años, los términos de sus grupos, su número de tesis y hasta tres tesis de ejemplo | 5, 6, 7, 8 | Hasta 2 por regla, 8 como máximo |

Las muestras A, B y D son las que pide la especificación de este kit (T31). La muestra C y las recomendaciones de la muestra B
vienen de §3.8.3, que nombra "grupos temáticos con sus documentos y términos representativos,
ejemplos de vacíos identificados y casos de recomendaciones generadas".

## Cómo se eligen los casos

- **Sorteo con semilla, declarado antes de generar.** La regla de cada muestra, su semilla y sus
  estratos se fijan antes de ver ningún caso y quedan en el manifiesto del sorteo.
- **Sin elegir a mano.** Ningún caso se cambia después de verlo. Un caso inutilizable (por
  ejemplo, sin resumen) se reemplaza por el siguiente del sorteo, y el reemplazo se registra.
- **A:** hasta 2 tesis de 2025–2026 por carrera, al azar dentro de cada carrera.
- **B:** 1 caso por carrera, al azar entre los de A.
- **C:** 6 grupos al azar entre los de la corrida de referencia, sin las tesis que no quedan en
  ningún grupo; en cada uno, 5 tesis al azar entre sus miembros, no las más cercanas al centroide.
- **D:** hasta 2 vacíos estables al azar por cada regla (a) a (d).

Así la muestra es representativa e incluye los errores del sistema, que también se evalúan.

## Anonimización

| Campo | ¿Se muestra? | Motivo |
|---|---|---|
| Nombres de autores, asesores y jurados | No | §3.10 |
| ORCID y DNI | No | §3.10; el DNI ya se descarta al cosechar los metadatos |
| Códigos de autor y de asesor | No | No hacen falta para evaluar |
| Código del documento (`DOC-…`), uuid y handle | No | Cada caso lleva un código de muestra (`A01`, `B03`, `C02`, `D05`); la clave que lo vincula con el documento queda en `data/` |
| Enlace a la página del repositorio o al PDF | No | Esas páginas muestran a los autores (como en D34 (2)) |
| Título | Sí | Sin él no se puede juzgar el contenido; §3.8.3 pide mostrar los grupos "con sus documentos" |
| Resumen y objetivos | Solo en las propuestas de B | Son lo que recibe la vista de consulta (§3.9); en el resto, el título basta y la lectura cabe en la sesión |
| Palabras clave | En A y B | Ayudan a juzgar la asignación |
| Carrera y año | Sí | Los piden el ítem 3 y las reglas de vacíos (c) y (d) |
| Términos de los grupos | Sí | Ya están en los registros versionados de T19 |
| Similitud coseno y posición | En B | Es el resultado de la consulta |
| Veredicto de redundancia | No, mientras τ no exista | T22 aún no calibra el umbral |
| Un nombre de persona, un ORCID, un código de tesis, un handle o un enlace dentro de un texto | La generación se detiene y no escribe ninguna muestra; el aviso no repite el texto encontrado | Como el candado de nombres de los libros de etiquetado (D34 (1)) |

**Los títulos se muestran.** El etiquetado del conjunto evaluado (D34 (2)) muestra título,
resumen, objetivos y palabras clave, sin autores, asesores ni enlaces, porque un asesor puede
conocer una tesis y la página del repositorio muestra a sus autores. Las muestras siguen la misma
lógica: se muestran los textos y no las personas ni los enlaces. A diferencia del etiquetado, los
expertos no producen la verdad de referencia, sino una valoración de la utilidad percibida, y
§3.8.3 pide mostrar los grupos con sus documentos; sin títulos no se podrían responder los ítems
1, 9 y 11. Esto no cambia el plan de tesis: §3.10 anonimiza a las personas, no los títulos.

## Verificaciones antes de usar las muestras

- [ ] Cada texto de las muestras pasó la guarda de nombres de D34 (1): se busca, sin límites de
      palabra, cada forma de nombre de persona y cada ORCID del snapshot, y cualquier coincidencia
      detiene la generación.
- [ ] Ninguna muestra contiene un código `DOC-…`, un uuid, un handle ni un enlace.
- [ ] Las muestras y su clave están bajo `data/expert_session/<snapshot_id>/muestras/`, excluida
      de git. Lo único versionado es la composición agregada (conteos por muestra, carrera y
      regla), que pasa `check_privacy` de `gaps/results_store.py`, como en T20a y T24a.
- [ ] El texto "τ pendiente" de la muestra B coincide con el estado real de T22.
- [ ] Cada juego impreso lleva un número para recogerlo al final de la sesión.

---

## Plantillas de las muestras

Lo que sigue es lo que se imprime. Cada bloque se repite por caso.

### Muestra A. Asignaciones de tesis de 2025–2026 al mapa de 2021–2024

Cada caso es una tesis de 2025 o 2026, tratada como propuesta nueva, y el grupo temático del mapa
de 2021–2024 al que el sistema la asigna.

#### Caso {{A_CODIGO}}

- **Carrera y año:** {{CARRERA}}, {{ANIO}}
- **Título:** {{TITULO}}
- **Palabras clave:** {{PALABRAS_CLAVE}}
- **Grupo asignado:** {{GRUPO}}. Términos: {{TERMINOS_GRUPO}}
- **Tesis del grupo (2021–2024):** {{TITULOS_GRUPO}}
- **Propuesta atípica:** {{ATIPICA}}
- **Saturación del grupo:** {{CUARTIL_SATURACION}}

### Muestra B. Consultas simuladas

Cada caso es una tesis de 2025 o 2026 ingresada como propuesta en la vista de consulta y
comparada con las tesis de 2021 a 2024.

#### Caso {{B_CODIGO}}

**Propuesta** ({{CARRERA}}, {{ANIO}})

- **Título:** {{TITULO}}
- **Resumen:** {{RESUMEN}}
- **Objetivos:** {{OBJETIVOS}}
- **Palabras clave:** {{PALABRAS_CLAVE}}

**Grupo asignado:** {{GRUPO}}. Términos: {{TERMINOS_GRUPO}}

**Trabajos previos más similares**

| Posición | Título | Carrera | Año | Similitud |
|:-:|---|---|:-:|:-:|
| 1 | {{TITULO_1}} | {{CARRERA_1}} | {{ANIO_1}} | {{SIMILITUD_1}} |
| … | … | … | … | … |
| {{K}} | {{TITULO_K}} | {{CARRERA_K}} | {{ANIO_K}} | {{SIMILITUD_K}} |

**Veredicto de redundancia:** {{VEREDICTO}}. Mientras τ no esté calibrado: "No disponible: el
umbral de redundancia aún no está calibrado."

**Antecedentes sugeridos:** {{ANTECEDENTES}}

**Líneas alternativas:** {{LINEAS_ALTERNATIVAS}}

### Muestra C. Grupos temáticos

#### Grupo {{C_CODIGO}}

- **Tesis en el grupo:** {{TAMANIO}}. Por carrera: {{DISTRIBUCION_CARRERAS}}
- **Términos representativos:** {{TERMINOS}}

| Tesis de ejemplo | Carrera | Año |
|---|---|:-:|
| {{TITULO_1}} | {{CARRERA_1}} | {{ANIO_1}} |
| … | … | … |

### Muestra D. Vacíos de investigación

#### Vacío {{D_CODIGO}}

- **Tipo:** {{REGLA_TEXTO}}
- **Carreras:** {{CARRERAS}}. **Años:** {{ANIOS}}
- **Grupos y términos:** {{GRUPOS_TERMINOS}}
- **Tesis en el vacío:** {{N_TESIS}}
- **Tesis de ejemplo:** {{TITULOS_EJEMPLO}}

`{{REGLA_TEXTO}}` toma uno de estos textos fijos:

| Regla (D32) | Texto para el experto |
|---|---|
| (a) | Grupo pequeño pero coherente: tiene pocas tesis y sus términos son consistentes entre sí. |
| (b) | Zona entre dos grupos vecinos: casi ninguna tesis conecta sus temas. |
| (c) | Tema presente en una carrera y ausente en una carrera afín, donde podría ser transferible. |
| (d) | Tema en descenso: su participación bajó entre 2021–2023 y 2024–2026. |

Para el investigador, los valores declarados (D32): (a) un tamaño de, a lo sumo, el primer cuartil
de los tamaños de los grupos y un NPMI de, al menos, el piso declarado; (b) dos grupos entre los 3
más cercanos uno del otro, con a lo sumo 1 tesis puente; (c) presente con al menos 5 tesis y el
5 % de la carrera de origen y ausente con 0 tesis en una carrera de afinidad 1 o 2 en la matriz
U4; (d) una caída de al menos 2.5 puntos porcentuales en la proporción de tesis, con al menos 3
tesis en 2021–2023. Un vacío es estable si aparece en al menos 4 de las 5
corridas con un solapamiento de Jaccard de al menos 0.5.

## Marcadores

| Marcador | Contenido |
|---|---|
| `{{A_CODIGO}}`, `{{B_CODIGO}}`, `{{C_CODIGO}}`, `{{D_CODIGO}}` | Código de muestra del caso, asignado al azar; nunca el código del documento |
| `{{CARRERA}}`, `{{ANIO}}` | Nombre de la carrera (de `config/default.yaml`) y año de la tesis |
| `{{TITULO}}`, `{{RESUMEN}}`, `{{OBJETIVOS}}`, `{{PALABRAS_CLAVE}}` | Textos de la ficha, después de la guarda de nombres |
| `{{GRUPO}}`, `{{TERMINOS_GRUPO}}`, `{{TERMINOS}}` | Número del grupo en la corrida usada y sus 10 términos representativos |
| `{{TITULOS_GRUPO}}`, `{{TITULOS_EJEMPLO}}` | Títulos de tesis, uno por línea, cada uno con su carrera y año |
| `{{ATIPICA}}` | "Sí" o "No", según la regla de O06 |
| `{{CUARTIL_SATURACION}}` | Q1 a Q4, según T23 |
| `{{K}}`, `{{TITULO_K}}`, `{{CARRERA_K}}`, `{{ANIO_K}}`, `{{SIMILITUD_K}}` | Posición, título, carrera, año y similitud coseno de cada trabajo del ranking, con 2 decimales |
| `{{VEREDICTO}}` | "No disponible" mientras τ no exista |
| `{{ANTECEDENTES}}`, `{{LINEAS_ALTERNATIVAS}}` | Salida del recomendador (T26): títulos de antecedentes, y grupos vecinos y vacíos cercanos con sus términos |
| `{{TAMANIO}}`, `{{DISTRIBUCION_CARRERAS}}`, `{{N_TESIS}}` | Conteos |
| `{{REGLA_TEXTO}}`, `{{CARRERAS}}`, `{{ANIOS}}`, `{{GRUPOS_TERMINOS}}` | Del registro del vacío (T24a) |
