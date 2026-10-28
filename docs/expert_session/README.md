# Expert session kit

Documents for the expert validation of the prototype (plan T30 to T32; thesis plan §3.8 and
§3.10). Everything here is a **draft**: the facilitator-, judge- and participant-facing documents
are in neutral Spanish, and this index is for developers.

## Files, in order of use

| Step | File | What it is | Audience | Status |
|:-:|---|---|---|---|
| 1 | [formulario_jueces.md](formulario_jueces.md) | The judges' content-validity form (Aiken's V): clarity and relevance per item, sufficiency per dimension; part B maps its sheets to `aiken_validity` | Judges, researcher | Draft; the rating scale is a blank (N6) |
| 2 | [consentimiento_informado.md](consentimiento_informado.md) | The informed-consent template, for judges and panel experts | Participants | Draft template with `[completar: …]` blanks |
| 3 | [guion_sesion.md](guion_sesion.md) | The session script: timing, what the facilitator says and must not say, the three views' demo, materials, session log | Facilitator | Draft; its defaults are listed at its end |
| 4 | [muestras_anonimizadas.md](muestras_anonimizadas.md) | The specification of the result samples (A temporal-split assignments, B simulated queries, C topics, D gaps), their anonymization and their `{{…}}` placeholders | Researcher, samples unit | Specification; no sample yet |
| 5 | [cuestionario.md](cuestionario.md) | A faithful transcription of the thesis plan's Anexo A questionnaire; part B maps the answer sheet to `likert_analysis` | Panel, researcher | Draft pending the judges' validation (T30) |
| 6 | [protocolo_datos.md](protocolo_datos.md) | How answers are coded, stored under the gitignored `data/` tree, transcribed, withdrawn and analyzed | Researcher | Draft; no command reads the sheets yet |
| — | [plantillas/](plantillas/) | Header-only CSV templates of the six answer sheets | Researcher | Fixed by the layouts above |

## Placeholder conventions

| Marker | Filled by | Example |
|---|---|---|
| `[completar: …]` | The researcher, by hand, before use: contact data, dates, retention periods, institutional facts, the judges' scale | `[completar: nombre completo]` |
| `{{…}}` | The samples unit, from data | `{{TITULO}}` |

No document states an institutional fact (a committee, an approval number, a date) that the plan
does not record. The consent says only that the institutional authorization was requested.

## Sheet layouts and the analysis functions

| Sheet | Columns | Read by |
|---|---|---|
| `cuestionario/respuestas.csv` | `codigo`, `I01`–`I16`; integers 1–5; blank = missing | `likert_analysis` (T32) |
| `cuestionario/abiertas.csv` | `codigo`, `P17`, `P18`; free text | Qualitative synthesis only |
| `jueces/claridad.csv`, `jueces/pertinencia.csv` | `codigo`, `I01`–`I16`; integers on the judges' scale; blank = missing | `aiken_validity` (T30) |
| `jueces/suficiencia.csv` | `codigo`, `D1`–`D4` | `aiken_validity`, one "item" per dimension (N6) |
| `jueces/observaciones.csv` | `codigo`, `elemento`, `observacion` | Rewording of items (§3.8.2) |

Each item column, read top to bottom, becomes one list of the function's mapping, so every list
shares the row order; the `codigo` column is not passed. The proposed reading is the snippet in
`protocolo_datos.md`, checked on synthetic sheets outside the repository; the unit that closes T30
and T32 turns it into a tested command. The sheets live under
`data/expert_session/<snapshot_id>/` (gitignored); no name ever reaches a digital file.

## Pending

- **U6:** the panel (six to eight teachers), the judges and their consent.
- **N6:** the judges' rating scale and its anchors; the form leaves it as a blank.
- **The samples:** the second part of T31 generates them once T27 (the temporal split and its
  expert sample) and T28 (the query view) exist and T22 confirms the winner; it also runs the name
  guard of D34 (1) and `check_privacy` on any tracked output.
- **τ:** while T22 has not calibrated the redundancy threshold, the query view and sample B show
  similarity rankings without a verdict, and the script says so.
- **Decisions for the user** recorded with this kit: item 11 while τ does not exist, a per-case
  rating of the temporal-split sample (O04), and a second judges' round for reworded items.
