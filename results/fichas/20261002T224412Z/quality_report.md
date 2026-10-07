# Fichas quality report of snapshot 20261002T224412Z

Counts, public handles and lemmas only. The full figures are in `quality_report.json`;
every count except the first four covers the included theses.

## At a glance

| Measure | Value |
|---|---:|
| Records, one ficha each | 766 |
| Included theses | 744 |
| Excluded as `duplicate` | 2 |
| Excluded as `non_thesis_type` | 20 |
| Objectives extracted | 728 |
| Metadata only (title and abstract) | 16 |

## Included theses

| Field | Theses |
|---|---|
| Program | sistemas 65, industrial 339, electronica 46, mecanica 191, minas 103 |
| Issue year | 2021 139, 2022 116, 2023 105, 2024 98, 2025 160, 2026 126 |
| Objectives status | extracted 728, no_pdf 16 |
| Section source | title_abstract 16, title_abstract_objectives 728 |
| PDF status | already_present 582, downloaded 147, no_thesis_file 1, restricted 14 |
| Source format | digital 84, mixed 644, none 16 |
| Rights | embargoed 8, open 730, restricted 6 |
| Quality notes | advisor ORCID iD is malformed or fails its check digit; the advisor code comes from the name 21, objectives locator flag: ocr_page 4, PDF on disk belongs to another item 1, objectives locator flag: general_too_long 1, objectives locator flag: specific_missing 1 |

## People and dates

- Advisor ORCID iD on 712 theses: 691 valid, 21 malformed or failing the check digit.
- Advisor codes from the ORCID iD 691, from the name 53, none 0; 138 distinct advisor codes and 784 distinct author codes.
- Future-dated: `20.500.12920/16319`, `20.500.12920/16433`.
- Field completeness: abstract 744, objectives 728, issue_year 744, keywords 744, ocde_codes 744, language 744, embargo_end 9, advisor_code 744, author_codes 744, pdf_sha256 729.

| Text | Theses | Median characters | P95 characters |
|---|---:|---:|---:|
| title | 744 | 147.0 | 217 |
| abstract | 744 | 1720.5 | 2886 |
| objectives | 728 | 702.5 | 1306 |

## D24 duplicates

| Stays | Duplicate | Stays is the smaller uuid | Same PDF | Equal objectives |
|---|---|---|---|---|
| `20.500.12920/15247` | `20.500.12920/15265` | yes | no | yes |
| `20.500.12920/16449` | `20.500.12920/16450` | yes | yes | yes |

## Domain stopwords in the objectives

Over the 728 included theses with objectives, by the full cleaner (spaCy es_core_news_md) without domain stopwords; a lemma counts once per thesis.

- 79 domain stopwords; 3 of them occur in no objectives.
- Most frequent listed: determinar 284, evaluar 282, analizar 270, identificar 257, actual 252, propuesta 238, arequipa 189, situación 151, permitir 146, aplicar 136, proponer 134, basado 127, utilizar 117, establecer 101, resultado 95.
- 102 lemmas not on the list reach 5% of the theses. The 25 most frequent: diseñar 282, sistema 272, proceso 262, empresa 260, mejora 251, costo 208, desarrollar 206, análisis 200, implementar 182, económico 179, implementación 168, gestión 163, mejorar 157, optimizar 141, modelo 135, diseño 130, metodología 130, elaborar 124, beneficio 122, herramienta 119, aplicación 118, tiempo 116, área 110, control 105, estudio 102.
- Shaped like an infinitive, candidates only for the user's review (the list is unchanged): diseñar, desarrollar, implementar, mejorar, optimizar, elaborar, reducir, diagnosticar, seleccionar, validar, comparar, incrementar, definir, medir.
