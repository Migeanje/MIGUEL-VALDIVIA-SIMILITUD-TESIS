# Gaps of snapshot 20261002T224412Z

Rules (a) to (d) of D32 and their stability filter on the 5 seed runs of T19's winner C4 (sentence-transformers/paraphrase-multilingual-mpnet-base-v2, hdbscan 10), at commit `1e8fe3abaf7fb94e0a2d4ff9249d886a2718df17`. The reference run is `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42` (seed 42), T19's medoid run, whose map is shown.

The winner is provisional until T22 closes stage 2 of the selection (D30 E3); if it changes, the cuts and the gaps are computed again by the same rules.

Topics are named by run and topic id; their terms are T19's (`topics.parquet`). Which theses form each gap is in the gitignored `data/processed/` only.

## Rule (a) cuts

Rule (a) size cut 15.0: the first quartile of the sizes of the 92 topics of the 5 runs.
Rule (a) NPMI floor 0.1389: the 0.25 quantile of their 92 defined per-topic NPMI values (primary setting).

## Gaps per run

| Seed | (a) | (b) | (c) | (d) | All | Stable |
|---|---|---|---|---|---|---|
| 7 | 3 | 5 | 23 | 2 | 33 |  |
| 13 | 6 | 10 | 27 | 4 | 47 |  |
| 21 | 5 | 6 | 22 | 2 | 35 |  |
| 42 (reference) | 1 | 6 | 22 | 2 | 31 | 24 of 31 |
| 73 | 5 | 10 | 22 | 1 | 38 |  |

## Stable gaps of the reference run

Stable gaps of the reference run: 24 of its 31, present in at least 4 of the 5 runs at a Jaccard overlap of at least 0.5: (a) 1, (b) 2, (c) 19, (d) 2.

### Rule (a)

- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__a_t2`, in 5 of 5 runs: topic 2 has 15 theses, at most the bottom-quartile size 15 of the topics of every run (17 in this map), and its NPMI 0.3118 reaches the floor 0.138871.
  - Topic 2 (15 theses): estudiante, educativo, aprendizaje, virtual, físico, realidad virtual, realidad, alumno, escolar, computacional

### Rule (b)

- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__b_t2-t5`, in 5 of 5 runs: topics 2 and 5 are among each other's 3 nearest topics (rank 3, centroid cosine 0.6164) and have 1 bridge thesis, at most 1.
  - Topic 2 (15 theses): estudiante, educativo, aprendizaje, virtual, físico, realidad virtual, realidad, alumno, escolar, computacional
  - Topic 5 (17 theses): ergonómico, puesto, puesto trabajo, trabajador, evaluación, trabajo, riesgo, mejora, personal, evaluación ergonómico
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__b_t7-t9`, in 4 of 5 runs: topics 7 and 9 are among each other's 3 nearest topics (rank 3, centroid cosine 0.7837) and have 0 bridge theses, at most 1.
  - Topic 7 (12 theses): ventilación, vapor, gas, sistema ventilación, sistema, caldera, gas natural, combustible, aire, natural
  - Topic 9 (27 theses): eléctrico, instalación eléctrico, instalación, sistema, tensión, protección, red, distribución, subestación, sistema eléctrico

### Rule (c)

Affinity 2, the high-confidence group of D32 (5): 6; affinity 1: 13.

- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t0_industrial-mecanica`, in 5 of 5 runs: topic 0 holds 29 of 339 industrial theses (8.6%) and 0 of 191 mecanica theses (0.0%, at most 0.0%); affinity 2, at least 1.
  - Topic 0 (30 theses): mercado, estudio, plan, producto, producción, empresa, proyecto, orgánico, económico, demanda
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t0_industrial-minas`, in 5 of 5 runs: topic 0 holds 29 of 339 industrial theses (8.6%) and 0 of 103 minas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 0 (30 theses): mercado, estudio, plan, producto, producción, empresa, proyecto, orgánico, económico, demanda
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t2_sistemas-industrial`, in 5 of 5 runs: topic 2 holds 14 of 65 sistemas theses (21.5%) and 0 of 339 industrial theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 2 (15 theses): estudiante, educativo, aprendizaje, virtual, físico, realidad virtual, realidad, alumno, escolar, computacional
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t2_sistemas-mecanica`, in 5 of 5 runs: topic 2 holds 14 of 65 sistemas theses (21.5%) and 0 of 191 mecanica theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 2 (15 theses): estudiante, educativo, aprendizaje, virtual, físico, realidad virtual, realidad, alumno, escolar, computacional
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t3_mecanica-sistemas`, in 5 of 5 runs: topic 3 holds 37 of 191 mecanica theses (19.4%) and 0 of 65 sistemas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 3 (41 theses): diseño, estructural, sistema, componente, molino, máquina, estructura, material, fabricación, elemento
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t3_mecanica-minas`, in 5 of 5 runs: topic 3 holds 37 of 191 mecanica theses (19.4%) and 0 of 103 minas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 3 (41 theses): diseño, estructural, sistema, componente, molino, máquina, estructura, material, fabricación, elemento
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t6_mecanica-sistemas`, in 5 of 5 runs: topic 6 holds 14 of 191 mecanica theses (7.3%) and 0 of 65 sistemas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 6 (17 theses): módulo, temperatura, agua, medición, fluido, control, sistema, presión, combustión, laboratorio
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t6_mecanica-industrial`, in 5 of 5 runs: topic 6 holds 14 of 191 mecanica theses (7.3%) and 0 of 339 industrial theses (0.0%, at most 0.0%); affinity 2, at least 1.
  - Topic 6 (17 theses): módulo, temperatura, agua, medición, fluido, control, sistema, presión, combustión, laboratorio
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t6_mecanica-minas`, in 4 of 5 runs: topic 6 holds 14 of 191 mecanica theses (7.3%) and 0 of 103 minas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 6 (17 theses): módulo, temperatura, agua, medición, fluido, control, sistema, presión, combustión, laboratorio
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t8_mecanica-industrial`, in 5 of 5 runs: topic 8 holds 18 of 191 mecanica theses (9.4%) and 0 of 339 industrial theses (0.0%, at most 0.0%); affinity 2, at least 1.
  - Topic 8 (33 theses): control, sistema, robot, diseño, motor, módulo, controlador, artificial, prototipo, dispositivo
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t8_mecanica-minas`, in 5 of 5 runs: topic 8 holds 18 of 191 mecanica theses (9.4%) and 0 of 103 minas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 8 (33 theses): control, sistema, robot, diseño, motor, módulo, controlador, artificial, prototipo, dispositivo
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t9_mecanica-sistemas`, in 5 of 5 runs: topic 9 holds 25 of 191 mecanica theses (13.1%) and 0 of 65 sistemas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 9 (27 theses): eléctrico, instalación eléctrico, instalación, sistema, tensión, protección, red, distribución, subestación, sistema eléctrico
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t9_mecanica-industrial`, in 4 of 5 runs: topic 9 holds 25 of 191 mecanica theses (13.1%) and 0 of 339 industrial theses (0.0%, at most 0.0%); affinity 2, at least 1.
  - Topic 9 (27 theses): eléctrico, instalación eléctrico, instalación, sistema, tensión, protección, red, distribución, subestación, sistema eléctrico
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t9_mecanica-minas`, in 5 of 5 runs: topic 9 holds 25 of 191 mecanica theses (13.1%) and 0 of 103 minas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 9 (27 theses): eléctrico, instalación eléctrico, instalación, sistema, tensión, protección, red, distribución, subestación, sistema eléctrico
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t12_sistemas-mecanica`, in 5 of 5 runs: topic 12 holds 19 of 65 sistemas theses (29.2%) and 0 of 191 mecanica theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 12 (27 theses): web, sistema, módulo, proyecto, framework, desarrollo, proceso, gestión, privado, arquitectura
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t13_industrial-sistemas`, in 5 of 5 runs: topic 13 holds 49 of 339 industrial theses (14.5%) and 0 of 65 sistemas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 13 (51 theses): productividad, proceso, manufacturing, lean manufacturing, empresa, lean, mejora, herramienta, producción, textil
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t14_industrial-mecanica`, in 4 of 5 runs: topic 14 holds 51 of 339 industrial theses (15.0%) and 0 of 191 mecanica theses (0.0%, at most 0.0%); affinity 2, at least 1.
  - Topic 14 (56 theses): almacén, empresa, inventario, gestión, logístico, mejora, proceso, metodología, producto, compra
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t16_industrial-mecanica`, in 4 of 5 runs: topic 16 holds 35 of 339 industrial theses (10.3%) and 0 of 191 mecanica theses (0.0%, at most 0.0%); affinity 2, at least 1.
  - Topic 16 (39 theses): lean, área, proceso, cliente, empresa, servicio, mejora, atención, service, lean service
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__c_t16_industrial-minas`, in 4 of 5 runs: topic 16 holds 35 of 339 industrial theses (10.3%) and 0 of 103 minas theses (0.0%, at most 0.0%); affinity 1, at least 1.
  - Topic 16 (39 theses): lean, área, proceso, cliente, empresa, servicio, mejora, atención, service, lean service

### Rule (d)

- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__d_t3`, in 5 of 5 runs: topic 3 fell from 26 of 360 theses in 2021-2023 (7.2%) to 15 of 384 in 2024-2026 (3.9%), a decline of 3.3%, at least 2.5%.
  - Topic 3 (41 theses): diseño, estructural, sistema, componente, molino, máquina, estructura, material, fabricación, elemento
- `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42__d_t8`, in 4 of 5 runs: topic 8 fell from 21 of 360 theses in 2021-2023 (5.8%) to 12 of 384 in 2024-2026 (3.1%), a decline of 2.7%, at least 2.5%.
  - Topic 8 (33 theses): control, sistema, robot, diseño, motor, módulo, controlador, artificial, prototipo, dispositivo

## Sensitivity

Reported beside the declared rules and never used to select a gap (D32 (1)). A cell counts the run's gaps of the rule; for `literal_b`, the theses it flags, those that are noise, and the Jaccard index with the reference run's.

| Variant | Rule | Setting | Seed 7 | Seed 13 | Seed 21 | Seed 42 | Seed 73 | Stable |
|---|---|---|---|---|---|---|---|---|
| primary | (a) | declared values (gaps) | 3 | 6 | 5 | 1 | 5 | 1 |
| primary | (b) | declared values (gaps) | 5 | 10 | 6 | 6 | 10 | 2 |
| primary | (c) | declared values (gaps) | 23 | 27 | 22 | 22 | 22 | 19 |
| primary | (d) | declared values (gaps) | 2 | 4 | 2 | 2 | 1 | 2 |
| literal_a | (a) | size below 5 % of 744 theses (at most 37), no NPMI floor | 11 | 15 | 13 | 10 | 15 | 10 |
| literal_b | (b) | mean cosine distance to the centroids above the P75 | 186 (54 noise, J 0.908) | 186 (55 noise, J 0.888) | 186 (55 noise, J 0.927) | 186 (60 noise, J 1.000) | 186 (54 noise, J 0.918) |  |
| rare_max_share | (c) | rare_max_share 0.01 (declared 0) | 27 | 31 | 28 | 28 | 26 | 25 |
| min_overlap | (a) | stability_min_overlap 0.75 (declared 0.5) | 3 | 6 | 5 | 1 | 5 | 1 |
| min_overlap | (b) | stability_min_overlap 0.75 (declared 0.5) | 5 | 10 | 6 | 6 | 10 | 2 |
| min_overlap | (c) | stability_min_overlap 0.75 (declared 0.5) | 23 | 27 | 22 | 22 | 22 | 16 |
| min_overlap | (d) | stability_min_overlap 0.75 (declared 0.5) | 2 | 4 | 2 | 2 | 1 | 2 |

Declared values: nearest_topics 3, bridge_max_docs 1, temporal_first_year 2021, temporal_split_year 2024, temporal_last_year 2026, temporal_min_count 3, stability_min_runs 4, npmi_floor_quantile 0.25, min_affinity 1, present_min_count 5, present_min_share 0.05, rare_max_share 0, min_decline 0.025, stability_min_overlap 0.5.
