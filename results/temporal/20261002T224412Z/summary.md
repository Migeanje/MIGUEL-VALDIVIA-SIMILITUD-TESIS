# Temporal split of snapshot 20261002T224412Z

T19's winner C4 (sentence-transformers/paraphrase-multilingual-mpnet-base-v2, hdbscan 10) fitted with no new sweep (D30 E3) on the 458 theses of 2021 to 2024, once per seed run of T19, at commit `b16101628937c4faa0a213f204345d08a276fdc6`. The 286 theses of 2025 to 2026 are assigned to its topics by nearest centroid (D14) and flagged atypical below O06's P5 of the member-to-own-centroid similarities (global, linear). Each seed's assignment is compared with T19's full-map labels of the same seed. The reference run is `sentence-transformers--paraphrase-multilingual-mpnet-base-v2__hdbscan_mcs10__seed42` (seed 42), T19's medoid run, whose map gives the expert sample.

The winner is provisional until T22 closes stage 2 of the selection (D30 E3); if it changes, this split is computed again by the same rules.

Which theses were assigned where, and the expert sample, are in the gitignored `data/processed/` only. O04 declares no pass threshold, so none is applied here.

## Periods

| Year | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|
| Theses | 139 | 116 | 105 | 98 | 160 | 126 |

| Program | 2021 to 2024 | 2025 to 2026 |
|---|---|---|
| sistemas | 51 | 14 |
| industrial | 195 | 144 |
| electronica | 32 | 14 |
| mecanica | 143 | 48 |
| minas | 37 | 66 |
| All | 458 | 286 |

2 of the later theses carry an issue date after the snapshot date (T06); they are assigned by their issue year, as every other thesis.

## Earlier maps

| Seed | Topics | Noise | Terms | Atypical threshold | Atypical | Full-map noise |
|---|---|---|---|---|---|---|
| 7 | 11 | 112 (24.5 %) | 12453 | 0.6171 | 11 of 286 (3.8 %) | 79 |
| 13 | 12 | 144 (31.4 %) | 12453 | 0.6225 | 12 of 286 (4.2 %) | 84 |
| 21 | 12 | 129 (28.2 %) | 12453 | 0.6304 | 14 of 286 (4.9 %) | 72 |
| 42 (reference) | 12 | 131 (28.6 %) | 12453 | 0.6311 | 13 of 286 (4.5 %) | 71 |
| 73 | 14 | 124 (27.1 %) | 12453 | 0.6227 | 11 of 286 (3.8 %) | 84 |

## Consistency of the later theses (O04), 2025 to 2026

| Seed | ARI own_label | ARI exclude_either | NMI own_label | NMI exclude_either | Agreement own_label | Agreement exclude_either |
|---|---|---|---|---|---|---|
| 7 | 0.301 | 0.634 | 0.594 | 0.781 | 0.524 | 0.705 |
| 13 | 0.300 | 0.650 | 0.592 | 0.789 | 0.385 | 0.545 |
| 21 | 0.346 | 0.623 | 0.599 | 0.765 | 0.451 | 0.570 |
| 42 (reference) | 0.367 | 0.680 | 0.631 | 0.805 | 0.479 | 0.614 |
| 73 | 0.350 | 0.761 | 0.637 | 0.836 | 0.545 | 0.772 |
| Mean (SD) | 0.333 (0.030) | 0.670 (0.055) | 0.611 (0.022) | 0.795 (0.027) | 0.477 (0.064) | 0.641 (0.095) |

ARI and NMI compare the later theses' topics with their full-map labels, whatever the label ids. Agreement is the share of the later theses whose full-map label is their topic's counterpart: the full-map label of most of the topic's own earlier members, learned from the earlier theses alone. `own_label` counts the full map's noise as one more label; `exclude_either` leaves out the theses that are noise in either map (D30 E4: `own_label` primary).

## Earlier theses on both maps, 2021 to 2024

| Seed | ARI own_label | ARI exclude_either | NMI own_label | NMI exclude_either |
|---|---|---|---|---|
| 7 | 0.315 | 0.612 | 0.633 | 0.860 |
| 13 | 0.406 | 0.816 | 0.681 | 0.930 |
| 21 | 0.414 | 0.755 | 0.676 | 0.899 |
| 42 (reference) | 0.404 | 0.829 | 0.674 | 0.935 |
| 73 | 0.406 | 0.870 | 0.693 | 0.936 |
| Mean (SD) | 0.389 (0.041) | 0.776 (0.101) | 0.672 (0.023) | 0.912 (0.033) |

How far the map itself moves when the later theses join it: the earlier theses' labels on the earlier map against their full-map labels.

## Expert sample (O04)

Expert sample: 40 of the 286 later theses on the reference run's map (seed 42), 10 of them atypical, drawn with seed 20261027: up to 10 atypical, the rest typical, each pool spread over the programs with at least 2 per program that has any, and shown in a seeded order.

| Pool | Program | Candidates | Drawn |
|---|---|---|---|
| atypical | sistemas | 5 | 3 |
| atypical | industrial | 3 | 2 |
| atypical | electronica | 2 | 2 |
| atypical | mecanica | 2 | 2 |
| atypical | minas | 1 | 1 |
| typical | sistemas | 9 | 2 |
| typical | industrial | 141 | 13 |
| typical | electronica | 12 | 3 |
| typical | mecanica | 46 | 5 |
| typical | minas | 65 | 7 |
