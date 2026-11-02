# Configuration and seeds

Every run reads one file, `config/default.yaml`, validated on load by `load_config`
(`shared/config.py`), plus two pinned inputs: `config/career_affinity.yaml` and
`config/stopwords_domain_es.txt`. The values were declared before the steps that use them, and
each change is logged in the plan. The experiment declaration is pinned by a SHA-256 that a
test and the experiment command both check.

Marks: (M) measured from a file or a command. Decision ids (D, O, U) refer to the plan,
`odd/tasks/thematic-redundancy-system.md`, "Accepted decisions", "Open decisions" and "Inputs
the user must secure"; SR ids are the decisions of the 2026-10-27 speed run that wait for the
user (plan "Next step").

## Quick facts

| Fact | Value |
|---|---|
| SHA-256 of `config/default.yaml` at `2e8f9df` | `cba05e394ecfa41475218aa682053dc3118fd2a74622feade47ba444a1f291e1` (M): the file since `2d41c2c`, which added `expert_samples` |
| Experiment declaration SHA-256 | `b974c54c3378c8fe02f7983323f0d58fa7672e4ca7d1999df15f708d2b76850b`, unchanged since D30 |
| Blocks | 23 (M), in the order of the table below |
| Loader | `load_config`: a YAML key written twice in one mapping is refused, and each block is a validated pydantic model with bounds |
| Changing a value | Log it in the plan's "Initial configuration"; a pinned value needs a new decision |

Each run manifest records the SHA-256 of the file it read, so a rerun can be matched with the
configuration of its commit. Since the gap run, the file changed five times (M: each value
equals `git show <commit>:config/default.yaml`):

| Commit | SHA-256 | What the commit added | Recorded by |
|---|---|---|---|
| `1e8fe3a` | `e77120585f75122418c5a07b367502a269a256884e69b897f44a55070737d9db` | — (the gap run's commit) | The gap run's manifest |
| `2eb3249` | `0dc4db287a812073d4ac38d8b60cfba4eb61e609e43b43373622ad2cc20810b6` | The `labeling` block | The labeled-set draw's manifest |
| `a9b8969` | `12155d3fd030c638f4c18f60faf1209081677674c2c88a6be6a0380150ba1128` | Four `redundancy` keys: T22's calibration values | No run |
| `ec2abf6` | `3ec8f5984f872a183be157179d6a3d5b63c9c172b29fddb3dd0aa972dcf7ab1f` | The `proposal` and `temporal` blocks | The temporal split's manifest (run at `b161016`) |
| `af46c30` | `27dec213f1b1703d88b9f78493298e946f22ff6c1e49e676049f09a96a950c1a` | The key `labeling.annotators` | The pilot workbook's manifest (run at `7a9fe30`) |
| `2d41c2c` | `cba05e394ecfa41475218aa682053dc3118fd2a74622feade47ba444a1f291e1` | The `expert_samples` block | The expert samples' gitignored manifest (run at `2e8f9df`, which holds this file; D: the command records `default.yaml` among its inputs) |

## Blocks of `config/default.yaml`

In file order. "Read by" names the commands of the [how-to-run guide](how_to_run.md).

| Block | Values | What it controls | Fixed by | Read by |
|---|---|---|---|---|
| `paths` | `data_dir` data, `results_dir` results, `tessdata_dir` tessdata | Where every command reads and writes, relative to the project root; a data directory that resolves outside the root is refused | T01 | Every command |
| `repository` | `base_url`, `faculty_community_uuid`, `page_size` 100 (1–1000), `request_interval_seconds` 1.0 (1.0–60.0), `max_retries` 3 (0–10) | The DSpace API and the politeness of every request | T04, D04 | `harvest`, `download_pdfs` |
| `snapshot` | `year_start` 2021, `year_end` 2026, `thesis_type` tesis, 5 `programs` (key, name, collection uuid) | The corpus filter and the program collections | D01, D02, T04 | `harvest`, `download_pdfs`, the fichas build, the gap run, the temporal split |
| `ocr` | `languages` spa+eng, `dpi` 300 | Tesseract on pages without a text layer | D07, T03 | `ocr_assets`, `extract_text` |
| `extraction` | `min_text_chars` 50, `ocr_window_pages` 100, `header_footer` (`edge_lines` 3, `min_share` 0.3, `min_pages` 5) | Low-text pages, the OCR window and running-line removal | T09, D25 | `extract_text` |
| `objectives` | `general_max_chars` 1200, `specific_max_chars` 4000, `min_chars` 40, `verification_sample` (`per_program` 12, `seed` 20261007, `workbook_dir` data/labels/objectives_check) | The objectives locator's bounds and the manual verification sample | T10, O05 | `locate_objectives`, `objectives_check` |
| `fichas` | `wrong_pdfs` (one item whose PDF is another item's), `expected` (766, 744, 2, 20, 728, 16 for `20261002T224412Z`) | PDFs whose text is not used, and the counts the build must reach | T12, D24, T05 finding | `build_fichas` |
| `chunking` | `max_tokens` 128, `expected` (744 documents, 5,217 chunks, 744 title and 4,473 body chunks) | The token budget per chunk under both tokenizers, and the counts the chunk build must reach | D09, D27, T13 | `build_chunks`; the app's query page, which chunks a proposal as a thesis |
| `embedding` | `models` (MiniLM, mpnet), `batch_size` 64, `device` cpu, `threads` 10, `document_vector` section_mean, `sensitivity_cosine_threshold` 0.95, `expected` (744 documents, 5,217 chunks) | The two models, how they run, which document vector later steps read, and the expected counts | D09, T14 | `build_chunks`, `build_embeddings`, `run_experiment`, `draw_pairs`; through T19's or T20a's loader, `run_temporal`, `run_expert_samples` and `run_redundancy`; the app's query page |
| `tfidf` | `max_ngram` 2, `min_df` 2, `max_df` 1.0, `sublinear_tf` true, `document_vector` section_mean, `sensitivity_cosine_threshold` 0.95, `expected` (744, 2,216 sections, 14,308 terms, 98,282 nonzeros) | The TF-IDF baseline and the counts its build must reach | D28, D10, D26, T15 | `build_tfidf`, `run_experiment`, `draw_pairs`; through T19's or T20a's loader, `run_temporal`, `run_expert_samples` and `run_redundancy` |
| `umap` | `n_neighbors` 15, `n_components` 5, `min_dist` 0.0, `metric` cosine | The reduction before clustering; part of the experiment declaration | D30 E1 | `run_experiment`, `run_temporal` (the winner refitted) |
| `kmeans` | `k_start` 8, `k_stop` 50, `k_step` 2 | The K sweep, 22 values; part of the declaration | D30 E1 | `run_experiment` |
| `hdbscan` | `min_cluster_sizes` [5, 10, 15, 20] | The `min_cluster_size` grid; part of the declaration | D30 E1 | `run_experiment` |
| `seeds` | [7, 13, 21, 42, 73] | The five seeds of every stochastic step of the experiment; part of the declaration | D30 E1 | `run_experiment`, `run_gaps`; `run_temporal` takes the five from T19's records |
| `experiment` | The D30 declaration: K-means and HDBSCAN arguments, the TF-IDF baseline, the K and size choices, c-TF-IDF, NPMI, ARI, the selection gates and fallbacks, the bootstrap of F1, the criterion-4 cap | How T19 runs and picks its winner, and how T22 closes stage 2 | D30, T18 | `run_experiment`, `run_redundancy` |
| `topic_map` | `projection` (`method` umap, `n_neighbors` 15, `min_dist` 0.0, `metric` cosine) | The 2-D map for display only; outside the declaration and its pin, and still waiting for the user's confirmation (plan T16) | T16 | `run_experiment` (its map); the app shows the saved map |
| `redundancy` | `cv_folds` 5, `cv_seed` 7, `cv_stratified` true, `f1_tie` highest_tau, `final_tau` all_pairs, `known_duplicates` (one pair) | The cross-validation that chooses τ: its folds, their seed, stratification by the binary label, the tie rule and the final τ from all the labeled pairs; and the pair known to be one thesis deposited twice | T22, D13, D15, O09; the four calibration keys were declared on 2026-10-27 as T22's notes 2 to 5 proposed, and wait for the user's confirmation before the real run (SR-T22-1) | `run_redundancy`; `draw_pairs` and the app's query page (the known pair) |
| `recommender` | `top_k` 5, `mmr_lambda` 0.7, `neighbor_topics` 3, `nearby_gaps` 3 | The antecedents and the alternative lines | Initial configuration, T26 | The app's query page (T28b), `run_expert_samples` (sample B) |
| `gaps` | `temporal_split_year` 2024, `temporal_min_count` 3, `bridge_max_docs` 1, `nearest_topics` 3, `stability_min_runs` 4, `npmi_floor_quantile` 0.25, `min_affinity` 1, `present_min_count` 5, `present_min_share` 0.05, `rare_max_share` 0.0, `min_decline` 0.025, `stability_min_overlap` 0.5 | Gap rules (a) to (d) and their stability filter | D16, D32, D29, U4, T24 | `run_gaps`; `run_expert_samples`, which computes the gap rules again to check them against the gap run's records. The app rebuilds the stable gaps with the values that the gap run's manifest records |
| `labeling` | `seed` 20261021, `sbert_high_per_model` 20, `tfidf_high` 20, `mid_range` 30, `random` 30, `top_share` 0.01, `mid_low` 0.90, `mid_high` 0.99, `stratum_floor` 2, `pilot_size` 10, `pilot_floor` 2, `records_dir` data/labels/labeled_set, `annotators` null | The pools, strata and pilot of the labeled-set draw; the workbooks' order, sides and assignment; the number of annotators, 3 or 4, `null` until the user confirms it, which keeps the annotators' workbooks from being made; the records directory must lie inside `paths.data_dir` | D34, D15, T20a; `annotators` T20b, U3, D34 (8) | `draw_pairs`, `pair_workbooks`; `run_redundancy` (the consensus labels under `records_dir`) |
| `proposal` | `atypical_scope` global, `atypical_method` linear | O06's atypical flag of a thesis assigned by nearest centroid (D14): below the 5th percentile of the member-to-own-centroid similarities over every clustered thesis of the map, with numpy's linear percentile | O06, T23 (the values the user confirmed on 2026-10-20), T27 | `run_temporal`. The app's query page uses the same two values as constants of `app/query.py`, not this block |
| `temporal` | `split_year` 2025, `sample` (`seed` 20261027, `size` 40, `atypical` 10, `program_floor` 2) | The temporal split: the winner fitted on 2021–2024, the theses of 2025–2026 assigned to it; and the expert sample's size, atypical quota and floor per program. Not gap rule (d)'s `gaps.temporal_split_year` | O04, D30 E3, T27 | `run_temporal` |
| `expert_samples` | `seed` 20261031, `assignments_per_program` 2, `group_examples` 3, `queries_per_program` 1, `ranking_size` 5, `topics` 6, `topic_examples` 5, `gaps_per_rule` 2, `gap_examples` 3, `terms` 10, `saturation_method` linear, `saturation_boundary` lower | The sizes and draws of samples A to D of the expert session | T31, `docs/expert_session/muestras_anonimizadas.md`; the user kept sample B's maps as built (SR-T31b-1, 2026-10-29) | `run_expert_samples` |

The `expected` counts of `fichas`, `chunking`, `embedding` and `tfidf` hold for the snapshot
they name only; another snapshot needs its own counts, or the build stops. The
[data card](data_card.md) explains them.

## The pinned experiment declaration

The declaration (D30, fixed on 2026-10-19 before any run) is the whole `experiment` block plus
the values it reuses from other blocks: `umap`, `kmeans`, `hdbscan`, `seeds`,
`embedding.models`, `embedding.document_vector` and `tfidf.document_vector`. The blocks added
since (`redundancy`'s new keys, `proposal`, `temporal`, `expert_samples` and
`labeling.annotators`) are outside it, so its SHA-256 did not change.

| Check | Where | What it does |
|---|---|---|
| Unit test | `tests/unit/test_experiment_declaration.py` | Builds the declaration from the loaded configuration as canonical JSON (keys sorted, no whitespace, UTF-8) and compares its SHA-256 with `D30_DECLARATION_SHA256`; also checks the run counts that D30 derived and that the fallback model is one of `embedding.models` |
| Run-time check | `check_declaration`, `topic_map/experiment.py` | `run_experiment` hashes `experiment_declaration()` (`shared/config.py`) and refuses to run unless it equals `DECLARED_SHA256` |
| Record | `results/experiments/20261002T224412Z/t19/manifest.json` | `declaration_sha256` holds the pinned value (M) |

A change to any pinned value fails the test and stops the experiment; it needs a new decision
in the plan, then a new pin.

## Other pinned inputs

| File | What it holds | Fixed by | Pin |
|---|---|---|---|
| `config/career_affinity.yaml` | The career-affinity matrix of gap rule (c): scores 0 to 2 per pair of programs; `electronica` is never a gap target | U4 (2026-10-19, before any topic result), D29 | `U4_PAIRS_SHA256` in `tests/unit/test_career_affinity.py` |
| `config/stopwords_domain_es.txt` | The 89 domain stopwords of the full cleaner (TF-IDF, c-TF-IDF, NPMI) | T11, D26 | `D26_SHA256` in `tests/unit/preprocessing/test_domain_stopwords.py` |

## Seeds

Every seeded random choice derives from one of these seeds, so the same configuration and
inputs give the same outputs.

| Seed | Configuration key | Used by | Where |
|---|---|---|---|
| 7, 13, 21, 42, 73 | `seeds` | The `random_state` of each UMAP fit (one per model and seed) and of each K-means run, the TF-IDF baseline included; HDBSCAN takes none | `topic_map/reduction.py`, `topic_map/clustering.py`, through `run_experiment` (T19) |
| The same five | `seeds` | The 2-D map: the representative run's seed as the projection's `random_state` (seed 42 for the T19 winner) | `topic_map/reduction.py` (`topic_map.projection`) |
| The same five | `seeds` | The gap run reads the five seed runs of the winner; a gap is stable when present in at least `gaps.stability_min_runs` (4) of them | `gaps/run_gaps.py` (T24a) |
| The same five | `seeds` | The temporal split refits each of the winner's five seed runs on 2021–2024 with that run's seed as UMAP's `random_state` | `topic_map/temporal.py`, through `run_temporal` (T27), which takes the runs and their seeds from T19's records |
| 20261007 | `objectives.verification_sample.seed` | The O05 sample of 60 theses: `random.Random("<seed>:<program>:<status>")` per stratum, then `"<seed>:order"` for the row order | `make_sample`, `labeling/objectives_check.py` (T10) |
| 7 | `experiment.selection.f1.bootstrap_seed` | The paired bootstrap of the labeled-set F1, 2,000 resamples: `numpy.random.default_rng(seed)` | `redundancy/comparison.py`, through `run_redundancy` (T22; not run yet, it needs the labels of T21) |
| 7 | `redundancy.cv_seed` | The fold split that chooses τ: `StratifiedKFold` (with `cv_stratified` true) or `KFold`, `shuffle=True`, `random_state=seed` | `redundancy/threshold.py`, through `run_redundancy` (T22; not run yet; the value waits for SR-T22-1) |
| 20261021 | `labeling.seed` | Every choice of the labeled-set draw: `random.Random("<seed>:<phase>:<pool>")`, with `:<model>` or `:<stratum>` appended where the pool needs it | `labeling/pairs.py`, through `draw_pairs` (T20a) |
| The same | `labeling.seed` | The workbooks: each workbook's pair order, `random.Random("<seed>:order:<workbook>")`, and each pair's sides, `random.Random("<seed>:sides:<pair id>")`; the assignment of the labeled pairs to the pairs of annotators, `"<seed>:assignment:<pool>"`; the order in which disputed pairs go to adjudicators, `"<seed>:adjudication"` | `labeling/workbook.py`, `labeling/assignment.py`, `labeling/label_import.py`, through `pair_workbooks` (T20b, T20c) |
| 20261027 | `temporal.sample.seed` | The expert sample of 40 later theses: `numpy.random.default_rng(seed)` | `topic_map/temporal.py`, through `run_temporal` (T27) |
| 20261031 | `expert_samples.seed` | Samples A to D: sample k draws from `numpy.random.default_rng([seed, k])`, k 1 to 4, so one sample's sizes leave the others' draws alone | `evaluation/expert_samples.py`, through `run_expert_samples` (T31; run on 2026-10-29) |

One random choice has no seed, by design: the blind code of each drawn pair (two letters and
three digits, `secrets.choice` in `labeling/workbook.py`) must be unrelated to the pair id. The
`pilot` step makes the codes once and keeps them in the gitignored `blind_codes.parquet`; a
rerun after moving the records away gives new codes, so its `blind_codes.parquet` and pilot
workbook have SHA-256 values other than the recorded ones, while the order and sides stay
those of `labeling.seed`.

The T19 manifest also records a repeatability check: the MiniLM UMAP fit and the K-means run
at K 8 with seed 7 were each repeated, and both gave identical SHA-256 values (M, `repeatability`
in `results/experiments/20261002T224412Z/t19/manifest.json`).
