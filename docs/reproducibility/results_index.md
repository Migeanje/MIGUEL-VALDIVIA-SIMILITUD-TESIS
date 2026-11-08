# Results index

Every file that git tracks under `results/` at `16866f9`, 36 files (M): the command
that wrote it, the commit it ran at, where its provenance is recorded and what it holds. All
but one are aggregates for snapshot `20261002T224412Z`; the other is the manifest of the
OpenAlex Topics download (T25), which records a public topic list. None holds a title, an
abstract, a person's name or an ORCID iD. A second table lists the tracked files that commands in `main` will write but have
not written yet, and a third the gitignored files under `data/` that the same commands write,
by name and role only.

> **Provisional.** The T19 records, the gap run and the temporal split are built on a
> provisional winner, C4 (mpnet with HDBSCAN, `min_cluster_size` 10). T22 confirms or changes
> the model once the labeled set is labeled (T21), by D30's stage 2 (UD3). If the model
> changes, those three are run again by the same rules. Every other result is final for this
> snapshot.

## Tracked results

"Ran at" is the commit the command ran at, as its record states, or the commit that holds the
file when the record names none. Commands are in the [how-to-run guide](how_to_run.md).

### Snapshot card (T06)

No command: written from the files on 2026-10-03, in commit `a5889ad`; last changed in
`3b05679` (a note on the wrong-file item).

| File | Holds |
|---|---|
| `results/snapshot/20261002T224412Z/data_card.md` | The freeze digests, the counts by program, year, type and rights, the embargo list, the duplicates, the PDF statuses and the text-layer profile |

### Metadata profile (T07, T07a)

```sh
uv run python experiments/eda_metadata.py
```

Ran at `7c6a4e1` with a clean tree (`provenance.git_commit`, `git_uncommitted_changes`
false, in `metadata_profile.json`); committed in `4628b25`. The four figures are byte-identical to those of the first run,
committed in `96c410c` (ledger, 2026-10-13).

| File | Holds |
|---|---|
| `results/eda/20261002T224412Z/metadata_profile.json` | The whole profile with its provenance: the snapshot and configuration SHA-256, the git commit and both tokenizers |
| `results/eda/20261002T224412Z/summary.md` | Tables and observations for chunking, title cleaning and topic modeling |
| `results/eda/20261002T224412Z/program_year.png` | Theses per program and year |
| `results/eda/20261002T224412Z/abstract_tokens.png` | Abstract lengths in tokens |
| `results/eda/20261002T224412Z/title_tokens.png` | Title lengths in tokens |
| `results/eda/20261002T224412Z/top_keywords.png` | The most frequent keywords |

### Stopword candidates (T11)

```sh
uv run python experiments/domain_stopwords.py
```

Ran at `f21743f` with uncommitted changes (`provenance.git_uncommitted_changes` true);
committed in `e9f4f70`.

| File | Holds |
|---|---|
| `results/eda/20261002T224412Z/stopword_candidates.json` | The 411 lemmas found in at least 5 % of the 744 distinct theses, each with its document count, and the method; the source of the hand-made list in `config/stopwords_domain_es.txt` |

### Text extraction (T09)

```sh
uv run python -m thematic_redundancy.extraction.extract_text \
    --summary-out results/extraction/20261002T224412Z/summary.json
```

The file records no commit; its `updated_at` is 2026-10-06T03:57:20Z, and it is committed in
`0390ff7`. It copies the settings and counts of the gitignored page manifest.

| File | Holds |
|---|---|
| `results/extraction/20261002T224412Z/summary.json` | The settings, their fingerprint and `extraction_version` 2; 731 PDFs, 141,076 pages by source (138,598 text layer, 1,958 OCR, 520 empty), 0 errors, and the seconds |

### Objectives (T10)

```sh
uv run python -m thematic_redundancy.extraction.locate_objectives \
    --summary-out results/objectives/20261002T224412Z/summary.json
uv run python -m thematic_redundancy.labeling.objectives_check \
    --report-out results/objectives/20261002T224412Z/verification.json
```

The second command imports the workbook of the manual verification. The files are committed
in `6f1ba96` and `c121d78`.

| File | Holds |
|---|---|
| `results/objectives/20261002T224412Z/summary.json` | The locator's manifest, numbers only: settings, the table's SHA-256, 731 extracted and 0 not found, by program, heading pattern and flag; created 2026-10-07T15:38:27Z |
| `results/objectives/20261002T224412Z/verification.json` | The O05 accuracy report: 60 of 60 general objectives judged correct, accuracy 1.0 with a Wilson 95 % interval from 0.9398, `meets_target` true; the workbook's SHA-256; imported 2026-10-07T21:43:43Z |

### Fichas (T12)

```sh
uv run python -m thematic_redundancy.corpus.build_fichas \
    --report-out results/fichas/20261002T224412Z
```

Created 2026-10-07T22:36:24Z (`provenance.created_at`), with the SHA-256 of the metadata, the
objectives table and the fichas table; committed in `73f1676`. Later builds ran without
`--report-out` and wrote the same fichas table (ledger, 2026-10-11).

| File | Holds |
|---|---|
| `results/fichas/20261002T224412Z/quality_report.json` | The counts of the [data card](data_card.md) §3, the quality notes, the advisor ORCID checks, completeness, text lengths, the duplicate pairs by public handle, and the stopword re-check |
| `results/fichas/20261002T224412Z/quality_report.md` | The same, readable |

The stopword re-check in this report counted the 79 lemmas of the list at that time; D26 raised
the list to 89 lemmas on 2026-10-11, after the report.

### Experiment (T19), provisional

```sh
uv run python -m thematic_redundancy.topic_map.run_experiment \
    --project-root . --snapshot-id 20261002T224412Z
```

Offline (`HF_HUB_OFFLINE=1`). Ran at `bf0ca62` with a clean tree (`git_commit` in the
manifest), 2026-10-20T14:41:36Z to 14:48:05Z; committed in `9d75b11`.

| File | Holds |
|---|---|
| `results/experiments/20261002T224412Z/t19/manifest.json` | Written last: snapshot, `declaration_sha256`, git commit, the SHA-256 of the three input manifests, `provisional` true, winner C4, counts, the repeatability check, library versions, CPU, times and the SHA-256 of every other file, the gitignored ones included |
| `results/experiments/20261002T224412Z/t19/runs.parquet` | One row per clustering run: representation, model revision, algorithm with K or `min_cluster_size`, seed, clusters and noise, the internal metrics, NPMI for the final runs, and run times |
| `results/experiments/20261002T224412Z/t19/configurations.parquet` | One row per configuration and hyperparameter: seed means and SDs of every metric, the ARI mean, minimum and matrix, and the chosen flags |
| `results/experiments/20261002T224412Z/t19/topics.parquet` | The final runs' topics: id, size, the top 10 c-TF-IDF terms with their scores, and topic NPMI; vocabulary lemmas, no thesis text |
| `results/experiments/20261002T224412Z/t19/selection.json` | Every comparison of the selection with the step that decided it, the winner, criteria 1 and 4, the representative run, the map's run and the findings |

### Gap run (T24a), provisional

```sh
uv run python -m thematic_redundancy.gaps.run_gaps --snapshot-id 20261002T224412Z
```

Offline, twice. Ran at `1e8fe3a` (`git_commit` in the manifest), 2026-10-21T22:58:15Z;
committed in `925600a`. The committed manifest is the second run's; the two runs' other files
are byte-identical (ledger).

| File | Holds |
|---|---|
| `results/gaps/20261002T224412Z/manifest.json` | Written last: snapshot, git commit, the SHA-256 of T19's records and of every input, the values used, `provisional` true, counts (184 gaps over the 5 runs, 24 stable), library versions, CPU, times and the SHA-256 of every output |
| `results/gaps/20261002T224412Z/cuts.json` | Rule (a)'s size cut 15.0 and NPMI floor 0.1389 over the 92 topics of the winner's 5 seed runs |
| `results/gaps/20261002T224412Z/gaps.parquet` | One row per gap of every run: rule, run, topics as `(run_id, topic)`, programs and years with their counts, shares, affinity, and the support and stability of the reference run's gaps; no doc code, no term |
| `results/gaps/20261002T224412Z/sensitivity.parquet` | D32's sensitivity table beside the declared values; never used for selection |
| `results/gaps/20261002T224412Z/summary.md` | The stable gaps with their topics' T19 terms, the counts per rule and the sensitivity table |

### Labeled-set draw (T20a)

```sh
uv run python -m thematic_redundancy.labeling.draw_pairs --snapshot-id 20261002T224412Z
```

Offline, once. Ran at `2eb3249` (`git_commit` in the manifest), 2026-10-22T02:52:26Z; committed
in `90fedf2`. Not provisional: it scores the pairs under both models and TF-IDF, not under the
winner.

| File | Holds |
|---|---|
| `results/labeled_set/20261002T224412Z/draw/manifest.json` | Written last: snapshot, git commit, seed 20261021 and the values, the SHA-256 of every input and output, counts (276,396 pairs, 10 pilot and 120 labeled pairs), library versions, CPU and times |
| `results/labeled_set/20261002T224412Z/draw/composition.parquet` | Each pool's candidates, quota and drawn pairs per stratum, for the pilot and the labeled set |
| `results/labeled_set/20261002T224412Z/draw/ranges.parquet` | Each pool's score and percentile-rank ranges under each representation and of the mean rank |
| `results/labeled_set/20261002T224412Z/draw/cuts.json` | The pairs per stratum, each representation's top-1 % cut score with its strata, and the mid-range's candidates |
| `results/labeled_set/20261002T224412Z/draw/summary.md` | The same, readable, with the values used |

The "Declared values" line of this `summary.md` prints the seed as `2.0261e+07`: at the draw's
commit, `_settings_text` (`labeling/draw_pairs.py`) formatted every value with `:g`. Since
`8aa5a90` it prints an integer exactly, so a new draw would print `20261021` (README,
"Labeled-set draw"); the recorded draw is left as it was written. The exact seed is in
`manifest.json`.

### Temporal split (T27), provisional

```sh
uv run python -m thematic_redundancy.topic_map.run_temporal --snapshot-id 20261002T224412Z
```

Offline (`HF_HUB_OFFLINE=1`), once, after a dry run at the same commit; a second real run
refused. Ran at `b161016` (`git_commit` in the manifest), 2026-10-28T17:14:55Z to 17:15:28Z;
committed in `eea584a`. Every record says `provisional` true. The six files hold 36,162 B (D).

| File | Holds |
|---|---|
| `results/temporal/20261002T224412Z/manifest.json` | Written last: snapshot, git commit, the SHA-256 of T19's four records (its gitignored labels included) and of every input, winner C4 and `provisional` true, T19's five seed runs and their five refits, the values used, counts (744 theses, 458 fitted on 2021–2024, 286 assigned from 2025–2026, 2 of them issued after the snapshot; 12,453 of T15's 14,308 terms; the sample of 40 with 10 atypical; per program and per year), library versions, CPU, times and the SHA-256 of every output, the gitignored ones included |
| `results/temporal/20261002T224412Z/runs.parquet` | 5 rows (M), one per seed run: theses fitted and assigned, topics, noise and its share, terms, the full map's topics and noise among the assigned, O06's threshold, the atypical theses and their share, and the step times |
| `results/temporal/20261002T224412Z/consistency.parquet` | 20 rows (M), one per seed run, period and noise handling (`own_label`, `exclude_either`): the points compared, ARI, NMI and, for the later theses, the counterpart agreement |
| `results/temporal/20261002T224412Z/topics.parquet` | 61 rows (M), one per topic of each earlier map: size, the top terms and their scores, O06's threshold, the counterpart under each noise handling, and the later theses assigned and flagged atypical; vocabulary lemmas, checked by hand against the snapshot's person-name tokens before the commit (D30 E6, plan T27) |
| `results/temporal/20261002T224412Z/composition.parquet` | 10 rows (M), one per pool (atypical, typical) and program: the expert sample's candidates and quota |
| `results/temporal/20261002T224412Z/summary.md` | The periods by year and program, the earlier maps, the metrics per seed with their mean and SD, and the sample's composition |

### Labeled-set workbooks (T20b)

```sh
uv run python -m thematic_redundancy.labeling.pair_workbooks pilot --snapshot-id 20261002T224412Z
```

Offline, once, after a dry run at the same commit; a second run refused. Ran at `7a9fe30`
(`git_commit` in the manifest), 2026-10-28T21:02:10Z; committed in `58198e5`. Only the `pilot`
step has run: the pilot workbook waits for the calibration meeting.

| File | Holds |
|---|---|
| `results/labeled_set/20261002T224412Z/workbooks/pilot_manifest.json` | Written last: the step, snapshot, git commit, seed 20261021 with `annotators` null and the records directory, counts (130 drawn pairs; 10 pilot pairs over 20 theses, 5 shown with their sides swapped, 0 cells redacted; the name guard's 3,312 strings over 144 distinct texts), the SHA-256 of its 5 inputs and of its 3 gitignored records, library versions and times; no CPU record. No pair id, blind code or doc code |

### OpenAlex Topics reference (T25)

```sh
uv run python -m thematic_redundancy.reference.openalex_topics
```

Online, once, under the user's authorization (U8, granted 2026-11-04 for one anonymous download
of the full Topics list). Cursor paging got HTTP 401 on its second page on 2026-11-05 and wrote
nothing; the user then chose page-number paging (2026-11-06). Ran at `a26fa9e` with T25's first
slice uncommitted (`git_commit` and the 14 paths of `git_dirty` in the manifest),
2026-11-06T23:21:05Z, 38.4 s (M); committed in `16866f9`. It belongs to no snapshot:
it records the public Topics list (CC0) that its gitignored data file holds.

| File | Holds |
|---|---|
| `results/reference/openalex/topics/20261106T232105Z/manifest.json` | Written last: the download id, the retrieval time, the URL, the first request's query (`per-page` 200, `page` 1) and the User-Agent; the `openalex` values and the configuration's SHA-256; `meta.count` 4,516, 4,516 records written with 4,516 distinct ids, 23 pages and 23 requests (M); the path, size (38,214,080 B) and SHA-256 of the gitignored `topics.jsonl`; the git commit and changed paths, and the versions of Python, httpx and pydantic. No thesis data |

## Tracked results not written yet

The README names these tracked records; at `2e8f9df` no run has written them, because each
waits for people or for an earlier record (the [package index](README.md#pending)). `<id>` is
the snapshot id. `run_expert_samples` writes no tracked file at all: its outputs are
gitignored (below).

| Command | Files | Waits for |
|---|---|---|
| `pair_workbooks import-pilot`, `main` | `results/labeled_set/<id>/workbooks/pilot_labels_manifest.json`, `main_manifest.json` | The calibration meeting (D34 (7)); then U3, `labeling.annotators` |
| `pair_workbooks import-main`, `adjudication`, `import-adjudication` | `results/labeled_set/<id>/workbooks/main_labels_manifest.json`, `adjudication_manifest.json`, `adjudication_labels_manifest.json` | The labeling round (T21) |
| `run_redundancy` | `results/experiments/<id>/t22/representations.parquet`, `folds.parquet`, `curves.parquet`, `comparisons.parquet`, `selection.json`, `summary.md`, `manifest.json` | The consensus labels; SR-T22-1 |

## Gitignored files the commands write

Under `data/` (the `paths.data_dir`), `tessdata/` and the Hugging Face cache. Names and roles
only: these files hold thesis text, codes that name theses, or large arrays, and never enter
git. `<id>` is the snapshot id.

| Command | Files | Role |
|---|---|---|
| `ocr_assets` | `tessdata/spa.traineddata`, `tessdata/eng.traineddata`, `tessdata/manifest.json` | The Tesseract language files and their record |
| `harvest` | `data/raw/<id>/metadata.jsonl`, `data/raw/<id>/manifest.json` | The snapshot's records and its manifest |
| `download_pdfs` | `data/raw/<id>/pdfs/<item uuid>.pdf`, `data/raw/<id>/pdfs/manifest.json` | The thesis PDFs and their statuses and checksums |
| `extract_text` | `data/interim/<id>/pages/<item uuid>.parquet`, `data/interim/<id>/pages/manifest.json` | The page texts of each PDF and the extraction manifest |
| `locate_objectives` | `data/interim/<id>/objectives.parquet`, `data/interim/<id>/objectives_manifest.json` | The objectives of each PDF and their manifest |
| `locate_objectives --make-sample` | `data/labels/objectives_check/<id>/objectives_check.xlsx`, `data/labels/objectives_check/<id>/sample.json` | The O05 verification workbook and its sample |
| `build_fichas` | `data/interim/<id>/fichas.parquet`, `data/interim/<id>/exclusions.parquet`, `data/interim/<id>/fichas_manifest.json`, `data/interim/pseudonym.key` | The corpus table, the exclusion log, their manifest, and the pseudonym key (created once) |
| `build_chunks` | `data/interim/<id>/chunks.parquet`, `data/interim/<id>/chunks_manifest.json` | The sentence-aligned chunks and their manifest |
| `build_embeddings` | `data/interim/<id>/embeddings/<model>/documents_section_mean.npy`, `documents_plain_mean.npy`, `sections.npy`; `data/interim/<id>/embeddings/embedded_sections.parquet`, `embeddings_manifest.json`; `data/cache/chunk_vectors/<model>.npy` | The document vectors (both variants) and section vectors per model, the sections table, the manifest, and the raw chunk-vector cache |
| `build_tfidf` | `data/interim/<id>/tfidf/documents_section_mean.npz`, `documents_concatenated.npz`, `full_clean_sections.parquet`, `tfidf_terms.parquet`, `tfidf_manifest.json` | Both TF-IDF matrices, the full-clean tokens per section, the terms and the manifest |
| `run_experiment` | `data/processed/<id>/experiments/t19/labels.parquet`, `map.parquet`, `umap/<model>__seed<seed>.npy` with a `.json` beside each | Every run's labels per doc code, the 2-D map and PCA coordinates, and the 10 cached reductions |
| `run_gaps` | `data/processed/<id>/gaps/members.parquet` | The doc codes of each gap's theses, joined to `gaps.parquet` by `gap_id` |
| `draw_pairs` | `data/labels/labeled_set/<id>/pairs.parquet` | The drawn pairs: pair id, phase, pool, stratum, the two doc codes, scores, ranks and the known-duplicate flag |
| `run_temporal` | `data/processed/<id>/temporal/assignments.parquet`, `data/processed/<id>/temporal/sample.parquet` | Every thesis on every seed run's maps with its period and labels, and for a later thesis its similarity and atypical flag; the expert sample of 40 in presentation order, with doc codes and titles |
| `run_expert_samples` | `data/expert_session/<id>/muestras/muestra_a.md` to `muestra_d.md`, `clave.csv`, `composicion.json`, `manifest.json` | The printed session materials, samples A to D; their key, which maps the case codes to the theses' doc codes, groups and gaps; their composition in counts; and the manifest. Never committed |
| `pair_workbooks pilot` | `data/labels/labeled_set/<id>/blind_codes.parquet`, `pilot_layout.parquet`, `workbooks/etiquetado_piloto.xlsx` | The blind code of every drawn pair, the pilot workbook's pairs and sides, and the pilot workbook |
| `pair_workbooks import-pilot`, `main` | `data/labels/labeled_set/<id>/pilot_labels.parquet`, `assignment.parquet`, `main_layout.parquet`, `workbooks/etiquetado_anotador_<k>.xlsx` | The pilot's agreed labels, each labeled pair's pair of annotators, and the annotators' workbooks with their layout |
| `pair_workbooks import-main`, `adjudication`, `import-adjudication` | `data/labels/labeled_set/<id>/labels/labels.parquet`, `labels/fleiss_inputs.parquet`, `adjudication_layout.parquet`, `workbooks/desempate_anotador_<k>.xlsx`, `labels/adjudication_labels.parquet`, `labels/consensus.parquet` | Each annotator's labels, the Fleiss' kappa inputs, the adjudication workbooks, the third labels, and the consensus labels that `run_redundancy` reads |
| `openalex_topics` | `data/raw/openalex/topics/<download id>/topics.jsonl` | Every topic record of the OpenAlex Topics list (CC0) as the API served it, one per line; public data, kept out of git for its size |
| `run_redundancy` | `data/processed/<id>/experiments/t22/pair_scores.parquet` | Each labeled pair's labels, fold, scores and predictions under each representation |
| `download_pdfs`, `extract_text`, `run_gaps`, `draw_pairs`, `run_temporal`, `pair_workbooks`, `run_redundancy`, `run_expert_samples`, `openalex_topics` | A `.lock` file in or beside the output folder (`results/**/.lock` is gitignored; the samples' lock is in their folder under `data/`, and the OpenAlex download's is `data/raw/openalex/topics/.lock`) | Run locks, so that only one run at a time uses a folder |

At `2e8f9df` the files of `run_temporal`, of the `pilot` step and of `run_expert_samples` (its
real run of 2026-10-29) exist; those of the later `pair_workbooks` steps and of
`run_redundancy` do not (names listed, M).

The manifests of T19, the gap run, the draw, the temporal split, the pilot step and the
OpenAlex download record the SHA-256 of their gitignored files (`data_files`), so a rerun can be compared with the recorded
one without publishing the files.
