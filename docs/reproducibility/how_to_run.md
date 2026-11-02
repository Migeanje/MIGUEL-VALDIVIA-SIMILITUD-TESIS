# How to run

The pipeline is 30 steps in a fixed order, from the harvest to the redundancy threshold: 25
commands and 5 steps done by people. A Streamlit app browses the results; it is not a step,
since it writes nothing. The steps check their inputs against the manifests that wrote them,
so a step whose inputs are missing or changed stops with an `error:` line instead of reading
stale data.

Only the machine setup and the corpus acquisition need the network. Steps 3 to 30 and the app
run offline, from the frozen snapshot, the pseudonym key and, for steps 9, 22, 26 and 29, the
workbooks that people filled.

Times are wall-clock times from the ledger, [`docs/processing_costs.md`](../processing_costs.md),
on the laptop described in the [environment](environment.md). Commands are the README's, unless
the step says otherwise. On snapshot `20261002T224412Z`, steps 1 to 20 have run (step 19 on
2026-10-29, at `2e8f9df`); steps 21 to 30 have not.

## Before you start

| Need | Why |
|---|---|
| The [environment](environment.md): `uv sync`, the tessdata files, both models in the Hugging Face cache | Every command runs from `.venv/`; the commands load the models from the cache only |
| `HF_HUB_OFFLINE=1` in the shell from step 3 on (`$env:HF_HUB_OFFLINE=1` in PowerShell) | No model request reaches the network |
| The pseudonym key, `data/interim/pseudonym.key`, of the recorded runs | Without it, step 10 gives every person a new code, so the fichas table and every manifest that records its SHA-256 differ ([data card](data_card.md#6-codes-instead-of-identities)) |
| A clean git tree before the real runs of the commands of steps 15 to 30 | Their real runs refuse changed or untracked paths outside their own results folder (step 19: anywhere in the tree), because the manifest records the commit |
| The snapshot under `data/raw/20261002T224412Z/` | A new harvest gives a new snapshot, not this one (the repository is live, D23) |

Every command takes `--project-root` (default: the current directory) and, except
`ocr_assets`, `--snapshot-id` (default: the snapshot harvested last). `pair_workbooks` takes
them after its step's name, and the app after a `--`. Run `--help` on any of them for the rest.

## Network and authorization

| Step | Network | Authorization |
|---|---|---|
| `uv sync` | PyPI, the PyTorch CPU index, GitHub | — |
| `ocr_assets` | GitHub (`tessdata_best`) | — |
| First smoke run | Hugging Face (MiniLM) | — |
| mpnet model download (no command) | Hugging Face | U8, granted 2026-10-07 |
| 1. `harvest`, 2. `download_pdfs` | The UCSM repository | U8, granted for T04 and T05 |
| Fetching the 7 expired-embargo PDFs | The UCSM repository | Needs its own U8; not done |
| Steps 3 to 30, the app, the checks | None; the app serves `localhost` only | — |

## One-time steps

These refuse to run twice, by design:

| Step | What it refuses |
|---|---|
| 1. `harvest` | A snapshot that already exists: snapshots are never overwritten. A new run writes a new snapshot, named by default after its UTC harvest time |
| 7. `locate_objectives --make-sample` | An existing verification workbook, which may hold verdicts |
| 10. `build_fichas`, first run | Nothing, but it creates `data/interim/pseudonym.key`, which is never overwritten; a key file of the wrong size stops every later build |
| 17. `draw_pairs` | An earlier draw of the snapshot: a draw is a one-time event. To draw again, move the earlier draw's records away first |
| 18. `run_temporal` | An earlier temporal split of the snapshot (its manifest, assignments or sample), since its expert sample is a one-time draw |
| 19. `run_expert_samples` | Earlier samples of the snapshot |
| 20, 22, 24, 26, 27, 29. `pair_workbooks` steps | Each step's earlier records: a workbook is never overwritten. `main` also waits for `labeling.annotators` and the pilot's labels |
| 30. `run_redundancy` | An earlier run of the snapshot |

To run one of the commands from step 17 on again, move its earlier records away first. Step
20 draws new blind codes then (`secrets`, by design), so its records get other SHA-256 values
([configuration](configuration.md#seeds)). Every other step can be run again: it rebuilds or
resumes its outputs. The reruns that the plan records (T07a, T12, T13, T14, T15 and T24a)
wrote byte-identical data files; the second real runs of steps 18 and 20 refused, as designed
(plan T27, T20b).

## The pipeline

### Machine setup, once per machine

```sh
uv sync
uv run python -m thematic_redundancy.extraction.ocr_assets
uv run pytest -m smoke
```

| Step | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| `uv sync` | `pyproject.toml`, `uv.lock` | `.venv/`, 1.91 GB | Repeatable | Not measured; over 1 GB to download |
| `ocr_assets` | `paths.tessdata_dir` | `tessdata/spa.traineddata`, `eng.traineddata`, `manifest.json`, 29.0 MB | Repeatable: it fetches only missing or changed files; `--force` fetches both again | Not measured |
| First smoke run | — | MiniLM in the Hugging Face cache, 479.7 MB | Later runs are offline | The download was not timed (T02) |

The mpnet model (1.13 GB) has no command; it was downloaded once under U8 in 287.6 s, its first
load included ([environment](environment.md#embedding-models)).

### Corpus acquisition (P1), online, under U8

```sh
uv run python -m thematic_redundancy.corpus.harvest
uv run python -m thematic_redundancy.corpus.download_pdfs
```

| Step | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| 1. `harvest` (T04) | `repository`, `snapshot` | `data/raw/<id>/metadata.jsonl`, `manifest.json` | One-time per snapshot | 47 s, 14 requests |
| 2. `download_pdfs` (T05) | The snapshot | `data/raw/<id>/pdfs/<item uuid>.pdf`, `pdfs/manifest.json` | Resumable: a rerun keeps every verified PDF and retries failures | 12 h 29 min elapsed, about 3 h 9 min active (derived); about 1,463 requests (estimate) |

Then the snapshot is frozen (T06): it is never modified or downloaded again, and its card,
`results/snapshot/20261002T224412Z/data_card.md`, records its digests. There is no command for
this step.

### Data preparation (P2), offline

```sh
uv run python experiments/eda_metadata.py
uv run python experiments/domain_stopwords.py
uv run python -m thematic_redundancy.extraction.extract_text
uv run python -m thematic_redundancy.extraction.locate_objectives \
    --summary-out results/objectives/<snapshot_id>/summary.json
uv run python -m thematic_redundancy.extraction.locate_objectives --make-sample
uv run python -m thematic_redundancy.labeling.objectives_check \
    --report-out results/objectives/<snapshot_id>/verification.json
uv run python -m thematic_redundancy.corpus.build_fichas \
    --report-out results/fichas/<snapshot_id>
```

| Step | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| 3. `eda_metadata.py` (T07) | `metadata.jsonl`, `manifest.json`; the tokenizers in the cache | `results/eda/<id>/`: `metadata_profile.json`, `summary.md`, 4 figures | Repeatable; refuses an output that holds a title, abstract, name or ORCID iD | 16.5–17.4 s |
| 4. `domain_stopwords.py` (T11) | `metadata.jsonl` | `results/eda/<id>/stopword_candidates.json` | Repeatable; the same privacy refusal | Not measured |
| 5. `extract_text` (T09) | `pdfs/manifest.json`, the PDFs, `tessdata/` | `data/interim/<id>/pages/`; with `--summary-out results/extraction/<snapshot_id>/summary.json`, the tracked summary | Resumable at any point; start it in your own terminal | 5,341.1 s (1 h 29 min) for 721 PDFs, almost all of it OCR |
| 6. `locate_objectives` (T10) | The page files | `data/interim/<id>/objectives.parquet` and its manifest; the tracked summary | Repeatable: it rebuilds from scratch | 5.977 s; 8.5 s for the whole command |
| 7. `locate_objectives --make-sample` (O05) | The objectives, the snapshot | `data/labels/objectives_check/<id>/objectives_check.xlsx`, `sample.json` | One-time | 1.39 s |
| 8. Manual verification | The workbook, in Excel | The filled workbook | A person's work | About 2 h for 60 rows (reported by the user) |
| 9. `objectives_check` (O05) | The filled workbook | `results/objectives/<id>/verification.json` | Repeatable import; refuses a blank or invalid verdict | Not measured |
| 10. `build_fichas` (T12) | The snapshot, the PDF and page manifests, the objectives table | `data/interim/<id>/fichas.parquet`, `exclusions.parquet`, `fichas_manifest.json`; the tracked quality report | Repeatable: the reruns wrote the same tables; the first run creates the pseudonym key | 0.6 s for the build; 11.1 s for the whole command with the quality report |

Steps 3 and 4 read the metadata only, so they can run any time after the harvest.

### Representation and topic map (P3), offline

```sh
uv run python -m thematic_redundancy.representation.build_chunks
uv run python -m thematic_redundancy.representation.build_embeddings
uv run python -m thematic_redundancy.representation.build_tfidf
uv run python -m thematic_redundancy.topic_map.run_experiment --dry-run
uv run python -m thematic_redundancy.topic_map.run_experiment
```

The README has no section for these four commands yet; their forms come from their `--help`
and the plan (T13, T14, T15 and T19 evidence).

| Step | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| 11. `build_chunks` (T13) | The fichas; both tokenizers in the cache | `data/interim/<id>/chunks.parquet`, `chunks_manifest.json` | Repeatable; stops unless it reaches `chunking.expected` | 22.8–23.5 s for the build; 25.7–26.9 s for the whole command |
| 12. `build_embeddings` (T14) | The chunks; both models in the cache | `data/interim/<id>/embeddings/`, `data/cache/chunk_vectors/` | Repeatable, from the cache | 549.0 s from an empty cache (encoding: MiniLM 127.2 s, mpnet 306.3 s); 52.8 s from the cache |
| 13. `build_tfidf` (T15) | The fichas, the chunks' sections, spaCy | `data/interim/<id>/tfidf/` | Repeatable; the second run wrote byte-identical data files | 25.9–29.1 s |
| 14. `run_experiment --dry-run` (T19) | The three representations and their manifests | Nothing: it checks the declaration and the inputs and prints the forecast | Repeatable | 3.7–6 s |
| 15. `run_experiment` (T19) | The same; a clean tree | `results/experiments/<id>/t19/`, `data/processed/<id>/experiments/t19/` | Repeatable: it refuses a dirty tree outside its results folder, and reuses its cached reductions per model and seed | 417.0 s for the whole command; 389.2 s in the manifest |

The experiment refuses to run unless its declaration matches the D30 pin
([configuration](configuration.md#the-pinned-experiment-declaration)).

### Gaps and labeled-set draw (P4, P5), offline

```sh
uv run python -m thematic_redundancy.gaps.run_gaps --dry-run
uv run python -m thematic_redundancy.gaps.run_gaps
uv run python -m thematic_redundancy.labeling.draw_pairs --dry-run
uv run python -m thematic_redundancy.labeling.draw_pairs
```

| Step | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| 16. `run_gaps` (T24a) | T19's records, checked against its manifest; the mpnet vectors; the fichas; `config/career_affinity.yaml`; the metadata, for the privacy guard | `results/gaps/<id>/`, `data/processed/<id>/gaps/members.parquet` | Repeatable: the second real run wrote byte-identical files but for the manifest's times; `--dry-run` writes nothing | Dry run 6.4–10.3 s; real run 6.77–9.47 s |
| 17. `draw_pairs` (T20a) | The fichas; the MiniLM, mpnet and TF-IDF document vectors; the metadata, for the privacy guard | `results/labeled_set/<id>/draw/`, `data/labels/labeled_set/<id>/pairs.parquet` | One-time; `--dry-run` writes nothing and can be repeated | Dry run 8.37–19.6 s; real run 27.97 s |

Both real runs refuse a dirty tree outside their own results folder, hold that folder's run
lock, and pass every tracked output through the privacy guard before writing anything. The gap
run builds on T19's provisional winner ([results index](results_index.md)).

### Temporal split and expert samples (P5, P7), offline

```sh
uv run python -m thematic_redundancy.topic_map.run_temporal --dry-run
uv run python -m thematic_redundancy.topic_map.run_temporal
uv run python -m thematic_redundancy.evaluation.run_expert_samples --dry-run
uv run python -m thematic_redundancy.evaluation.run_expert_samples
```

| Step | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| 18. `run_temporal` (T27) | T19's records, its gitignored labels included, checked against its manifest; T19's inputs through T19's loader; the fichas; the metadata, for the privacy guard; `temporal`, `proposal` | `results/temporal/<id>/`, `data/processed/<id>/temporal/` | One-time; `--dry-run` writes nothing and can be repeated | Real run 32.8 s in the run, 38 s for the whole command; dry run at the same commit 64.7 s in the run, 71 s for the whole command (2026-10-28) |
| 19. `run_expert_samples` (T31) | T19's records, T27's records and T24a's records, each checked against its manifest; T19's inputs; the fichas; the metadata, for the name guard; `expert_samples`, `recommender` | `data/expert_session/<id>/muestras/` only | One-time; it refuses to run, the dry run included, until step 18's records exist; `--dry-run` writes nothing | Real run 2.4 s, as the command reported it, at `2e8f9df` on 2026-10-29 (the plan's T31 entry of 2026-10-29). Dry run 4.0 s for `main`, with T27's split computed in memory before step 18 had run |

Step 18 needs only T19's records, so it can run any time after step 15. Step 19 needs steps 16
and 18. Both build on T19's provisional winner.

### Labeled set: workbooks, labeling and import (P4), offline, with people

```sh
uv run python -m thematic_redundancy.labeling.pair_workbooks pilot --dry-run
uv run python -m thematic_redundancy.labeling.pair_workbooks pilot
uv run python -m thematic_redundancy.labeling.pair_workbooks import-pilot --dry-run
uv run python -m thematic_redundancy.labeling.pair_workbooks import-pilot
uv run python -m thematic_redundancy.labeling.pair_workbooks main --dry-run
uv run python -m thematic_redundancy.labeling.pair_workbooks main
uv run python -m thematic_redundancy.labeling.pair_workbooks import-main --dry-run
uv run python -m thematic_redundancy.labeling.pair_workbooks import-main
uv run python -m thematic_redundancy.labeling.pair_workbooks adjudication
uv run python -m thematic_redundancy.labeling.pair_workbooks import-adjudication
```

Each step of `pair_workbooks` reads the draw's tracked manifest and the gitignored
`pairs.parquet` it vouches for, the included fichas and the metadata, and the records of the
steps before it. Its records go under `data/labels/labeled_set/<id>/` and its manifest under
`results/labeled_set/<id>/workbooks/` ([results index](results_index.md)). A workbook is
returned by saving it under its own name in its folder.

| Step | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| 20. `pair_workbooks pilot` (T20b) | The draw; `labeling.seed` | The 130 blind codes, the pilot workbook of 10 pairs, `pilot_manifest.json` | One-time; stops before writing anything if the name guard finds a person's name or ORCID iD | Real run 0.763 s in the step, 5 s for the whole command; dry run 1.14–1.25 s for the whole command |
| 21. Calibration meeting (D34 (7)) | The pilot workbook | The pilot workbook with its agreed labels and justifications | People's work; its date is still to be set | Not measured |
| 22. `pair_workbooks import-pilot` (T20b) | The returned pilot workbook | The pilot's labels, `pilot_labels_manifest.json` | One-time; refuses the workbook, listing every problem, if a pair or a fixed cell changed, a label is blank or outside the scale, or a level of the scale has no pair | Not measured |
| 23. Set `labeling.annotators` (U3) | The user's decision: 3 or 4 annotators | `config/default.yaml`, committed | The user's decision, then a commit, since the next steps need a clean tree | — |
| 24. `pair_workbooks main` (T20b) | The pilot's labels; `labeling.annotators` | The assignment, one workbook per annotator, `main_manifest.json` | One-time; waits for steps 22 and 23 | Not measured |
| 25. Labeling round (T21) | The annotators' workbooks | The same workbooks, labeled | People's work: two annotators per pair | Not measured; an open measurement of the ledger |
| 26. `pair_workbooks import-main` (T20c) | Every returned annotator's workbook | The labels, Fleiss' kappa inputs and, with no disputed pair, the consensus labels; `main_labels_manifest.json` with Fleiss' and Cohen's kappas | One-time; refuses every workbook, listing every problem, unless each holds its pairs unchanged and a label on every pair | Not measured; its dry run refused in 0.02 s before any workbook existed |
| 27. `pair_workbooks adjudication` (T20c) | The disputed pairs | One adjudication workbook per adjudicator, `adjudication_manifest.json` | One-time; only when some pair is disputed | Not measured; its dry run refused at once (0.00 s) before any workbook existed |
| 28. Adjudication (D34 (9)) | The adjudication workbooks | The same workbooks, labeled | People's work: an annotator who did not label the pair | Not measured |
| 29. `pair_workbooks import-adjudication` (T20c) | The returned adjudication workbooks | The third labels, `labels/consensus.parquet`, `adjudication_labels_manifest.json` | One-time; refuses the workbooks as step 26 does | Not measured; its dry run refused at once (0.00 s) before any workbook existed |

Every workbook is in Spanish and shows each pair under its blind code, with no score, pool,
link, doc code, pair id or person; the manifests pass the privacy guard with the doc codes, pair
ids and blind codes added ([data card](data_card.md#8-privacy-guards)).

### Redundancy threshold and stage 2 (P4), offline

```sh
uv run python -m thematic_redundancy.redundancy.run_redundancy --dry-run
uv run python -m thematic_redundancy.redundancy.run_redundancy
```

| Step | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| 30. `run_redundancy` (T22) | The draw and its `pairs.parquet`; the consensus labels of step 26 or 29; T19's manifest and selection; both models' document vectors and the TF-IDF matrix through T20a's loader; `redundancy`, `experiment.selection.f1` | `results/experiments/<id>/t22/`, `data/processed/<id>/experiments/t22/pair_scores.parquet` | One-time; `--dry-run` writes nothing, and `--labels PATH` reads another labels file in a dry run only | Real run not measured (not run: it waits for the labels and SR-T22-1). Dry run with synthetic labels 1.77 s for `main` |

Stage 2 confirms T19's provisional winner, changes it, or is undefined. If it changes, the
records of steps 16 and 18 are moved away and those steps run again by the same rules (README,
"Temporal split"; the plan's "Next step" holds both provisional until T22).

## The app

```sh
uv run streamlit run src/thematic_redundancy/app/main.py
uv run streamlit run src/thematic_redundancy/app/main.py -- --snapshot-id 20261002T224412Z
```

| Command | Inputs | Outputs | Nature | Time |
|---|---|---|---|---|
| `streamlit run src/thematic_redundancy/app/main.py` (T28), from the project root, so that it reads `.streamlit/config.toml` | T19's records and map, the gap run's records and members, the fichas, each checked against its manifest; for the query page, the winner's document vectors, the chunk manifest, the winner's model and every configured tokenizer from the cache | None: it writes no file and keeps no query; it serves `http://localhost:8501/` (the map), `/consulta` (the query) and `/vacios` (the gaps) | Read-only and repeatable, any time after step 16; it shows no redundancy verdict until step 30 calibrates τ | Server healthy after 3.1–3.6 s; the first query after a start 31.9–65.3 s (the model's cold load); later queries 0.15–0.20 s in process and about 1 s in the browser |

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
uv run pytest -m smoke
```

| Check | What it covers | Time |
|---|---|---|
| `ruff check .`, `ruff format --check .` | Lint and format of every Python file, and of the Markdown files too (ruff 0.16) | Not recorded in the ledger |
| `pytest -q` | The unit tests, on synthetic data; the smoke suite is left out (`tests/conftest.py`) | At `2e8f9df` in the main tree: 3,694 passed, 3 skipped and 36 smoke tests deselected, in 253.23 s; 261.62 s for the same count at `108f6e7` (2026-10-29). Times vary with the machine state: 79.29 s for 3,088 tests on 2026-10-21 |
| `pytest -m smoke` | The locked stack on this machine: spaCy, both models, UMAP, K-means, HDBSCAN, BERTopic, the metrics, OCR of a page without a text layer | Offline: `test_stack.py` and `test_ocr.py`, 17 tests, 32.8–39.0 s (T02a); per file, `test_chunker.py` 25.0–58.3 s, `test_embeddings.py` 27.3–58.1 s, `test_tfidf.py` 34.7–51.6 s, `test_topic_map_build.py` 43.3–49.7 s, `test_experiment_runner.py` 115.12–166.97 s |

- The OCR smoke test needs the tessdata files and fails with the fetch command when they are
  missing (README "Smoke test").
- The chunker, embedding and TF-IDF smoke tests read the real fichas and chunk table; they skip
  when those files or a model are not on the machine (`tests/smoke/`).
- `uv run pytest --smoke -k ocr` runs the OCR tests of both suites; `-m` and `-k` filter the
  smoke tests like any other (README "Smoke test").
- This package has its own structural test, `tests/unit/test_reproducibility_docs.py`: every
  command of this guide must exist with a `main` and the options it names, and every tracked
  result must appear in the [results index](results_index.md).
