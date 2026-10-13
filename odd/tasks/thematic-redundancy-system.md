# Feature: Thematic Redundancy System (MVP)

- **Locator:** `odd/tasks/thematic-redundancy-system.md`
- **Status (2026-10-12):** Plan approved by the user on 2026-10-02, including O02 and O08 (direct commits on `main`). Phases 0, 1 and 2 are done (T01 to T12). T10 met O05 with 60/60 general objectives correct, and T12 shipped the 766 fichas (744 included theses) as `73f1676`. D26 (ten more domain stopwords) and the mpnet model download (U8) are recorded, and T12a was committed as `ad2c6f0`. Next: the overnight stack of follow-ups (T08a, T09b and T12b prepared, T09b open for the user's decision on edge lines of digits only; then T11b with T07a, T05a, T04a and T02a), then Phase 3 (T13).
- **Source:** the thesis plan v3 `Plan_de_Tesis_UCSM_v3_Directiva.pdf` (119 pages, kept outside the repo), plus the decisions recorded below.

## Objective

Build a local, reproducible Python prototype that analyzes thematic redundancy and research gaps across the public defended theses of UCSM. Coverage is FCIFF, five programs, 2021–2026. The prototype includes a Streamlit interface for students and advisors.

It must also produce the evidence the thesis needs:
- a comparative experiment;
- labeled-set metrics;
- a temporal-split evaluation;
- expert validation.

## Problem / Why

Institutional integrity tools such as Turnitin detect textual overlap. They do not detect thematic redundancy, where two works tackle the same problem with the same approach but use different words. No tool maps the repository's research lines or surfaces unexplored areas (plan §1.1).

## Scope

**In:**
- Corpus acquisition from the public UCSM repository (DSpace 7.6.1, REST + OAI-PMH).
- Title, abstract, keywords, date, program and advisor taken from metadata; objectives parsed from the PDF.
- Two cleaning variants: light (for embeddings) and full (for TF-IDF / c-TF-IDF / coherence).
- Chunked multilingual SBERT embeddings and a TF-IDF lexical baseline.
- Topic map: UMAP + K-means vs HDBSCAN inside BERTopic, compared in a declared experiment.
- Direct-redundancy module with a calibrated threshold, a topic-saturation signal, and topic assignment for new proposals.
- Gap detection (five types plus a stability filter) and a recommender.
- Streamlit app with three views, and an Excel-based labeling workflow.
- Evaluation: internal clustering metrics, labeled pairs, temporal split, analysis of the expert questionnaire.
- A Dockerfile for the demo (M5–M6).

**Out (future work):**
- Production deployment, authentication, integration with UCSM systems (plan §3.1.2).
- A database server; persisting user queries.
- A full external literature corpus; fine-tuning models; BETO/MarIA.

## Constraints

- **Local only:** Windows 11, Python 3.13, CPU-only torch. No cloud processing.
- **Data stays out of git and images:** `data/` and `tessdata/` are gitignored and dockerignored.
- **License (CC BY-NC-ND 4.0):** research use only, no redistribution of texts. Every displayed thesis links to its repository item.
- **Privacy:**
  - Drop `renati.author.dni` and `renati.advisor.dni` at ingestion.
  - Never show author names in the app or reports; use codes.
  - Never persist user queries.
- **Language:** technical artifacts (code, comments, docs, tests) are in English. The UI copy language is pending (O01).
- **Remote operations need explicit user authorization per destination:** T04, T05 (repositorio.ucsm.edu.pe) and T25 (api.openalex.org).
- **Reproducibility:** `uv.lock`, fixed seeds, versioned configs, a frozen dated snapshot, and results tagged with config hash, git commit and snapshot id.

## Accepted decisions

| ID | Decision | Rationale / evidence |
|---|---|---|
| D01 | The corpus is public defended theses only; thesis plans are dropped | User decision 2026-10-02. No plans are public: 0 of 126 repository collections hold them |
| D02 | Census of all eligible theses (≈745), no sampling | 745 docs is computationally small, and sampling would create artificial gaps. Plan §3.3.3 must be updated |
| D03 | Metadata (title, abstract, keywords, date, advisor) comes from the API; the program comes from the collection; objectives are parsed from the PDF | Metadata keys verified 2026-10-02. There is no objectives field |
| D04 | Work from a frozen, dated local snapshot; update incrementally via OAI-PMH `from` | Reproducibility, availability, and politeness to the server |
| D05 | Storage is Parquet + `.npy` + PDFs on disk. No database in the MVP. Storage sits behind a port so a DB adapter can be added at deployment | Only ≈745 rows, the app is read-only, and labels arrive through Excel import |
| D06 | Stack as listed below; Python 3.13 with uv | PyPI wheels verified 2026-10-01. Python 3.14 is excluded because gensim 4.4.0 ships no cp314 Windows wheel |
| D07 | OCR uses PyMuPDF's built-in Tesseract plus tessdata files; no Tesseract install | Tested 2026-10-01 on a rasterized page using `spa.traineddata` only |
| D08 | Develop natively; add the Dockerfile at M5–M6 | Docker/WSL are not installed, and `uv.lock` already pins dependencies |
| D09 | Chunking: per section, sentence-aligned, ≤128 tokens under BOTH tokenizers. Section vector = mean of its chunks. Document vector = equal-weight mean of the sections, L2-normalized | Both models have `max_seq_length` 128. Vocab sizes differ (250037 vs 250002). Measured on 2026-10-11, those are the models' `config.json` `vocab_size` (embedding rows); the two tokenizers hold the same 250,002 tokens (see T13) |
| D10 | Strip the place/year suffix from titles. For TF-IDF, also remove domain stopwords and legal suffixes (S.A.C., E.I.R.L.) | 10 of the 20 most recent titles end with a place and/or year |
| D11 | Pass precomputed embeddings to BERTopic | Decouples BERTopic 0.17.4 (2025-12-03) from sentence-transformers 6 and transformers 5, both released later |
| D12 | Metrics: silhouette with cosine on the original embeddings; DB/CH on L2-normalized vectors; NPMI coherence on the top-10 c-TF-IDF terms; HDBSCAN noise excluded from internal metrics with the noise % reported; ARI for stability across seeds | sklearn's DB/CH accept only Euclidean distance |
| D13 | Report two separate signals: direct redundancy (calibrated τ) and topic saturation (informative only) | User decision 2026-10-02 |
| D14 | A new proposal is assigned to the nearest topic centroid (cosine, original space). No re-clustering at query time | Same rule for K-means and HDBSCAN. BERTopic's safetensors serialization drops UMAP/HDBSCAN |
| D15 | Labeled set (details below) | 20–30 pairs leave a recall margin of about ±0.28 |
| D16 | Gap rules (five types) plus a stability filter (present in ≥4 of 5 seeds); experts are the final filter | Plan §3.6.6 allows refining the criteria during development |
| D17 | "Never touched" reference: OpenAlex Topics (4516 topics, CC0), filtered by the subfields relevant to each program | The official research lines are too broad to reveal untouched topics |
| D18 | Official research lines are used as a coverage view | — |
| D19 | Code architecture is hexagonal-lite with a screaming layout. Notebooks only orchestrate | The experiment requires swappable components |
| D20 | Project repository: this repo, the single official repo for the whole system (remote `origin` on GitHub). No push without a user decision | Confirmed by the user 2026-10-02. One repo is enough; the user creates any extra repo on request |
| D21 | The earlier `TesisSimilitud` repo (Feb 2025) is ignored entirely | The user says it was only a test |
| D22 | Always construct BERTopic with `language="spanish"` (never the default) | With `embedding_model=None`, BERTopic 0.17.4 keeps `language="english"` and strips every character outside `[A-Za-z0-9 ]` before c-TF-IDF, so `fragmentación` becomes `fragmentacin`. Found and proven by the T02 smoke test |
| D23 | No backup copy of the raw snapshot; U7 is dropped | User decision 2026-10-03, risk R8 accepted. Partial mitigation: the metadata and PDF manifests record item uuids, bitstream uuids and sha256, so a later re-harvest can be diffed and documented |
| D24 | Deduplicate exact duplicate records before anything else | T07 found 2 pairs of `#tesis` records with identical title, abstract, author, advisor and date under different item uuids, so there are 744 distinct theses, not 746. T12 keeps the record accessioned first (`dc.date.accessioned`; a tie goes to the smaller item uuid) and logs the other as `duplicate_of`. Duplicates never enter the labeled pairs (T20) or the redundancy evaluation, where they would be trivial positives |
| D25 | OCR is selective: only low-text pages among the first `extraction.ocr_window_pages` pages (default 100) of a PDF are OCR'd. Other low-text pages keep their few text-layer characters, or are `empty`, with their low-text flag | The PDF text serves only the objectives (T10); title and abstract come from metadata (D03, D09). Measured 2026-10-03: the first text-layer page with "objetivo general" was found in 658 of 731 PDFs, at median page index 7, P95 25, P99 102. The window cuts OCR from about 8,600 low-text pages to about 1,960. It is configurable, so a wider pass can run later |
| D26 | Add ten research-action verbs to the domain stopwords: diseñar, desarrollar, implementar, elaborar, diagnosticar, seleccionar, validar, comparar, definir, medir, so the list goes from 79 to 89 lemmas. The outcome verbs mejorar, optimizar, reducir and incrementar stay out until the T16 review. The list is frozen before T18 | User decision 2026-10-09, from the T12 objectives re-check. Theses whose objectives hold each verb, of 728: diseñar 282, desarrollar 206, implementar 182, elaborar 124, diagnosticar 77, seleccionar 74, validar 67, comparar 65, definir 47, medir 39 (`results/fichas/20261002T224412Z/quality_report.json`). They play the role of the listed reporting verbs determinar (284), evaluar (282) and analizar (270). The nouns are separate lemmas (diseño 130, implementación 168), so their topic signal stays. Only the lexical side changes (TF-IDF, c-TF-IDF, NPMI), not the SBERT embeddings. T11a had removed `desarrollar` with a group that fit no criterion; it returns under the reporting-verb criterion. The outcome verbs may name a topic: mejorar 157, optimizar 141, reducir 86, incrementar 63 |

D15, labeled set, in detail:
- ≈120 pairs drawn from four pools: SBERT-high, TF-IDF-high, mid-range, and random.
- Sampling is stratified by program.
- Annotators use a 3-level scale; each pair gets 2 annotators.
- Target Cohen's κ ≥ 0.6.
- τ is chosen by 5-fold cross-validation, with a bootstrap CI.
- SBERT is compared against TF-IDF.

## Open decisions

| ID | Question | Proposal | Status | Needed by |
|---|---|---|---|---|
| O01 | Language of the UI copy | Spanish, because the end users are UCSM students and advisors. Code and docs stay in English | **Accepted 2026-10-02** | T28 (M5) |
| O02 | How annotators label pairs | Run a short calibration meeting (guide + 5 pilot pairs together). Then each annotator fills one Excel workbook at home: guide sheet, side-by-side texts, dropdowns, randomized order, no system scores shown. We import and validate it. No server or DB needed | **Accepted 2026-10-02** | T20 (M3) |
| O03 | How the winning configuration is chosen | The embedding model is chosen by labeled-set F1, then NPMI. The clustering algorithm is chosen by within-space silhouette, then NPMI, stability and noise coverage. The rule is declared before T19 | **Accepted 2026-10-02** | T18 (M3) |
| O04 | How to measure success on the temporal split | Assignment consistency between the 2021–2024 map and the full map, plus a sample rated by experts | **Accepted 2026-10-02** | T27 |
| O05 | Acceptance criterion for objectives extraction | ≥90% correct on a manual, stratified sample of ~60 theses. Otherwise fall back to title + abstract and document it | **Accepted 2026-10-02** | T10 |
| O06 | Rule for flagging an atypical proposal | Flag it when its similarity to the nearest centroid is below the P5 of member-to-own-centroid similarities | **Accepted 2026-10-02** | T23 |
| O07 | Where the expert demos run | The local laptop, with the Docker image ready. Revisit at M5 | **Accepted 2026-10-02** | T29 |
| O08 | PR chain strategy | Work units go directly on `main`, with no PR chain. The user makes every commit and push manually in GitHub Desktop. The developer prepares each verified work unit and announces when a commit is due, with the file list and a Conventional Commit message | **Accepted 2026-10-02 (user decision; revised the same day)** | First commit |

## Inputs the user must secure (lead time)

| ID | Input | Needed by |
|---|---|---|
| U1 | Confirm the official research lines per school in a browser (the site blocks automated fetches) | T25 (M5) |
| U2 | Advisor approval of the population change: theses only, census, ≈745 docs, plus the plan-section updates | Before T19 (M3) |
| U3 | 3–4 advisors as annotators (≈80 pairs and 2–3 h each) | T21 (M3) |
| U4 | A career-affinity matrix agreed with advisors BEFORE seeing any results | Before T24 |
| U5 | Review of the OpenAlex subfield mapping per program | Before T25 |
| U6 | Expert panel (6–8 people), V de Aiken judges, consent forms | T30 (M6) |
| U7 | ~~A backup location for the raw snapshot~~. Dropped by the user on 2026-10-03 (D23) | — |
| U8 | Authorization for each remote operation: UCSM harvesting (T04/T05, granted), the mpnet model download (granted 2026-10-07, done) and the OpenAlex topics download (T25) | At each task start |

## Architecture and data layout

```
MIGUEL-VALDIVIA-SIMILITUD-TESIS/
├── pyproject.toml · uv.lock · .python-version
├── config/                      # YAML: paths, ocr, chunking, models, umap, grids, seeds, thresholds, gaps
├── data/                        # gitignored
│   ├── raw/<snapshot_id>/       # metadata.jsonl, pdfs/, manifest.json (immutable)
│   ├── interim/<snapshot_id>/   # fichas, light/full texts, exclusions (parquet)
│   ├── processed/<snapshot_id>/ # embeddings/<model>.npy, chunks/, tfidf/, maps/<config>/, gaps/
│   └── labels/                  # annotator workbooks (out/in), labels.parquet
├── results/                     # experiment tables and figures (no raw texts)
├── tessdata/                    # spa/eng traineddata (gitignored, fetched by script)
├── src/thematic_redundancy/
│   ├── corpus/                  # harvesting, bitstream selection, snapshot manifest, ficha schema
│   ├── extraction/              # PDF text, OCR fallback, objectives locator
│   ├── preprocessing/           # light/full cleaners, stopwords, title suffix stripping
│   ├── representation/          # chunker, embedder port + adapters, tfidf
│   ├── topic_map/               # umap, clustering adapters, bertopic wrapper, assignment
│   ├── redundancy/              # similarity search, threshold calibration, saturation
│   ├── gaps/                    # gap rules, stability filter, openalex reference
│   ├── recommendation/          # antecedents (MMR), alternative lines
│   ├── evaluation/              # clustering metrics, coherence, PR/F1, kappa, temporal split, likert
│   ├── labeling/                # pair sampler, excel workbook io
│   └── shared/                  # domain types, config models, storage ports + parquet adapters
├── experiments/                 # notebooks/scripts that only call src/
├── app/                         # Streamlit (map, query, gaps)
├── tests/                       # unit, fixtures, smoke
├── docker/                      # Dockerfile (M5–M6)
└── odd/tasks/                   # this document
```

## Stack

Versions were verified on PyPI on 2026-10-01 and 2026-10-02. Major versions get pinned after the T02 smoke test.

Python 3.13 · uv · pymupdf 1.28 · pdfplumber 0.11 (fallback) · httpx 0.28 · pandas 3.0 · numpy 2.5 · pyarrow 25 · pydantic 2.13 · pyyaml · spacy 3.8 + es_core_news_md 3.8.0 (URL dependency) · sentence-transformers 6.1 · transformers 5 · torch 2.14 (CPU) · umap-learn 0.5.12 · hdbscan 0.8.44 · scikit-learn 1.9 · bertopic 0.17.4 · gensim 4.4 · scipy 1.18 · pingouin 0.7 · statsmodels 0.15 · openpyxl 3.1 · streamlit 1.64 · plotly 7.1 · matplotlib 3.11 · seaborn 0.13 · jupyterlab 4.6 · pytest 9.1 · ruff

## Initial configuration (declared ex ante; every change is logged here)

| Key | Initial value |
|---|---|
| snapshot filter | `dc.date.issued` from 2021 to 2026, `renati.type` = tesis, the 5 program collections |
| programs.collection_uuid | sistemas `695b14ab-e5b2-49b5-9edf-f709e883b73b`, industrial `f0217548-ce48-4bae-b163-21b5e2504ef6`, electronica `a8bdd7c0-f1aa-4797-b04e-112e27f60da3`, mecanica `cf0ca97e-9b81-484d-a325-611b8a8d8223`, minas `9d9ecbe5-56c5-4cdf-81a3-b9a3736a47b3` (added in T04) |
| repository | `page_size` 100 (1–1000), `request_interval_seconds` 1.0 (1.0–60.0), `max_retries` 3 (0–10) (added in T04) |
| ocr | `spa+eng`, 300 dpi, `full=True` on pages without a text layer |
| extraction | `min_text_chars` 50 (≥1): a page under 50 stripped characters is low-text. `ocr_window_pages` 100 (≥0; 0 turns OCR off) (D25). (added in T09) |
| extraction.header_footer | `edge_lines` 3 (1–10), `min_share` 0.3 (>0–1), `min_pages` 5 (≥2): a line among the first or last 3 non-empty lines of a page that repeats on at least 30% of the pages with text, in a PDF with at least 5 such pages, is a running line (added in T09) |
| objectives | `general_max_chars` 1200 (≥1), `specific_max_chars` 4000 (≥1), `min_chars` 40 (≥1, below `general_max_chars`): longer texts are cut and flagged, a shorter general objective is flagged (added in T10) |
| objectives.verification_sample | `per_program` 12 (≥1), `seed` 20261007 (0 to 2^32−1), `workbook_dir` `data/labels/objectives_check` (relative, no `..`, and inside `paths.data_dir`, since the workbook holds thesis text) (added in T10) |
| fichas | `wrong_pdfs`: `20.500.12920/11777` carries the PDF of `20.500.12920/11776`. `expected`, for snapshot `20261002T224412Z`: 766 records, 744 included, 2 duplicates, 20 non-thesis, 728 with objectives, 16 metadata-only; the counts must add up, and the build stops when one is missed (added in T12) |
| chunk.max_tokens | 128 under both tokenizers, special tokens included |
| doc_vector | equal-weight mean of the section vectors, L2-normalized |
| models | paraphrase-multilingual-MiniLM-L12-v2, paraphrase-multilingual-mpnet-base-v2 |
| umap | `n_neighbors` 15, `n_components` 5, `min_dist` 0.0, metric cosine. A separate 2D map for visualization |
| kmeans.k | sweep 8–50, step 2, scored by silhouette. Final range confirmed in T18 |
| hdbscan.min_cluster_size | {5, 10, 15, 20}, with `min_samples` left at its default |
| seeds | 5 fixed seeds |
| redundancy.tau | max-F1 point of the PR curve, via 5-fold CV |
| saturation | quartiles of topic size |
| recommender | top-k 5, MMR λ 0.7, 3 neighbor topics, 3 nearby gaps |
| gaps.a | bottom quartile of topic size, plus an NPMI floor |
| gaps.b | pairs among each topic's 3 nearest topics with ≤1 bridge document |
| gaps.c | per the career-affinity matrix (U4) |
| gaps.d | change in share between 2021–2023 and 2024–2026, minimum count 3 |
| gaps.stability | present in ≥4 of 5 seeds |

## Task checklist

Owner is the developer unless noted. Route: D = delegated, I = inline. Risk: P = passive, M = medium, H = high.

### P0 — Foundation (M1)
- [x] **T01** Scaffold the repo. Contents:
  - layout, `pyproject.toml` (PyTorch CPU index, spaCy model as a URL dependency);
  - ruff/pytest config;
  - `.gitignore` / `.dockerignore`;
  - config models.
  - Route D. Risk M.
  - Checks: `uv sync`, `uv run ruff check .`, `uv run pytest -q`.
  - Accept: a clean install from the lock works on Windows.
  - Evidence (2026-10-02). Implemented and verified; the commit is pending O08.
    - RED: collection failed with `ModuleNotFoundError` for `thematic_redundancy.shared.config`. GREEN: 43 passed.
    - `uv lock`: 192 packages. `uv sync`: OK. `ruff check` and `ruff format --check` are clean.
    - The import check printed `2.14.1+cpu False`.
    - Independent re-run: `uv run pytest -q` gave 43 passed, and `ruff check` passed.
  - Resolved versions: torch 2.14.1+cpu, sentence-transformers 6.1.0, transformers 5.18.0, bertopic 0.17.4, pandas 3.0.6, numpy 2.5.3, scikit-learn 1.9.1, umap-learn 0.5.12, hdbscan 0.8.44, gensim 4.4.0, spacy 3.8.16, streamlit 1.64.0, plotly 7.1.0. No bound was changed.
  - Size: 692 authored lines, excluding `uv.lock`. Slice plan:
    - A: scaffold, ≈159 lines (pyproject, ignores, README, package skeleton).
    - B: config unit, ≈533 lines (`config.py` 245, `default.yaml` 78, tests 209). B gets a `size:exception` recommendation: it is one cohesive unit, the tests share a base fixture loaded from `default.yaml`, and 39% of it is tests.
  - Notes for later tasks:
    - T04: pydantic normalizes `base_url` with a trailing slash.
    - `load_config` reads YAML as UTF-8 because of the accents.
    - Run uv with `UV_PYTHON_DOWNLOADS=never` so it uses the local CPython 3.13.2.
  - Commits on `main`: `04bd609` (plan doc), `c87eba2` (scaffold + lock), `cdf72e4` (config unit). The user pushed them to `origin/main` through GitHub Desktop at 14:01:35.
  - Review:
    - The range was assessed as medium risk.
    - The full range exceeded the reviewer context budget, because `uv.lock` is 2,803 generated lines.
    - The config commit alone was reviewed with consent granted. It was approved and acknowledged.
    - `c87eba2` has no automated review, since the generated lock cannot be split. Its authored part was verified with `uv lock --check` and ruff.
  - Lesson: commit future `uv.lock` changes separately (`build(deps): ...`) so code commits stay reviewable.
- [x] **T01a** Config hardening, from the four non-blocking review findings:
  - enforce an inclusive k-means stop (`(k_stop - k_start) % k_step == 0`);
  - reject paths that escape the project root (`..`);
  - reject duplicate YAML keys;
  - use strict types (no string-to-number coercion).
  - Route D. Trigger: 2 non-trivial files (`config.py`, `test_config.py`). Risk M.
  - Evidence (2026-10-02):
    - RED → GREEN per change: 3 failed → 47 passed; 4 failed → 53; 2 failed → 55; 6 failed → 63.
    - `ruff check` and `ruff format --check` are clean, and `pytest` gives 63 passed (independent re-run).
    - Diff (code + tests): 224 insertions, 30 deletions.
    - Review: high risk, four review lenses, approved before and after the commit.
  - Behavior notes:
    - `k_values()` returns the inclusive sweep.
    - `resolve_against` also verifies that resolved paths stay inside the root. As a result, a `data/` symlink or junction pointing to another drive is rejected. This was kept deliberately; revisit it if the data must live on another disk.
    - Integer fields reject floats such as `300.0`.
    - YAML exponents need a dot and a signed exponent (`7.0e-1`). Superseded by T01b: plain scientific notation now loads as a float.
  - Commits: `30282fa` (plan progress) and `f39343b` (config hardening), both made by the user.
- [x] **T01b** Config follow-ups from the non-blocking T01a review findings:
  - build explicit `KMeansConfig` objects in the sweep tests instead of relying on the shipped defaults;
  - rename `_require_inside_project_root` to `_require_relative_without_parent_segments`, because it only performs lexical checks;
  - test YAML `<<` merge keys: a merged key overridden by a written key loads, a duplicate inside a merge source is refused, and an anchor reused across mappings keeps working;
  - add a real symlink containment test, skipped when the platform cannot create symlinks;
  - make plain scientific notation (`1e-3`) load as a float, or fail with an actionable message.
  - Route D. Trigger: 2 non-trivial files. Risk M.
  - Evidence (2026-10-02):
    - Bug found and fixed. A key written twice inside a `<<` merge source (inline, list item, or merge-only anchor) was silently accepted. The duplicate check now runs once per mapping in `flatten_mapping`; RED was `DID NOT RAISE` on 2 cases.
    - Plain scientific notation (`1e-3`, `1E3`) now loads as a float through a resolver on the custom loader only. The global `yaml.SafeLoader` is unchanged, and int fields still refuse `300.0`, `3e2` and `3.0e+2`. RED: 8 failed.
    - The sweep tests build explicit `KMeansConfig` objects. The validator was renamed.
    - A real symlink test is skipped on this machine (WinError 1314, no Developer Mode). An equivalent junction check was refused manually.
    - `ruff check` and `ruff format --check` are clean. `pytest`: 83 passed, 1 skipped (independent re-run).
    - Diff (code + tests): 216 insertions, 34 deletions.
  - Known limits:
    - A `<<` key written twice in the same mapping is still accepted, and the later merge wins.
    - `yaml.safe_dump` writes strings like `1e-3` unquoted, so they would round-trip as floats.
  - Review: medium risk, one lens, approved before the commit. After the commit it was assessed as `under_budget`, so it stays pending in the slice.
  - Commit: `2575007`, made by the user.
  - Follow-ups from the review (non-blocking):
    - The symlink test has never executed, so it must run on Linux (T29). This also checks its message regex.
    - Add a test that pins what happens when an exponent-shaped unquoted scalar lands in a string field.
- [x] **T02** Smoke-test the full stack on a synthetic mini corpus. The run covers embeddings → UMAP → K-means/HDBSCAN → BERTopic → NPMI → an AppTest import. Pin any major version that breaks.
  - Route D. Risk M.
  - Accept: the run completes end to end and the pins are documented.
  - Evidence (2026-10-02):
    - `tests/smoke/test_stack.py` holds 14 tests marked `smoke`. The default run excludes them via `addopts`; run them with `uv run pytest -m smoke`.
    - All 14 pass, also offline with `HF_HUB_OFFLINE=1` (~30 s). The default run still gives 83 passed, 1 skipped.
    - No pins were needed; pandas 3, plotly 7, transformers 5, sentence-transformers 6 and numpy 2.5 all work with BERTopic 0.17.4.
  - Observed values:
    - MiniLM: `max_seq_length` 128, dimension 384. Paraphrase cosine 0.598 vs. 0.002 for an unrelated pair.
    - K-means: 3 topics, each pure. HDBSCAN: 3 topics, 0 outliers.
    - Metrics are finite: silhouette 0.444, DB 1.454, CH 15.94, NPMI 0.092.
    - Seeded UMAP and K-means are bitwise deterministic across runs.
  - Findings:
    - BERTopic must get `language="spanish"` (D22).
    - In sentence-transformers 6, `get_embedding_dimension()` replaces the deprecated `get_sentence_embedding_dimension()`.
    - Run gensim NPMI with `processes=1`; on a small corpus the default spawns 15 workers.
    - NPMI can exceed 1 by about 1e-11, so range checks need a tolerance.
    - The model cache lives in `%USERPROFILE%\.cache\huggingface\hub`.
  - Route D. Size: 508 lines in one cohesive suite, about 75 of them the synthetic corpus. This exceeds the 400-line guide.
  - Review:
    - Before the commit: medium risk, one lens, approved.
    - After the commit, the range `f39343b..fdd81da` (T01b + T02, 838 lines) was declined by the user for this candidate only. Both commits had already been approved before they were committed.
  - Commit: `fdd81da`, made by the user.
- [ ] **T02a** Phase 0 follow-ups from the non-blocking T02 and T03 reviews. Deferred by the user's priority on Phase 1; none of them blocks.
  - Make smoke exclusion robust to later `-m` filters (for example an opt-in flag or a collection hook instead of `addopts -m`).
  - Add an explicit assertion that accented tokens survive in BERTopic topic words (direct proof of D22).
  - The OCR assets CLI should report `ValueError`/validation errors as one `error:` line with exit code 1.
  - Make the tessdata size check per language (or pinned sha256) instead of one 5 MB floor.
  - Route D. Risk P/M.
- [x] **T03** Script that fetches tessdata, plus an OCR fixture test that reads the tessdata path only from config.
  - Route D. Risk M.
  - Evidence (2026-10-02):
    - `extraction/ocr_assets.py` is a one-time, idempotent tessdata_best fetch. It takes an injected downloader, uses atomic `.part` → rename, has a 5 MB size floor, records sha256 in `tessdata/manifest.json`, and runs as `python -m thematic_redundancy.extraction.ocr_assets`.
    - Unit tests: 33, no network. RED was `ModuleNotFoundError`, then GREEN. Three throwaway mutants were each caught.
    - OCR smoke test: an image-only page with Spanish text was recovered exactly, accents and ñ included, at 300 dpi in 0.27 s. If tessdata is missing it fails with an actionable message.
  - Downloads:
    - `spa.traineddata`: 13,570,187 bytes, sha256 `e2c1ffda…ab2c`.
    - `eng.traineddata`: 15,400,601 bytes, sha256 `8280aed0…66ba`.
    - A second run skips both. No Tesseract installation is needed.
  - Checks: `ruff check` and `ruff format --check` are clean. `pytest`: 116 passed, 1 skipped. `pytest -m smoke`: 15 passed (independent re-run).
  - Size: 906 authored lines, about 57% tests. This exceeds the 400-line guide, but it is one cohesive unit.
  - Optional follow-ups:
    - pin the source URL to a tessdata commit instead of `main`;
    - pin the expected sha256 values in code;
    - clean stale `.part` files left by a hard kill.
  - Review: medium risk, one lens, approved both before and after the commit.
  - Findings (non-blocking, moved to T02a):
    - The CLI lets `ValueError`/validation errors escape as tracebacks.
    - The 5 MB size floor is calibrated on spa/eng only.
  - Commit: `3f9888f`, made by the user.

### P1 — Corpus acquisition (M1–M2)
- [x] **T04** Metadata harvester over DSpace REST for the 5 collections, with filters, DNI fields dropped, and a manifest. **Needs U8.**
  - Route D. Risk H.
  - Tests use recorded fixtures (no network).
  - Accept: counts reconcile with the repository totals at the snapshot date, and no DNI appears in any output.
  - Evidence (2026-10-02):
    - Code layout under `corpus/`:
      - `repository.py`: the port.
      - `dspace.py`: the httpx adapter. It paces requests at 1/s or more, retries 429/5xx with backoff, and verifies that the scope and filters were applied.
      - `snapshot.py`: models, the DNI filter, and an atomic, never-overwrite snapshot.
      - `harvest.py`: the use case and the CLI `python -m thematic_redundancy.corpus.harvest`.
    - Tests use synthetic fixtures only (no real names, DNI or texts). RED → GREEN for config and corpus. 11 throwaway mutants were all caught.
    - Checks: `ruff` clean. `pytest`: 233 passed, 1 skipped (independent re-run).
  - Real harvest, done once under U8:
    - Snapshot `20261002T224412Z`. 14 requests in total and 47 s wall time.
    - Records per program: sistemas 67, industrial 349, electronica 55, mecanica 192, minas 103. Total 766, which equals the faculty total.
    - `renati.type`: `#tesis` 746, `#trabajoAcademico` 1, and suficiencia profesional 19 across 5 spellings.
    - Dates: 2 items are issued after the harvest date (2026-12-01, 2026-12-04). Range 2021-01-06 to 2026-12-04.
    - Privacy check (independent): 0 `.dni` keys in 766 records. The manifest lists the dropped `renati.advisor.dni` and `renati.author.dni`.
  - Size: about 2,760 authored lines, about 1,600 of them tests and fixtures. Slice plan, in order, made by the user:
    - A: config unit (`config.py`, `default.yaml`, `test_config.py`).
    - B: port, adapter, `test_dspace.py`, fixtures. B depends on A.
    - C: snapshot, harvest, CLI, `test_harvest.py`, README, plan doc.
  - Commit: `ad160a5`, made by the user. It is a single commit, not the planned A/B/C slices, and its message describes only the config unit, yet it contains all of T04 (15 files). Its tree `373abce…` is byte-identical to the reviewed candidate. The message was left as is, because rewriting published history is not worth it for a cosmetic issue.
  - Reviews: medium risk, one lens, approved both before and after the commit.
- [ ] **T04a** Harvest robustness follow-ups from the T04 reviews:
  - Make offset paging stable with a unique tiebreak, or a listing that does not depend on undefined order among items with the same `dc.date.issued`. The current snapshot is unaffected: 766 unique uuids, reconciled with each listing total and with the faculty total.
  - Enforce, or clearly document, reconciliation against the faculty-wide total, which is currently only recorded.
  - Route D. Risk M.
- [x] **T05** Thesis PDF downloader. It picks the thesis bitstream by name (excluding `*.RT.pdf` and `Autorización_*`), rate-limits, resumes, and records sha256. Embargoed/restricted items are skipped with a reason. **Needs U8.**
  - Route D. Risk H.
  - Accept: every included item has exactly one thesis PDF or a recorded reason.
  - Evidence, implementation (2026-10-02):
    - Port and adapter extensions:
      - `list_bundles` uses `items/{uuid}?embed=bundles/bitstreams`, so each thesis costs 2 requests.
      - Downloads are streamed and share the pacing and retry core. 401/403/404 are final and never retried. Same-host redirects are followed up to 3 hops.
    - Pure selection rules in `thesis_files.py`: PDFs in ORIGINAL only, excluding `.RT.pdf` and `Autoriz*`. If none remains the status is `no_thesis_file`; if several remain it is `ambiguous`, and nothing is downloaded.
    - Each download is checked for the `%PDF` header, a size equal to `sizeBytes`, and the server MD5. sha256 is then recorded.
    - Files are named by item uuid. The manifest stores no original filenames.
    - Restricted or embargoed items are skipped by `dc.rights` without any request.
    - Runs resume and are incremental; an atomic manifest and an OS lock prevent concurrent runs.
    - CLI: `python -m thematic_redundancy.corpus.download_pdfs`.
    - Tests: 121 new. RED → GREEN. 24 mutants were tried; 2 survivors led to 4 added tests, after which all were caught.
    - Checks: `ruff` clean. `pytest`: 354 passed, 1 skipped (independent re-run).
  - Naming inspection (13 items across the 5 programs): each item has exactly one thesis PDF. 9 also have `.RT.pdf`, 7 have `Autorización_*.pdf`. 0 ambiguous, 0 `no_thesis_file`.
  - Sample (2 per program, 10 PDFs):
    - 65.3 MB in 20 requests and 114 s. All 10 were verified independently.
    - A rerun made 0 requests.
    - Extrapolated total: 4.5–10 GB, about 2.5–4.5 h at 1 request/s.
  - Full run, closing evidence (2026-10-03). It finished at 2026-10-03T12:18:20Z and covers all 746 `#tesis` items:
    - 731 thesis PDFs on disk, 11,401,121,329 bytes (11.40 GB).
    - 14 `restricted`: 8 embargoed (COAR `c_f1cf`) and 6 restricted (`c_16ec`). Of the 8 embargoes, 7 have already ended (2022-02-19, 2022-08-19, 2022-10-29, 2022-11-22, 2024-01-13, 2025-05-03, 2025-12-19) and 1 ends on 2027-03-18.
    - 1 `no_thesis_file` (industrial): the item has no ORIGINAL bundle.
    - 0 errors. The 3 `ConnectTimeout` errors of the interrupted run were retried and downloaded.
    - Per program (PDF + restricted + no file): sistemas 63 + 2; industrial 330 + 9 + 1; electronica 46; mecanica 189 + 3; minas 103.
  - Independent verification (2026-10-03):
    - All 731 files were re-hashed. Each passes the `%PDF` header check, its size equals `sizeBytes`, its sha256 equals the manifest, and its MD5 equals the server checksum.
    - The manifest's item uuids equal the 746 `#tesis` uuids in the metadata.
    - The only extra file is the OS lock file `.lock`, which is harmless.
    - The acceptance criterion is met.
  - Open decision: 7 of the 8 embargoed theses have a `dc.date.embargoEnd` already past (2022–2025). They are skipped without any request. Fetching them needs a separate authorization. Recommendation: do not fetch them for now; they stay in the census through their metadata (title and abstract).
  - Findings on the downloaded files (2026-10-03). T06, T09 and T12 point here.
    - **Wrong file in the repository.** The sistemas item with handle `20.500.12920/11777` carries a byte-identical copy of the PDF of item `20.500.12920/11776`, yet their metadata differ entirely: abstract similarity 0.017, and the share of the item's own title words found in the PDF's first 4 pages is 0.0, against 1.0 for 11776. Planned handling: T12 treats 11777 as metadata-only (no PDF text; objectives fall back to title + abstract per O05) with a quality note, and T06 lists it.
    - **D24 pairs and their PDFs.** The mecanica pair has byte-identical PDFs. The industrial pair has two different PDFs (4.5 MB and 4.6 MB), so T12's canonical rule must also choose which PDF to keep.
    - **Title check.** Title-word coverage over the first 4 pages has a median of 1.0, and 718 of 731 PDFs reach 0.9 or more. 4 are below 0.5: 11777 (the wrong file), and handles 11240, 10952 and 11177, whose PDFs hold their own full abstract. Those 3 are the right files with a different cover title.
    - **Valid own PDF.** 728 of the 744 distinct theses have one. 16 are metadata-only: 14 restricted, 1 without a file, 1 wrong file.
    - **Text layer (for T09):**
      - 141,076 pages in total; per thesis, median 174, p95 343, min 65, max 695.
      - 8,616 pages (6.1%) have fewer than 50 characters, and 8,595 of them hold images.
      - Documents: 604 text (<10% low-text pages), 126 mixed (10–80%), 1 scanned (≥80%, mecanica). Mixed by program: mecanica 56, industrial 37, minas 15, electronica 12, sistemas 6.
      - 0 encrypted files. Median file size 6.96 MB, max 412 MB. Every PDF has a text layer on its first 4 pages.
      - Implication: OCR selectively (only the sections the pipeline needs), and run long jobs as resumable CLIs in the user's own terminal.
    - **Models.** The mpnet model is not cached; only MiniLM is (about 480 MB). Downloading it from Hugging Face before T13 needs the user's authorization. (Granted and done on 2026-10-07; see the progress log.)
  - Size: about 3,400 authored lines, about 1,760 of them tests and fixtures. Commit slices:
    - A: port, adapter, fixture, `test_dspace`.
    - B: selection rules, manifest, use case, CLI, their tests, README, plan doc.
  - Commit: `54173b8`, made by the user. Its tree `dd756e9…` is identical to the reviewed one, and the post-commit review was approved.
- [ ] **T05a** Downloader follow-ups from the T05 reviews (non-blocking):
  - Classify deterministic integrity failures (unsupported checksum algorithm, a stale listing producing a size or checksum mismatch) so they neither retry forever nor trip the stop after 3 consecutive failures.
  - Add tests for refusing a PDF manifest that belongs to another snapshot.
  - Add tests for keeping a matching on-disk file while a previous manifest entry exists, whether it was an error or had a different bitstream.
  - Route D. Risk M.
- [x] **T06** Freeze the snapshot and write the data card (no backup, per D23) (counts by program and year, embargo list). The card must flag records whose `dc.date.issued` lies after the snapshot date. As of 2026-10-02 some items carry future dates (2026-12-04, 2026-12-01). The temporal split (T27) must handle them explicitly.
  - Route D (planned as I). Trigger: a new data card computed from the raw data, plus plan and README edits. Risk P.
  - Also list the T05 findings (2026-10-03): the PDF statuses, the wrong-file item 11777, the D24 PDF note and the 7 expired embargoes.
  - Evidence (2026-10-03):
    - The card is `results/snapshot/20261002T224412Z/data_card.md`: aggregates, handles and program keys only. The README's "Corpus snapshot" section points to it.
    - Freeze digests (SHA-256): `metadata.jsonl` `d8883901…d932` (equal to the snapshot manifest's), `manifest.json` `1ed0c205…94c8`, `pdfs/manifest.json` `4a1ecdba…816a`. Combined PDF digest `9c72910d…4586`: the SHA-256 of the sorted lines `<item_uuid> <pdf_sha256>\n` of the 731 PDFs, with the recipe in the card. The re-hashed files and the manifest's values give the same digest.
    - Every number was recomputed from the files, and each matches the T04, T05 and T07 evidence: 766 records (746 `#tesis`, 19 suficiencia, 1 trabajo académico); 744 distinct theses; the program × year table equals the metadata profile; 2 future-dated theses (`16433` 2026-12-01 and `16319` 2026-12-04, both industrial, open, with a PDF); rights 732 / 8 / 6; 731 PDFs, 11,401,121,329 bytes, all 731 re-hashed OK; 728 of 744 with a valid own PDF; 141,076 pages, median 174; 8,616 low-text pages; 604 text, 126 mixed, 1 scanned (`15595`).
    - No numeric mismatch. Notes on method:
      - The low-text count depends on the definition: 8,616 with the stripped text under 50 characters (the card's rule), 7,270 with the raw text, 10,070 with all whitespace removed.
      - The title check gives 718 PDFs at 0.9 or more with alphanumeric tokens (the card's rule), and 720 with letter-only tokens; the 4 below 0.5 are the same either way.
      - The abstract similarity 0.017 of 11777 was not recomputed. The card uses an abstract-token check instead: 1.0 for 11776 and for the 3 right files with a different cover title, 0.25 for 11777.
      - The live-repository count change (765 → 766) cannot be recomputed from the files, so the card cites the progress log without the numbers.
    - New details in the card: one open thesis (`12050`) carries a past `dc.date.embargoEnd`; both future-dated theses were accessioned in early 2026; the 20 non-thesis records are all open; the largest PDF is 412 MB (`15113`).
    - Privacy: the scan of the card and of the added plan and README lines for every title, abstract, abstract sentence of 40+ characters, author, advisor and juror name, and ORCID found 0 hits.
    - Checks: `pytest` 791 passed, 1 skipped (no code changed). `ruff check` and `ruff format --check` are clean. `ruff format` also checks Python code blocks inside Markdown, so the card's digest snippet follows ruff style; run from the repository root, it prints the combined digest above.
    - Size: 366 authored changed lines, 313 of them the new card.
- [x] **T07** EDA notebook: program/year distribution, abstract token lengths, keywords, title suffix patterns.
  - Route D. Risk P.
  - Evidence (2026-10-03):
    - `corpus/profile.py` holds the pure, tested profile functions (82 tests). `experiments/eda_metadata.py` writes aggregates only to `results/eda/20261002T224412Z/` (`metadata_profile.json`, `summary.md`, 4 figures).
    - RED → GREEN. 27 throwaway mutants were all caught.
    - Checks: `ruff` clean. `pytest`: 436 passed, 1 skipped (independent re-run).
    - Privacy: the writer's independent scan of every output for titles, abstracts, sentences, names and ORCIDs found 0 hits. A parent spot check of 60 random titles found 0 hits.
  - Key numbers:
    - Abstract tokens (MiniLM, special tokens included): median 366.5, P95 625.5, max 1,157. 745 of 746 (99.9%) exceed 128. Sentence packing needs a median of 4 chunks per abstract, max 13.
    - Title tokens: median 35, max 85, none above 128. 397 of 746 titles (53.2%) end with a place and/or year; Arequipa appears in 263. This supports D10.
    - Keywords: median 3 per thesis, 1,719 distinct, 84% used by a single thesis. They are too sparse to act as a topic signal.
    - Programs: industrial 340 (45.6%), mecanica 192, minas 103, sistemas 65, electronica 46. 2025 is the busiest year (162). Minas grows from 3 (2021) to 34 (2025).
    - Rights: 732 open, 8 embargoed (7 past end date), 6 restricted. 2 theses are future-dated.
  - Findings that change later tasks:
    - 2 exact-duplicate pairs, so there are 744 distinct theses (D24).
    - 716 of 746 abstracts are hard-wrapped, with 85.7% of breaks in mid-sentence, so T11 must unwrap them.
    - 44 advisor ORCIDs map to several name variants, so advisors must be identified by ORCID (T08/T12).
    - The mpnet tokenizer is not cached yet, so D09's check under both tokenizers is pending (T13).
  - Commit: `96c410c`, made by the user. Its tree `c1257fc…` is identical to the reviewed one.
  - Reviews: high risk, four lenses, approved both before and after the commit.
- [x] **T07a** EDA script follow-ups from the T07 reviews (non-blocking):
  - Write the outputs atomically (stage, then publish) so a failed write leaves no half-written results.
  - Guard against empty programs or years in `observations()`.
  - Derive the mpnet-tokenizer note at runtime instead of hardcoding it.
  - Tie the "top advisors" text to its constant.
  - Name the year-suffix patterns explicitly.
  - Add tests for the privacy write gate.
  - Document that text drawn inside the PNG figures is outside the gate's string search.
  - Route D. Risk M. Trigger: the profile script, the shared writer, `preprocessing/title_suffix.py`, and three test modules (one new), plus the README, this plan and the ledger; done together with T11b as one unit.
  - Evidence (2026-10-12):
    - Item 1, atomic outputs:
      - `shared/filesystem.py` gains `write_files_atomically(directory, contents)`. It writes each file to `<name>.part` and flushes it, all before the first rename; it then renames them in the given order and flushes the directory once (`fsync_directory`).
      - A failure or an interruption while staging leaves every previous file, raises the original error, and removes the staged files where the system allows it. A rename cannot be undone, so a failed rename leaves the files moved before it new; the docstring says so.
      - A key that is not a plain file name (`""`, `.`, `..`, a name with a directory part) is refused with `ValueError` before anything is written.
      - `write_bytes_atomically(path, content)` is its one-file form, and `write_text_atomically` now goes through it, writing the same bytes as before.
      - The script creates its folder and stages its 6 outputs together. Other files of the folder, such as `stopword_candidates.json`, stay. A write error gives two `error:` lines (the second since review fix U4-01) and exit code 1.
    - Item 2: `observations()` leaves out the programs sentence when no program has theses, and its busiest-year clause when no thesis has a readable issue date. The new test of a snapshot without theses also found that the program and year figure crashed on a legend of 0 columns; the figure now skips the legend then.
    - Item 3, the tokenizer note:
      - `main` records each distinct text that the profile tokenizes, with its count, through a wrapper around the counter.
      - `compare_tokenizer(model, counted, load, reference)` loads each other configured model's tokenizer through the given loader, from the local cache only, and counts the texts that it counts differently. A tokenizer missing from the cache is reported as not cached; one whose vocabulary size differs from the reference's is not compared (review fix U4-03).
      - The facts go into the provenance, as `other_tokenizers`. The note is rendered from them, with four texts: not cached (the old text), another vocabulary size (U4-03), the same count for every text, or another count for some texts.
      - Tests inject the facts and a stand-in loader, so no model loads.
    - Item 4: the advisor note reads `TOP_ADVISORS` from `corpus/profile.py`.
    - Item 5: a new test in `test_profile.py` names the year and place pattern sets one by one, checks both constants against them, and counts one title per pattern. `YEAR_SUFFIX_PATTERNS` in `preprocessing/title_suffix.py`, where T11 moved it, now names its four patterns instead of slicing `TITLE_SUFFIX_PATTERNS[:4]` (review item 4, below).
    - Item 6: `tests/unit/corpus/test_eda_metadata.py` (new) tests the gate in `main`:
      - a title in each of the 6 outputs;
      - an abstract sentence in upper case, a person's name, an ORCID iD, and the title of an item that is not a thesis;
      - a title inside binary bytes, in UTF-8 or in Latin-1;
      - each time, the error names the output and not the leak, and nothing is written; earlier outputs stay as they were.
    - Item 7: the module docstring, `_searchable` and the README say that the search reads the bytes of each file, so text drawn into a figure's pixels is outside it. The figures draw only fixed labels and profile values (program keys, years, counts, keywords), which `metadata_profile.json` also holds, and that file is searched.
    - Also fixed: the title-suffix method note named `src/thematic_redundancy/corpus/profile.py` as the home of `TRAILING_YEAR` and `TRAILING_PLACE`; since T11 it is `preprocessing/title_suffix.py`.
    - Test first:
      - `test_eda_metadata.py`, RED: 16 failed, 17 passed. The failures: `ValueError` from `max()` (3) and from the legend (1); `AttributeError` for `TOP_ADVISORS` and `compare_tokenizer` (4); the hardcoded or missing notes, including the old module path (4); `KeyError: 'other_tokenizers'` (2); and the `OSError` that escaped after 3 of the 6 previous outputs had been overwritten (1). One was a test defect: a person's name also leaks its family-name part, so the error counted 2 strings; the assertion now accepts any positive count. GREEN: 33 passed.
      - `test_filesystem.py`, RED: a collection error (`ImportError`, 1 error). GREEN: 37 passed (25 + 12).
      - Passing at once, as characterization tests: 15 of the gate and writing tests, the observation that names the busiest year, the not-cached note, and the pattern test. The gate test of a person's name passed once its assertion was fixed.
    - Mutants, each with a fresh bytecode cache, the module byte-identical after each restore:
      - 28 of the profile script and the shared writer, all killed. Profile script (22): the gate disabled; only the first output searched; the search without Latin-1 or without UTF-8; a message that echoes the leak; the strings of the theses only; the outputs written before the gate; direct writes one after another; a write error not caught; each of the two guards removed; the busiest year never named; the advisor count as the literal 10; equal counts reported as different; a missing tokenizer raising; the texts compared off by one; the note's verdict inverted; the first model compared with itself; the counts not recorded; the legend guard removed; the old module path; the note always "not cached". Shared writer (6): each file moved right after it is staged; staged files kept on failure; names not checked; `..` not checked; a line feed added to bytes; no directory flush.
      - 2 mutants of `preprocessing/title_suffix.py`, made on a copy of the source tree in a scratch folder and loaded through `PYTHONPATH` (the file itself was not touched): two patterns swapped so that the slice holds `place_after_separator`, and a slice of 3. Both were caught only by the new pattern test; the existing profile tests passed on both.
    - Real data, read-only and in memory, numbers only: the script's own functions profiled snapshot `20261002T224412Z` (766 items, 746 theses) and rendered the 6 outputs without writing, with the committed provenance plus the new `other_tokenizers`.
      - The mpnet tokenizer (`XLMRobertaTokenizer`, 250,002 tokens, 2 special tokens per sequence) is cached, and it gives the same count as MiniLM's for all 7,097 distinct texts (0 differ). The MiniLM facts equal the committed ones.
      - Byte-identical to the committed files: the 4 figures, and `metadata_profile.json` once its new key is removed. `summary.md` differs in exactly two method notes: the second-model note, now derived ("the same count … for each of the 7,097 distinct texts … holds for both models"), and the module path of the suffix patterns.
      - The old script, run the same way, reproduced all 6 committed files byte for byte, so the differences come from this change only.
      - Privacy gate on the new outputs: 6 outputs, 11,215 strings, 0 hits. The 7 committed files kept their SHA-256; nothing was written.
      - Times: MiniLM load 1.28 s, profile 1.42 s, mpnet load and comparison 2.77 s, render 0.48 s; 16.8 s for the whole command.
    - The committed `results/eda/20261002T224412Z/` files predate this change and stay as they are. A real run would keep the 4 figures; `metadata_profile.json` would gain `other_tokenizers` and record the current commit; `summary.md` would change its Code line and the two notes above. Regenerate them once this unit is committed, with one run of `experiments/eda_metadata.py` (the in-memory check took 16.8 s).
    - Review fixes (2026-10-12): an independent adversarial review raised three findings and asked to close item 5. One scoped correction fixed them, test first.
      - U4-01, a false claim. The README, the script's docstring and the `write_files_atomically` docstring said that a failed write leaves the previous outputs, which holds only for a failure while staging. A failed rename leaves the outputs renamed before it new: in the review's probe, a Windows reader holding `abstract_tokens.png` open gave WinError 5 after 3 outputs had been replaced. Ctrl+C between two renames does the same. The three texts now say so, and `main` adds a second error line: `some outputs in <folder> may already have been replaced; run the script again to write them all`. RED: the new test, with a rename that fails after the first output, failed without that line. GREEN: passed; it also checks that the first output is new, the other 5 are the previous ones, and no staged file is left.
      - U4-02, colliding names. `write_files_atomically` now refuses, before writing anything, a key that ends in `.part` in any letter case (`STAGED_SUFFIX`): `{"a.part": …, "a": …}` had left only `a`, holding the content meant for `a.part`. It also refuses two keys equal after `casefold()`: `A.json` and `a.json` name one file on Windows, where the probe got a `FileNotFoundError` after `A.json` had been replaced. RED: 3 failed (2 `DID NOT RAISE`, 1 `FileNotFoundError`). GREEN: passed. The review's probe, rerun on the fixed code: the held-open file still gives the documented mix, and both collisions are refused with nothing written.
      - U4-03, a wrong verdict. A partly downloaded mpnet cache, with its configuration files but without `tokenizer.json` or `sentencepiece.bpe.model`, loads offline as an `XLMRobertaTokenizer` of 5 tokens, whose counts would have produced a D09 verdict. `compare_tokenizer` now takes the reference tokenizer's description and compares counts only when both vocabularies have the same size. Otherwise the facts keep the vocabulary size without counts, and the note says that the cached tokenizer has a vocabulary of N tokens while the reference's has M, so their counts were not compared, that cache may be incomplete, and D09's check is still pending. RED: 6 failed (a `TypeError` for the new argument in 4 tests, `KeyError: 'texts_compared'` in the note, and counts compared by `main`). GREEN: passed. The review's probe on a real partial cache, rerun on the fixed code: vocabulary size 5 against 250,002, no counts, and that note.
      - Item 4, which closes T07a: `YEAR_SUFFIX_PATTERNS` names `place_and_year`, `year_after_separator`, `year_after_temporal_word` and `year_after_other_word`; `TITLE_SUFFIX_PATTERNS` keeps its patterns and their order. The pattern test in `test_profile.py` passed before and after, and `test_title_suffix.py` needed no change (122 passed together). Two patterns of `TITLE_SUFFIX_PATTERNS` swapped failed the pattern test before the change and pass it now: the sets no longer depend on that order.
      - RED in all: 10 failed, 67 passed in `test_eda_metadata.py` and `test_filesystem.py`. GREEN: 77 passed (37 and 40; 7 new tests).
      - 10 throwaway mutants, each with a fresh bytecode cache and each module byte-identical after its restore, all killed: no line about replaced outputs; that line without the folder; `.part` not refused; `.part` refused in lower case only; case-equal names not refused; names compared with their case; counts compared whatever the vocabulary; the note without its vocabulary branch; a place pattern counted as a year; a year pattern left out.
      - Real data, rerun read-only and in memory after the fixes: the same results as above (7,097 texts, 0 counted differently, both vocabularies of 250,002 tokens; the 4 figures byte-identical; the same two notes; 0 privacy hits; the 7 committed files kept their SHA-256), in 16.7 s.
    - Checks (T11b and T07a together, after the review fixes): `pytest` 1,376 passed, 3 skipped (1,319 + 57 new: 37 profile script, 15 shared writer, 4 candidates script, 1 profile) in 33.7–38.6 s (3 runs; before the fixes, 1,369 passed in 33.0–40.4 s); `ruff check` and `ruff format --check` clean; the attribution scan of the added lines found 0 hits.
    - Size (T11b and T07a together, after the review fixes): 1,417 authored changed lines. 974 are tests, 692 of them the new `test_eda_metadata.py`; 342 are source: 214 in the profile script, 84 in the shared writer, 40 in the candidates script (32 of them removed), and 4 in `title_suffix.py`; 101 are the README (20), this plan (79) and the ledger (2). The review correction accounts for about 180 of them, and T11b alone for about 133. That is above the 400-line guide because the new test module checks the privacy gate for each output and each kind of leak, the tokenizer facts, the empty-snapshot guards and the staged writing, each through `main`.
  - Commit: pending; prepared on 2026-10-12 as unit 4 of the overnight stack.

### P2 — Data preparation (M2–M3)
- [x] **T08** Ficha schema (pydantic) and its parquet adapter. It holds the Anexo B fields plus: handle URL, snapshot id, keywords, OCDE code, objectives status, section source, include/exclude with reason, file hash.
  - Route D. Risk M.
  - Evidence (2026-10-03):
    - `corpus/ficha.py` defines a frozen `Ficha` (`extra="forbid"`, inputs hidden in errors). Fields:
      - item uuid; `doc_code`, derived as `DOC-` + 8 hex of sha256(uuid);
      - handle URL, snapshot id, document type with the raw renati type;
      - title, abstract, objectives and objectives status;
      - program, raw issue date and year, the future-dated flag, keywords, OCDE codes, language;
      - rights and embargo end;
      - `advisor_code`, a single code;
      - `author_codes`, a tuple of at most 5 in repository order;
      - PDF status and sha256, source format, section source;
      - `include`, exclusion reason, `duplicate_of`, quality notes.
    - Cross-field rules:
      - exclusion iff not included;
      - `duplicate_of` iff the reason is `duplicate`;
      - `non_thesis_type` iff the document type is not tesis (D01);
      - objectives text iff the status is extracted or manual;
      - `pdf_sha256` iff the PDF is on disk.
  - `corpus/anonymize.py`:
    - keyed HMAC-SHA256 pseudonyms. The MAC includes the prefix, so ADV and AUT codes cannot be linked to each other.
    - Person identity uses a checksummed ORCID first and falls back to the normalized name; accents are dropped only for matching.
    - A 32-byte key lives under gitignored `data/interim/`. It is created once and never overwritten.
  - `shared/storage.py`: a table-store port plus a parquet adapter. Writes are atomic. UUIDs, tuples, None, booleans and accents round-trip. A schema mismatch raises `SchemaMismatchError`.
  - Data check (aggregate): 705 of the 746 theses have 1 author, 41 have 2, and every thesis has exactly 1 advisor. Hence `author_codes` is a tuple.
  - Tests: 202 new, RED → GREEN. 67 throwaway mutants were all caught. `ruff` clean. `pytest`: 638 passed, 1 skipped (independent re-run).
  - Notes for T12: check `doc_code` uniqueness across the dataset, and log a quality note when an ORCID fails its checksum.
  - Commit: `f21743f`, made by the user. Its tree `acaa52a…` is identical to the reviewed one.
  - Reviews: medium risk, one lens, approved both before and after the commit.
- [x] **T08a** Ficha/storage follow-ups from the T08 reviews (non-blocking):
  - Key creation must not depend on hard-link support. `os.link` fails on FAT/exFAT, some SMB shares and some container bind mounts, which matters for T29.
  - fsync the key's parent directory after publishing it.
  - The storage schema check should cover nullability and list item types, with tests for wrong list item types and for non-list columns.
  - Route D. Risk H, raised from M for this unit, since it handles the pseudonym key and the storage contract. Trigger: two source modules and their two test modules (4 non-trivial files), plus this plan, the ledger and the README.
  - Evidence (2026-10-12):
    - Item 1, key creation without hard links (`corpus/anonymize.py`, `load_or_create_key`):
      - The hard link stays where it works: the key is staged, flushed and linked into place, so it appears complete or not at all.
      - Any other link failure than an existing file leads to the fallback: `O_CREAT | O_EXCL` with mode `0o600`, then the write and an fsync. A FAT volume gives EPERM on Linux and EINVAL (winerror 1) on Windows; some shares give EOPNOTSUPP or winerror 50. Since the codes vary by system, the fallback takes any `OSError`; exclusive creation still never overwrites a file. A failure before the whole key is in the file removes the partial file; a later one keeps it (review fix W1 below).
      - A key file that is short, or (review fix W3) holds 32 zero bytes, may still be in writing by another run, so it is read again for up to `KEY_SETTLE_SECONDS` (2 s) and then refused, never replaced. A longer file is refused at once. `clock` and `sleeper` arguments, as in `corpus/dspace.py`, replace real time in tests.
      - A run that loses the race uses the winner's key. If the winner fails and removes its partial file, the loser raises `FileNotFoundError` and the next run creates the key.
      - Residual risk, in the docstring: a crash during the in-place write can leave a short file or, on FAT or exFAT, a full-length file of stale or zero bytes. Short and all-zero files are refused; other stale bytes cannot be told apart from a key.
    - Item 2: the run that publishes the key then fsyncs its directory with a local `_fsync_directory`. A directory that cannot be opened (Windows) or flushed (EINVAL, or any `OSError`) is skipped, since the key is already flushed and named. A run that loses the race, or reads an existing key, flushes nothing. `experiments/domain_stopwords.py` keeps its own helper, which T11b fixes.
    - Item 3 (`shared/storage.py`): `_check_columns` compares each column's nullability with its field's, both ways, and `_same_type` compares the type and the nullability of list items, whatever their name (Parquet reads `item` back as `element`). Exact nullability was chosen because the store writes it from the model, so a difference means another model version or another writer.
    - RED → GREEN:
      - Storage: 3 failed, 26 passed (`DID NOT RAISE` for list items that may be null, a nullable column for a required field, and a required column for an optional field) → 29 passed. A wrong item type and a plain column for a list were already refused; those 2 tests are characterization.
      - Key, counted over the whole module `test_anonymize.py` (the tests kept from HEAD included): 24 failed, 69 passed → 93 passed. Of the 24: 9 `TypeError` for the new `clock` argument (the settle tests, and the wrong-size test now on a fake clock), 9 with the link error propagating, 2 with that error instead of the simulated disk-full one, and 4 missing the directory steps. Three new tests passed at once, as characterization: an interrupted link leaves nothing, and reading an existing key or losing the race flushes nothing.
      - Written after GREEN: a test where the winner gives up (above), and a race of 4 real threads over 20 rounds, with and without hard links. A probe saw more than one creator in 200 of 200 rounds in both modes, but never a short read: 0 settle sleeps in 200 rounds of each mode. So the settle loop is covered by the fake-clock tests, not by the race test.
    - 28 throwaway mutants on the first version (22 key, 6 storage), each with a fresh bytecode cache, the modules byte-identical after each restore: all killed. They covered no fallback, a fallback on EPERM only or on interruptions, no `O_EXCL`, mode `0o666`, no removal or removal on `Exception` only, a loser that keeps its own key (exclusive, link, vanished file), no settle wait, a wait for long files or past the settle time, no directory fsync, an fsync before the publish, after a lost race or on an existing key, directory open or fsync errors raised, the descriptor not closed, the staged file kept, the fallback key not fsynced; and in storage, item nullability, column nullability (either way), the item type or the list check ignored, and swapped messages.
    - Real data, read-only, numbers only:
      - Through the stricter check, `load_fichas` read 766 fichas (744 included), the exclusions table 22 rows, the objectives table 731 rows, and the 731 page files 141,076 pages, with 0 refused, in 9.4 s.
      - The three tables kept their SHA-256 (`4918fbc6…`, `bb915e46…`, `e6e58f33…`).
      - The real key `data/interim/pseudonym.key` had the same SHA-256 (`7f5a5bbd…`) before and after, and again after the review fixes; key creation ran only in temporary directories.
    - Review fixes (2026-10-12): two independent adversarial reviews of this unit, one scoped correction.
      - W1 (warning, shown by both reviews with probes). The fallback removed the key file after any failure, even once all 32 bytes were visible. A run that read the key in that moment kept using it, while the next run created another key. Now the file is removed only while it holds fewer than 32 bytes on disk; after that, it is kept and the error is raised. Reading the size on disk rather than setting a flag also covers a stop between the flush and the flag, and a key that the closing of the file flushes.
      - W2. On Windows the removal fails while another run holds the file open, and its `PermissionError` replaced the original error. Errors of the removal are now suppressed; the short file stays, and the next run refuses it.
      - W3. The docstring now says that a crash during the in-place write on FAT or exFAT can leave a full-length file of stale or zero bytes. A file of 32 zero bytes counts as incomplete during the settle wait and is then refused with a `ValueError`. The key format is unchanged, so the real key stays valid.
      - W4. The docstring now says that mode `0o600` is requested: it binds on POSIX file systems, while FAT, exFAT and CIFS mounts take their mount options. A POSIX-only test checks `0o600` for a key made either way; it is skipped on this Windows machine, so it has not run here.
      - W5. A test pins that a short key file that disappears during the wait gives way to a new key.
      - RED → GREEN, over the whole module `test_anonymize.py`: 5 failed, 99 passed, 2 skipped → 104 passed, 2 skipped (the 2 POSIX-only tests). The 5: a zero key accepted, a zero key not waited for, the complete key removed in both cases of the in-use test (`FileNotFoundError`), and the `PermissionError` hiding the disk-full error. Passed at once, as characterization: the vanishing short file, and the 4 partial-write cases (0 or 16 bytes visible) that must still remove the file.
      - 6 new mutants, all killed: a complete key removed as well, a key of exactly 32 bytes removed, removal errors not suppressed, a zero key accepted, a zero key not waited for, and a vanished short file read as empty. The whole set then re-ran on the corrected code, with 3 earlier mutants adapted to it: 34 mutants (28 key, 6 storage), all killed, the modules byte-identical after each restore.
    - Checks: `pytest` 1,251 passed, 3 skipped (1,215 + 36 new: key 31, storage 5; plus the 2 POSIX-only skips) in 22.3–32.6 s (2 runs); `ruff check` and `ruff format --check` clean; the attribution scan of the added lines found 0 hits.
    - Size: 923 authored changed lines, 186 of them from the review correction: 658 in the two test modules (576 for the key), 200 in the two source modules, 60 in this plan, 3 in the README and 2 in the ledger. That is above the 400-line guide, because the key tests cover each failure path: four link errors, races with and without hard links, interruptions, partial and complete writes, the settle wait and the directory fsync.
  - Commit: pending; prepared on 2026-10-12 as unit 1 of the overnight stack.
- [x] **T09** PDF text extraction with text-layer detection and an OCR fallback. Note from T05: 6 of 13 inspected items also carry DSpace's extracted full text (`*.pdf.txt` in a TEXT bundle). It could serve as a cross-check, but it was not fetched and is not authorized yet.
  - Route D. Risk M.
  - Inputs: the T05 text-layer findings (2026-10-03), which call for selective OCR; and header and footer removal, moved here from T11. Long runs go in the user's terminal.
  - Trigger for route D: new extraction modules, CLI, config and tests.
  - Evidence (2026-10-05), implementation. Modules in `extraction/` (D19):
    - `page_text.py`, pure rules with no PyMuPDF import:
      - the low-text test and the OCR decision (low-text and page index below the window, D25);
      - line keys that fold case, whitespace and digits, so `Página 12` matches `Página 13`;
      - page numbers: arabic up to 3 digits, lower-case roman, `Página N`, `N de M`;
      - running header/footer removal across one document, never emptying a one-line page;
      - the `PageText` record: page index, text, source (`text_layer`, `ocr` or `empty`), char count, low-text flag, OCR attempted and failed, lines removed.
    - `pdf_reader.py`: the port (a page's text layer, OCR of one page, a warning count) and its two errors.
    - `pymupdf_reader.py`: the adapter.
      - One MuPDF text page per page gives both the data card's low-text measure (content order) and the text in reading order (blocks sorted top to bottom, a blank line between blocks).
      - OCR is `get_textpage_ocr(full=True)` with the tessdata path from config.
      - Every PyMuPDF exception becomes `PdfReadError` or `OcrError`, and MuPDF's messages are counted instead of printed.
    - `text_manifest.py`: the manifest model, the settings fingerprint and atomic JSON writes.
    - `extract_text.py`: the use case and the CLI `python -m thematic_redundancy.extraction.extract_text` (`--project-root`, `--snapshot-id`, `--limit`, `--only`, `--summary-out`).
  - Outputs, gitignored:
    - `data/interim/<snapshot>/pages/<item uuid>.parquet`: one `PageText` per page, written atomically through the table-store port.
    - `pages/manifest.json`: per PDF, the status, the PDF and output SHA-256, the settings fingerprint, pages by source, low-text pages, OCR tried and failed, unreadable pages, lines removed, MuPDF warnings, seconds and OCR seconds. Counts only, never text.
  - Behavior:
    - A rerun skips a PDF whose output is unchanged and was made from the same PDF SHA-256 with the same fingerprint.
    - A PDF whose bytes no longer match the PDF manifest is an `error` (`pdf_changed`) and loses its stale output. A corrupt PDF is `unreadable_pdf`, and the run goes on.
    - A page whose text layer cannot be read counts as unreadable; an OCR failure keeps the text layer.
    - An OS lock (`pages/.lock`) allows one run at a time, a run first deletes partial files, and Ctrl+C exits with 130 and nothing half-written.
  - Design choices:
    - A low-text page outside the window keeps its few text-layer characters (`text_layer`, low-text flag) instead of being blanked, so a separator page such as a chapter title survives for T10. Only a page left with no text is `empty`.
    - Every PDF is extracted, 11777 and the D24 pairs included, because T09 works per file. This supersedes the data card's line "T09 must not extract its PDF": T12 ignores the text of 11777.
    - Follow-up (non-blocking): the OS lock and the atomic JSON write repeat those of `download_pdfs`. Move both into `shared/` and reuse them; that was outside this task's edit surface.
  - Test-first evidence:
    - Config: RED 22 failed → GREEN 123 passed, 1 skipped.
    - `page_text`: RED was a collection error. A stubbed skeleton then failed per behavior (OCR decision 2, page numbers 13, running lines and records 10), and each behavior went GREEN in turn: 58 passed.
    - A share bug was caught test-first. In floating point 0.14 × 50 is 7.000000000000001, so the rule asked for 8 pages instead of 7. RED 1 failed → GREEN with exact decimal arithmetic, 59 passed.
    - Adapter: RED `ModuleNotFoundError` → GREEN 12 passed.
    - Use case and CLI: their tests were written before the module but first ran after it, so 20 throwaway mutants, each with a fresh bytecode cache, give the RED evidence. They cover skip logic (4), error isolation (3), page isolation (2), atomic writes (2), the lock, the OCR window, reading order, the low-text measure, the adapter's error mapping (2), the encrypted-PDF check, and the running-line guards (2). 19 were caught; the 1 survivor was an equivalent mutant, and the redundant check behind it was removed.
    - Smoke: real OCR through the adapter on a rasterized Spanish page, accents and ñ included.
  - Sample run, numbers only: 10 PDFs, 2 per program, with the fully scanned `15595` as mecanica's second.
    - 1,638 pages: 1,535 text layer, 103 OCR, 0 empty. 139 are low-text, 36 of them past the window. OCR: 103 tried, 0 failed. 2,333 running lines removed, 6 MuPDF warnings, 0 errors.
    - Seconds per PDF: 0.9, 1.5, 2.0, 7.4, 8.2, 8.9, 9.9, 18.0 and 71.2, then 355.3 for the scanned PDF (87 pages, 86 OCR).
    - Seconds per OCR page: 4.6 overall; 4.1 on the scanned PDF; 7.1 on the 17 pages of the other PDFs, spread from 1 to 17 s. Re-timed alone, 3 of the slowest pages took 3.8 s each, so the first session ran under slower machine conditions.
    - Work without OCR: 0.0056 s per page (hashing, text layer, header/footer removal, Parquet).
    - Resume: the session ended after 7 PDFs; the next run skipped those 7 and extracted the other 3. A third run did no work (10 skipped, 0.1 s).
    - Tesseract prints short diagnostics on some pages, such as "Line cannot be recognized!!". They hold no text.
  - Full-run estimate: about 13 min of text work (141,076 pages × 0.0056 s) plus OCR of about 1,960 pages at 4–5 s (2.2–2.7 h), so about 3 h. Under the slower conditions of the first session it could reach about 5.5 h. The run resumes, so it can be split.
  - Options to shorten it, each a config decision for the user (none taken): `ocr.languages: spa` saved 31% of the time for 1% fewer characters on 3 pages; 200 dpi saved 11%; a window of 60 pages means 1,353 OCR pages, and one of 30 pages 994.
  - Checks:
    - `pytest`: 918 passed, 1 skipped (791 + 127 new).
    - `pytest -m smoke`: 16 passed (15 + 1 new).
    - `ruff check` and `ruff format --check` are clean.
  - Size: about 3,200 authored changed lines, about 1,540 of them tests. Slice plan, three commits in order:
    - A: config and the pure page-text rules: `config.py`, `default.yaml`, `test_config.py`, `page_text.py`, `test_page_text.py`, and the plan's D25 and config rows (≈680 lines).
    - B: the port, the PyMuPDF adapter, the synthetic-PDF helper, the adapter tests and the OCR smoke test (≈500). B depends on A.
    - C: the manifest, the use case, the CLI, their tests, the README and this evidence (≈2,000). C gets a `size:exception` recommendation: one resumable CLI whose tests share one set of fakes, 43% of it tests.
  - Full run (2026-10-05, in the user's terminal; it finished at 2026-10-06T03:57:20Z):
    - This run:
      - 731 PDFs: 721 extracted, 10 skipped (the sample, already at rules version 2), 0 failed.
      - 139,438 pages: 137,063 text layer, 1,855 OCR, 520 empty; 8,477 of them low-text.
      - OCR: 1,858 pages tried, 0 failed, 5,042.4 s, so 2.71 s per page.
      - 146,279 running lines removed.
      - Wall time 5,341.1 s (1 h 29 min), well under the 3–6.5 h estimate.
    - Whole text manifest, checked independently:
      - All 731 PDFs are `ok`: 0 errors, 0 stale, 0 orphaned. They are exactly the 731 available PDFs of the PDF manifest.
      - 141,076 pages (138,598 text layer, 1,958 OCR, 520 empty) and 8,616 low-text pages, both equal to the data card.
      - OCR: 1,961 pages tried, 0 failed; the 3 tries that returned no text count as empty.
      - 148,601 running lines removed; 197 MuPDF warnings; 0 unreadable pages.
      - 731 parquet files plus the manifest: 90,660,188 bytes. No partial files were left.
    - `results/extraction/20261002T224412Z/summary.json` holds the whole manifest's counts and settings, with no text. Its settings fingerprint is `2b7c1adb…`, at extraction version 2.
    - It also holds whole-manifest timings: 6,568.3 s summed over the PDFs, of which 6,268.8 s were OCR, so 3.20 s per OCR page. They include the 10 sample PDFs from the slower T09a session.
    - The run-only timings above (5,341.1 s wall, 2.71 s per OCR page) come from the run's terminal output, not from that file.
    - The run's costs are recorded in `docs/processing_costs.md`.
  - Commit: `3b05679`, made by the user as one commit. Its tree `8451eb4…` is identical to the reviewed one.
  - Reviews: medium risk, one lens, approved before the commit. The user declined the post-commit review for this candidate.
- [x] **T09a** Fixes from the T09 review, needed before the full run. The settings fingerprint covers code only through a rules version raised by hand, and T09 left it at 1, so a later fix would not extract the PDFs again.
  - Route D. Risk M.
  - Trigger for route D: a rule fix plus manifest and fingerprint changes, with tests, across 3 or more files.
  - R3-001 (warning), `page_text.py`: an edge line of digits only goes only when it is a page number. Bare page numbers made `#` a running key, which also dropped a year such as `2024` at the foot of a cover page, or any other all-digit edge line. Running lines with words, such as `Página 12`, behave as before.
  - R3-003 (suggestion), `extract_text.py` and `text_manifest.py`: an entry whose item the PDF manifest no longer lists as on disk (restricted or removed since) is `orphaned`. A run reports it when it starts, and the summary counts it as `orphaned` and leaves it out of every other total. The entry and its page file are kept; nothing is deleted automatically.
  - R3-002 (suggestion): the skip trusts the PDF manifest's SHA-256 and does not hash the PDF on disk. This is kept by design, since the snapshot is frozen (T06). The README now says a rerun extracts again when the PDF manifest's record of the PDF changed, and points to the data card's combined PDF digest for verifying the files; the docstrings say the same.
  - Rules version: `EXTRACTION_VERSION` in `text_manifest.py`, already part of the settings fingerprint, goes from 1 (T09) to 2. Its docstring lists the versions and says to raise it whenever the page-text rules change, and `page_text.py` points to it. Documented in the README.
  - Evidence (2026-10-05):
    - R3-001: RED 1 failed (the year was dropped), 59 passed → GREEN 60 passed.
    - Rules version and R3-003: RED 4 failed, 35 passed. The rules-version test failed because both PDFs were skipped; the orphan tests failed because there was no orphan report. GREEN: 111 passed in `tests/unit/extraction/`.
    - Sample re-run, numbers only: the same 10 PDFs as T09, through `--only` with the 10 items of the T09 manifest.
      - All 10 were extracted again because of rules version 2: 10 extracted, 0 skipped, 0 failed.
      - Running lines removed: 2,333 → 2,322 (−11). That is one line kept per PDF, two in one PDF.
      - Otherwise the same: 1,638 pages (1,535 text layer, 103 OCR, 0 empty), 139 low-text, OCR 103 tried and 0 failed, 6 MuPDF warnings.
      - Seconds: 1,233.8 wall, of which 1,226.3 OCR, so 11.9 s per OCR page. The scanned PDF took 1,138.0 s (13.2 s per page), and the other 17 OCR pages took 5.4 s each.
      - That OCR speed is about 2.6 times slower than in T09 (4.6 s per page) on the same pages, so machine conditions vary widely. At this speed the full run's OCR would take about 6.5 h instead of 2.2–2.7 h; it resumes, so it can be split.
      - A second run skipped all 10 in 0.1 s. Stderr held 4 lines, all Tesseract diagnostics without text.
    - Checks:
      - `pytest`: 923 passed, 1 skipped (918 + 5 new).
      - `pytest -m smoke`: 16 passed.
      - `ruff check` and `ruff format --check` are clean.
      - Attribution scan of the added lines: 0 hits. Privacy scan of the changed files against the metadata titles, abstract sentences and names: 0 hits.
  - Commit: `7334e66`, made by the user. Its tree `0a7fd20…` is identical to the reviewed one.
  - Reviews: medium risk, one lens, approved before the commit. The post-commit check found it under the slice budget, so it stays pending in the slice.
  - Finding (non-blocking), moved to T09b: edge lines of digits only.
- [ ] **T09b** Follow-ups from the T09 and T09a reviews (non-blocking). Item 1 is done; the task stays open only for the user's decision on item 2.
  - Item 1 (done): move the OS lock and the atomic JSON write, which `extraction/` repeats from `download_pdfs` and `pdf_manifest`, into `shared/` and reuse them (from T09).
  - Item 2 (open): edge lines of digits only (from the T09a review):
    - A running footer made of digits only, such as `2024` on every page, is kept on every page.
    - A year with punctuation, such as `2024.` or `– 2024 –`, is only partly protected, because the line key folds the digits. It goes only when its folded key (`#.` or `– # –`) reaches the running-line threshold, counting the year: that key must sit at the edges of at least `max(2, ⌈min_share × pages with text⌉)` pages, in a document of at least `min_pages` pages with text (declared: `min_share` 0.3, `min_pages` 5). The pages counted are the year's own page and every page with an edge line that folds to the same key, such as `– 12 –` (a page number) or `12.` (not one). So the year goes when it repeats on enough pages, or when enough numbered lines share its key. It stays when they are on too few pages, as in a test with a share of 0.5: of 10 pages, the year with `– 2 –` and `– 3 –` puts the key on 3 pages of the 5 needed. It also stays when no other line shares its key.
    - It is harmless for the objectives: T10 met O05 with 60/60.
    - Why it stays open: any change to this behavior changes the page records of PDFs already extracted, so it must raise `EXTRACTION_VERSION`, which extracts all 731 PDFs again. The T09 full run took 1 h 29 min; at the slower OCR speed seen in T09a, its OCR alone would take about 6.5 h. T10's locator and T12's build would then run again (seconds each), and their tables would need to keep their SHA-256, or their changes reviewed. That cost is the user's decision.
    - Options:
      - (a) Keep the current rules, now pinned by tests (below). Nothing runs again.
      - (b) Compare an edge line that is a 4-digit year, bare or with punctuation, by its exact text instead of its folded key: a year that repeats on enough pages goes, a one-off year stays, and page numbers still go as page numbers. Other numbered lines keep today's rule; compared by exact text, running numbering such as `12.` on every page would no longer repeat, and would stay. Raise `EXTRACTION_VERSION` to 3 and run the extraction again.
      - (c) Extend the digits-only guard to a 4-digit year with punctuation, so that it is kept like a bare year. A running year footer then stays on every page, as `2024` does now. The guard must stay limited to years: widened to any digits with punctuation, it would also keep running numbering such as `12.` on every page, which goes today only as the running key `#.`. Same re-extraction.
  - Route D. Risk M. Trigger: a new shared module and its test module, 8 source modules and 5 test modules moved onto it or pinned, plus this plan and the ledger.
  - Evidence (2026-10-12):
    - Item 1, `shared/filesystem.py`:
      - `exclusive_run(directory, busy_error)` is a context manager that holds the OS lock on `<directory>/.lock` (`LOCK_FILE_NAME`) without waiting: `msvcrt.LK_NBLCK` on Windows, `flock(LOCK_EX | LOCK_NB)` elsewhere. When another run holds it, it raises the caller's own error from the system's `OSError`, so `download_pdfs` and `extract_text` keep `DownloadInProgressError` and `ExtractionInProgressError` with their messages. The lock file stays; the lock is released when the block ends, also on an exception or Ctrl+C, and by the system when the process dies.
      - `write_text_atomically(path, text)` replaces `text_manifest.write_json_atomically` and the copy inside `write_pdf_manifest`. It writes the text plus one line feed, in UTF-8 and in binary mode (no CRLF), to `<name>.part` beside the target; it flushes and fsyncs it, then calls `os.replace`. On any failure or interruption it raises the original error and removes the staged file where the system allows it (review fix F2). Same bytes as before.
      - Added: after the replace, `fsync_directory` flushes the directory, tolerating failures exactly as `_fsync_directory` in `corpus/anonymize.py` does. A directory that cannot be opened (Windows) or flushed (EINVAL, or any `OSError`) is skipped, and the descriptor is always closed. A rename is durable on POSIX only once its directory is flushed, as `experiments/domain_stopwords.py` and the key already do. On this Windows machine the step is skipped, so the files written are unchanged. Side effect: in `build_fichas`, the manifest's directory flush also covers the two tables renamed into that directory just before.
      - Callers moved: `corpus/download_pdfs.py` and `extraction/extract_text.py` (the lock, and the extraction summary), `corpus/pdf_manifest.py`, `extraction/text_manifest.py`, `extraction/locate_objectives.py`, `labeling/objectives_check.py`, `corpus/fichas_quality.py` and `corpus/build_fichas.py`. The two lock copies, the two writer copies, and the imports of the writer from `extraction/` into `corpus/` and `labeling/` are gone. No re-export was needed: nothing outside these modules used them. `test_build_fichas.py` now patches `build_fichas.write_text_atomically`.
      - Left as they are, as possible later reuse, since they write other kinds of files: the workbook save in `objectives_check.py`, the Parquet replace in `shared/storage.py`, `extraction/ocr_assets.py`, and the PDF download's partial-file replace. `_fsync_directory` in `corpus/anonymize.py` now repeats `fsync_directory`. The one in `experiments/domain_stopwords.py` does not quite: it lets an `os.fsync` error propagate after the output is published, which T11b fixes. Reusing `fsync_directory` there changes that experiment's failure behavior, as noted under T11b.
    - Item 1, test first:
      - `tests/unit/shared/test_filesystem.py`: RED was a collection error (`ModuleNotFoundError`, 1 error) → GREEN 23 passed; 25 after review fix F2. The lock is checked from a second handle and from a child process: refused at once with the caller's error and its `OSError` cause, released after the block, after an error and after Ctrl+C, not blocked by a lock file left behind, and freed when a holder process is killed. The writer is checked for its exact bytes, the replace, a staged file left behind, a missing directory, the order fsync → replace → directory flush, the directory steps that fail, and failures or interruptions of the fsync or the replace.
      - Characterization in the callers, passing on the old code before the move and again after it (5 tests): a download and an extraction are refused, with their exact messages and an `OSError` cause, while the shared lock holds their directory's `.lock`, which also pins the lock file name; both manifests are their JSON plus one line feed; and both quality-report files are UTF-8 text ending in one line feed.
      - 23 throwaway mutants, each with a fresh bytecode cache, the modules byte-identical after each restore; 22 killed. Lock (7): another file name, no `OSError` cause, a waiting lock (`LK_LOCK`, caught by the 2 s bound of the child-process test), a generic error, a held lock ignored, and a lock file kept open after an exception. Writer (10): no final line feed, text mode, cp1252, no fsync, staged file kept on failure or staged elsewhere, no directory flush, a failed directory flush or a directory that cannot be opened raised, the descriptor not closed. Callers (6): a changed download message, the snapshot or the parent directory locked, either manifest without indentation, the Markdown report's trailing line breaks kept. The survivor is equivalent on CPython: dropping both the `with` and the `finally` still releases the lock on an exception, because the generator's frame is cleared and the file closed. After review fix F2, W5 (no removal of the staged file) was adapted and re-run: killed.
    - Item 2, characterization of the open behavior (`test_page_text.py`, 9 tests, passing at once; `page_text.py` and `EXTRACTION_VERSION` unchanged): a running footer `2024` is kept on every page; a running `2024.` or `– 2024 –` goes from every page; a cover page's `– 2024 –` or `2024.` goes when its key is on enough pages, counting it; it stays when the key is on too few pages (review fix F1), and when no other line shares it. 5 mutants of `page_text.py`, all killed by these 9 tests alone: the digits-only guard removed, the guard widened to punctuated digits, keys that keep the digits, 4-digit page numbers, and a running threshold that ignores the share.
    - Real data, numbers only: a rerun of the extraction with `--only 00e24b7c…`, an `ok` item, and without `--summary-out`.
      - The code skips such an item: status `ok`, the PDF's SHA-256 matching the PDF manifest, the settings fingerprint matching (`2b7c1adb…`, computed again), and the page file's SHA-256 matching the manifest. The manifest is rewritten only when its fingerprint or summary is out of date.
      - Before: `pages/manifest.json` `7a06f50f…`, 3 page files `d9d0d4a2…`, `7b0d04b2…` and `41e615aa…`, and the SHA-256 and modification times of all 731 page files.
      - The run: 1 selected, 0 extracted, 1 skipped, 0 failed, exit code 0, the shared lock taken and released. The command's own wall time was 0.0 s; a second, identical run took 1.2 s including start-up.
      - After both runs: the manifest, the 3 files and all 731 page files are byte-identical, with the same modification times; the manifest was not rewritten (`updated_at` still `2026-10-06T03:57:20Z`), `.lock` was untouched, and no `.part` file was left. `results/` was not written.
    - Review fixes (2026-10-12): an independent adversarial review confirmed the refactor behavior-identical and raised four accuracy and robustness findings. One scoped correction fixed them.
      - F1. Item 2's text said a year with punctuation goes whenever numbered edge lines share its key. It goes only when that key reaches the running-line threshold, counting the year; the text above is corrected. The test is renamed `test_open_behavior_a_punctuated_year_goes_when_its_key_is_on_enough_pages_counting_it`, and a new one pins the reviewer's case: of 10 pages at a share of 0.5, the year with `– 2 –` and `– 3 –`, or with `2.` and `3.`, puts the key on 3 pages of the 5 needed, so the year stays. It passed at once (2 tests); a mutant whose threshold ignores the share (`needed = 2`) was caught only by it among the 9 open-behavior tests.
      - F2. When the removal of the staged file failed, as with a Windows sharing violation from an antivirus or an indexer, its `PermissionError` replaced the original error, and an interrupt surfaced as an `OSError`. The removal now suppresses `OSError`, as unit 1 did for the key, so the original error or `KeyboardInterrupt` always propagates; the docstring says the staged file is removed where the system allows it. RED, with `os.replace` and the removal both patched to fail: 2 failed, 23 passed (the `PermissionError` instead of the disk-full error, and instead of `KeyboardInterrupt`) → GREEN 25 passed. The test also checks that the next write takes over the staged file left behind.
      - F3. The writer of `experiments/domain_stopwords.py` does not repeat `fsync_directory` exactly; the text above and T11b now say so, with the change in failure behavior that reusing it brings.
      - F4. Option (c) is limited to 4-digit years, and its side effect on running numbering such as `12.` is named. Option (b) is limited the same way, since comparing every numbered line by its exact text would also keep such numbering.
    - Checks: `pytest` 1,290 passed, 3 skipped (1,251 + 39 new: 25 shared, 5 in the callers, 9 page-text) in 23.1–25.0 s (2 runs); `ruff check` and `ruff format --check` clean; the attribution scan of the added lines found 0 hits.
    - Size: 980 authored changed lines, 70 of them from the review correction: 603 in tests (428 in the new module), 316 in source (128 new; 188 in the existing modules, 155 of them removed), and 61 in this plan (59) and the ledger (2). That is above the 400-line guide because the lock tests cover each path a long run depends on, across processes, and the writer tests cover each failure path.
  - Commit: pending; prepared on 2026-10-12 as unit 2 of the overnight stack.
- [x] **T10** Objectives locator plus a manual verification sample (~60, stratified by program), producing an accuracy report (O05).
  - Route D. Risk M.
  - Trigger for route D: a new locator, a CLI, workbook input and output, config and tests.
  - Evidence (2026-10-07), implementation:
    - `extraction/objectives.py` holds the pure locator (no I/O) and the table row `ObjectivesRecord`.
      - Lines are matched in lower case with accents folded; the objectives keep their original text, whitespace collapsed, control characters dropped, private-use bullet glyphs shown as `•`, and accents that LaTeX printed before their letter (`Espec´ıficos`) joined back.
      - Headings: `objetivo general`, `objetivos generales`, `objetivo principal`, a line naming the general objective that ends in `:`, and a bare `General` below an `Objetivos` block (or below a mislabeled `Objetivos específicos` block). Specific: `objetivo(s) específico(s)` or `secundario(s)`. OCR confusions (`0`/`o`, `1`/`l`/`i`) and the typo `objeticos` are tolerated.
      - Skipped: contents lines and pages (dotted leaders, trailing page numbers, a contents title), dedication and acknowledgement pages. Abstract pages are a fallback only (`abstract_only`).
      - The general objective is the first body heading's text up to the first line that ends a sentence or the next section, across page breaks. The specific objectives follow within 30 lines, one item per line, up to the next section. Items numbered at least three levels deep under their heading (`1.7.2.1`) are items, not sections.
      - Statuses `extracted` and `not_found` are a subset of the ficha's `ObjectivesStatus`. Flags: `general_too_short`, `general_too_long`, `specific_too_long`, `specific_missing`, `abstract_only`, `block_fallback`, `ocr_page`. `LOCATOR_VERSION` 1 is recorded with the table.
    - `extraction/locate_objectives.py`: the use case and the CLI `python -m thematic_redundancy.extraction.locate_objectives` (`--project-root`, `--snapshot-id`, `--summary-out`, `--make-sample`). It checks each page file's SHA-256 against the text manifest, then writes `data/interim/<snapshot>/objectives.parquet` through the table-store port and `objectives_manifest.json`, both atomically. The run takes seconds, so it always rebuilds the table; it needs no lock because both files are replaced atomically.
    - `labeling/objectives_check.py`: the sample, the workbook, the import and the report, plus the import CLI `python -m thematic_redundancy.labeling.objectives_check` (`--report-out`).
  - Test-first evidence:
    - Config: RED 30 failed, 122 passed → GREEN 152 passed, 1 skipped. One existing test moved `paths.data_dir` alone; it now moves the workbook directory along, as the new rule requires.
    - Locator: RED was a collection error with the module absent. Against a stub that always answers `not_found`: 58 failed, 9 passed. GREEN: 67 passed.
    - Labeling: RED was a collection error. The first run found a real bug, caught by a test: the Wilson upper bound came out as 0.9999999999999999 for 10 of 10 instead of 1. Fixed by pinning both ends. GREEN: 33 passed.
    - Use case and CLI: RED was a collection error → GREEN 6 passed.
    - Round-2 rules: 12 new tests, RED 12 failed, 67 passed → GREEN 79 passed.
  - Real runs, numbers only, all 731 page files (141,076 pages), no OCR:
    - Round 1: 727 extracted, 4 not found (0.5%); sistemas 61/63, minas 101/103, the other programs complete. 16 `specific_missing`. 6.6 s.
    - Round 1 inspection (snippets of at most 300 characters, never written anywhere): the 4 misses were an `Objetivo del Proyecto` block with a `General.` sub-heading (twice: 11776 and 11777 carry the same PDF), a heading followed by a template remark `(Solo va el título)`, and a block mislabeled `Objetivos específicos` that holds a `General` sub-heading. Of the 16 missing specifics, most had an introduction line right below the heading, `Objetivos secundarios`, items numbered `1.7.2.1.`, the typo `Objeticos`, or a LaTeX split accent.
    - Round 2: 731 extracted, 0 not found; 1 `specific_missing` and 1 `general_too_long`, both the same thesis, whose first `Objetivo general` heading introduces a quoted strategic objective; 4 `ocr_page`. Patterns: `objetivo_general` 635, `general_subheading` 58, `objetivo_principal` 35, `objetivo_general_intro` 3, `objetivos_block` 0. 6.7 s.
    - Plausibility checks: heading page index median 20 (P90 28, max 86), none before page index 10; no general objective starts with a lower-case letter, and most start with an infinitive (round 1: diseñar 150, desarrollar 87, proponer 68); general objective median 174 characters (P90 263); ten long specific lists each ended at the next section. A third round was not needed.
    - Final run: the same counts in 5.977 s (about 122 PDFs/s); the whole command, with the workbook, took 8.5 s. `results/objectives/20261002T224412Z/summary.json` holds the settings, the counts and the seconds, no text.
  - Verification sample and workbook (`data/labels/objectives_check/20261002T224412Z/`, gitignored):
    - Design: 12 per program, seed 20261007, drawn from the 728 theses with a valid own PDF (sistemas 62, industrial 329, electronica 46, mecanica 188, minas 103, equal to the data card). Left out: the wrong file `20.500.12920/11777` and, of each D24 pair, the record with the larger uuid (2 handles). Each (program, status) stratum has its own derived seed, `not_found` documents enter in proportion by largest remainder, and the 60 rows are shuffled.
    - The draw: 60 rows, 12 per program, all `extracted`, because the final run found no `not_found` document.
    - The workbook has a `Guía` sheet in Spanish (task, how to judge, 4 to 6 minutes per row) and a `Verificación` sheet: N.º, program, handle linked to the repository, a relative link to the local PDF, the PDF pages counted from 1, the status, both extracted texts, two drop-down verdict columns and notes. The header is frozen, long texts wrap, and no cell is a formula.
    - Checks: all 60 PDF links resolve to a file; 0 hits for 3,096 forms of the 1,066 author, advisor and juror names of the snapshot, and 0 ORCID hits; the name guard redacted 0 cells. Built again in memory in 1.39 s.
    - `sample.json` records the draw, the population, the exclusions and the provenance (locator version, settings fingerprint, table SHA-256), without text.
  - Decision, D24 placeholder: until T12 fixes the canonical record, the sample keeps the member of each pair with the lexicographically smaller uuid. T12 may pick the other one; the sample is not drawn again for that, because the two records are the same thesis.
  - Decision, strict accuracy rule (O05): accuracy = rows judged `Correcto` / judged rows, with a Wilson 95% interval, overall and per program. `Parcial` counts as not correct. The report also gives an informative figure that counts a `not_found` thesis judged `No existe en la tesis` as correct; it does not replace the strict figure. `meets_target` compares the point accuracy with 0.90.
    - Confirmed against the code on 2026-10-07, with no code change. The O05 target is checked on the point estimate, `meets_target=correct / total >= TARGET_ACCURACY` (`labeling/objectives_check.py:859`, with `TARGET_ACCURACY = 0.9` at line 81). `correct` counts only the `Correcto` rows (line 816), since `Parcial` maps to `partial` (line 92). The Wilson 95% interval is always reported next to it (line 845), and the report states the rule (`RULE`, lines 794–798). The line numbers are those of `6f1ba96`.
  - Checks: `pytest` 1,070 passed, 1 skipped (923 + 147 new: config 29, locator 79, CLI 6, labeling 33); `ruff check` and `ruff format --check` clean.
  - Size: about 4,200 authored changed lines, about 1,730 of them tests.
  - Delivery: two reviewed commits instead of the four planned slices (A config, B locator, C workbook, D use case and CLI).
    - The review budget refused the whole candidate at 4,182 lines. A 3,198-line candidate had passed, so the limit lies between the two.
    - Slice 1 (A + B): `4e74a7a` `feat(extraction): agregar el localizador de objetivos de las tesis`, with `config/default.yaml`, `shared/config.py`, `test_config.py`, `extraction/objectives.py` and `test_objectives.py` (1,641 insertions, 16 deletions). The user committed it. Its tree `e6e9a52…` is identical to the reviewed one. Approved before and after the commit, so the reviewed boundary is now `4e74a7a`.
    - Slice 2 (C + D, plus the fixes of the review findings below): `6f1ba96` `feat(labeling): agregar la verificación manual de los objetivos extraídos`, with `labeling/objectives_check.py` and its tests, `extraction/locate_objectives.py` and its tests, `results/objectives/20261002T224412Z/summary.json`, the README, the ledger and this plan, plus `shared/config.py`, `test_config.py` and `test_objectives.py`. The user committed it. Approved before and after the commit, so the reviewed boundary is now `6f1ba96`.
  - Review findings: both reviews of slice 1 raised the same two non-blocking warnings. Both are fixed in slice 2, test first.
    - Warning 1, the record validator had no test: `ObjectivesRecord._require_consistent_fields` was untested.
      - Added 24 characterization tests in `test_objectives.py`:
        - 5 consistent records accepted;
        - 16 contradictory records refused, one per branch: an `extracted` record without the general objective, `page_start`, `page_end` or `pattern`; a `not_found` record with a general or specific text, a page, a pattern or flags; `general_chars` or `specific_chars` off; `page_end` before `page_start`; a repeated flag;
        - the rule order;
        - errors that never echo the input.
      - They passed at once, as characterization tests do. 13 throwaway mutants were all killed and then reverted: each condition removed or inverted, the flags rule moved ahead of the content rule, and `hide_input_in_errors` turned off.
      - No bug was found, so `LOCATOR_VERSION` stays 1.
      - Rule order, pinned: a `not_found` record reports its content rule first, then its flags rule. Any flag already breaks a `not_found` record, so the repeated-flag rule, which runs last, only ever reports on `extracted` records.
    - Warning 2, workbook containment: `resolve_workbook_dir` checked only that the resolved workbook directory stays inside the project root, and `AppConfig` checked containment in `paths.data_dir` only on the text. A symbolic link or junction under `data_dir` that points to a tracked folder could therefore route thesis text into git.
      - Fix: `resolve_workbook_dir(project_root, data_dir)` now also requires the resolved workbook directory to lie inside the resolved `data_dir`, and raises `ValueError` otherwise. Its one caller, `workbook_directory()`, passes `paths.data_dir`.
      - RED: with the old signature, 5 failed with a `TypeError`. With the new signature but no check, 2 failed with `DID NOT RAISE`:
        - a real junction `data/labels` → `results/labels`, made with `_winapi.CreateJunction`, which needs no privilege (symbolic links fail here with WinError 1314); the test skips only when neither a symbolic link nor a junction can be made;
        - a variant that patches `Path.resolve`, so the check runs on every platform.
      - GREEN: 5 passed. A `data_dir` that is itself a link inside the root still holds the workbook, since both sides are compared once resolved.
      - 3 throwaway mutants were all killed and then reverted: the check removed, the check inverted, and `data_dir` compared without resolving it.
    - Checks after the fixes: `pytest` 1,097 passed, 1 skipped (1,070 + 24 + 3); `ruff check` and `ruff format --check` clean.
  - Commits: `4e74a7a` (slice 1) and `6f1ba96` (slice 2), both made by the user. The closure below awaits its commit, together with T10b.
  - Verification (O05), 2026-10-07:
    - Effort: the user filled the 60-row workbook in about 2 h (reported by the user, not timed), so about 2 minutes per row (derived), below the 4 to 6 minutes estimated.
    - Report: `results/objectives/20261002T224412Z/verification.json`, from the official import of the workbook with `workbook_sha256` `8e8ffe83…eda53`, imported at 2026-10-07T21:43:43Z. It holds numbers only.
    - General objective: 60/60 `Correcto` = 100%, Wilson 95% CI 94.0%–100%. The O05 target of at least 90% is met (`meets_target` true). 0 partial, incorrect, missed or absent.
    - Per program: 12/12 in each of the 5 programs, each with a Wilson 95% CI of 75.8%–100%. These intervals are wide, so only the overall figure supports the ≥90% claim.
    - Specific objectives: 60/60 `Correcto`, Wilson 95% CI 94.0%–100%.
    - The workbook holds 0 notes and 0 redaction markers.
    - The lower bounds were checked by hand: with no failures, the Wilson lower bound is n/(n + z²), so 60/(60 + 1.96²) = 0.9398 and 12/(12 + 1.96²) = 0.7575.
  - Consequence: the extracted objectives are accepted for T12 and for the representation (D09). The title + abstract fallback is needed only for the 16 metadata-only theses (744 distinct theses, 728 with a valid own PDF).
  - Limitations:
    - A single verifier, the author, so there is no inter-rater agreement.
    - Every sampled row was `extracted`, because the corpus has 0 `not_found`; the `not_found` path was not verified on real theses.
    - The per-program intervals are wide (lower bound 75.8%).
- [x] **T10b** Fixes from the slice-2 reviews of T10: two warnings and one suggestion, all non-blocking.
  - Route D. Risk M.
  - Trigger for route D: a module fix with tests, plus the plan, the ledger and the README.
  - Warning 1, an unreadable workbook escaped the import command as a raw traceback. `main` caught only `WorkbookValidationError`, `OSError`, `ValueError` and `yaml.YAMLError`, and openpyxl raises other exceptions for a damaged or non-xlsx file.
    - Fix: `import_verification` loads the workbook through `_load_workbook`, which turns those exceptions into one `ValueError`: "`<path>` is not a readable xlsx workbook (`<reason>`)". The command prints it as one `error:` line and exits with code 1. A missing file is still a `FileNotFoundError`. The exceptions, as probed on openpyxl 3.1.5:
      - `BadZipFile`: garbage bytes or a truncated zip;
      - `KeyError`: a zip without the workbook parts;
      - `SyntaxError`: broken XML, since `ParseError` is a subclass;
      - `TypeError`: a value of the wrong type in a part;
      - `InvalidFileException`: an extension openpyxl does not support.
    - RED: 5 command-line cases (garbage, truncated zip, zip without a workbook, broken XML, invalid value) escaped as `BadZipFile` twice, `KeyError`, `ParseError` and `TypeError`. GREEN: 5 passed.
    - `InvalidFileException` comes only from openpyxl's extension check. The command never reaches it, since the workbook name is fixed, so a direct test imports an xlsx saved under a `.xls` name. It was added after GREEN and is proven by its mutant below.
    - 5 throwaway mutants, one per exception class dropped from the list, were all killed and then reverted.
  - Suggestion, two refusal branches of `import_verification` had no test: a row whose handle is not in the sample, and a row whose `Estado` was edited.
    - Added one characterization test per branch. Each asserts its exact problem message.
    - Both passed at once, as characterization tests do. 4 throwaway mutants were all killed and then reverted: each problem not recorded, the unknown-handle message without its row, and the status check inverted.
  - Warning 2, the name redactor matched without word boundaries, ignoring case, so a name form could match inside a longer word of the objective text.
    - Fix: a form is replaced only where no letter stands right before or after it. The boundary is a lookaround on `_LETTER`: a word character other than a digit, `_` or a superscript digit, so accented letters and ñ count as letters. Python's `\w` is Unicode-aware for `str` patterns.
    - Why letters, not `\w` or `\b`:
      - A digit glued to a name, such as a footnote number (`1` or `²`), must not stop the redaction. With `\w` lookarounds it did, which would leak the name.
      - A form that ends with an initial's period needs no letter after it, but `\b` would demand one. One name of the snapshot ends that way.
      - With `re.ASCII`, `ñ` would end a word.
    - RED 1, the old pattern: a synthetic family name that ends one longer word and begins another came out as `pre[nombre omitido] con la [nombre omitido]ña`.
    - RED 2, after a first fix with `\w` lookarounds: two names followed by a footnote number were left unredacted.
    - GREEN: all passed.
    - 6 throwaway mutants were all killed and then reverted: the lookbehind dropped, the lookahead dropped, ASCII-only `\b`, Unicode `\b`, `\w` lookarounds, and superscript digits counted as letters.
    - No effect on the completed verification. The workbook holds 0 redaction markers, so the old pattern matched no form anywhere in its 60 rows. Every match of the new pattern is also a match of the old one, so a rebuild would redact 0 cells too. The workbook was not rebuilt.
  - Every mutant ran with a fresh bytecode cache (`PYTHONPYCACHEPREFIX`), and the module was byte-identical after each restore.
  - Checks: `pytest` 1,107 passed, 1 skipped (1,097 + 10 new: CLI 5, extension 1, characterization 2, redactor 2); `ruff check` and `ruff format --check` clean.
  - Commit: `c121d78`, made by the user together with the T10 closure. Approved before and after the commit, so the reviewed boundary is now `c121d78`.
- [ ] **T10c** Follow-ups from the T10b reviews (non-blocking):
  - `zlib.error` and `EOFError`, raised by a corrupted deflate member of the workbook, escape the unreadable-workbook handling (`labeling/objectives_check.py:709`).
  - The letters-only boundary of the name redactor no longer redacts a name glued to an adjacent letter, such as a preposition fused to a surname in PDF text. Decide the policy (over- or under-redaction) before any new workbook is generated.
  - The broad `KeyError`/`TypeError` catch could mask a library bug; pin it with a test.
  - Route D. Risk M.
- [x] **T11** Light and full cleaners. They handle:
  - title suffix stripping, header/footer removal and hyphenation;
  - domain stopwords (by document frequency, then manual review);
  - legal suffixes;
  - spaCy lemmatization (full cleaner only).
  - Accents and ñ are kept.
  - Route D. Risk M.
  - Evidence (2026-10-03). Modules in `preprocessing/`:
    - `light_cleaner.py`:
      - runs without spaCy and is idempotent;
      - unwraps lines in linear time, keeping sentence and paragraph breaks;
      - joins words split by an end-of-line hyphen while keeping real compounds;
      - normalizes quotes and dashes conservatively.
    - `title_suffix.py`:
      - holds the shared patterns, moved from `profile.py`; the 82 profile tests are unchanged and pass;
      - strips repeatedly, never below 3 words, and returns a quality note.
      - On real data: 397 titles stripped, 0 refused, 0 left with a suffix.
    - `full_cleaner.py`:
      - spaCy `es_core_news_md` lemmas, with spaCy and domain stopwords removed;
      - legal suffixes (S.A.C., E.I.R.L., …), URLs and emails dropped;
      - keeps "Lean", which spaCy mislemmatized as *leer* in 61 of 172 cases;
      - throughput 25–76 docs/s.
    - `stopwords.py` loads `config/stopwords_domain_es.txt`: 79 curated lemmas covering genre, reporting verbs, institution, place and time, and generic words. It shipped with 84; T11a removed 5, and D26 added 10 (89).
  - The candidates come from `experiments/domain_stopwords.py`. The output is aggregate only: 411 lemmas with DF ≥ 5% over 744 documents, after the D24 dedupe. Privacy: 0 hits.
  - Tests: 124 new, RED → GREEN. `ruff` clean. `pytest`: 762 passed, 1 skipped (independent re-run).
  - Moved to T09: header and footer removal, which needs page-level PDF text.
  - For review at T16, once topics exist: borderline frequent words deliberately left out of the list, such as análisis (414), implementación (375), trabajo (332), estudio (315) and metodología (282). Also the 5 that T11a removed: mejora (361), empresa (335), mejorar (291), desarrollar (272) and desarrollo (224).
  - Commit: `e9f4f70`, made by the user. Its tree `4daf041…` is identical to the reviewed one.
  - Reviews: high risk, four lenses, approved both before and after the commit.
- [x] **T11a** Follow-ups from the T11 reviews and the user's observations (2026-10-03):
  - A1. Privacy gate tests: `privacy_problems` of `experiments/domain_stopwords.py` is the only barrier between the snapshot text and the published results file, and it had no test.
  - A2. Tests for the D24 dedupe (`distinct_theses`) and the document builder (`thesis_document`).
  - A3. Prove each characterization test of A1 and A2 can fail, with throwaway mutants.
  - A4. Guard the throughput report: an empty snapshot or zero elapsed time raised `ZeroDivisionError` after the output was written.
  - A5. Durable publish: fsync the staged file before `os.replace`, and the parent directory after it, tolerating platforms that cannot open a directory (Windows).
  - A6. `NOTE_MAX_LENGTH` in `title_suffix.py` repeats `QUALITY_NOTE_MAX_LENGTH` of `corpus/ficha.py`, because `preprocessing` may not import `corpus`. Comment why, and keep a test that holds the two equal.
  - B. Stopword list: the "Organization and improvement" group fits none of the file's criteria. `universidad` moves to the institution, place and time group; `empresa`, `mejora`, `mejorar`, `desarrollo` and `desarrollar` leave the list ("When in doubt, a word stays out"); 84 → 79 lemmas.
  - Route D. Trigger: tests plus script plus config plus plan, 4+ non-trivial files. Risk M.
  - Evidence (2026-10-03):
    - B: RED, the new test `test_the_curated_list_leaves_out_frequent_words_that_fit_no_criterion` failed on the shipped list (1 failed, 15 passed). GREEN after the edit (16 passed). The list now holds 79 lemmas, `universidad` included. The header lists the 5 removed words among the frequent words that stay out.
    - A1/A2: 22 characterization tests with synthetic records (privacy gate 11, dedupe 8, document builder 3). They passed on the existing code. 23 throwaway mutants were all caught, for example: leak check disabled, sensitive strings of no record, a leak message that echoes the leak, an empty lemma allowed, the 40-character bound as `>=` or `+ 1`, tabs not counted as whitespace, `any` instead of `all` in the dedupe key, no case folding, no whitespace or NFC folding, accents folded away, the kept order reversed, a key that ignores the title or the abstract, the last value instead of the first, the title suffix kept, title or abstract not light-cleaned, and the abstract always appended.
    - A4: RED, `ZeroDivisionError` in both cases (two theses, and an empty snapshot) with a clock that stands still. GREEN: the run exits 0, writes the output, and prints "throughput unavailable". A second test pins the documents-per-second figure. Mutants: `>=` instead of `>`, and "always unavailable"; both caught.
    - A5: RED, the recorded calls were only `replace`. GREEN: flush and fsync of the staged file (3 bytes on disk at fsync time), `replace`, then the directory open, fsync and close; a directory that cannot be opened is skipped. The repo had no directory-fsync helper to reuse (T08a lists one for the key file), so the script has its own `_fsync_directory`. 6 mutants caught: no fsync, no flush, no directory fsync, open failure not tolerated, descriptor not closed, directory flushed before the move.
    - A6: the comment was added, and the equality check moved into its own test, `test_the_note_length_limit_equals_the_ficha_quality_note_limit`. Mutants 199 and 250 were both caught.
    - Every mutant ran with a fresh bytecode cache. A same-size mutant restored within the same second can leave a stale `.pyc` that still looks valid.
    - Checks: `pytest` 791 passed, 1 skipped (762 + 29 new). `ruff check` and `ruff format --check` are clean.
    - Size: about 590 authored changed lines, 428 of them in `test_domain_stopwords.py` and 97 in this plan (T05 closing evidence included). This exceeds the 400-line guide, because the characterization tests cover each branch of the privacy gate and the dedupe.
  - Commit: `633dfd2`, made by the user. Its tree `1b22471…` is identical to the reviewed one.
  - Reviews: medium risk, one lens, approved both before and after the commit.
- [x] **T11b** Follow-ups from the T11a reviews (non-blocking):
  - `_fsync_directory` in `experiments/domain_stopwords.py` tolerates only a failure to open the directory. An `OSError` from `os.fsync` on an opened directory (EINVAL on some network, FUSE or overlay filesystems) propagates after `os.replace` has already published the output, which contradicts its docstring. Tolerate it, and add a test. Both reviews raised it.
    - `fsync_directory` in `shared/filesystem.py` (T09b) can replace it. That changes the experiment's failure behavior: an EINVAL, or any other `OSError`, from flushing a FUSE or network directory is then swallowed instead of raised, which is the fix above, so its tests must expect it.
  - The documents-per-second test couples to the number of `perf_counter` readings in `main`. Assert only the format, or control the two readings that bracket the cleaner.
  - Pin the D26 list in a test, by its 89-lemma count or the file's SHA-256, so the freeze before T18 is observable (review R3-001 of the closure unit `b6943fa`).
  - Route D. Risk M. Trigger: the candidates script and its test module, on top of the shared writer that T07a extends; done together with T07a as one unit.
  - Evidence (2026-10-12):
    - Item 1, the directory flush:
      - `_write_atomically` now creates the output folder and calls `write_bytes_atomically` from `shared/filesystem.py` (T07a). Its directory flush is therefore `fsync_directory`: a directory that cannot be opened (Windows) or flushed (EINVAL on some network, FUSE or overlay file systems, or any other `OSError`) is skipped, since the output is flushed and in place by then.
      - This is the change in failure behavior noted above: such an error is now swallowed instead of raised after the output was published. The script's own `_fsync_directory` and its `os` and `tempfile` imports are gone.
      - The bytes written are the same. The staged file is now `<name>.part` beside the output instead of a `mkstemp` name, removed on failure where the system allows it.
      - RED: the new test `test_write_atomically_publishes_where_the_directory_cannot_be_flushed` (EINVAL and EIO) gave 2 failed, 47 passed: the `OSError` was raised after the replace. GREEN: 49 passed. A test that the folder is created passed at once.
    - Item 2, the throughput test: the stand-in clock now moves only inside the stand-in model load (1.5 s) and the stand-in cleaner (4.0 s), and the test checks `2 documents in 4.0 s (0.5 documents/s` and `model load 1.5 s`, however often `main` reads the clock. With an extra clock reading added to `main`, the old test failed and the new one passed; both files were restored byte-identical.
    - Item 3, the D26 pin: `test_the_curated_list_is_frozen_as_decided_in_d26` checks 89 lemmas and the SHA-256 of the sorted lemmas joined by line feeds, `8a889062a2f117e2…`. Its docstring says that a deliberate change, such as one from the T16 review, must update both values together with a decision in this plan. Proven on temporary copies of the list, which itself was not touched: a lemma added, a lemma removed, or one `ñ` dropped fails; a comment edited, a blank line added and two lemmas swapped pass.
    - Mutants, each with a fresh bytecode cache, the script byte-identical after each restore: 5, all killed (no folder created, seconds per document, the clock read after the cleaner, the cleaner timed from the model load, the load timed after it).
    - Real data: none; the script writes under `results/`, and its tests cover it.
    - Review fixes: none in this script. Its one-file write goes through `write_files_atomically`, which now also refuses a name that ends in `.part` (U4-02 under T07a); the candidates file is `stopword_candidates.json`, so it writes as before.
    - Checks and size: under T07a, which closes the same unit.
  - Commit: pending; prepared on 2026-10-12 as unit 4 of the overnight stack.
- [x] **T12** Build the fichas dataset, the exclusion log, and a quality report. Inclusion requires an exact `renati.type` fragment `#tesis` (746 in snapshot `20261002T224412Z`). Suficiencia profesional appears in 5 spellings and is excluded with its reason.
  - Route D. Risk M.
  - Inputs from the T05 findings (2026-10-03): 11777 is metadata-only with a quality note, and the D24 canonical rule must also choose the industrial pair's PDF.
  - Trigger for route D: a builder, a quality report, config and tests across 6 or more files.
  - Evidence (2026-10-07), implementation:
    - `corpus/build_fichas.py`: the pure rules, the enforced checks, the use case and the CLI `python -m thematic_redundancy.corpus.build_fichas` (`--project-root`, `--snapshot-id`, `--report-out`). It writes `fichas.parquet` (all 766), `exclusions.parquet` (no text) and `fichas_manifest.json` into `data/interim/<snapshot>/`, atomically, and only after every check passes.
    - `corpus/fichas_quality.py`: the quality report, the stopword re-check, and a privacy gate that refuses a report holding a title, an abstract, a name or an ORCID iD of the snapshot, or a lemma that is not one word.
    - Config `fichas` (row above): the wrong PDF of 11777 and the expected counts, which must add up. The build first checks that they belong to the snapshot being built.
  - Rules:
    - D24, decided by the parent: the record accessioned first stays, and a tie goes to the smaller item uuid. `duplicate_of` holds the item uuid that stays, as `Ficha` types it.
    - Objectives: the general objective, a blank line, then the specific objectives. Each locator flag becomes a quality note.
    - 11777 is `no_pdf` with its PDF hash kept and the note "PDF on disk belongs to another item (20.500.12920/11776)". The build confirms that both PDFs have the same SHA-256.
    - Source format: `digital`, `ocr` or `mixed` from the text manifest's pages; `none` when no PDF text is used.
    - Excluded records get their fields by the same rules, so every ficha describes its record fully.
  - Test-first evidence:
    - Config: RED 10 failed, 154 passed → GREEN 164 passed, 1 skipped.
    - Builder and report: RED was a collection error with each module absent. The first GREEN run gave 23 failed, 55 passed, all from test-helper defects: titles and abstracts differed per record, so no D24 group formed; a page-text entry had OCR pages without OCR attempts; a helper passed a keyword twice; tests used the real wrong-PDF declaration; one expected message order. GREEN after fixing the helpers: 78 passed.
    - 19 throwaway mutants, each with a fresh bytecode cache, the module byte-identical after each restore: 18 killed. The survivor reversed the uuid text, which keeps the order of the counter uuids, so it was equivalent; a true order flip was then killed.
  - Real run (2026-10-07, numbers only):
    - 766 fichas: 744 included, 22 excluded (20 `non_thesis_type`, 2 `duplicate`). 728 `extracted`, 16 `no_pdf` (14 restricted, 1 without a file, 1 wrong file), so 16 metadata-only. 766 unique doc codes. Every enforced check passed.
    - Advisor ORCID iD on 712 included theses: 691 valid, 21 malformed or failing the check digit, each with a quality note. Advisor codes: 691 from the ORCID iD, 53 from the name; 138 distinct advisor codes and 784 distinct author codes.
    - Quality notes besides the ORCID ones: `ocr_page` 4, `general_too_long` 1, `specific_missing` 1, wrong PDF 1. Future-dated: `16319` and `16433`.
    - Text lengths in characters (median, P95): title 147, 217; abstract 1,720.5, 2,886; objectives 702.5, 1,306.
    - Build 0.6 s; quality report 8.5–9.2 s with the spaCy load; whole command 11.1 s. A second run wrote the same fichas table (same SHA-256). The first run created the pseudonym key.
    - No-names check: 0 hits for 3,065 name forms and 131 ORCID values in the 19,367 text cells of `fichas.parquet`.
  - D24 picks: `15247` stays over `15265` (industrial, different PDFs: the PDF of `15247` is kept), and `16449` over `16450` (mecanica, the same PDF). In both pairs the record that stays was accessioned first and also has the smaller uuid, so the rule agrees with the T10 placeholder: the T10 sample rows refer to the canonical records. The objectives texts of both twins have equal hashes.
  - Stopword re-check over the objectives of the 728 theses, with the full cleaner without domain stopwords (no list change):
    - 27 of the 79 domain stopwords reach 5% of the objectives; 3 never occur (`concluir`, `evaluó`, `posteriormente`). The most frequent: determinar 284, evaluar 282, analizar 270, identificar 257.
    - 102 lemmas not on the list reach 5%, led by diseñar 282, sistema 272, proceso 262, empresa 260 and mejora 251.
    - 14 infinitive-shaped candidates, for the user to decide: diseñar 282, desarrollar 206, implementar 182, mejorar 157, optimizar 141, elaborar 124, reducir 86, diagnosticar 77, seleccionar 74, validar 67, comparar 65, incrementar 63, definir 47, medir 39. T11a removed `desarrollar` and `mejorar` on purpose; some candidates may name topics (optimizar, diagnosticar).
    - Decided on 2026-10-09 as D26: the 10 research-action verbs join the list, and mejorar, optimizar, reducir and incrementar wait for the T16 review. The committed quality report describes this run, with the 79-lemma list.
  - Checks: `pytest` 1,194 passed, 1 skipped (1,107 + 87 new: config 9, builder 67, report 11) in 22.5 s; `ruff check` and `ruff format --check` clean. The attribution scan and the privacy scan of the tracked new and changed files found 0 hits; the synthetic test names share no word with the snapshot names.
  - Size: 3,161 authored changed lines (2,913 in new files), 1,326 of them tests and 391 the report files. That is above the 2,500-line target and close to the review limit, so two slices:
    - Slice 1 (849 lines): the config unit and the quality report: `shared/config.py`, `config/default.yaml`, `test_config.py`, `corpus/fichas_quality.py`, `test_fichas_quality.py`. Checked alone on a copy of `c121d78`: 1,127 passed, 1 skipped.
    - Slice 2 (2,312 lines): the builder, its CLI and the outputs: `corpus/build_fichas.py`, `test_build_fichas.py`, `results/fichas/20261002T224412Z/`, the README, the ledger and this plan. It depends on slice 1; both together give the 1,194 above.
  - Commit: `73f1676`, made by the user as one commit of 12 files instead of the two planned slices, since the whole 3,161-line candidate fit the review budget. Its tree `760a5e9…` is identical to the reviewed one. Approved before and after the commit, so the reviewed boundary is now `73f1676`.
- [x] **T12a** Follow-ups from the T12 reviews (non-blocking):
  - The three outputs of `build_dataset` (`fichas.parquet`, `exclusions.parquet`, `fichas_manifest.json`) are each written atomically, but not as a set (`corpus/build_fichas.py`, lines 615–637 at `73f1676`): a failure between writes leaves a mix of new and old files. Publish them as one unit, or have readers check the manifest's SHA-256 values.
  - When `--report-out` fails after the build has published the fichas, `write_quality_report` raises "…; nothing was written" (`corpus/fichas_quality.py:255`), which reads as if the fichas were not written either. Say which files were written.
  - Some command-line paths of `build_fichas.main` have no test: an interruption (exit code 130), a refused or failed `--report-out` after the build (exit code 1, see above), and an unreadable configuration (`yaml.YAMLError`). The refusals for undeclared or missed counts and for a changed objectives table are tested.
  - `_advisor_orcid` (`corpus/fichas_quality.py:347`) counts an advisor name as present when its metadata list is not empty, even when every value is blank, so `advisor_code_from_name` can overcount. Count only non-blank values.
  - Route D. Risk M. Trigger: the builder, the quality report and their two test modules (4 non-trivial files), plus this plan, the ledger and the README.
  - Evidence (2026-10-11):
    - Finding 1, one publish unit. `build_dataset` stages both tables as `<name>.staged` through the table-store port and takes their SHA-256 from the staged files. It then moves both into place and writes `fichas_manifest.json` last, as the commit point.
      - A failure or an interruption before the moves removes the staged files and leaves the previous outputs untouched, so the command's "nothing half-written was left" stays true.
      - A stop between the two renames, or before the manifest, leaves tables that the manifest does not describe.
      - The new reader `verify_fichas_outputs` checks both tables against the manifest's SHA-256 values and raises on a missing file or a mismatch. `load_fichas` reads the fichas through it; T13 and T14 read them with it.
    - Finding 2. `write_quality_report` now ends a refusal with "the quality report was not written". When the report fails after the build, `main` prints one line, "error: the fichas outputs were written to <dir>, but the quality report was not: <reason>", and returns 1. An interruption there says the same and returns 130.
    - Finding 3. New command-line tests: an interruption while saving (exit 130), a refused and a failed `--report-out` after the build (exit 1), an interruption during the report (exit 130), and an unreadable configuration, with bad syntax or a repeated key (exit 1).
    - Finding 4. `_advisor_orcid` counts a name only when one advisor value is a non-blank string, as the builder reads it. The other counters are unchanged.
    - RED → GREEN:
      - Finding 4: 3 failed, 13 passed (a blank value, an empty and a tab value, and a null value were counted as names) → 16 passed.
      - Finding 2, the report: 1 failed, 15 passed, on the old message "…; nothing was written" → 16 passed.
      - Command line and publish: 5 failed, 69 passed. The two stopped-build tests failed because the old code had already replaced `fichas.parquet`; the other three were the report-failure and report-interruption messages. The two configuration tests passed at once, as characterization.
      - Reader: a collection error, with `load_fichas` absent → 97 passed in the two modules.
    - 17 throwaway mutants, each with a fresh bytecode cache, the modules byte-identical after each restore: all killed. They covered staged files not removed, the manifest written before the moves, the fichas table written straight to its final name, the reader skipping the exclusions or never comparing hashes, `load_fichas` not verifying, the wrong directory or exit code in the report messages, YAML errors not caught or not kept to one line, the old refusal message, and three weaker advisor rules (any string, a non-empty list, the first value only).
    - Real data, numbers only:
      - Before: `fichas.parquet` `4918fbc6…` and `exclusions.parquet` `bb915e46…`, equal to the manifest of 2026-10-07 (0.628 s).
      - Re-run without `--report-out`: the same two SHA-256 values. Build 0.843 s by the manifest, 2.8 s for the whole command. No staged or temporary file was left.
      - The reader accepted the real outputs in 0.002 s and loaded 766 fichas, 744 included, in 0.46 s.
      - Finding 4 on the 744 included theses, in memory: 53 advisor codes from the name under the old rule and under the new one; 0 advisor lists hold only blank values. The committed `quality_report.json` (53) is unaffected, and nothing under `results/` changed.
    - Checks: `pytest` 1,215 passed, 1 skipped (1,196 + 19 new: builder 14, report 5) in 42.7–48.0 s (2 runs); `ruff check` and `ruff format --check` clean; the attribution scan of the added lines found 0 hits.
    - Size: 473 authored changed lines: 245 in the two test modules, 182 in the two source modules, 38 in this plan, 6 in the README and 2 in the ledger. That is above the 400-line guide, because the tests cover each failure path of the publish and of the command line.
    - Known limit, outside the review findings: the two quality-report files (`quality_report.json`, then `quality_report.md`) are each written atomically but not as a pair. A stop between them leaves a new JSON beside an old Markdown, and the message "the quality report was not written" is then too strong. Harmless for the committed report; revisit if the report is regenerated.
  - Commit: `ad2c6f0`, made by the user. Its tree `11e3857…` is identical to the reviewed one. Approved before the commit (one lens); its three suggestions became T12b.
- [x] **T12b** Follow-ups from the T12a review (non-blocking):
  - An interruption (`KeyboardInterrupt`) between the moves of the staged tables and the manifest write reaches the build handler of `main`, which prints "nothing half-written was left": not true there, since the tables were replaced and the manifest still describes the previous ones. The data stay safe because `verify_fichas_outputs` refuses the mismatch, but the message should say a re-run is required, and no test covers an interruption inside that window.
  - No test corrupts or truncates `fichas_manifest.json` to show that `verify_fichas_outputs` raises `ValueError` (it relies on pydantic's `ValidationError` being a `ValueError` subclass).
  - `load_fichas` hashes the fichas table and then reads it a second time, so a concurrent rebuild could swap the file between the check and the read. Read the bytes once, then hash and parse them.
  - Route D. Risk M. Trigger: the builder and the storage port with their two test modules (4 non-trivial files), plus this plan, the ledger and the README.
  - Evidence (2026-10-12):
    - Item 1, the interruption message. `build_dataset` raises `PublishInterrupted`, a `KeyboardInterrupt` subclass that carries `interim_dir`, for an interruption from the first move of a staged table until the manifest write returns. An earlier one stays a plain `KeyboardInterrupt`, and any caller that handles one handles both.
      - `main` then prints "interrupted: the outputs in <dir> were being replaced, so the tables may not match fichas_manifest.json, and readers refuse them until they do; run the command again", and returns 130. Before the moves it still prints "nothing half-written was left".
      - The message says "may": an interruption that lands just before the first move, or after the manifest's rename while its directory is flushed, falls inside the window although the outputs still match.
    - Item 2, a damaged manifest. Pydantic's error is a `ValueError`, but its message took 3 lines, named neither the file nor the command, and quoted a refused snapshot id. `verify_fichas_outputs`, and so `load_fichas`, now raise one line: "fichas_manifest.json in <dir> is damaged (<field>: <issue type>; …); run <build command> again", such as `json_invalid` or `snapshot_id: value_error`. The error is not chained to the validation error, so a traceback does not quote the file either.
    - Item 3, one read, in its final form after review fix F3. The table-store port gains `read_verified(model_cls, path, *, sha256)`. `ParquetTableStore` reads the file once, compares the SHA-256 of those bytes, and parses the same bytes through `pa.BufferReader`, with the column and row checks and the messages of `read`. `read(model_cls, path)` is unchanged. `load_fichas` reads the fichas table through it. The exclusions table is checked as before and not returned.
    - RED → GREEN:
      - Storage, first version (`read_from_bytes`, replaced in F3): 9 failed (`AttributeError`), 29 passed → 38 passed.
      - Builder: first a collection error (`ImportError`), with `PublishInterrupted` absent. With the class added but not raised: 10 failed, 82 passed → 92 passed. The two interruptions inside the window printed "nothing half-written was left"; `build_dataset` raised a plain `KeyboardInterrupt`; the reader returned the table put in place right after its check; and the six damaged manifests (truncated, garbled, a refused snapshot id, each through both readers) got pydantic's 3-line message, which quoted the marker in the snapshot-id case.
      - An interruption just before the moves, from the manifest's clock with both tables staged, passed at once, as characterization.
    - 17 throwaway mutants in 18 runs, each with a fresh bytecode cache, the modules byte-identical after each restore (SHA-256): all killed.
      - Builder (14): no wrapper; the window without the moves; the window with the staging (killed again by the characterization test alone); the window without the manifest write; the wrong directory in the message; a wrapper that is no `KeyboardInterrupt`; the old message for the window; the damaged-manifest error chained, keeping pydantic's messages, without the build command, or not caught; `load_fichas` reading the table a second time; the exclusions not checked; the hash never compared.
      - Storage (3), against the first version: bytes reported under another name, bytes without the column check, and a bad row without its number.
    - Real data, read-only, numbers only: `load_fichas` on `data/interim/20261002T224412Z/` returned 766 fichas, 744 included, in 0.498 s; `verify_fichas_outputs` alone took 0.002 s. `fichas.parquet` `4918fbc6…`, `exclusions.parquet` `bb915e46…` and `fichas_manifest.json` `cf5926cb…` kept their SHA-256, sizes and modification times, and the directory listing did not change. Nothing was built, and nothing under `results/` was written.
    - Review fixes (2026-10-12): an independent adversarial review raised a false claim, a warning and a design finding. One scoped correction fixed all three, test first.
      - F1. The damaged-manifest message took each issue's location from pydantic's `loc`, which holds keys taken from the file: an unknown top-level key, or a key of `counts`. Such a key was quoted, and a `counts` key with a line break made the error two lines. The location now names a `FichasManifest` field only, and any other part shows as `<key>`, as in `<key>: extra_forbidden` or `counts.<key>: greater_than_equal`. The test now pins the exact message for 5 kinds of damage, through both readers.
      - F2. A second Ctrl+C during the removal of the staged files replaced `PublishInterrupted` with a plain `KeyboardInterrupt`, so the command printed "nothing half-written was left" over tables that the manifest did not describe. A flag now marks the span from the first move until the manifest is written, and it is checked after the cleanup, so an interruption there keeps the label of the moves. After a complete publish the flag is clear, and an interruption during the cleanup still prints "nothing half-written was left", which is then true.
      - F3. `read_from_bytes` put the bytes of a Parquet file on the generic table-store port, which a later database adapter (D05, D19) could not honor. `read_verified(model_cls, path, *, sha256)` replaces it, as described under item 3. A mismatch raises `ChecksumMismatchError`, a `ValueError`, before anything is parsed. `load_fichas` turns it into its usual message, "fichas.parquet differs from fichas_manifest.json …".
      - RED → GREEN. Builder: 5 failed, 93 passed → 98 passed. The two new kinds of damage, through both readers, quoted the key or broke the line, and the second interruption printed "nothing half-written was left". An interruption during the cleanup of a complete publish passed at once, as characterization. Storage, with `ChecksumMismatchError` added but not used: 12 failed (`AttributeError`: no `read_verified`), 29 passed → 41 passed. The 9 tests of `read_from_bytes` became 12 tests of `read_verified`, the equivalence with `read` included.
      - 14 throwaway mutants, each with a fresh bytecode cache, the modules byte-identical after each restore: all killed.
        - F1 (3): any first key named, deeper keys named, the locations dropped.
        - F2 (5): the flag cleared when the cleanup starts, not cleared after the manifest, set only after the moves, every interruption labeled, and the flag checked before the cleanup, as before the fix. The double-interruption test alone kills the last one.
        - F3 (6): the SHA-256 never compared; the file read a second time (also killed by the store's read-once test alone); another mismatch message; the exclusions' SHA-256 passed for the fichas; the mismatch not turned into the reader's message; the exclusions not checked.
      - The review's probes now give one line without the key for both manifest cases, and the moves' message for the double interruption.
      - Real data, re-run read-only: 766 fichas, 744 included, in 0.385 s; the check alone took 0.001 s. The three files kept their SHA-256 (`4918fbc6…`, `bb915e46…`, `cf5926cb…`), sizes and modification times.
    - Checks: `pytest` 1,319 passed, 3 skipped (1,290 + 29 new: storage 12, builder 17) in 26.3–26.7 s (2 runs, after the review fixes; 1,310 before them); `ruff check` and `ruff format --check` clean; the attribution scan of the added lines found 0 hits.
    - Size: 731 authored changed lines for the whole unit: 399 in the two test modules, 275 in the two source modules, 47 in this plan, 8 in the README and 2 in the ledger. It was 490 before the review fixes; the correction changed 439 lines of that version, part of them the staging block re-indented under the new handler. That is above the 400-line guide, because the tests cover each interruption point, each kind of damage through both readers, and the store's verified read.
    - Known limit, outside the findings: a failure inside the same window, rather than an interruption, such as a refused rename of the second table, still prints only the system's error. The reader refuses those outputs too, and a new run fixes them.
  - Commit: pending; prepared on 2026-10-12 as unit 3 of the overnight stack.

### P3 — Representation and topic map (M3–M5)
- [ ] **T13** Sentence-aligned chunker that is valid for both tokenizers. Property tests check that every chunk fits both tokenizers and no text is lost.
  - Route D. Risk M.
  - Inputs (2026-10-11): both models are cached (U8), so the chunker can run offline (`HF_HUB_OFFLINE=1`). Measured offline with transformers 5.18.0: the MiniLM tokenizer (model type `bert`) and the mpnet tokenizer (`xlm-roberta`) hold the same 250,002-token vocabulary, with the same 5 special tokens, and gave the same token ids for 3 Spanish sample sentences with accents, ñ and symbols. D09's 250,037 vs 250,002 are the models' `config.json` `vocab_size`. Keep the check under both tokenizers, since it is cheap and guards a model swap; comparing both tokenizations over the whole corpus would confirm the equality.
  - Input (2026-10-12, T07a): over the metadata of snapshot `20261002T224412Z`, the two tokenizers give the same token count, special tokens included, for each of the 7,097 distinct texts that the metadata profile tokenizes: titles, abstracts, abstract sentences and the empty text. Counts were compared, not token ids. The PDF objectives were not compared.
- [ ] **T14** Embedding pipeline: section means, an equal-weight document vector, L2 normalization, cached chunk vectors, and a plain-mean sensitivity flag.
  - Route D. Risk M.
- [ ] **T15** TF-IDF baseline representations, used for both clustering and redundancy.
  - Route D. Risk M.
  - Input: the 89-lemma domain stopword list of D26. Since the list is frozen before T18, record its SHA-256 with the outputs.
- [ ] **T16** Topic map builder (must pass `language="spanish"` to BERTopic, see D22):
  - UMAP + K-means/HDBSCAN + BERTopic, using precomputed embeddings and the full-clean documents;
  - outputs topics, assignments, centroids and 2D coordinates;
  - applies the noise policy.
  - Route D. Risk M.
- [ ] **T17** Metrics module (silhouette cosine, DB/CH on L2, NPMI, noise %, ARI), tested on synthetic data with known structure.
  - Route D. Risk M.
- [ ] **T18** Declare the experiment grid and the winner-selection rule (O03) BEFORE any run.
  - Route I. Risk P.
- [ ] **T19** Run the experiment: 2 models × 2 algorithms, plus the TF-IDF baseline, × 5 seeds, plus the grids. Produce the results table and select the winner by the T18 rule. **Needs U2.**
  - Route D. Risk M.

### P4 — Labeled set and redundancy (M3–M5)
- [ ] **T20** Pair sampler, plus the Excel workbook generator and importer with validation (O02).
  - Route D. Risk M.
- [ ] **T21** Labeling round (U3): 2 annotators per pair, then κ, then consensus labels. Owner: the user (with a developer import).
  - Route I. Risk P.
- [ ] **T22** Redundancy module: cosine search, τ from the PR curve via 5-fold CV, a bootstrap CI, and the SBERT vs TF-IDF comparison.
  - Route D. Risk M.
- [ ] **T23** Proposal handling: nearest-centroid topic assignment, the atypical flag (O06), and saturation quartiles.
  - Route D. Risk M.

### P5 — Gaps and recommendation (M5)
- [ ] **T24** Gap rules a–d plus the stability filter. **Needs U4.**
  - Route D. Risk M.
- [ ] **T25** OpenAlex Topics reference:
  - download the list once (**needs U8**);
  - map subfields to programs (**needs U5**);
  - embed and assign topics, then rank untouched topics adjacent to existing ones.
  - Also build the official-lines coverage view (U1).
  - Route D. Risk M.
- [ ] **T26** Recommender: antecedents (top-k with MMR) and alternative lines (neighbor topics plus nearby gaps).
  - Route D. Risk M.
- [ ] **T27** Temporal-split evaluation: build the map on 2021–2024, assign the 2025–2026 theses, compute consistency metrics, and prepare an expert sample (O04).
  - Route D. Risk M.

### P6 — Prototype (M5–M6)
- [ ] **T28** Streamlit app with the map, query and gaps views. It loads saved artifacts, persists no queries, links to the repository, and shows no author names. UI language per O01. Disable Streamlit usage statistics (`browser.gatherUsageStats = false`).
  - Route D. Risk M.
  - Checks: an AppTest smoke test.
- [ ] **T29** Dockerfile (python:3.13-slim + uv, CPU torch, data mounted as a volume) and a run guide (O07).
  - Route D. Risk M.

### P7 — Validation (M6–M7)
- [ ] **T30** V de Aiken computation and finalization of the questionnaire with the judges (U6).
  - Route I. Risk P.
- [ ] **T31** Expert session kit: anonymized result samples, a consent template, and a session script.
  - Route D. Risk P.
- [ ] **T32** Questionnaire analysis (medians, frequencies, Cronbach α) and the check against success criteria a–d.
  - Route D. Risk M.

### P8 — Closure (M7–M8)
- [ ] **T33** Reproducibility package: data card, environment, configs, seeds, results index, and a how-to-run guide.
  - Route D. Risk P.
- [ ] **T34** List of updates to the thesis document, for the user and the advisor:
  - plan sections: Resumen/Abstract, 1.7.2, 3.1, 3.3.2–3.3.3, 3.5.2–3.5.3, 3.6.1, 3.6.6, 3.7, 3.10, Anexo B;
  - 512 → 128 tokens;
  - broken tables, the index, the Chapter II vs III inconsistency, and removing Colab from the budget.
  - Route I. Risk P.

## Feature acceptance criteria (plan §3.6.8, adapted)

1. The winning semantic configuration beats the TF-IDF baseline on NPMI coherence and on labeled-set F1. Silhouette is reported within each space and is not compared across spaces.
2. Direct-redundancy F1 is ≥ 0.70 at the selected τ, cross-validated and reported with a bootstrap CI.
3. Each of the 4 questionnaire dimensions has a median ≥ 4.
4. The winning configuration is stable across the 5 seeds (low SD, with ARI reported).
5. Engineering: lint, format and tests all pass. The app runs from a clean clone plus the snapshot, and the Docker image runs with mounted data.

If a criterion is not met, that is reported as a finding. It is never hidden.

## Checks

- `uv run ruff check .`
- `uv run ruff format --check .`
- `uv run pytest -q`
- Test-first: deterministic modules follow RED → GREEN → REFACTOR with `uv run pytest`. Exceptions: EDA notebooks, experiment runs, manual labeling, and the Docker run check.
- Processing costs: every long or measured operation adds a row to `docs/processing_costs.md` in the same work unit.

## Risks

| ID | Risk | Mitigation |
|---|---|---|
| R1 | Objectives extraction falls below the target | Fall back to title + abstract and document it (O05) |
| R2 | Fresh library majors break BERTopic | Smoke test in T02; pin the previous major |
| R3 | Low annotator agreement | A guide with examples, a 10-pair pilot, and re-labeling of disputed pairs |
| R4 | F1 < 0.70, or another criterion unmet | Report it honestly as a result |
| R5 | Industrial dominates the topics (46% of the corpus) | Report per program and normalize gap rules per program |
| R6 | Annotators or experts are unavailable | Request them now (U3, U6) |
| R7 | Cross-lingual matching against OpenAlex is weak | Use nearest-topic assignment and manually check a sample |
| R8 | The snapshot is lost | Accepted by the user, with no backup (D23). The manifests' uuids and sha256 allow a documented re-harvest |

## Delivery strategy

- **Forecast:** about 8,000–9,000 authored changed lines (source plus tests), which exceeds the ~400-line slice budget.
- **Strategy:** direct work-unit commits on `main`, with no PR chain (O08, user decision). Slices are commits, each one coherent on its own.
- **Commits:** Conventional Commits only, with no trailers. The user writes the Summary and Description in Spanish, and the type keywords stay in English (user decision, 2026-10-02). Pushing is the user's decision.
- **Slice boundaries (commits per task):** recorded in each task's evidence.

## Progress log

- 2026-10-02: Plan drafted. It draws on:
  - an exhaustive analysis of plan v3;
  - stack verification on PyPI (2026-10-01/02);
  - an OCR test;
  - repository research (745 theses, metadata keys, DNI caveat);
  - a review of the existing local repositories.
  - Status: awaiting user approval.
- 2026-10-02: The user approved the plan and O01, O03–O07. The user confirmed this repo as the official one and asked to ignore `TesisSimilitud`. O02 was explained again and proposed as a calibration meeting followed by Excel labeling at home.
- 2026-10-02: The user accepted O02. T01 started:
  - uv 0.12.22 was installed via winget. It is on the user PATH but needs a terminal restart; this session calls it by absolute path.
  - Route: D. Trigger: 3+ non-trivial files (pyproject, config module, tests).
- 2026-10-02: The user chose O08: direct commits on `main`, with no PR chain. This makes the slice plan for T01 three commits: the plan doc, the scaffold, and the config unit.

- 2026-10-02: T01 closed. It is three commits on `main`, with the config unit reviewed and approved and T01a added from the review findings.

- 2026-10-02: T01a implemented and verified (63 tests). Its commit was handed to the user.
- 2026-10-02: The user committed T01a as `f39343b`. Its review was approved before and after the commit. T01b was created from the findings, and commit messages now use Spanish.

- 2026-10-02: T01b implemented and verified (83 passed, 1 skipped). Its commit was handed to the user.
- 2026-10-02: The user committed T01b as `2575007`. T02 started. Route: D. Trigger: new smoke suite plus pytest config.
- 2026-10-02: T02 implemented and verified (14 smoke tests, no pins). D22 was added, and the commit was handed to the user.
- 2026-10-02: The user committed T02 as `fdd81da` and declined the post-commit review. T02a was added. T03 started.
  - Route: D. Trigger: new OCR assets module plus unit and smoke tests.
  - Downloads: two public tessdata files from GitHub (`tesseract-ocr/tessdata_best`), no credentials.

- 2026-10-02: T03 implemented and verified (116 unit, 15 smoke). Its commit was handed to the user.
- 2026-10-02: The user committed T03 as `3f9888f`. The post-commit review was approved, so the reviewed boundary is now `3f9888f`.
  - The user prioritized Phase 1, so T02a is deferred.
  - The repository count for 2021–2026 (all types) moved from 765 to 766 during the day, and future `dc.date.issued` values were observed. Both are recorded under T06.

- 2026-10-02: U8 was granted for T04 only. The scope is read-only GET requests to the public DSpace REST API of `repositorio.ucsm.edu.pe`, with no credentials, to download the metadata (no PDFs) of the 5 program collections for 2021–2026.
  - Rate: about 1 request/s.
  - DNI fields are dropped at ingestion, and the output stays in gitignored `data/raw/`.
  - T05 (PDF download) needs a separate authorization.
  - T04 started. Route: D. Trigger: new corpus package plus config and tests.

- 2026-10-02: T04 implemented and verified, and the real snapshot `20261002T224412Z` was taken (766 records, 746 theses, 0 DNI). Three commits (A, B, C) were handed to the user.

- 2026-10-02: The user committed T04 as the single commit `ad160a5`, and the post-commit review was approved, so the reviewed boundary is now `ad160a5`. T04a was added. T05 was proposed, awaiting a new U8 authorization:
  - only the thesis PDF of each of the 746 theses;
  - about 3 requests per thesis at 1/s;
  - an estimated 3–8 GB, measured first on a sample of 10 (183 GB free);
  - restricted items skipped with a reason.

- 2026-10-02: U8 was granted for T05. The scope:
  - read-only GET requests to `repositorio.ucsm.edu.pe`, with no credentials;
  - only the thesis PDF of each of the 746 `#tesis` items in snapshot `20261002T224412Z`;
  - no similarity report or authorization form;
  - at least 1 s between requests, sequential;
  - restricted or embargoed items skipped with a recorded reason;
  - files go to gitignored `data/raw/<snapshot>/pdfs/` with checksums;
  - research use only, no redistribution (CC BY-NC-ND).

  T05 started. Route: D. Trigger: adapter extension, new use case and CLI, plus tests. A writer implements the downloader and runs a 10-file sample; the parent runs the full download in the background.

- 2026-10-02: T05 downloader implemented and verified on a 10-PDF sample. The full download is running in the background, and the commits were handed to the user.

- 2026-10-03: The user committed T05 as `54173b8`. Its tree `dd756e9…` is identical to the reviewed one. The post-commit review was approved, so the reviewed boundary is now `54173b8`. T05a was added from the review findings.

- 2026-10-03: The user dropped the snapshot backup (D23, U7 removed). T07 was started ahead of T06, because the EDA needs only the metadata while the PDF download runs. T06 follows when the download ends, since the data card needs the PDF counts. T07 route: D. Trigger: new profiling module, tests and an experiment script.

- 2026-10-03: T07 implemented and verified. D24 (deduplication) was added. Notes for T08, T11, T12 and T13 are recorded under T07. The commit was handed to the user. The T05 download was at 434 of 746 (6.57 GB), with 1 `no_thesis_file`.

- 2026-10-03: The user committed T07 as `96c410c`. The post-commit review was approved, so the reviewed boundary is now `96c410c`. T07a was added. The plan's stale next step was fixed, and the decision rows were reordered.
  - The background T05 download stopped at the tool's maximum time limit, at 470 of 746 (7.22 GB). One `.part` was left over; the downloader overwrites it when it retries that item.
  - The session must not relaunch a job that hit the maximum limit, so the user resumes it from a terminal with the same CLI. It is resumable.

- 2026-10-03: T08 implemented and verified, including the `author_codes` change from measured data. Its commit was handed to the user.

- 2026-10-03: The user committed T08 as `f21743f`. The post-commit review was approved, so the reviewed boundary is now `f21743f`. T08a was added.
  - The user resumed the T05 download from a terminal, and it is running. The `.part` item `2cf456c1…` was re-downloaded and verified, which confirms safe resume after a hard kill.
  - T11 started ahead of T09/T10, because it needs only metadata texts. Route: D. Trigger: new preprocessing modules, tests and a stopword-candidates script.

- 2026-10-03: Session handoff to a new working session.
  - Progress is about 22% of the 34 planned tasks. Phase 0 is done, Phase 1 is about 75% and Phase 2 is about 30%.
  - The last reviewed boundary is `f21743f`.
  - T11 is implemented but not yet committed. The T05 download is still running in the user's terminal.

- 2026-10-03: The user committed T11 as `e9f4f70`. The post-commit review was approved, so the reviewed boundary is now `e9f4f70`. T11a was added from the review findings and the user's observations on the stopword list.

- 2026-10-03: The T05 download finished at 12:18:20Z and was verified independently.
  - All 746 `#tesis` items are covered: 731 PDFs (11.40 GB), 14 restricted, 1 without a file, 0 errors.
  - Findings, recorded under T05: item 11777 carries the wrong file; the industrial D24 pair has two different PDFs; 728 of the 744 distinct theses have a valid own PDF; 126 PDFs are mixed text/image and 1 is scanned, so T09 should OCR selectively.

- 2026-10-03: T11a started. Route: D. Trigger: new tests, script fixes, the stopword list and the plan.

- 2026-10-03: The user committed T11a as `633dfd2`. The post-commit review was approved, so the reviewed boundary is now `633dfd2`. T11b was added from the review findings.

- 2026-10-03: T06 started. Route: D (the plan said I). Trigger: a new data card computed from the raw data, plus plan and README edits.

- 2026-10-03: T09 started. Route: D. Trigger: new extraction modules, CLI, config and tests. D25 (selective OCR window) was added.

- 2026-10-05: T09 implemented and verified on a 10-PDF sample. The session ended mid-sample, and the next run resumed it. Its commits are handed to the user, and the full run is pending in the user's terminal.

- 2026-10-05: The user committed T09 as `3b05679`. The post-commit review was declined for this candidate, so the reviewed boundary is now `3b05679`.
  - T09a started. Route: D. Trigger: a rule fix plus manifest and fingerprint changes, with tests, across 3 or more files.
  - The full run waits for T09a, because the fingerprint covers code only through a rules version raised by hand, so a later fix would not extract the PDFs again.

- 2026-10-05: The user committed T09a as `7334e66`. The post-commit check found it under the slice budget, so the reviewed boundary stays `3b05679`.
  - The user started the full T09 extraction in their terminal.
  - The processing-cost ledger `docs/processing_costs.md` was added at the user's request, for the thesis write-up.

- 2026-10-05: The user committed the ledger as `3c19358`; its tree `080f3f9…` is identical to the reviewed one. The full T09 extraction finished in 1 h 29 min, with 731 of 731 PDFs `ok` and 0 OCR failures. It was verified independently, and T09 is closed. Phase 2 continues with T10.

- 2026-10-07: T10 started. Route: D. Trigger: a new locator, a CLI, workbook input and output, config and tests.

- 2026-10-07: T10 implemented and verified. Two tuning rounds on all 731 page files took not_found from 4 to 0. The 60-row verification workbook is written; the manual verification by the user is pending, so T10 stays open. The commits are handed to the user.

- 2026-10-07: The review budget refused the whole T10 candidate at 4,182 lines (a 3,198-line candidate had passed), so T10 ships as two reviewed commits.
  - The user committed slice 1 as `4e74a7a`. Its tree `e6e9a52…` is identical to the reviewed one. It was approved before and after the commit, so the reviewed boundary is now `4e74a7a`.
  - Both reviews raised the same two non-blocking warnings: the untested record validator and the workbook containment. Both were fixed test first in slice 2, as recorded under T10, and the strict accuracy rule was confirmed against the code.
  - Slice 2 is handed to the user. T10 stays open until the manual verification.

- 2026-10-07: The user committed T10 slice 2 as `6f1ba96`. It was approved before and after the commit, so the reviewed boundary is now `6f1ba96`.
  - The user filled the verification workbook in about 2 h (reported, not timed). The official import gave 60/60 general objectives `Correcto` (Wilson 95% CI 94.0%–100%), so O05 is met and T10 is closed.
  - T10b was added from the slice-2 reviews (two warnings and one suggestion) and implemented test first. It awaits its commit together with the T10 closure.

- 2026-10-07: The user committed the T10 closure and T10b as `c121d78`. It was approved before and after the commit, so the reviewed boundary is now `c121d78`. T10c was added from the T10b review findings.

- 2026-10-07: T12 started. Route: D. Trigger: a builder, a quality report, config and tests across 6 or more files.
  - T12 implemented and verified: 766 fichas, 744 included theses, every enforced check passed on the real snapshot. The D24 rule agrees with the T10 placeholder in both pairs. The stopword re-check proposes candidates only; the list is unchanged.
  - Phase 2 is done. The T12 commits are handed to the user as two slices.

- 2026-10-07: The user committed T12 as the single commit `73f1676`. Its tree `760a5e9…` is identical to the reviewed one. It was approved before and after the commit, so the reviewed boundary is now `73f1676`, and Phase 2 is closed. The four review findings became T12a.

- 2026-10-07: U8 was granted for the mpnet model. The scope: an anonymous, read-only download of `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` from the Hugging Face Hub into the user's model cache, with no credentials.
  - The download and the first load took 287.6 s together, from 2026-10-08T04:08:21Z. The cache's last write, 04:13:09Z, agrees.
  - Checked again offline on 2026-10-11: dimension 768, `max_seq_length` 128, a 250,002-token tokenizer, 11 files and 1,126,358,977 bytes on disk, at revision `4328cf26390c`. The model loads from the cache in 3.9 s.
  - The costs are in `docs/processing_costs.md`.

- 2026-10-09: The user accepted D26: ten research-action verbs join the domain stopwords, and four outcome verbs wait for the T16 review.

- 2026-10-11: Closure unit. D26 applied to `config/stopwords_domain_es.txt`, test first; U8 and the mpnet download recorded here and in the ledger; T12a added.
  - Route I. Trigger check: one non-trivial file (the stopword test), plus mechanical edits to the list, this plan and the ledger; the parent already held the whole plan.
  - RED: the new test `test_the_curated_list_holds_the_research_action_verbs_of_the_objectives` failed on the 79-lemma list (1 failed, 6 passed). GREEN after the edit: 7 passed, and the list loads 89 lemmas. Two throwaway mutants were both caught and reverted, the list byte-identical afterwards: `medir` dropped, and `optimizar` added.
  - `mejorar` moved from the test's frequent words left out to the deferred outcome verbs, with optimizar, reducir and incrementar. Both sets are tested to stay off the list.
  - Measured on the way: the MiniLM and mpnet tokenizers share one vocabulary (noted under D09 and T13).
  - Checks: `pytest` 1,196 passed, 1 skipped (1,194 + 2 new); `ruff check` and `ruff format --check` clean.

- 2026-10-11: The user committed the closure unit as `b6943fa`. Its tree `810f8a5…` is identical to the reviewed one. It was approved before the commit; the post-commit check found it under the slice budget (medium risk, 115 lines), so the reviewed boundary stays `73f1676`.

- 2026-10-11: T12a started and implemented on top of `b6943fa`, the closure unit. Route: D. Trigger: the builder, the quality report and their two test modules, plus this plan, the ledger and the README. The closure unit's review finding R3-001 joined T11b.
  - The fichas outputs are now published as one unit, with the manifest as the commit point, and later steps read the fichas through the verified reader `load_fichas`.
  - The real re-run wrote the same fichas and exclusions tables (same SHA-256). The committed quality report is unaffected by the advisor fix.
  - Checks: `pytest` 1,215 passed, 1 skipped; `ruff check` and `ruff format --check` clean. The commit is handed to the user.

- 2026-10-11: The user committed T12a as `ad2c6f0`. Its tree `11e3857…` is identical to the reviewed one. The post-commit check from `73f1676` (581 lines, medium risk) found `slice_budget_reached`, so a post-commit review is due; its consent waits for the user. The reviewed boundary stays `73f1676`.

- 2026-10-12: The user approved preparing four follow-up units overnight, as a stack of verified, uncommitted units, each committed by the user after its review: T08a; T09b; T12b; then T11b with T07a, T05a, T04a and T02a.

- 2026-10-12: T08a implemented and verified on top of `ad2c6f0`, as unit 1 of the overnight stack. Route: D. Trigger: two source modules and their two test modules, plus this plan, the ledger and the README.
  - The pseudonym key no longer needs hard links: exclusive creation is the fallback, a short key file gets 2 s to fill before it is refused, and the run that creates the key fsyncs its directory where the system allows it.
  - The storage schema check now covers nullability and the type and nullability of list items. Every real table on disk still passes it: 766 fichas, 22 exclusions, 731 objectives rows and 731 page files, 0 refused. The real key is unchanged (same SHA-256).
  - Two independent adversarial reviews raised one warning, that a complete key could be removed while in use, and four smaller findings. One scoped correction fixed them all, test first, as recorded under T08a.
  - Checks: `pytest` 1,251 passed, 3 skipped; `ruff check` and `ruff format --check` clean. The unit is uncommitted, for the user's review and commit.

- 2026-10-12: T09b item 1 implemented and verified on top of unit 1, as unit 2 of the overnight stack. Route: D. Trigger: a new shared module and its tests, 8 source modules and 5 test modules.
  - The run lock and the atomic text write now live once, in `shared/filesystem.py`, and every caller uses them; the copies and the imports of the writer from `extraction/` into `corpus/` and `labeling/` are gone. The writer now also flushes its directory, skipped where the system refuses, as on this machine.
  - A rerun of the extraction for one `ok` item skipped it and changed nothing: the manifest and all 731 page files are byte-identical.
  - Item 2, edge lines of digits only, is now pinned by 9 characterization tests and stays open for the user: changing it means raising `EXTRACTION_VERSION` and extracting all 731 PDFs again.
  - An independent adversarial review confirmed the refactor behavior-identical and raised four findings: the exact rule for a year with punctuation, a cleanup error that could hide the original one, and two wording fixes. One scoped correction fixed them, as recorded under T09b.
  - Checks: `pytest` 1,290 passed, 3 skipped; `ruff check` and `ruff format --check` clean. The unit is uncommitted, for the user's review and commit.

- 2026-10-12: T12b implemented and verified on top of unit 2, as unit 3 of the overnight stack. Route: D. Trigger: the builder and the storage port with their two test modules.
  - An interruption after the tables have begun to move now says that the outputs may not match the manifest and that the command must run again; before the moves, the message is unchanged. The build signals the window with `PublishInterrupted`, which stays a `KeyboardInterrupt`.
  - A damaged `fichas_manifest.json` is refused in one line that names the file and the build command, without quoting its content.
  - `load_fichas` reads the fichas table once and parses the bytes it checked, through the new `read_verified` of the table-store port.
  - The real outputs load as before (766 fichas, 744 included, 0.385–0.498 s), and their three files kept their SHA-256.
  - An independent adversarial review raised three findings: keys of a damaged manifest still reached its message, a second Ctrl+C during the cleanup brought back the old message, and the first byte-level read did not fit the table-store port. One scoped correction fixed them, as recorded under T12b.
  - Checks: `pytest` 1,319 passed, 3 skipped; `ruff check` and `ruff format --check` clean. The unit is uncommitted, for the user's review and commit.

- 2026-10-12: T11b and T07a implemented and verified on top of unit 3, as unit 4 of the overnight stack. Route: D. Trigger: the two experiment scripts, the shared writer, `preprocessing/title_suffix.py`, and four test modules (one new).
  - `shared/filesystem.py` now also writes bytes, and several files of one folder staged together before any moves into place. Both experiment scripts use it: the stopword candidates no longer fail after publishing when a directory cannot be flushed, and the metadata profile stages all its outputs before it renames any.
  - The D26 list is pinned by a test (89 lemmas and their SHA-256), and the throughput test no longer depends on how often the script reads the clock.
  - The metadata profile now derives its second-tokenizer note at runtime, guards a snapshot without theses or without readable dates, ties the advisor note to its constant, and has tests for its privacy gate. Rendered in memory from the real snapshot, the 4 figures are byte-identical to the committed ones, and the two tokenizers count all 7,097 distinct texts the same.
  - An independent adversarial review raised three findings: the docs claimed that any failed write leaves the previous outputs, while a failed rename leaves a mix; names that collide through `.part` or letter case were not refused; and a partly downloaded tokenizer cache could produce a false D09 note. One scoped correction fixed them, test first, as recorded under T07a. It also names the year-suffix patterns in `preprocessing/title_suffix.py`, which closes T07a.
  - Checks: `pytest` 1,376 passed, 3 skipped; `ruff check` and `ruff format --check` clean. The unit is uncommitted, for the user's review and commit.

## Next step

Resume checklist, in order:

1. **Re-sync.**
   - Read this document fully.
   - Run `git status`, `git log -5`, `uv run pytest -q`, `uv run ruff check .` and `uv run ruff format --check .`.
   - Confirm that HEAD is `ad2c6f0` (T12a, tree `11e3857…`) or a later commit of the overnight stack. The last reviewed boundary is `73f1676`: the post-commit review of `b6943fa` and `ad2c6f0` together (581 lines, `slice_budget_reached`) is due and waits for the user's consent. Once it is acknowledged, or declined as for T09, the boundary moves to `ad2c6f0`.
2. **Follow-ups: the overnight stack (2026-10-12).** Each unit is verified and left uncommitted, for the user to review and commit in order:
   - Unit 1, T08a: prepared (key creation without hard links, the key directory fsync, and the stricter storage schema check).
   - Unit 2, T09b: prepared (the run lock and the atomic text write moved into `shared/filesystem.py`, and characterization tests for edge lines of digits only). T09b stays open for the user's decision on those lines: options (a) to (c) under T09b; (b) and (c) mean extracting all 731 PDFs again.
   - Unit 3, T12b: prepared, with its review fixes (an accurate message for an interruption while the fichas tables move, a one-line refusal of a damaged fichas manifest, and `load_fichas` parsing the bytes it checked through the port's `read_verified`).
   - Unit 4, T11b and T07a: prepared, with its review fixes (the D26 pin, the tolerant directory flush of the stopword candidates, a clock-independent throughput test; the metadata profile's staged outputs with an honest message for a failed rename, refused colliding names, a runtime tokenizer note that compares counts only for equal vocabulary sizes, empty-snapshot guards, privacy-gate tests, and the year-suffix patterns named in the source). Once the unit is committed, run `experiments/eda_metadata.py` again so that `results/eda/20261002T224412Z/` records the tokenizer comparison (the figures stay byte-identical; see T07a).
   - Unit 5, T05a, T04a and T02a: next.
   - T10c waits for the user's redactor policy, which is needed before any new workbook, so before T20.
3. **Phase 3.** T13, the chunker (both models are cached), reading the fichas through `load_fichas`; T14; T15, with the 89-lemma stopword list of D26; then T16 to T19.

Pending user inputs: U1; U2 (before T19); U3 (month 3); U4, which must be agreed before any topic result exists, so before T16; U5; U6; U8 for any new download (the OpenAlex topics for T25 and, optionally, the 7 expired-embargo PDFs); the T10c redactor policy (before T20).
