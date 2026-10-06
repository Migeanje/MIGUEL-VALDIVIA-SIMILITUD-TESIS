# Feature: Thematic Redundancy System (MVP)

- **Locator:** `odd/tasks/thematic-redundancy-system.md`
- **Status (2026-10-03):** Plan approved by the user on 2026-10-02, including O02 and O08 (direct commits on `main`). Phase 0 is done. T04, T05, T07, T08, T11 and T11a are done. T06 is done and awaits the user's commit. Next: T09.
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
| D09 | Chunking: per section, sentence-aligned, ≤128 tokens under BOTH tokenizers. Section vector = mean of its chunks. Document vector = equal-weight mean of the sections, L2-normalized | Both models have `max_seq_length` 128. Vocab sizes differ (250037 vs 250002) |
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
| D24 | Deduplicate exact duplicate records before anything else | T07 found 2 pairs of `#tesis` records with identical title, abstract, author, advisor and date under different item uuids, so there are 744 distinct theses, not 746. T12 keeps one canonical record per pair (rule fixed in T12) and logs the other as `duplicate_of`. Duplicates never enter the labeled pairs (T20) or the redundancy evaluation, where they would be trivial positives |
| D25 | OCR is selective: only low-text pages among the first `extraction.ocr_window_pages` pages (default 100) of a PDF are OCR'd. Other low-text pages keep their few text-layer characters, or are `empty`, with their low-text flag | The PDF text serves only the objectives (T10); title and abstract come from metadata (D03, D09). Measured 2026-10-03: the first text-layer page with "objetivo general" was found in 658 of 731 PDFs, at median page index 7, P95 25, P99 102. The window cuts OCR from about 8,600 low-text pages to about 1,960. It is configurable, so a wider pass can run later |

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
| U8 | Authorization for the remote operations: UCSM harvesting (T04/T05) and the OpenAlex topics download (T25) | At each task start |

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
    - **Models.** The mpnet model is not cached; only MiniLM is (about 480 MB). Downloading it from Hugging Face before T13 needs the user's authorization.
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
- [ ] **T07a** EDA script follow-ups from the T07 reviews (non-blocking):
  - Write the outputs atomically (stage, then publish) so a failed write leaves no half-written results.
  - Guard against empty programs or years in `observations()`.
  - Derive the mpnet-tokenizer note at runtime instead of hardcoding it.
  - Tie the "top advisors" text to its constant.
  - Name the year-suffix patterns explicitly.
  - Add tests for the privacy write gate.
  - Document that text drawn inside the PNG figures is outside the gate's string search.
  - Route D. Risk M.

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
- [ ] **T08a** Ficha/storage follow-ups from the T08 reviews (non-blocking):
  - Key creation must not depend on hard-link support. `os.link` fails on FAT/exFAT, some SMB shares and some container bind mounts, which matters for T29.
  - fsync the key's parent directory after publishing it.
  - The storage schema check should cover nullability and list item types, with tests for wrong list item types and for non-list columns.
  - Route D. Risk M.
- [ ] **T09** PDF text extraction with text-layer detection and an OCR fallback. Note from T05: 6 of 13 inspected items also carry DSpace's extracted full text (`*.pdf.txt` in a TEXT bundle). It could serve as a cross-check, but it was not fetched and is not authorized yet.
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
  - Full run: pending in the user's terminal (command in the README). T09 stays unchecked until it is done and verified.
  - Commit: pending; the user commits in GitHub Desktop.
- [ ] **T10** Objectives locator plus a manual verification sample (~60, stratified by program), producing an accuracy report (O05).
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
    - `stopwords.py` loads `config/stopwords_domain_es.txt`: 79 curated lemmas covering genre, reporting verbs, institution, place and time, and generic words. It shipped with 84; T11a removed 5.
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
- [ ] **T11b** Follow-ups from the T11a reviews (non-blocking):
  - `_fsync_directory` in `experiments/domain_stopwords.py` tolerates only a failure to open the directory. An `OSError` from `os.fsync` on an opened directory (EINVAL on some network, FUSE or overlay filesystems) propagates after `os.replace` has already published the output, which contradicts its docstring. Tolerate it, and add a test. Both reviews raised it.
  - The documents-per-second test couples to the number of `perf_counter` readings in `main`. Assert only the format, or control the two readings that bracket the cleaner.
  - Route D. Risk M.
- [ ] **T12** Build the fichas dataset, the exclusion log, and a quality report. Inclusion requires an exact `renati.type` fragment `#tesis` (746 in snapshot `20261002T224412Z`). Suficiencia profesional appears in 5 spellings and is excluded with its reason.
  - Route D. Risk M.
  - Inputs from the T05 findings (2026-10-03): 11777 is metadata-only with a quality note, and the D24 canonical rule must also choose the industrial pair's PDF.

### P3 — Representation and topic map (M3–M5)
- [ ] **T13** Sentence-aligned chunker that is valid for both tokenizers. Property tests check that every chunk fits both tokenizers and no text is lost.
  - Route D. Risk M.
- [ ] **T14** Embedding pipeline: section means, an equal-weight document vector, L2 normalization, cached chunk vectors, and a plain-mean sensitivity flag.
  - Route D. Risk M.
- [ ] **T15** TF-IDF baseline representations, used for both clustering and redundancy.
  - Route D. Risk M.
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

## Next step

Resume checklist, in order:

1. **Re-sync.**
   - Read this document fully.
   - Run `git status`, `git log -5`, `uv run pytest -q`, `uv run ruff check .` and `uv run ruff format --check .`.
   - Confirm whether HEAD contains the T09 commits.
2. **T09 commit handoff.** Hand the user the three slices A, B and C listed under T09, with their file lists and Conventional Commit messages; the user commits in GitHub Desktop.
3. **T09 full run, in the user's terminal.**
   - `uv run python -m thematic_redundancy.extraction.extract_text --summary-out results/extraction/20261002T224412Z/summary.json` (about 3 h; it resumes if stopped).
   - Then verify the manifest (731 PDFs, 0 errors, 0 stale), commit the numbers-only summary, and check T09.
4. **T10.** Objectives locator. Manual check of about 60 theses stratified by program; target at least 90% correct (O05).
5. **T12.** Build the fichas dataset:
   - D24 dedupe, including the industrial pair's PDF choice;
   - 11777 as metadata-only;
   - advisor codes keyed by ORCID;
   - `author_codes`;
   - the exclusion log;
   - `doc_code` uniqueness;
   - re-check the domain stopwords against the objectives text.

Batched follow-ups, to be scheduled once Phase 2 is done: T02a, T04a, T05a, T07a, T08a, T11b.

Pending user inputs: U1; U2 (before T19); U3 (month 3); U4, which must be agreed before any topic result exists, so before T16; U5; U6; U8 for any new download, including the mpnet model before T13 and, optionally, the 7 expired-embargo PDFs.
