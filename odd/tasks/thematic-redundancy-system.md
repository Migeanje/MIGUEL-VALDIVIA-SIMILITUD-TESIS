# Feature: Thematic Redundancy System (MVP)

- **Locator:** `odd/tasks/thematic-redundancy-system.md`
- **Status:** Plan approved by the user on 2026-10-02, including O02 and O08 (direct commits on `main`). T01 is implemented and verified, and is being committed.
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
| O08 | PR chain strategy | Commit work units directly on `main`, with no PR chain. Each task closes with one or more Conventional Commits | **Accepted 2026-10-02 (user decision)** | First commit |

## Inputs the user must secure (lead time)

| ID | Input | Needed by |
|---|---|---|
| U1 | Confirm the official research lines per school in a browser (the site blocks automated fetches) | T25 (M5) |
| U2 | Advisor approval of the population change: theses only, census, ≈745 docs, plus the plan-section updates | Before T19 (M3) |
| U3 | 3–4 advisors as annotators (≈80 pairs and 2–3 h each) | T21 (M3) |
| U4 | A career-affinity matrix agreed with advisors BEFORE seeing any results | Before T24 |
| U5 | Review of the OpenAlex subfield mapping per program | Before T25 |
| U6 | Expert panel (6–8 people), V de Aiken judges, consent forms | T30 (M6) |
| U7 | A backup location for the raw snapshot (external drive or personal cloud; the data is public) | T06 (M2) |
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
| ocr | `spa+eng`, 300 dpi, `full=True` on pages without a text layer |
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
- [ ] **T01** Scaffold the repo. Contents:
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
- [ ] **T02** Smoke-test the full stack on a synthetic mini corpus. The run covers embeddings → UMAP → K-means/HDBSCAN → BERTopic → NPMI → an AppTest import. Pin any major version that breaks.
  - Route D. Risk M.
  - Accept: the run completes end to end and the pins are documented.
- [ ] **T03** Script that fetches tessdata, plus an OCR fixture test that reads the tessdata path only from config.
  - Route D. Risk M.

### P1 — Corpus acquisition (M1–M2)
- [ ] **T04** Metadata harvester over DSpace REST for the 5 collections, with filters, DNI fields dropped, and a manifest. **Needs U8.**
  - Route D. Risk H.
  - Tests use recorded fixtures (no network).
  - Accept: counts reconcile with the repository totals at the snapshot date, and no DNI appears in any output.
- [ ] **T05** Thesis PDF downloader. It picks the thesis bitstream by name (excluding `*.RT.pdf` and `Autorización_*`), rate-limits, resumes, and records sha256. Embargoed/restricted items are skipped with a reason. **Needs U8.**
  - Route D. Risk H.
  - Accept: every included item has exactly one thesis PDF or a recorded reason.
- [ ] **T06** Freeze the snapshot, back it up (U7), and write the data card (counts by program and year, embargo list).
  - Route I. Risk P.
- [ ] **T07** EDA notebook: program/year distribution, abstract token lengths, keywords, title suffix patterns.
  - Route D. Risk P.

### P2 — Data preparation (M2–M3)
- [ ] **T08** Ficha schema (pydantic) and its parquet adapter. It holds the Anexo B fields plus: handle URL, snapshot id, keywords, OCDE code, objectives status, section source, include/exclude with reason, file hash.
  - Route D. Risk M.
- [ ] **T09** PDF text extraction with text-layer detection and an OCR fallback.
  - Route D. Risk M.
- [ ] **T10** Objectives locator plus a manual verification sample (~60, stratified by program), producing an accuracy report (O05).
  - Route D. Risk M.
- [ ] **T11** Light and full cleaners. They handle:
  - title suffix stripping, header/footer removal and hyphenation;
  - domain stopwords (by document frequency, then manual review);
  - legal suffixes;
  - spaCy lemmatization (full cleaner only).
  - Accents and ñ are kept.
  - Route D. Risk M.
- [ ] **T12** Build the fichas dataset, the exclusion log, and a quality report.
  - Route D. Risk M.

### P3 — Representation and topic map (M3–M5)
- [ ] **T13** Sentence-aligned chunker that is valid for both tokenizers. Property tests check that every chunk fits both tokenizers and no text is lost.
  - Route D. Risk M.
- [ ] **T14** Embedding pipeline: section means, an equal-weight document vector, L2 normalization, cached chunk vectors, and a plain-mean sensitivity flag.
  - Route D. Risk M.
- [ ] **T15** TF-IDF baseline representations, used for both clustering and redundancy.
  - Route D. Risk M.
- [ ] **T16** Topic map builder:
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
- [ ] **T28** Streamlit app with the map, query and gaps views. It loads saved artifacts, persists no queries, links to the repository, and shows no author names. UI language per O01.
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
| R8 | The snapshot is lost | Back it up (U7) |

## Delivery strategy

- **Forecast:** about 8,000–9,000 authored changed lines (source plus tests), which exceeds the ~400-line slice budget.
- **Strategy:** direct work-unit commits on `main`, with no PR chain (O08, user decision). Slices are commits, each one coherent on its own.
- **Commits:** Conventional Commits only, with no trailers. Pushing is the user's decision.
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

## Next step

Commit T01 on `main`: plan doc, then scaffold, then config unit. After that, start T02 (stack smoke test) and T03 (OCR assets).
