# Processing costs

The computational cost of this project: wall times, throughput, requests to remote servers,
data sizes on disk, and money spent. It feeds the methodology and results chapters of the
thesis. It is a living ledger: every work unit that runs a long or measured operation adds its
row here, in the same commit.

## At a glance

| Measure | Value |
|---|---|
| Money spent | 0: local CPU only, open models and libraries, no cloud, no GPU |
| Requests to the UCSM repository | 14 for the metadata, plus about 1,463 for the PDFs (estimate) |
| Raw snapshot on disk | 11.41 GB: 731 PDFs (11.40 GB) plus metadata and manifests |
| PDF download | about 3 h 9 min of active work (derived), within a 12 h 29 min elapsed span |
| Text layer, no OCR | about 464 pages/s (whole-corpus profile) |
| OCR, real pages | 2.71 s per page over 1,858 pages in the full run; 4.6 to 11.9 s per page in the earlier samples, depending on the machine state |
| Full PDF text extraction (T09) | 1 h 29 min (5,341.1 s) for 721 PDFs and 139,438 pages, 0 failures |
| Objectives locator (T10) | 6.0 s for 731 PDFs and 141,076 pages, no OCR: about 122 PDFs/s (derived) |
| Manual verification of the objectives (T10) | about 2 h of the user's time for 60 theses (reported by the user, not timed): about 2 min per thesis (derived) |
| Fichas build (T12) | 0.6 s for 766 records; the quality report with the spaCy stopword re-check, 8.5–9.2 s |
| mpnet model download (U8) | 287.6 s, the first load included, for 1,126,358,977 B (1.13 GB) on disk |
| Chunk build (T13) | 22.8–23.5 s for 744 theses and 5,217 chunks under both tokenizers, tokenizer loading included; 46.2–46.9 s in a slower machine state |
| Embeddings (T14) | 549.0 s from an empty cache for 5,211 distinct texts with both models (encoding: MiniLM 127.2 s, mpnet 306.3 s); 52.8 s on a rerun from the cache |
| TF-IDF baseline (T15) | 25.9–29.1 s for the whole command over 744 theses and 2,216 sections (cleaning 16.1–16.7 s); 14,308 terms and 98,282 nonzeros per matrix |

## 1. Purpose and conventions

**What gets recorded.** Every long or measured operation: harvests, downloads, text
extraction, embeddings, experiment runs, labeling and expert sessions, builds, and test-suite
runs. The row is added in the same work unit that runs the operation, from that unit's
evidence.

| Convention | Rule |
|---|---|
| Measured values only | A bare value is a measurement. "estimate" marks a projection, "derived" a value computed from measurements (such as a throughput), and "not measured" an unknown. "—" means not applicable |
| Times | Wall-clock time of the operation, with its unit (s, min, h). An elapsed span that includes pauses says so |
| Sizes | Bytes as measured. MB = 10^6 bytes and GB = 10^9 bytes, as in the data card. GiB (2^30 bytes) only for RAM |
| Dates | Local calendar days (UTC−5), as in the plan. Timestamps that end in `Z` are UTC |
| Evidence | Each row cites a plan task id, a results file or a manifest field. "Run output" means a console measurement first recorded in this ledger |
| Machine state | Record the power plan and the power source of each long run (§2) |

**How to add a row.**

1. Take the wall time from the run's own timer or its summary file, and the counts from its
   manifest.
2. Add one row to the ledger (§3) in date order, citing the evidence.
3. Update §4 if the operation writes data, and tick §6 if it closes an open measurement.

## 2. Machine and environment

Every operation ran on one laptop. Hardware data come from `Get-CimInstance` (Windows CIM
classes) and Python's `os.cpu_count()`; versions come from importing each library. All were
read on 2026-10-05.

| Component | Value |
|---|---|
| Machine | MSI Cyborg 15 A12VF laptop |
| CPU | 12th Gen Intel Core i7-12650H: 10 cores, 16 threads, 2.3 GHz base clock as reported by Windows, 24 MB L3 cache |
| RAM | 16 GiB, one module at 4800 MT/s; 16,868,970,496 bytes visible to Windows |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU and Intel UHD Graphics. **Not used:** torch is the CPU build (`torch.cuda.is_available()` is `False`) and runs 10 threads by default |
| Storage | 1 TB NVMe SSD (Kingston SNV2S1000G). Drive C: 999.0 GB, 166.0 GB free during the T09 run |
| OS | Windows 11 Home 10.0.26200, 64-bit |
| Power | Windows power plan "Balanced", on AC power, during the T09 run |
| Python | CPython 3.13.2, 64-bit, managed with uv 0.12.22 |

| Library | Version |
|---|---|
| pymupdf (MuPDF) | 1.28.2 |
| torch | 2.14.1+cpu |
| sentence-transformers | 6.1.0 |
| transformers | 5.18.0 |
| bertopic | 0.17.4 |
| spacy | 3.8.16, model `es_core_news_md` 3.8.0 |
| scikit-learn | 1.9.1 |
| umap-learn | 0.5.12 |
| hdbscan | 0.8.44 |
| gensim | 4.4.0 |
| numpy | 2.5.3 |
| pandas | 3.0.6 |
| pyarrow | 25.0.1 |
| httpx | 0.28.1 |
| streamlit | 1.64.0 |

**The machine state changes throughput.** OCR of the same 103 pages, with the same OCR
settings, took 4.6 s per page in the T09 sample and 11.9 s per page in the T09a re-run: 2.6
times slower. The cause was not isolated. A laptop's power mode, power source, temperature and
background load all change CPU throughput, so record them with each long run and compare times
across sessions with care.

## 3. Ledger

| Date | Task | Operation | Input / scope | Wall time | Throughput | Requests | Notes | Evidence |
|---|---|---|---|---|---|---|---|---|
| 2026-10-02 | T01 | Dependency lock (`uv lock`) | 192 packages resolved | not measured | — | not measured | `uv sync` downloads over 1 GB (estimate) | plan T01; README "Setup" |
| 2026-10-02 | T02 | Smoke suite (`pytest -m smoke`) | 14 tests on a synthetic mini corpus, offline | about 30 s | — | 0 | The first run fetched MiniLM; that download was not timed | plan T02 |
| 2026-10-02 | T03 | tessdata download | `spa` 13,570,187 B and `eng` 15,400,601 B (tessdata_best) | not measured | — | 2 files; redirects not counted | `eng` finished 2 s after `spa` | `tessdata/manifest.json` (`downloaded_at`); plan T03 |
| 2026-10-02 | T03 | OCR smoke | 1 synthetic image-only page, 300 dpi | 0.27 s | 0.27 s/page | 0 | A synthetic page; real pages are slower (T09 rows) | plan T03 |
| 2026-10-02 | T04 | Metadata harvest | 5 collections, 2021–2026: 766 records; `metadata.jsonl` 4,744,184 B | 47 s | about 16 records/s (derived) | 14 | At least 1 s between requests | plan T04; `os.stat` |
| 2026-10-02 | T05 | PDF download, sample | 10 PDFs, 2 per program: 65.3 MB | 114 s | about 0.57 MB/s (derived) | 20 | A rerun made 0 requests | plan T05 |
| 2026-10-02 to 2026-10-03 | T05 | PDF download, full | 746 items: 731 PDFs (11,401,121,329 B), 14 restricted, 1 without a thesis file | 12 h 29 min elapsed span, not continuous work; about 3 h 9 min active (derived) | about 1.0 MB/s and 3.9 PDFs/min over the active time (derived) | about 1,463, plus retries (estimate) | 2 interruptions; 3 `ConnectTimeout` errors retried later (notes) | `pdfs/manifest.json` (`downloaded_at`); plan T05 |
| 2026-10-03 | T05 | Re-hash of every PDF (SHA-256 and MD5) | 731 PDFs, 11.40 GB | 75.8 s | about 150 MB/s, 9.6 PDFs/s (derived) | 0 | All 731 matched the manifest and the server checksums | plan T05 (independent verification); time: run output |
| 2026-10-03 | T05, T06 | Text-layer profile (PyMuPDF `get_text`) | 141,076 pages of 731 PDFs | 304.3 s | about 464 pages/s (derived) | 0 | Gave the data card's page and low-text counts | plan T05 findings; data card; time: run output |
| 2026-10-03 | T05, D25 | Title check and "objetivo general" scan | 731 PDFs | 319 s | not derived | 0 | A second pass; gave the D25 page indices | plan T05 findings and D25; time: run output |
| 2026-10-03 | T07 | Metadata profile (EDA script) | 746 theses, metadata only | not measured | — | 0 | Offline | plan T07 |
| 2026-10-03 | T11 | Full cleaner (spaCy `es_core_news_md`) | thesis metadata texts | not measured | 25–76 docs/s | 0 | The light cleaner needs no model | plan T11 |
| 2026-10-03 | T11 | Stopword candidates script | 744 distinct theses | not measured | not recorded; the script prints docs/s | 0 | Offline | plan T11 and T11a (A4) |
| 2026-10-04 | T09 | OCR exploration | 6 real low-text pages; 300 dpi, `spa+eng`, tessdata_best | 0.4, 1.4, 2.7, 3.4, 4.1 and 4.2 s per page | mean 2.7 s/page (derived) | 0 | One page at a time | run output |
| 2026-10-05 | T09 | OCR setting trials | 3 real pages | not recorded | `spa` alone: 31% less time for 1% fewer characters; 200 dpi: 11% less time | 0 | Options for the user; none taken | plan T09 |
| 2026-10-05 | T09 | Text extraction, sample | 10 PDFs, 1,638 pages: 1,535 text layer, 103 OCR | 483.3 s, summed over the PDFs (derived) | OCR 4.6 s/page; other work 0.0056 s/page | 0 | Split across 2 sessions; per-PDF times in the notes | plan T09 |
| 2026-10-05 | T09a | Text extraction, sample re-run (rules version 2) | the same 10 PDFs | 1,233.8 s | OCR 11.9 s/page, 2.6 times slower; other work about 0.0046 s/page (derived) | 0 | The scanned PDF took 1,138.0 s; a rerun skipped all 10 in 0.1 s | plan T09a |
| 2026-10-05 | T09 | Text extraction, full run | 731 PDFs: 721 extracted, 10 skipped (the sample); 139,438 pages in this run: 137,063 text layer, 1,855 OCR, 520 empty | 5,341.1 s (1 h 29 min) | OCR 2.71 s/page (1,858 pages in 5,042.4 s); about 26.1 pages/s overall, and about 0.0021 s per page outside OCR (derived) | 0 | 0 failures. Every PDF in the text manifest: 141,076 pages, 1,958 OCR, 8,616 low-text, 197 MuPDF warnings. The manifest's own timings cover all 731 PDFs, the slower sample included: 6,568.3 s summed over the PDFs and 3.20 s per OCR page. The estimate was 3–6.5 h. The power plan was not recorded | run output (wall time, OCR time); `results/extraction/20261002T224412Z/summary.json` (counts and whole-manifest timings) |
| 2026-10-03 to 2026-10-05 | all | Test suite (`pytest -q`) | 762, 918 and 923 tests, plus 1 skipped each time | 12.5 s; 16.6–28.4 s; 47.6 s | — | 0 | The 47.6 s run was in a slow machine state | plan T11, T09 and T09a; times: run output |
| 2026-10-07 | T10 | Objectives locator, tuning rounds 1 and 2 | 731 page files (141,076 pages) each time | 6.6 s; 6.7 s | about 110 PDFs/s (derived) | 0 | Round 1: 727 extracted, 4 not found; round 2: 731 extracted, 0 not found | plan T10; run output |
| 2026-10-07 | T10 | Objectives locator, final run | 731 page files (141,076 pages) | 5.977 s; the whole command took 8.5 s, start-up and workbook included | about 122 PDFs/s and 23,600 pages/s (derived) | 0 | 731 extracted, 0 not found; reading and hashing the page files is most of the time | `results/objectives/20261002T224412Z/summary.json` (`seconds`); run output |
| 2026-10-07 | T10 | Verification sample and workbook | 60 theses drawn from 728 eligible; 60-row workbook | 1.39 s, timed again in memory | — | 0 | Includes the name guard over 3,096 name forms; 0 cells redacted | plan T10; run output |
| 2026-10-07 | all | Test suite (`pytest -q`) | 1,070 tests, plus 1 skipped | 20.6 s | — | 0 | — | plan T10; time: run output |
| 2026-10-07 | T10 | Manual verification of the objectives, by the user | 60 rows of the workbook, one verifier | about 2 h (reported by the user, not timed) | about 2 min per row (derived) | 0 | Below the 4 to 6 min per row estimate | `results/objectives/20261002T224412Z/verification.json`; the user's report |
| 2026-10-07 | T10 | Workbook import and accuracy report | the filled 60-row workbook | not measured | — | 0 | Imported at 2026-10-07T21:43:43Z | `results/objectives/20261002T224412Z/verification.json` (`imported_at`) |
| 2026-10-07 | all | Test suite (`pytest -q`) | 1,107 tests, plus 1 skipped | 22.1–31.5 s (2 runs) | — | 0 | — | plan T10b; times: run output |
| 2026-10-07 | T12 | Fichas build (`build_fichas`) | 766 records, 731 objectives rows: 766 fichas (744 included), 22 exclusions | 0.6 s (0.628 s), 2 runs; the whole command took 11.1 s, start-up and quality report included | about 1,200 records/s (derived) | 0 | Every enforced check passed. The first run created the pseudonym key; both runs wrote the same fichas table (same SHA-256) | `data/interim/20261002T224412Z/fichas_manifest.json` (`seconds`); run output |
| 2026-10-07 | T12 | Quality report with the stopword re-check (spaCy) | objectives text of the 728 included theses that have it | 8.5 s and 9.2 s (2 runs), spaCy model load included | about 80 theses/s (derived) | 0 | Full cleaner without domain stopwords | run output |
| 2026-10-07 | all | Test suite (`pytest -q`) | 1,194 tests, plus 1 skipped | 22.5 s | — | 0 | — | plan T12; time: run output |
| 2026-10-07 | T13 (U8) | mpnet model download from the Hugging Face Hub, then its first load | `paraphrase-multilingual-mpnet-base-v2`: 11 files, 1,126,358,977 B on disk | 287.6 s, download and first load together | about 3.9 MB/s, the load included (derived) | not measured | Anonymous GET, no credentials. Started 2026-10-08T04:08:21Z; the cache's last write is 04:13:09Z, which agrees. The power plan was not recorded | run output; Hugging Face cache (`scan_cache_dir`, file sizes) |
| 2026-10-11 | T13 (U8) | Offline load of both models from the cache (`SentenceTransformer`, CPU) | MiniLM and mpnet | 3.0 s; 3.9 s | — | 0 | `HF_HUB_OFFLINE=1`; checked dimensions 384 and 768, `max_seq_length` 128 and 250,002-token tokenizers | plan T13 and progress log; time: run output |
| 2026-10-11 | all | Test suite (`pytest -q`) | 1,194 tests, plus 1 skipped | 38.6 s | — | 0 | The resume check, before the closure unit | time: run output |
| 2026-10-11 | all | Test suite (`pytest -q`) | 1,196 tests, plus 1 skipped | 23.7 s | — | 0 | After the closure unit (D26) | plan progress log; time: run output |
| 2026-10-11 | T12a | Fichas build re-run (`build_fichas`, without `--report-out`) | 766 records, 731 objectives rows: 766 fichas (744 included), 22 exclusions | 0.8 s (0.843 s); the whole command took 2.8 s, start-up included | about 910 records/s (derived) | 0 | The new one-unit publish wrote the same fichas and exclusions tables as on 2026-10-07 (same SHA-256) and left no staged file. The verified reader then checked the outputs in 0.002 s and loaded the 766 fichas in 0.46 s | `data/interim/20261002T224412Z/fichas_manifest.json` (`seconds`); plan T12a; run output |
| 2026-10-11 | all | Test suite (`pytest -q`) | 1,215 tests, plus 1 skipped | 42.7–48.0 s (2 runs) | — | 0 | After T12a | plan T12a; time: run output |
| 2026-10-12 | T08a | Read-only schema re-check of every real table through the stricter reader | `fichas.parquet` (766 rows, through `load_fichas`), `exclusions.parquet` (22), `objectives.parquet` (731) and 731 page files (141,076 pages) | 9.4 s (9.37 s), one run | about 15,000 pages/s (derived) | 0 | 0 tables refused; the three tables kept their SHA-256, and nothing was written | plan T08a; time: run output |
| 2026-10-12 | all | Test suite (`pytest -q`) | 1,251 tests, plus 3 skipped | 22.3–32.6 s (2 runs) | — | 0 | After T08a and its review fixes | plan T08a; time: run output |
| 2026-10-12 | T09b | Text extraction re-run of one already extracted item (`extract_text --only`, without `--summary-out`), 2 runs | 1 PDF selected of 731 | 0.0 s (the command's own wall time); 1.2 s for the whole command, start-up included (second run) | — | 0 | Both runs skipped the item (0 extracted) through the shared run lock. The text manifest and all 731 page files kept their SHA-256 and modification times; the manifest was not rewritten | plan T09b; time: run output |
| 2026-10-12 | all | Test suite (`pytest -q`) | 1,290 tests, plus 3 skipped | 23.1–25.0 s (2 runs) | — | 0 | After T09b item 1, the characterization tests of item 2, and the review fixes | plan T09b; time: run output |
| 2026-10-12 | T12b | Read-only load of the real fichas through the one-read reader (`load_fichas`) | `fichas.parquet` (766 rows), with `exclusions.parquet` and `fichas_manifest.json` checked | 0.4–0.5 s (0.498 s, then 0.385 s after the review fixes), 2 runs; the check alone (`verify_fichas_outputs`) took 0.001–0.002 s | about 1,540–1,990 fichas/s (derived) | 0 | 766 fichas, 744 included, in both runs. The three files kept their SHA-256, sizes and modification times; nothing was built or written | plan T12b; time: run output |
| 2026-10-12 | all | Test suite (`pytest -q`) | 1,319 tests, plus 3 skipped | 26.3–26.7 s (2 runs) | — | 0 | After T12b and its review fixes (1,310 tests in 25.8–29.3 s before them) | plan T12b; time: run output |
| 2026-10-12 | T07a | Read-only, in-memory render of the metadata profile outputs of the real snapshot, with the second tokenizer's comparison, against the committed files | 766 items (746 theses); 7,097 distinct texts tokenized by both tokenizers; 6 outputs | 16.8 s for the whole command, two script imports included; MiniLM tokenizer load 1.28 s, profile 1.42 s, mpnet tokenizer load and comparison 2.77 s, render 0.48 s | about 2,560 texts/s for the mpnet load and comparison together (derived) | 0 | 0 of 7,097 texts counted differently. The 4 figures are byte-identical to the committed ones; the profile JSON differs only by its new `other_tokenizers` key, and `summary.md` by two method notes. Privacy gate: 0 hits. Nothing was written; the 7 committed files kept their SHA-256. Repeated after the review fixes with the same results in 16.7 s | plan T07a; time: run output |
| 2026-10-12 | all | Test suite (`pytest -q`) | 1,376 tests, plus 3 skipped | 33.7–38.6 s (3 runs) | — | 0 | After T11b, T07a and their review fixes (1,369 tests in 33.0–40.4 s, 3 runs, before the fixes; 28.2 s for the 1,319 tests before the unit). The new test module of the profile script takes 8.0–8.6 s alone: it imports matplotlib and transformers and draws figures | plan T07a; time: run output |
| 2026-10-12 | T04a, T05a | Read-only check of the real snapshot and its PDF manifest against the new harvest and download guards | `metadata.jsonl` (766 records), `manifest.json`, `pdfs/manifest.json` (746 entries) | 1.0 s for the whole command, start-up included (second run) | — | 0 | 766 distinct uuids, 0 repeated; each program equals its manifest count; programs 766 = faculty 766, so the enforced faculty check passes with no accepted difference. The PDF manifest still loads with the new `unverifiable` status (583 already present, 148 downloaded, 14 restricted, 1 no thesis file). The three files kept their SHA-256. Rerun after the review fixes, with the faculty check called on the real totals and the 766 fichas loaded through `load_fichas` as well: the same results, 6 files unchanged, 2.4 s for the whole command | plan T04a, T05a; time: run output |
| 2026-10-12 | T02a | Read-only check of the real tessdata files against the new pins and the layout check | `spa.traineddata` 13,570,187 B and `eng.traineddata` 15,400,601 B | 1.0 s for the whole command, start-up included; 0.02 s for `ensure_tessdata`, which hashed both files | about 1.4 GB/s hashed (derived from the rounded 0.02 s) | 0 | Both files equal their pins and pass the layout check (24 entries, 7 present). `ensure_tessdata` with a downloader that refuses every request downloaded nothing; the 3 files kept their SHA-256 | plan T02a; time: run output |
| 2026-10-12 | T02a | Smoke suite through the new selection (`pytest -m smoke`) | 17 tests (16 + the D22 test), offline (`HF_HUB_OFFLINE=1`) | 32.8–39.0 s (32.75 s, then 38.97 s after the review fixes), 2 runs | — | 0 | 17 passed, 1,425 and then 1,434 deselected. The mutant run of the D22 test alone, with BERTopic's default language, took 65.1 s and failed as expected | plan T02a; time: run output |
| 2026-10-12 | all | Test suite (`pytest -q`) | 1,431 tests, plus 3 skipped; 17 smoke tests deselected by the new collection hook | 35.0–37.9 s (2 runs); `pytest -q -m "not slow"` 35.7 s, the same counts | — | 0 | After T05a, T04a, T02a and their review fixes (unit 5); 1,422 tests in 35.4–42.8 s (3 runs) before the fixes. Before the unit: 1,376 tests in 35.2 s, one run, with 16 smoke tests deselected through `addopts` | plan T02a; time: run output |
| 2026-10-13 | T07a | Metadata profile regeneration (`experiments/eda_metadata.py`), offline (`HF_HUB_OFFLINE=1`), 3 runs | 766 items (746 theses); 7,097 distinct texts counted by both tokenizers; 6 outputs, 318,974 B | 17.4 s and 16.5 s for the whole command (runs 2 and 3); run 1 not timed | — | 0 | Runs 1 and 3, each from a clean tree at `7c6a4e1`, wrote byte-identical outputs (same SHA-256). Run 2, with run 1's outputs still uncommitted in the tree, differed only by `git_uncommitted_changes` in `metadata_profile.json` and the matching line of `summary.md`. The 4 figures are byte-identical to the committed ones in every run. Privacy gate: 6 outputs, 11,215 strings, 0 hits in each run. The 6 outputs went from 318,424 B to 318,974 B: `metadata_profile.json` 20,499 → 20,841 B, `summary.md` 12,584 → 12,792 B | plan progress log (2026-10-13); `results/eda/20261002T224412Z/metadata_profile.json`; times: run output |
| 2026-10-13 | all | Test suite (`pytest -q -p no:cacheprovider`) | 1,431 tests, plus 3 skipped; 17 smoke tests deselected | 35.4 s, one run | — | 0 | The closing unit of the stack: no source or test changed since `7c6a4e1`. The 15 privacy-gate test cases of `test_eda_metadata.py`, selected with `-k`, took 8.2 s alone | plan progress log (2026-10-13); time: run output |
| 2026-10-15 | T13 | Chunk build (`build_chunks`), offline (`HF_HUB_OFFLINE=1`), 5 runs | 744 included theses: 2,216 sections, 5,217 chunks (744 title, 4,473 body), each counted by both tokenizers | 23.5 s, 23.2 s and 22.8 s for the build (runs 3 to 5); 26.4 s, 26.9 s and 25.7 s for the whole command, start-up included. Runs 1 and 2, in a slower machine state and before later edits that change no rule: 46.9 s and 46.2 s; 50.1 s and 49.3 s | about 32–33 theses/s and 222–229 chunks/s over the build (derived, runs 3 to 5) | 0 | Phases, one profiling run in each state: tokenizer loading, with the import of transformers and torch, 11.2 s (22.0 s in the slower state); chunking 8.6 s (19.3 s), 19,853 counts per tokenizer; checks 2.5 s (5.6 s). Runs 3 to 5 wrote byte-identical tables. 0 chunks above 128 under either tokenizer; all 2,216 sections recovered. The power plan was not recorded | `data/interim/20261002T224412Z/chunks_manifest.json` (`seconds`); plan T13; times: run output |
| 2026-10-15 | T13 | Chunker smoke tests (`pytest -m smoke tests/smoke/test_chunker.py`), offline, 3 runs | 5 tests over the real fichas with both real tokenizers | 28.4 s and 25.0 s (30.9 s and 27.8 s for the whole command); the first run, in the slower state, 58.3 s (62.1 s) | — | 0 | 5 passed in every run | plan T13; times: run output |
| 2026-10-15 | all | Test suite (`pytest -q -p no:cacheprovider`) | 1,702 tests, plus 3 skipped; 22 smoke tests deselected | 46.3 s and 43.4 s (49.4 s and 46.4 s for the whole command), 2 runs | — | 0 | After T13: 271 new tests, 266 in `tests/unit/representation` and 5 in the configuration tests | plan T13; time: run output |
| 2026-10-15 | T13a | Chunk build (`build_chunks`) after the `no.` rule, offline (`HF_HUB_OFFLINE=1`), 3 runs | 744 included theses: 2,216 sections, 5,217 chunks (744 title, 4,473 body), each counted by both tokenizers | The final run 23.5 s for the build (26.4 s for the whole command); the first run, in a slower machine state, 45.6 s (49.0 s); a mutant run that recorded a wrong tokenizer SHA-256 for the smoke test's RED, 76.4 s (whole command not timed) | about 32 theses/s and 222 chunks/s over the final build (derived) | 0 | The same counts and chunk ends as T13 (sentence 4,985, clause 66, comma 162, window 4). The first and final runs wrote a `chunks.parquet` byte-identical to T13's (SHA-256 `2e579834…`, 1,375,134 B); the manifest (1,336 B) changes with `chunker_version` 2. The power plan was not recorded | `data/interim/20261002T224412Z/chunks_manifest.json` (`seconds`); plan T13a; times: run output |
| 2026-10-15 | T13a | Chunker smoke tests (`pytest -m smoke tests/smoke/test_chunker.py`), offline, 1 run | 6 tests over the real fichas with both real tokenizers | 26.7 s (29.2 s for the whole command) | — | 0 | 6 passed: T13's 5, with the tokenizer comparison now always run, plus the manifest's tokenizer SHA-256 check | plan T13a; time: run output |
| 2026-10-16 | T14 | Full embedding build (`build_embeddings`), offline (`HF_HUB_OFFLINE=1`), empty cache, run by the user in his terminal (PowerShell) | 744 theses: 2,216 sections, 5,217 chunks, 5,211 distinct texts per model; MiniLM and mpnet | `Seconds: 549.0` as printed; encoding 127.2 s (MiniLM) and 306.3 s (mpnet) | 41.0 and 17.0 texts/s (D) | 0 | The design note estimated about 102 s and 328 s of encoding (E). The other 115.5 s (D) is the imports, the model loads, the pooling, the checks and the writes. Cosine of the two document vectors per thesis: MiniLM min 0.8349, median 0.9765, 67 below 0.95; mpnet min 0.8795, median 0.9822, 30 below 0.95 | plan T14; run output |
| 2026-10-16 | T14 | Embedding rerun from the cache (`build_embeddings`), offline | Same scope | `Seconds: 52.8`; 0.1 s of encoding per model | — | 0 | 5,211 texts from the cache and 0 embedded per model; all vector, section and cache files byte-identical to the first run (SHA-256 compared); only the manifest's run fields changed | plan T14; run output and SHA-256 comparison |
| 2026-10-15 | all | Test suite (`pytest -q`) | 1,722 tests, plus 3 skipped; 23 smoke tests deselected | 52.2 s and 41.9 s (54.9 s and 45.2 s for the whole command), 2 runs | — | 0 | After T13a: 20 new tests in `tests/unit/representation` | plan T13a; time: run output |
| 2026-10-16 | T14 | Embedding smoke tests (`pytest -m smoke tests/smoke/test_embeddings.py`), offline (`HF_HUB_OFFLINE=1`), 2 runs | 4 tests: 11 real chunks of 2 theses embedded by MiniLM and mpnet into a temporary cache, then read back from it | 58.1 s and 49.6 s (64.3 s and 54.8 s for the whole command); in the first run, 48.9 s was the first setup, the imports of torch, transformers and sentence-transformers and the loading of both models | — | 0 | 4 passed in each run. Not the full run: embedding the 5,217 chunks is left to the user's terminal (about 7.2 min, estimate, from the T13 design note). The power plan was not recorded | plan T14; time: run output |
| 2026-10-16 | all | Test suite (`pytest -q`) | 1,810 tests, plus 3 skipped; 27 smoke tests deselected | 99.4 s and 87.7 s (103.0 s and 91.0 s for the whole command), 2 runs by the writer; 61.7 s (64.7 s for the whole command), 1 run by the parent after the full embedding build | — | 0 | After T14: 88 new tests, 74 in `tests/unit/representation` (2.6 s alone) and 14 in the configuration tests. The slowest tests were older ones (at most 1.7 s each); the cause of the longer runs was not isolated | plan T14; time: run output |
| 2026-10-16 | T14a | Embedding smoke tests (`pytest -m smoke tests/smoke/test_embeddings.py`), offline (`HF_HUB_OFFLINE=1`), 1 run | 4 tests: the chunks of the same 2 real theses as T14, embedded by MiniLM and mpnet into a temporary cache, then read back from it | 27.3 s (31.8 s for the whole command) | — | 0 | 4 passed. The `sample` fixture now skips when the chunk table has no three-section or no metadata-only thesis. The power plan was not recorded | plan T14a; time: run output |
| 2026-10-16 | all | Test suite (`pytest -q`) | 1,817 tests, plus 3 skipped; 27 smoke tests deselected | 88.1 s and 47.2 s (50.5 s for the whole second command; the first not recorded), 2 runs | — | 0 | After T14a: 7 new tests in `tests/unit/representation`; `test_build_embeddings.py` alone, 29 tests, took 2.8 s. The cause of the slower first run was not isolated | plan T14a; time: run output |
| 2026-10-19 | T15 | TF-IDF build (`build_tfidf`), offline (`HF_HUB_OFFLINE=1`), 2 runs | 744 included theses: 2,216 sections full-cleaned by spaCy (126,481 tokens); 14,308 terms and 98,282 nonzeros per matrix | 29.1 s and 25.9 s for the whole command; the manifest's `seconds` 23.6 and 21.0, of which cleaning 16.7 s and 16.1 s and vectorizing 0.26 s and 0.28 s | about 133–138 sections/s while cleaning (derived) | 0 | The design note estimated about 45 s (estimate). The second run wrote byte-identical data files; only the manifest's run fields changed. Cosine between the two document vectors per thesis: min 0.776063, median 0.908563, 719 of 744 below 0.95. The power plan was not recorded | `data/interim/20261002T224412Z/tfidf/tfidf_manifest.json` (`seconds`); plan T15; times: run output |
| 2026-10-19 | T15 | TF-IDF smoke tests (`pytest -m smoke tests/smoke/test_tfidf.py`), offline (`HF_HUB_OFFLINE=1`), 2 runs | 5 tests: two real builds into temporary projects, the reader on both variants, and 150 sections cleaned again in a batch of their own | 51.6 s and 34.7 s (55.0 s and 37.8 s for the whole command) | — | 0 | 5 passed in each run. The power plan was not recorded | plan T15; time: run output |
| 2026-10-19 | all | Test suite (`pytest -q`) | 1,916 tests, plus 3 skipped; 32 smoke tests deselected | 79.9 s (83.5 s for the whole command), 1 run | — | 0 | After T15: 99 new tests, 69 in `test_build_tfidf.py`, 29 in the configuration tests and 1 in the full cleaner tests; the targeted run of `tests/unit/representation`, `tests/unit/preprocessing` and the configuration tests, 808 tests, took 22.3 s and 25.4 s | plan T15; time: run output |
| 2026-10-19 | all | Test suite (`pytest -q`) | 2,040 tests, plus 3 skipped; 32 smoke tests deselected | 87.4 s and 55.9 s (91.1 s and 59.2 s for the whole command), 2 runs | — | 0 | After T17: 118 new tests in `tests/unit/evaluation`, which took 2.4–5.1 s alone; the U4 test was changed, not added. The 18 mutant runs of `tests/unit/evaluation` took 68.9 s in all. The cause of the slower first run was not isolated | plan T17; time: run output |
| 2026-10-19 | all | Test suite (`pytest -q`) | 2,134 tests, plus 3 skipped; 32 smoke tests deselected | 91.4 s and 61.2 s (94.9 s and 65.2 s for the whole command), 2 runs | — | 0 | After T18: 94 new tests, 87 in the configuration tests, 4 in `test_experiment_declaration.py` and 3 in `test_npmi_coherence.py`; the targeted run of the configuration, declaration and evaluation tests, 425 tests, took 9.5 s (11.6 s whole). An earlier full run, before the fallback-model rule left the loader, gave 4 failed and 2,130 passed in 103.7 s (107.9 s whole). The 14 mutant runs took 208.7 s in all (210.9 s for a first round). The cause of the slower first run was not isolated | plan T18; time: run output |
| 2026-10-19 | T16 | Topic-map smoke tests (`pytest -m smoke tests/smoke/test_topic_map_build.py`), offline (`HF_HUB_OFFLINE=1`), 2 runs | 3 tests on 63 synthetic theses (no thesis read): the real UMAP, K-means, HDBSCAN and BERTopic in its manual mode | 49.7 s (53.8 s for the whole command), then 43.3 s (47.7 s) | — | 0 | The first run failed 1 test, whose synthetic outliers UMAP had placed inside clusters; the second passed all 3 after that test was changed. Imports, each in a fresh process: BERTopic 52.1 s (cold), UMAP 12.6 s cold and 11.2 s with scikit-learn already loaded. The power plan was not recorded | plan T16; times: run output |
| 2026-10-19 | all | Test suite (`pytest -q`) | 2,245 tests, plus 3 skipped; 35 smoke tests deselected | 90.6 s, 76.6 s and 67.4 s (95.1 s, 80.7 s and 71.2 s for the whole command), 3 runs, the last on the final files | — | 0 | After T16: 111 new tests, 98 in `tests/unit/topic_map` (4.7–6.0 s alone) and 13 in the configuration tests. The 18 mutant runs of `tests/unit/topic_map` took 161.2 s in all, and 141.5 s in a second round on the final tests | plan T16; time: run output |
| 2026-10-19 | all | Test suite (`pytest -q`) | 2,300 tests, plus 3 skipped; 35 smoke tests deselected | 112.05 s and 145.71 s (117.3 s and 154.9 s for the whole command), 2 runs | — | 0 | After T19's first slice, the selection rules: 55 new tests in `tests/unit/topic_map/test_experiment_selection.py` (0.14–0.17 s alone); `tests/unit/topic_map`, 153 tests, took 8.09 s (17.2 s whole). The 23 mutant runs of `tests/unit/topic_map` took 310.5 s in all. No smoke test and no experiment run in this slice. The cause of the slower second run was not isolated | plan T19; time: run output |
| 2026-10-20 | T19 | Experiment dry run (`run_experiment --dry-run`), offline (`HF_HUB_OFFLINE=1`), 2 runs | Snapshot `20261002T224412Z`: reads and verifies both models' document vectors, the TF-IDF matrix (744 × 14,308, 98,282 nonzeros) and the full-clean sections; fits, clusters and writes nothing | 6 s, then 3.7 s for the whole command | — | 0 | Ran through a wrapper with the main repository as the project root (the worktree's `data/` junction is refused by the path containment check; config files byte-identical). Forecast: 12 UMAP fits, 1 PCA fit, 371 clusterings (+5 if the matched K lies outside the sweep), 25 c-TF-IDF fits (+5 if it is not B0's K*), 370 internal-metric calls, 40 reassigned-noise silhouettes, 50 NPMI and 148 ARI evaluations; the real run is estimated at about 4 min, 4 to 12 min with real-data convergence (E, D30 E6) | plan T19, second slice; dry-run output |
| 2026-10-20 | T19 | Experiment-runner smoke test (`pytest -m smoke tests/smoke/test_experiment_runner.py`), offline, 2 runs | 1 test: the full declared grid twice through the command, on 108 synthetic theses (no thesis read), with the real UMAP, K-means, HDBSCAN and BERTopic; the second run reuses the 10 cached reductions | 166.97 s (171 s for the whole command), then 115.12 s (117.8 s) on the final files | — | 0 | A profiling probe of one run of the same project outside pytest took 83.4 s: UMAP's first fit 15.5 s (its JIT compilation), the sweeps and their metrics done at 45.7 s, the first final configuration's c-TF-IDF 35.1 s (BERTopic's import with it), the rest 2.6 s; 376 clusterings, 30 c-TF-IDF fits | plan T19, second slice; times: run output |
| 2026-10-20 | T19 | Mutation check of the runner (23 throwaway mutants against `tests/unit/topic_map`), 2 full batches and 2 partial reruns | `experiment.py`, `results_store.py`, `run_experiment.py`; each source restored byte-identical after each mutant | 335.4 s (22 mutants), 57.4 s (4), 15.8 s (1), 328.3 s (23, the final batch) | about 14 s per mutant (D) | 0 | First batch 19 of 22 killed; after two test scenarios were added, the final batch killed all 23 | plan T19, second slice |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree | 2,352 tests, plus 3 skipped; 36 smoke tests deselected | 135.24 s (138.9 s for the whole command), 1 run | — | 0 | After T19's second slice: 52 new unit tests (44 in `test_experiment_runner_unit.py`, 8 in `test_results_store.py`; `tests/unit/topic_map` 205 tests in 16.49 s) | plan T19, second slice |
| 2026-10-20 | T19 | Experiment, real run (`run_experiment`), offline (`HF_HUB_OFFLINE=1`), at commit `bf0ca62` | 744 theses: MiniLM and mpnet section-mean vectors and the TF-IDF matrix; 5 seeds; 22 K, 4 min_cluster_size | 417.0 s for the whole command; 389.2 s in the manifest | — | 0 | 12 UMAP fits, 1 PCA fit, 371 clusterings, 30 c-TF-IDF fits, 370 internal-metric calls, 40 reassigned-noise silhouettes, 60 NPMI and 148 ARI evaluations; repeatability identical. Against the forecast of about 4 min, 4 to 12 min (E, D30 E6). Records: 232,566 B tracked, 446,679 B gitignored | plan T19; run output and manifest |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree after T22 | 2,482 tests, plus 3 skipped; 36 smoke tests deselected | 95.78 s (99.7 s for the whole command), 1 run | — | 0 | After T22: 130 new tests (122 in `tests/unit/redundancy`, 8 in `test_config.py`); `tests/unit/redundancy`, `test_config.py` and the declaration pin 446 passed, 1 skipped in 19.47 s | plan T22 |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree after T23 | 2,573 tests, plus 3 skipped; 36 smoke tests deselected | 111.94 s (116.4 s for the whole command), 1 run | — | 0 | After T23: 91 new tests (56 in `test_proposal_assignment.py`, 35 in `test_topic_saturation.py`); with `tests/unit/redundancy` 213 passed in 1.11 s | plan T23 |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree after T24 | 2,654 tests, plus 3 skipped; 36 smoke tests deselected | 103.18 s (108.3 s for the whole command), 1 run | — | 0 | After T24: 81 new tests in `tests/unit/gaps`; with the U4 pin 87 passed in 4.75 s | plan T24 |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree after T26 | 2,710 tests, plus 3 skipped; 36 smoke tests deselected | 114.52 s (118 s for the whole command), 1 run | — | 0 | After T26: 56 new tests in `tests/unit/recommendation`, which pass alone in 13.16 s (19.1 s for the whole command) | plan T26 |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree after the statistics for T21, T30 and T32 | 2,840 tests, plus 3 skipped; 36 smoke tests deselected | 94.50 s (99 s for the whole command), 1 run | — | 0 | After the statistics: 130 new tests in `tests/unit/evaluation`, where all 251 pass in 6.49 s (19.2 s for the whole command) | plan T21 |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree after D31 (Fleiss' κ) | 2,858 tests, plus 3 skipped; 36 smoke tests deselected | 159.87 s (164 s for the whole command), 1 run | — | 0 | After D31: 18 new tests in `tests/unit/evaluation/test_rater_agreement.py`; all 269 evaluation tests pass in 4.67 s (13.4 s for the whole command); the 6 mutant runs took 2.89–3.27 s each | plan T21 |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree after T12c | 2,907 tests, plus 3 skipped; 36 smoke tests deselected | 150.90 s (157 s for the whole command), 1 run | — | 0 | After T12c: 49 new tests (D: 2,907 − 2,858) | plan T12c |
| 2026-10-20 | T12c | Chunker smoke tests (`pytest -m smoke tests/smoke/test_chunker.py`), offline (`HF_HUB_OFFLINE=1`), in the main tree | the real chunk table and both tokenizers of the HF cache | 55.17 s (60 s for the whole command), 1 run | — | 0 | 6 passed; item 11 changed the manifest test, which the worktree could not run without `data/` | plan T12c |
| 2026-10-20 | T24 | Gap-threshold proposal (read-only analysis): reproduction of the 5 C4 runs in memory and the rule grids, offline | the T19 records and the 744 theses' mpnet vectors; no file written in the repository | 53.7 s for the reproduction (58 s on the wall clock); about 4, 4 and 2 s for the analysis scripts; about 881 s for the whole analyst session | — | 0 | The reproduction matched the T19 labels, UMAP arrays and per-topic NPMI exactly; seed 7's UMAP took 21.7 s, likely numba's compilation (E) | plan D32 |
| 2026-10-20 | all | Test suite (`pytest -q`), in the main tree after D32 | 2,941 tests, plus 3 skipped; 36 smoke tests deselected | 142.28 s (147 s for the whole command), 1 run | — | 0 | After D32: 34 new tests (D: 2,941 − 2,907); the 7 mutant runs took 2.74–5.06 s of pytest time each | plan T24 |
| 2026-10-21 | all | Test suite (`pytest -q`), in the main tree at `760013d`, at the session's start | 2,941 tests, plus 3 skipped; 36 smoke tests deselected | 139.70 s, 1 run | — | 0 | A re-check of the tree D32 left; no code changed | plan progress log 2026-10-21 |
| 2026-10-21 | T24a | Mutation check of the gap runner (21 throwaway mutants against `tests/unit/gaps`), 1 batch and 1 rerun of the changed lines | `run_gaps.py`, `results_store.py`, `sensitivity.py` | 336.9 s for 19 mutants; 14.4 s for 2 | — | 0 | 18 of 19 killed; the survivor was equivalent, the code was simplified and its 2 replacement mutants were killed | plan T24a |
| 2026-10-21 | all | Test suite (`pytest -q`), in the main tree after T24a's runner | 2,995 tests, plus 3 skipped; 36 smoke tests deselected | 124.52 s, 1 run | — | 0 | After T24a's runner: 54 new tests (D: 2,995 − 2,941) | plan T24a |
| 2026-10-21 | T24a | Gap-run dry run (`run_gaps --dry-run`), offline (`HF_HUB_OFFLINE=1`), 3 runs before the runner's commit | Snapshot `20261002T224412Z`: T19's records, the 744 theses' mpnet vectors, the fichas, and the snapshot's metadata for the privacy guard; nothing written | 10.3 s and 6.4 s for the whole command (5.7 s and 2.3 s of computing); a third run, 2.4 s of computing (its whole command not timed) | — | 0 | The first run includes the cold reads of the inputs | plan T24a |
| 2026-10-21 | T24a | Gap run, real run (`run_gaps`), offline (`HF_HUB_OFFLINE=1`), at commit `1e8fe3a`, 2 runs | 744 theses; the 5 seed runs of C4; rules (a)–(d), the stability filter and the sensitivity table | 9.47 s and 6.77 s for the whole command; 1.694 s and 1.479 s in the manifest | — | 0 | 184 gaps over the 5 runs, 24 stable in the reference run. The second run's tables, summary and members are byte-identical to the first's, and its manifest differs only in its times; the committed manifest is the second run's. Privacy guard: 0 hits against 11,215 strings. Records: 44,043 B tracked, 14,354 B gitignored | plan T24a; run output and manifest |
| 2026-10-21 | T20a | Mutation check of the sampler (28 throwaway mutants against `tests/unit/labeling` and `tests/unit/test_config.py`) | `labeling/pairs.py`, `labeling/draw_pairs.py`, `shared/config.py` | 162 s | — | 0 | 28 of 28 killed, each file restored byte-identical | plan T20 |
| 2026-10-21 | all | Test suite (`pytest -q`), in the main tree after T20a's sampler, 2 runs | 3,085 and 3,088 tests, plus 3 skipped; 36 smoke tests deselected | 235 s for 3,085 (3 tests were added after it); 79.29 s for 3,088 | — | 0 | A third run, after the last 3 tests, was stopped by the system for low memory at 86 % with no failure up to then; the parent's run of 3,088 is the closing one | plan T20 |
| 2026-10-21 | T20a | Labeled-set draw dry run (`draw_pairs --dry-run`), offline (`HF_HUB_OFFLINE=1`), 2 runs before the sampler's commit | Snapshot `20261002T224412Z`: the 744 theses' MiniLM, mpnet and TF-IDF vectors, the fichas and the snapshot's metadata for the privacy guard; 276,396 pairs scored three times; nothing written | 19.6 s and 8.37 s for the whole command; 8.4 s and 3.6 s of computing | — | 0 | — | plan T20 |
| 2026-10-21 | T20a | Labeled-set draw, real run (`draw_pairs`), offline (`HF_HUB_OFFLINE=1`), at commit `2eb3249`, once | 276,396 pairs under 3 representations; 130 pairs drawn (10 pilot, 120 labeled) over 217 theses | 27.97 s for the whole command; 5.3 s in the run (load 1.5 s, universe 3.6 s, draw 0.2 s) | — | 0 | The first run after a session restart. Composition identical to the dry runs; a second real run refused with exit 1 and left the records byte-identical. Privacy guard: 0 hits. Records: 16,602 B tracked, 18,282 B gitignored | plan T20; run output and manifest |
| 2026-10-27 | all | Test suite (`pytest -q`), in the t12c10 worktree, through the lock | 3,089 tests: 3,087 passed and the worktree's 2 environmental failures; 3 skipped, 36 smoke tests deselected | 123.98 s, 1 run (lock held 126.5 s) | — | 0 | After T12c item 10's follow-up: 1 new test (D: 3,089 − 3,088); the 2 failures are the worktree-junction cases of `test_anonymize.py`, which pass in the main repository | plan T12c (t12c10) |
| 2026-10-27 | all | Test suite (`pytest -q`), in the main tree after T12c item 10's follow-up | 3,089 passed, 3 skipped; 36 smoke tests deselected | 114.47 s (117 s for the whole command), 1 run | — | 0 | — | plan T12c |
| 2026-10-27 | T28a | Real run of the Streamlit app (`streamlit run src/thematic_redundancy/app/main.py`, headless, port 8531), driven by AppTest and headless Chrome, offline, read only | Snapshot `20261002T224412Z`: T19's records and map (744 theses), T24a's 24 stable gaps and their members, the fichas | 43.1 s for the whole check (lock held); server healthy after 3.1 s (6.99 s cold); first page run 4.63 s; each headless Chrome visit 3.4 to 19.1 s (12 visits over 3 runs) | — | 0 | Both pages without exception; 0 of 3,196 person strings in 1.95 M rendered characters; port free after; no file written | plan T28 (t28a) |
| 2026-10-27 | T28a | Mutation check of the app (12 throwaway mutants against `tests/unit/app`) | `app/data.py`, `charts.py`, `ui.py`, `map_view.py` | 135.1 s | — | 0 | 12 of 12 killed, each file restored byte-identical | plan T28 (t28a) |
| 2026-10-27 | all | Test suite (`pytest -q`) in the t28a worktree, through the lock | 3,166 tests, plus 3 skipped; 36 smoke tests deselected | 182.38 s (lock held 185.3 s) | — | 0 | 3,164 passed; the 2 failures are the worktree-junction cases of `test_anonymize.py`, which pass in the main repository; 78 new tests (D: 3,166 − 3,088) | plan T28 (t28a) |
| 2026-10-27 | all | Test suite (`pytest -q`), in the main tree after T28a | 3,167 passed, 3 skipped; 36 smoke tests deselected | 105.63 s (lock held 108.7 s), 1 run | — | 0 | After T28a: 78 new tests (D: 3,167 − 3,089) | plan T28 |
| 2026-10-27 | T28b | Read-only probe: T24a's gap objects rebuilt from the saved records and assessed again by T24's stability filter | Snapshot `20261002T224412Z`: 184 gap rows, 5,592 member rows, 5 runs | 22.7 s (lock held) | — | 0 | 184 of 184 reasons and 31 of 31 supports equal; 24 stable | plan T28 (t28b) |
| 2026-10-27 | T28b | Real run of the query view: server health, AppTest with the real model on one invented proposal and two more, 12 real theses re-represented, offline, read only | The winner's 744 document vectors, T19's and T24a's records, the fichas, the mpnet model from the local cache | 122.3 s (lock held); first query 65.31 s cold, next 0.15 s and 0.20 s; model reload 10.94 s; index 0.24 s; represent median 0.754 s | — | 0 | Representation equal to the corpus (cosine 1 − 1.1e-14); 0 person strings; 0 proposal words in logs | plan T28 (t28b) |
| 2026-10-27 | T28b | The same proposal submitted through the real server in headless Chrome (DevTools protocol) | As above | 63.9 s (lock held); first query 40.52 s cold, next 1.00 s and 1.01 s | — | 0 | The server's log holds none of the proposal's words; ports free after | plan T28 (t28b) |
| 2026-10-27 | T28b | Mutation check of the query view (19 throwaway mutants) | `app/query.py`, `gap_supports.py`, `tables.py`, `query_view.py`, `ui.py` | 198.5 s, then 23.2 s | — | 0 | 17 of 19 killed, the 2 survivors killed after 2 tests were added | plan T28 (t28b) |
| 2026-10-27 | all | Test suite (`pytest -q`) in the t28b worktree, through the lock | 3,200 tests, plus 3 skipped; 36 smoke tests deselected | 224.97 s (lock held 228.5 s) | — | 0 | 3,198 passed; the 2 failures are the worktree-junction cases of `test_anonymize.py`; 34 new tests (D: 3,200 − 3,088 − 78) | plan T28 (t28b) |
| 2026-10-27 | all | Test suite (`pytest -q`) run by the independent verifier of T28b | 3,198 passed and 2 environmental failures, plus 3 skipped; 36 smoke tests deselected | 174.61 s, 1 run | — | 0 | No blocker; 8 of 8 checks passed | plan T28 (t28b) |
| 2026-10-27 | T28b | The parent's real run of the app (port 8551, headless Chrome, an invented proposal), offline, read only | Snapshot `20261002T224412Z`: the three pages | health 3.2 s; map page 6.7 s; gaps page 1.0 s; first query 31.9 s | — | 0 | Five sections present; the proposal absent from the server log; ports freed | plan T28 (t28b) |
| 2026-10-27 | all | Test suite (`pytest -q`), in the main tree after T28b | 3,201 passed, 3 skipped; 36 smoke tests deselected | 155.64 s (lock held 158.5 s), 1 run | — | 0 | After T28b: 34 new tests (D: 3,201 − 3,167) | plan T28 |
| 2026-10-27 | T22 | Read-only probe of T20a's scores (`pair_cosines` over the 130 drawn pairs), offline, through the lock | Snapshot `20261002T224412Z`: T20a's loader (fichas, both models' vectors, TF-IDF), `pairs.parquet` | 0.92 s to load; lock held 16.5 s | — | 0 | Bit-identical to `pairs.parquet` for the three representations | plan T22 (t22run) |
| 2026-10-27 | T22 | Dry run of the T22 command (`run_redundancy --dry-run --labels`), offline, through the lock and a read-only wrapper, synthetic labels in the scratch folder | 744 theses' vectors, T20a's 120 labeled pairs, T19's records | 1.77 s for `main()` (load 0.7, check 0.1, compute 0.1); lock held 6.2 s | — | 0 | Nothing written; stage 2 confirmed C4 at the NPMI step on synthetic labels | plan T22 (t22run) |
| 2026-10-27 | T22 | Mutation check of the T22 command (16 throwaway mutants against the three new test files), 1 batch and 1 rerun | `labels.py`, `stage_two.py`, `run_redundancy.py`, `results_store.py` | 178.5 s for 16; 14.7 s for 1 | — | 0 | 15 of 16 killed, then the survivor killed by an added test; each file restored byte-identical | plan T22 (t22run) |
| 2026-10-27 | all | Test suite (`pytest -q`), in the t22run worktree, through the lock | 3,213 passed and 2 failed (environmental, the worktree's `data` junction), plus 3 skipped; 36 smoke tests deselected | 218.43 s, 1 run (lock held 221.3 s) | — | 0 | After T22's run command: 127 new tests (D: 3,215 − 3,088) | plan T22 (t22run) |
| 2026-10-27 | all | Test suite (`pytest -q`) run by the independent verifier of T22's run command | 3,213 passed and 2 environmental failures, plus 3 skipped; 36 smoke tests deselected | 199.39 s, 1 run | — | 0 | No blocker; 7 of 7 checks passed | plan T22 (t22run) |
| 2026-10-27 | all | Test suite (`pytest -q`), in the main tree after T22's run command | 3,328 passed, plus 3 skipped; 36 smoke tests deselected | 173.48 s (lock held 176.7 s), 1 run | — | 0 | After T22's run command: 127 new tests | plan T22 |
| 2026-10-27 | T27 | Temporal-split dry run (`run_temporal --dry-run`), offline, through a read-only wrapper with the main repository as project root, 2 runs | Snapshot `20261002T224412Z`: T19's records, the 744 theses' inputs through T19's loader, the fichas and the snapshot's metadata; 5 UMAP fits on 458 × 768, 5 HDBSCAN runs, 5 c-TF-IDF fits, 1,430 assignments; nothing written | 113.3 s and 70.9 s for `main` (118.4 s and 76.9 s for the wrapper's process) | about 1 s per seed after the fixed costs | 0 | First run: seed 7's UMAP 29.7 s (JIT) and c-TF-IDF 75.9 s (BERTopic's import); second run 22.5 s and 41.8 s; the other seeds 0.74 to 1.01 s and 0.24 to 0.69 s | plan T27 |
| 2026-10-27 | T27 | Parity check of T19's path, in the dry-run wrapper | 1 UMAP fit and HDBSCAN mcs 10 on all 744 theses, seed 42 | 1.5 s and 1.8 s | — | 0 | Labels identical to T19's medoid run | plan T27 |
| 2026-10-27 | T27 | Mutation check of the temporal split (throwaway mutants against its 3 test modules), 2 batches | `evaluation/temporal.py`, `topic_map/temporal.py`, `temporal_store.py`, `run_temporal.py` | 200.7 s for 22; 160.9 s for 23 | — | 0 | 20 of 22, then 23 of 23 killed; the 2 survivors were equivalent and led to a code and a scenario change | plan T27 |
| 2026-10-27 | all | Test suite (`pytest -q`), in the t27 worktree after T27 | 3,198 tests, plus 3 skipped; 36 smoke tests deselected | 129.48 s (152 s for the whole command), 1 run | — | 0 | 110 new tests (D: 3,198 − 3,088); 3,196 passed, the 2 failures are the worktree's known `test_anonymize` cases | plan T27 |
| 2026-10-27 | all | Test suite (`pytest -q`) run by the independent verifier of T27's runner | 3,196 passed and 2 environmental failures, plus 3 skipped; 36 smoke tests deselected | 109.75 s, 1 run | — | 0 | No blocker; 8 of 8 checks passed | plan T27 |
| 2026-10-27 | all | Test suite (`pytest -q`), in the main tree after T27's runner | 3,438 passed, plus 3 skipped; 36 smoke tests deselected | 181.18 s (lock held 183.5 s), 1 run | — | 0 | After T27's runner: 110 new tests | plan T27 |

### Notes on the ledger

**T05, full download.** The span runs from the first to the last `downloaded_at` in
`pdfs/manifest.json`: 2026-10-02T23:49:23Z to 2026-10-03T12:18:20Z, 12 h 28 min 57 s
(44,937 s). It is an elapsed span, not continuous work: the completion times fall into four
active segments.

| Segment | Files | First to last completion (UTC) | Active time | GB |
|---|---:|---|---:|---:|
| The sample, kept by the full run | 10 | 23:49:23 to 23:51:04 | 101 s | 0.065 |
| Background run, stopped by a 2 h time limit | 447 | 23:55:26 to 01:55:08 | 7,182 s | 7.160 |
| Resume from the user's terminal | 126 | 02:37:46 to 03:14:01 | 2,175 s | 2.212 |
| Final run; the 3 `ConnectTimeout` errors were retried and downloaded | 148 | 11:47:19 to 12:18:20 | 1,861 s | 1.964 |
| **Total** | **731** | | **11,319 s (3 h 9 min)** | **11.401** |

- **Active time** runs from the first to the last completion within a segment. It leaves out
  the time before each segment's first file, so it slightly understates the work. The sample's
  extrapolation had forecast 2.5–4.5 h.
- **Requests** are not logged. The estimate counts 2 per downloaded PDF (the item's file
  listing and the file), as measured on the sample (20 for 10), plus 1 listing for the item
  without a thesis file: 731 × 2 + 1 = 1,463. It leaves out the retries of the 3 timeouts, any
  redirects, and the re-fetched partial file of the first interruption. Restricted items cost
  no request.

**T09, sample.** Seconds per PDF: 0.9, 1.5, 2.0, 7.4, 8.2, 8.9, 9.9, 18.0 and 71.2, then 355.3
for the fully scanned PDF (87 pages, 86 of them OCR). The first session ended after 7 PDFs and
the next one extracted the other 3, so the wall time is the sum of the per-PDF times.

- OCR averaged 4.6 s per page: 4.1 s on the scanned PDF and 7.1 s on the 17 pages of the other
  PDFs. Re-timed alone, 3 of the slowest pages took 3.8 s each.
- Work without OCR (hashing, text layer, header and footer removal, Parquet) took 0.0056 s per
  page.

**T09a, sample re-run.** The same 10 PDFs, all extracted again because the rules version went
from 1 to 2. Of the 1,233.8 s, OCR took 1,226.3 s: 11.9 s per OCR page. The scanned PDF took
13.2 s per page, and the other 17 OCR pages 5.4 s each. The remaining 7.5 s cover the other
work on 1,638 pages.

**T09, full run.** The estimate adds about 13 min of text work (141,076 pages × 0.0056 s) to
the OCR of about 1,960 pages, the D25 window. At 4–5 s per page OCR takes 2.2–2.7 h; at the
T09a speed, about 6.5 h. When the run ends, fill the row from
`results/extraction/20261002T224412Z/summary.json`: wall time, pages by source, OCR pages and
seconds, errors. If the run was split across sessions, record the elapsed span as well.

## 4. Data on disk

Measured on 2026-10-05 with `os.stat` (no file was hashed) and, for the repository,
`git ls-tree -r -l HEAD`.

| Location | Content | Bytes | Size |
|---|---|---:|---:|
| `data/raw/20261002T224412Z/metadata.jsonl` | 766 metadata records | 4,744,184 | 4.7 MB |
| `data/raw/20261002T224412Z/manifest.json` | snapshot manifest | 1,857 | 1.9 kB |
| `data/raw/20261002T224412Z/pdfs/*.pdf` | 731 thesis PDFs | 11,401,121,329 | 11.40 GB |
| `data/raw/20261002T224412Z/pdfs/manifest.json` | PDF manifest | 487,971 | 0.49 MB |
| `data/interim/20261002T224412Z/pages/` | page texts of 731 PDFs (parquet) and their manifest (T09) | 90,660,188 | 90.7 MB |
| `data/interim/20261002T224412Z/objectives.parquet` and `objectives_manifest.json` | objectives table of 731 PDFs and its manifest (T10; measured 2026-10-07) | 360,893 | 0.36 MB |
| `data/labels/objectives_check/20261002T224412Z/` | verification workbook (30,042 B) and `sample.json` (11,817 B) (T10; measured 2026-10-07) | 41,859 | 0.04 MB |
| `data/interim/20261002T224412Z/` fichas files | `fichas.parquet` (1,206,201 B), `exclusions.parquet` (3,868 B) and `fichas_manifest.json` (702 B) (T12; measured 2026-10-07) | 1,210,771 | 1.21 MB |
| `data/interim/pseudonym.key` | the pseudonym key (T08), created by the first T12 run | 32 | 32 B |
| `data/interim/20261002T224412Z/` chunk files | `chunks.parquet` (1,375,134 B) and `chunks_manifest.json` (1,336 B) (T13; measured 2026-10-15) | 1,376,470 | 1.38 MB |
| `data/interim/20261002T224412Z/embeddings/` | Document vectors (both variants) and section vectors for both models, the sections table and the manifest (T14; measured 2026-10-16) | 17,089,753 | 17.1 MB |
| `data/cache/chunk_vectors/` | Raw chunk vectors keyed by D27's cache key: MiniLM 8,337,728 B, mpnet 16,341,824 B (T14; measured 2026-10-16) | 24,679,552 | 24.7 MB |
| `data/interim/20261002T224412Z/tfidf/` | Full-clean tokens per section (321,359 B), the terms (246,664 B), both document matrices (1,183,621 B each) and the manifest (2,454 B) (T15; measured 2026-10-19) | 2,937,719 | 2.94 MB |
| `tessdata/*.traineddata` | `spa` and `eng`, tessdata_best | 28,970,788 | 29.0 MB |
| Hugging Face cache | `paraphrase-multilingual-MiniLM-L12-v2` | 479,729,050 | 479.7 MB |
| Hugging Face cache | `paraphrase-multilingual-mpnet-base-v2` (measured 2026-10-11) | 1,126,358,977 | 1.13 GB |
| `.venv/` | the locked environment, spaCy model included (54,361,316 B) | 1,907,548,544 | 1.91 GB |
| Repository at `7334e66` | 79 tracked files, of which `results/` holds 8 (371,040 B) | 1,479,941 | 1.48 MB |

`data/`, `tessdata/` and `.venv/` are gitignored. The Hugging Face cache lives outside the
repository, in `%USERPROFILE%\.cache\huggingface\hub`. Each of its two rows sums every file in
the model's cache folder; `huggingface_hub.scan_cache_dir` reports 40 B less for each, since it
leaves out the 40-byte `refs/main` file.

## 5. Monetary and external costs

- **Money: 0.** No cloud processing, no paid API, no paid license and no GPU. Every model and
  library is open and runs locally.
- **Network.** Every request was an anonymous, read-only GET:
  - the public UCSM repository (DSpace REST API): 14 metadata requests and about 1,463 PDF
    requests (estimate), at least 1 s apart; 11.40 GB of PDFs and 4.7 MB of metadata stored;
  - one-time downloads: the two tessdata files from GitHub, the MiniLM and mpnet models from
    Hugging Face (479.7 MB and 1.13 GB on disk), and the packages of `uv sync` from PyPI, the
    PyTorch CPU index and GitHub (the spaCy model).
- **Electricity: not measured.** An estimate would take energy (kWh) = average power draw (W)
  × wall time (h) / 1,000, with the draw read from a plug-in power meter during a
  representative run, or bounded by the processor's rated power. Any such figure is an
  estimate and must be labeled so.
- **People's time.** Manual verification (T10), labeling (T21) and expert sessions (T30) cost
  hours of work; record them in the ledger like machine time.

## 6. Open measurements

- [x] T09 full extraction: wall time, OCR pages and seconds per OCR page, errors, from the run
  summary (2026-10-05).
- [x] Size of `data/interim/20261002T224412Z/pages/` after T09 (2026-10-05).
- [x] T10 manual verification time for the 60 theses of the workbook: about 2 h, reported by
  the user and not timed, against the estimate of 4 to 6 h (2026-10-07).
- [x] mpnet model download size and time: 1,126,358,977 B and 287.6 s, the first load
  included (2026-10-07, under U8).
- [x] T14 embeddings for the 2 models: 549.0 s from an empty cache (encoding 127.2 s with
  MiniLM and 306.3 s with mpnet, 41.0 and 17.0 texts/s), 17.1 MB of vectors and a 24.7 MB
  chunk-vector cache (2026-10-16, the user's run).
- [x] T19 experiment grid: 417.0 s for the whole real run (389.2 s in the manifest), 371 clusterings, 2026-10-20, run by the parent.
- [ ] T21 labeling workload: annotator hours per pair and in total.
- [x] T28 app latency: server healthy after 3.1 to 3.6 s; the first query after a start 31.9 to
  65.3 s (the model's cold load), later queries 0.15 to 0.20 s in process and about 1 s in the
  browser (a 1-second polling floor) (2026-10-27, T28a and T28b).
- [ ] T29 Docker build: build time and image size.
- [x] Time the EDA script (T07): 17.4 s and 16.5 s for the whole command, offline, in the
  regeneration after T07a (2026-10-13).
- [ ] Optional: time the stopword-candidates (T11) script on its next run.
