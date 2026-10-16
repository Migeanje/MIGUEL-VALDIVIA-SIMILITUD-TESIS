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
- [ ] T19 experiment grid: time per configuration and in total.
- [ ] T21 labeling workload: annotator hours per pair and in total.
- [ ] T28 app latency: start-up time and time per query.
- [ ] T29 Docker build: build time and image size.
- [x] Time the EDA script (T07): 17.4 s and 16.5 s for the whole command, offline, in the
  regeneration after T07a (2026-10-13).
- [ ] Optional: time the stopword-candidates (T11) script on its next run.
