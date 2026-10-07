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
| `tessdata/*.traineddata` | `spa` and `eng`, tessdata_best | 28,970,788 | 29.0 MB |
| Hugging Face cache | `paraphrase-multilingual-MiniLM-L12-v2` | 479,729,050 | 479.7 MB |
| Hugging Face cache | `paraphrase-multilingual-mpnet-base-v2` | not downloaded yet | pending (T13) |
| `.venv/` | the locked environment, spaCy model included (54,361,316 B) | 1,907,548,544 | 1.91 GB |
| Repository at `7334e66` | 79 tracked files, of which `results/` holds 8 (371,040 B) | 1,479,941 | 1.48 MB |

`data/`, `tessdata/` and `.venv/` are gitignored. The Hugging Face cache lives outside the
repository, in `%USERPROFILE%\.cache\huggingface\hub`.

## 5. Monetary and external costs

- **Money: 0.** No cloud processing, no paid API, no paid license and no GPU. Every model and
  library is open and runs locally.
- **Network.** Every request was an anonymous, read-only GET:
  - the public UCSM repository (DSpace REST API): 14 metadata requests and about 1,463 PDF
    requests (estimate), at least 1 s apart; 11.40 GB of PDFs and 4.7 MB of metadata stored;
  - one-time downloads: the two tessdata files from GitHub, the MiniLM model from Hugging Face
    (479.7 MB on disk), and the packages of `uv sync` from PyPI, the PyTorch CPU index and
    GitHub (the spaCy model).
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
- [ ] T10 manual verification time for about 60 theses.
- [ ] mpnet model download size and time (T13; needs U8).
- [ ] T14 embeddings for the 2 models: wall time, chunks per second, vector file sizes.
- [ ] T19 experiment grid: time per configuration and in total.
- [ ] T21 labeling workload: annotator hours per pair and in total.
- [ ] T28 app latency: start-up time and time per query.
- [ ] T29 Docker build: build time and image size.
- [ ] Optional: time the EDA (T07) and stopword-candidates (T11) scripts on their next run.
