# Thematic Redundancy

Integrity tools such as Turnitin detect textual overlap, but not thematic redundancy: two works
that tackle the same problem with the same approach in different words. This project is a local,
reproducible Python prototype that maps the research lines of the public defended theses of five
engineering programs (FCIFF, 2021–2026) at Universidad Católica de Santa María (UCSM). It flags
direct redundancy for new proposals, surfaces research gaps, and offers a Streamlit interface for
students and advisors.

## Requirements

- Python 3.13 (3.14 is not supported yet: gensim ships no cp314 Windows wheel).
- [uv](https://docs.astral.sh/uv/).
- Windows 11 or Linux. PyTorch runs on CPU only.

## Setup

```sh
uv sync
```

The first sync downloads CPU-only PyTorch from the PyTorch index and the spaCy Spanish model
(`es_core_news_md`) from GitHub, so expect a download of over 1 GB.

The run configuration lives in `config/default.yaml` and is validated on load.

## OCR assets

Scanned theses have pages without a text layer. PyMuPDF reads them with the Tesseract engine it
bundles, so no Tesseract installation is needed, only its language files. Fetch them once per
machine:

```sh
uv run python -m thematic_redundancy.extraction.ocr_assets
```

The command downloads `spa.traineddata` and `eng.traineddata` from Tesseract's
[`tessdata_best`](https://github.com/tesseract-ocr/tessdata_best) models (about 29 MB in total)
into `tessdata/` (the `paths.tessdata_dir` setting), which is gitignored. It records the source,
size, SHA-256 and download time of each file in `tessdata/manifest.json`. Later runs check the
files against the manifest and download only the ones that are missing or changed; `--force`
downloads them all again.

## Corpus snapshot

The corpus comes from the public DSpace 7 REST API of the UCSM repository. Harvest the
metadata of the program collections once per snapshot:

```sh
uv run python -m thematic_redundancy.corpus.harvest
```

The command lists every item of each program collection (`snapshot.programs`) issued from
`snapshot.year_start` to `snapshot.year_end`, and writes a new snapshot under `data/raw/`:

- `<snapshot_id>/metadata.jsonl`: one JSON object per item, with its uuid, handle, program,
  collection, harvest time, and metadata.
- `<snapshot_id>/manifest.json`:
  - the source URL and query, and the harvest time;
  - the item counts per program and per `renati.type`, plus the faculty-wide total;
  - the number of items issued after the harvest date, and the dropped DNI keys;
  - the SHA-256 of `metadata.jsonl`.

How a snapshot behaves:

- **Id.** It defaults to the UTC harvest time, such as `20261002T150405Z`. `--snapshot-id`
  chooses another name.
- **Immutable.** A snapshot is never overwritten. It appears complete or not at all.
- **All types kept.** Every item type stays in the snapshot; the theses are selected later,
  with an exclusion log.
- **Reconciled.** If a listing does not match the repository's own total, or lists an item
  twice, the harvest stops before anything is written.

The harvest is polite to the server:

- It sends one request at a time, at least `repository.request_interval_seconds` apart
  (1 s or more).
- Each request carries a User-Agent that names the project, and a timeout.
- After HTTP 429, HTTP 5xx or a network error, it retries at most `repository.max_retries`
  times, with exponential backoff.
- It reads metadata only. The PDFs come from a separate command, below.

The frozen snapshot `20261002T224412Z` is described in its data card,
[`results/snapshot/20261002T224412Z/data_card.md`](results/snapshot/20261002T224412Z/data_card.md):
freeze digests, counts by program and year, access rights, PDF statuses, and known issues.

## Thesis PDFs

Once a snapshot exists, fetch the PDF of each of its theses:

```sh
uv run python -m thematic_redundancy.corpus.download_pdfs
```

The command reads the snapshot harvested last (`--snapshot-id` picks another one) and takes
its theses: the items whose `renati.type` fragment is `snapshot.thesis_type` (`#tesis`). It
writes into `data/raw/<snapshot_id>/pdfs/`:

- `<item uuid>.pdf`: the thesis PDF, named after its item. Original file names are never
  used or stored, because they may hold author names.
- `manifest.json`:
  - one entry per thesis: its status and reason, the repository file it came from, the size
    and checksum that the repository lists, and the MD5 and SHA-256 of the file on disk;
  - counts by status and by program, and the total size of the files.

**Which file.** The `ORIGINAL` bundle of a UCSM thesis may hold three PDFs: the thesis, a
similarity report (`*.RT.pdf`) and a publication authorization form (`Autorización_*.pdf`).
The command drops the last two, ignoring letter case and accents.

- One PDF left: it is downloaded.
- None left: the thesis is recorded as `no_thesis_file`, with nothing downloaded.
- Several left: it is recorded as `ambiguous`, with nothing downloaded.

How a run behaves:

- **Restricted items.** An item whose `dc.rights` is embargoed, restricted or metadata-only
  access is recorded as `restricted` (`rights_restricted`) without any request.
  - HTTP 401 and 403 also give `restricted`, and HTTP 404 gives `not_found`.
  - None of them is asked again, in this run or in later ones.
- **Verified.** A download goes to `<item uuid>.pdf.part`. It is renamed into place only when
  it starts with `%PDF` and matches the size and checksum (MD5) that the repository lists.
  Otherwise it is deleted and recorded as `integrity_error`.
- **Resumable.** The manifest is rewritten atomically after every thesis, so a run can stop
  at any point. A rerun:
  - keeps every PDF whose size and SHA-256 still match the manifest, without any request;
  - retries the theses recorded as `error` or `integrity_error`.
- **Bounded.** `--limit N` takes the first N theses in snapshot order, and `--per-program N`
  the first N of each program.
- **Guarded.** Only one run at a time may use a PDF directory, and a run stops after 3 failed
  theses in a row, since that points at the server or the network.

It is as polite as the harvest: one request at a time, at least
`repository.request_interval_seconds` apart, with the same User-Agent and retry policy. It
follows redirects only within the repository's host. It sends about two requests per thesis:
one to list the item's files and one to download the thesis.

## PDF text extraction

Once the PDFs are on disk, extract their text page by page. A full run takes about 3 hours
on the development laptop, almost all of it OCR, and can stop and resume at any point, so
start it in your own terminal:

```sh
uv run python -m thematic_redundancy.extraction.extract_text
```

The pipeline needs PDF text only to find the objectives (T10); titles and abstracts come from
the metadata. The command reads the snapshot harvested last (`--snapshot-id` picks another
one), takes every PDF that `pdfs/manifest.json` lists as on disk, and writes into
`data/interim/<snapshot_id>/pages/`, which is gitignored:

- `<item uuid>.parquet`: one record per page. It holds the page text, its source
  (`text_layer`, `ocr` or `empty`), its length, a low-text flag, whether OCR was attempted or
  failed, and how many running lines were removed.
- `manifest.json`: one entry per PDF, plus a summary. An entry holds the status, the SHA-256 of
  the PDF and of its page file, the settings fingerprint, pages by source, OCR pages tried and
  failed, lines removed, library warnings and seconds. It holds counts only, never text.

What happens to each page:

| Step | Rule | Settings |
|---|---|---|
| Text layer | Read in reading order: blocks top to bottom, a blank line between blocks | — |
| Low-text page | Under 50 characters once stripped, the data card's measure | `extraction.min_text_chars` |
| OCR | Only low-text pages among the first 100 pages of a PDF; the whole page at 300 dpi in `spa+eng` | `extraction.ocr_window_pages`, `ocr` |
| Running lines | A line among the first or last 3 lines of a page that repeats on at least 30% of the pages with text, ignoring case, spacing and numbers, so `Página 12` matches `Página 13`. Standalone page numbers at those edges go too. A line of digits only goes only when it is a page number (at most 3 digits), so a year such as `2024` stays | `extraction.header_footer` |

**Why an OCR window (D25).** The objectives come early: the first page that mentions
"objetivo general" has a median index of 7 and a P99 of 102. The window cuts OCR from about
8,600 low-text pages to about 1,960. A low-text page outside it keeps its few text-layer
characters, or is `empty`. To OCR more, widen the window and run the command again.

How a run behaves:

- **Resumable.** The manifest is rewritten atomically after every PDF. A rerun skips each PDF
  whose page file is unchanged and was made from the same PDF, with the same settings and
  extraction rules. It extracts a PDF again when the PDF manifest's record of the PDF (its
  SHA-256), a setting, the extraction rules version or the page file changed. The skip trusts
  the PDF manifest and does not hash the PDFs on disk, because the snapshot is frozen; to
  verify the files on disk, use the combined PDF digest in the data card's freeze record.
- **Versioned rules.** The settings fingerprint includes `EXTRACTION_VERSION`
  (`extraction/text_manifest.py`). It is raised whenever a code change alters the page
  records, such as a page-text rule, so the next run extracts every PDF again. Version 2 keeps
  an edge line of digits only, such as a year, unless it is a page number.
- **Orphaned entries.** An entry whose PDF `pdfs/manifest.json` no longer lists as on disk,
  such as a thesis restricted since, is reported as `orphaned` when the run starts and in the
  summary. It is kept, and so is its page file, but it is left out of every other total.
  Nothing is deleted automatically.
- **Isolated failures.** A PDF that cannot be read is recorded as `error` (`unreadable_pdf`,
  `pdf_missing` or `pdf_changed`) and the run goes on; the next run tries it again. A page whose
  OCR fails keeps its text layer. MuPDF's warnings are counted per PDF instead of printed.
- **Interruptible.** Ctrl+C stops the run without half-written files; run the command again to
  resume.
- **Guarded.** Only one run at a time may use a pages directory.
- **Bounded.** `--limit N` takes the first N PDFs in snapshot order, and `--only UUID ...` the
  PDFs of those items.
- **Quiet about text.** Each PDF prints one line of counts, with an ETA, and the run ends with a
  summary. `--summary-out results/extraction/<snapshot_id>/summary.json` also writes the
  manifest's settings and counts, without items or text. Tesseract may print short
  diagnostics such as `Line cannot be recognized!!`; they hold no text and need no action.

Every PDF is extracted, including the D24 duplicates and the wrong file of item 11777; choosing
which text a thesis uses is the job of the fichas (T12).

## Objectives

Find the general and specific objectives of every thesis in its extracted page text (T10).
There is no OCR at this step, so the run takes about 6 seconds for the 731 PDFs:

```sh
uv run python -m thematic_redundancy.extraction.locate_objectives \
    --summary-out results/objectives/<snapshot_id>/summary.json
```

It reads the snapshot harvested last (`--snapshot-id` picks another one), checks that each page
file is still the one extracted (its SHA-256), and rebuilds these files from scratch:

| File | Content | In git |
|---|---|---|
| `data/interim/<snapshot_id>/objectives.parquet` | One row per PDF: the status (`extracted` or `not_found`, as in the ficha's `objectives_status`), the general objective, the specific objectives (one item per line), the page range, the heading pattern, the quality flags, character counts, and the SHA-256 of the page file read | No: it holds thesis text |
| `data/interim/<snapshot_id>/objectives_manifest.json` | The settings, the locator version, the table's SHA-256, the seconds taken, and the counts | No |
| `results/objectives/<snapshot_id>/summary.json` | The same manifest, written by `--summary-out`: numbers only | Yes |

How the locator (`extraction/objectives.py`) reads a document:

| Step | Rule |
|---|---|
| Matching | Lines are compared in lower case with accents folded; the objectives keep their original text, with whitespace collapsed |
| Headings | `Objetivo general`, `Objetivos generales` or `Objetivo principal`, with optional numbering (`1.3.1.`, `a)`, `II.`) and a trailing `:`. Also a line such as `El objetivo general es el siguiente:`, and a bare `General` below an `Objetivos` heading |
| Skipped pages | Contents pages (dotted leaders, page numbers), dedication and acknowledgement pages. Abstract pages are a fallback only, flagged `abstract_only` |
| General objective | The first general heading in the body. Its text runs to the first line that ends a sentence, or to the next section, across page breaks |
| Specific objectives | The first `Objetivos específicos` (or `secundarios`) heading after it, one item per line, up to the next section |
| Next section | Another objectives heading, multi-level numbering such as `1.4`, a roman-numeral or `CAPÍTULO` heading, or a keyword such as `Justificación`, `Hipótesis` or `Marco teórico` |
| Bounds | Texts are cut at `objectives.general_max_chars` (1,200) and `objectives.specific_max_chars` (4,000), and flagged; a general objective under `objectives.min_chars` (40) is flagged too |

Quality flags: `general_too_short`, `general_too_long`, `specific_too_long`, `specific_missing`,
`abstract_only`, `block_fallback` (first sentence of an `Objetivos` block) and `ocr_page`.
`LOCATOR_VERSION` in `extraction/objectives.py` is raised whenever a rule change alters the
results; the manifest records it.

### Manual verification (O05)

The objectives are used only if at least 90% of a stratified sample of about 60 theses is
judged correct; otherwise the pipeline falls back to title and abstract.

1. **Draw the sample and write the workbook, once:**

   ```sh
   uv run python -m thematic_redundancy.extraction.locate_objectives --make-sample
   ```

   This writes `objectives_check.xlsx` and `sample.json` into
   `data/labels/objectives_check/<snapshot_id>/` (`objectives.verification_sample`). The draw takes
   12 theses per program with a fixed seed, from the theses with a valid own PDF. It leaves out
   the wrong file of item 11777 and the larger uuid of each D24 duplicate pair, and it includes
   `not_found` documents in proportion. An existing workbook is never overwritten.
2. **Fill the workbook in Excel.** The `Guía` sheet explains the task in Spanish. Each row of
   `Verificación` links to the repository item and to the local PDF and gives the PDF pages. Pick
   one verdict for the general objective and one for the specific objectives from the
   drop-down lists. Count 4 to 6 minutes per row, so 4 to 6 hours for 60 rows; it can be done in
   several sessions. Note the time you spend for the cost ledger.
3. **Import it and compute the accuracy:**

   ```sh
   uv run python -m thematic_redundancy.labeling.objectives_check \
       --report-out results/objectives/<snapshot_id>/verification.json
   ```

   The import refuses the workbook, and lists every problem, if a verdict is blank or not
   allowed or a row was lost or repeated. A file that is not a readable xlsx workbook is
   reported in one `error:` line. Accuracy is the rows judged `Correcto` over the judged rows,
   with a Wilson 95% interval, overall and per program; `Parcial` counts as not correct.
   The report holds numbers only: the general-objective accuracy with its interval, overall and
   per program, the overall verdict counts and `meets_target`, the specific-objectives
   accuracy, and the workbook's SHA-256. The official report of snapshot `20261002T224412Z` is
   `results/objectives/20261002T224412Z/verification.json`.

## Metadata profile

Profile the metadata of a snapshot, offline:

```sh
uv run python experiments/eda_metadata.py
```

The script profiles the snapshot harvested last (`--snapshot-id` picks another one) and writes
into `results/eda/<snapshot_id>/`:

- `metadata_profile.json`: the whole profile, with its provenance (snapshot and configuration
  checksums, git commit, tokenizer).
- `summary.md`: tables and observations for chunking, title cleaning and topic modeling.
- Four figures: theses per program and year, abstract and title lengths in tokens, and the most
  frequent keywords.

How it works:

- **Inputs.** It reads only `metadata.jsonl` and `manifest.json`, and refuses a snapshot whose
  metadata no longer matches its manifest. The profiling logic lives in
  `src/thematic_redundancy/corpus/profile.py`.
- **Tokens.** It counts tokens, special tokens included, with the tokenizer of
  `paraphrase-multilingual-MiniLM-L12-v2` from the local Hugging Face cache. The
  [smoke test](#smoke-test) fetches that model once.
- **Aggregates only.** No title, abstract or person's name appears in the outputs. Before
  writing, the script searches every output for each title, abstract, abstract sentence,
  author, advisor and juror name, and ORCID of the snapshot. A single hit stops it with nothing
  written.

## Text cleaning

Two cleaning variants live in `src/thematic_redundancy/preprocessing/`:

- **Light** (`clean_light`), the input of the embedding models. It joins hard-wrapped lines and
  keeps a line break only where a sentence or a paragraph ends. It also rejoins words split by
  a hyphen at a line end, collapses whitespace, and straightens curly quotes. Casing, accents,
  ñ and punctuation stay, and it needs no language model.
- **Full** (`clean_full`, `clean_full_batch`), the input of TF-IDF, c-TF-IDF and NPMI. It
  light-cleans and lowercases the text, then lemmatizes it with spaCy `es_core_news_md`. It
  drops spaCy's Spanish stopwords, the domain stopwords, punctuation, numbers, one-character
  tokens, and legal suffixes such as S.A.C. or E.I.R.L. It returns lemma tokens, which
  `join_tokens` turns into one string. Accents and ñ stay.

`strip_title_suffix` cuts the trailing place and year of a title, such as `, Arequipa 2023`,
and reports what it cut for the quality notes.

The domain stopwords are in `config/stopwords_domain_es.txt`: genre and boilerplate lemmas
such as `tesis` or `arequipa`, picked by hand. The file's header states the criteria. The
candidates come from this script, which runs offline:

```sh
uv run python experiments/domain_stopwords.py
```

It writes `results/eda/<snapshot_id>/stopword_candidates.json`: each lemma found in at least 5%
of the theses, with its document count. Like the metadata profile, it writes nothing if the
output holds a title, an abstract, a person's name or an ORCID.

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## Smoke test

A separate suite checks that the locked stack works together on this machine. It covers spaCy,
the embedding model, UMAP with K-means/HDBSCAN inside BERTopic, the metrics, plotly/Streamlit,
the stats and file libraries, and OCR of a page without a text layer. The OCR check needs the
files from [OCR assets](#ocr-assets) and fails with the fetch command when they are missing.
The default test run excludes the suite. Run it with:

```sh
uv run pytest -m smoke
```

The first run downloads the embedding model `paraphrase-multilingual-MiniLM-L12-v2` (about
0.5 GB) into the Hugging Face cache. Later runs reuse that copy, take under a minute, and also
work offline with `HF_HUB_OFFLINE=1`.

## Data policy

- `data/` and `tessdata/` are never committed; both are gitignored and excluded from Docker builds.
- The theses are licensed CC BY-NC-ND 4.0. They are used for research only, their texts are not
  redistributed, and every displayed thesis links to its repository item.
- Thesis PDFs are saved under the uuid of their item. Their original file names, which may hold
  author names, are never stored.
- DNI fields (every metadata key ending in `.dni`, such as `renati.author.dni` and
  `renati.advisor.dni`) are dropped at ingestion, before anything is written. The snapshot
  manifest lists the dropped keys.
- Author names are never shown in the app or in reports, and user queries are never persisted.
- Fichas never store a person's name. Advisors and authors appear only as codes keyed by a local
  secret, `data/interim/pseudonym.key`, which is created on first use and never committed.
  Without it, the codes can be neither re-derived nor reversed.
- `results/` holds aggregates only. The metadata profile refuses to write an output that holds a
  title, an abstract, a person's name or an ORCID from the snapshot.
- Extracted page texts live only under the gitignored `data/interim/`. The extraction manifest
  and its summary hold counts only.
- The objectives table and the verification workbook hold thesis text, so they live only under
  the gitignored `data/`; the configuration refuses a workbook directory outside
  `paths.data_dir`. The workbook replaces every author, advisor and juror name of the snapshot
  with `[nombre omitido]`. The objectives summary and the accuracy report hold numbers only.

## Plan

Scope, decisions, and the task checklist live in
[`odd/tasks/thematic-redundancy-system.md`](odd/tasks/thematic-redundancy-system.md).
Processing times, request counts, data sizes and costs are recorded in
[`docs/processing_costs.md`](docs/processing_costs.md), one row per long or measured operation.
