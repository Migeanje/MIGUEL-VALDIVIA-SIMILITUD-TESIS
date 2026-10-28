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

How the files are checked:

- **Pinned.** `spa.traineddata` and `eng.traineddata` must be exactly the files that the OCR
  was validated with, fetched on 2026-10-02: 13,570,187 and 15,400,601 bytes, with the
  SHA-256 values in `PINNED_TRAINEDDATA` (`extraction/ocr_assets.py`). A download that
  differs is refused, and a kept file that differs is downloaded again. If `tessdata_best`
  ever changes one of them, review the new file and update the pin.
- **Other languages.** A file of a language without a pin must be laid out as a Tesseract
  language file: an entry count, then an offset table whose entries lie after the table and
  inside the file. An error page or a cut download is refused.
- **Errors.** A failed download or a configuration error is reported on one `error:` line,
  with exit code 1.

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
- **Faculty total enforced.** The harvest also counts the items of the whole faculty
  community under the same year filter. If the program collections add up to another
  number, a program collection may be missing from `snapshot.programs`, so it stops before
  anything is written and reports both totals and their difference. Once the difference is
  checked and known to be out of scope, run it again with `--accept-faculty-difference D`,
  where D is the faculty total minus the programs' sum that it reported (negative when the
  faculty holds fewer items). Any other difference is still refused, so a later change on
  either side stops the harvest again. A snapshot whose two totals differ in its manifest
  therefore records an accepted difference.

**Paging.** The search pages by offset and sorts by `dc.date.issued` alone, so the server
decides the order of items that share a date on each page request. If that order changed
between two requests, a listing would repeat one item and skip another. Both are caught: an
item listed twice stops the harvest with a message that names the item and its two pages,
and a listing that holds the reported total without a repeat cannot miss an item. One case
remains unseen: an item withdrawn while another is added to the same collection between two
page requests keeps the total and repeats nothing. The snapshot `20261002T224412Z` is
unaffected (766 distinct items, each listing equal to its total, the programs equal to the
faculty total). A future harvest can narrow or rule out that case, after checking against
the live server that it works as intended:

- list each collection one issue year at a time, through the year filter that the harvest
  already sends, and check that each year's listing fits in one page, which needs no paging;
- or harvest twice and compare the item uuids of both runs.

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
  Otherwise it is deleted:
  - a body that is not a PDF, usually an error page, is recorded as `integrity_error`;
  - so is a body longer than listed, even when its listed part is right: only that part
    could be compared, so it is retried like any other failure;
  - a PDF shorter than listed, or of the listed size with another checksum, is downloaded
    once more. If the second download sends the very same bytes, the repository's listing
    looks stale and the thesis is recorded as `unverifiable`, with reason `size_mismatch` or
    `checksum_mismatch`; otherwise as `integrity_error`.
  - A checksum in an algorithm that cannot be verified is `unverifiable`
    (`unsupported_checksum_algorithm`), with nothing downloaded.
- **Resumable.** The manifest is rewritten atomically after every thesis, so a run can stop
  at any point. A rerun:
  - keeps every PDF whose size and SHA-256 still match the manifest, without any request;
  - keeps a PDF on disk that matches the repository's current listing, also after an earlier
    failure or under another bitstream. A thesis refused with HTTP 401, 403 or 404, or
    closed by its `dc.rights`, is not asked again, so a file on disk does not change it;
  - retries the theses recorded as `error` or `integrity_error`;
  - fetches an `unverifiable` thesis again only once the repository lists its file
    differently, or with `--recheck-unverifiable`, which downloads each one again (twice at
    most) even when its listing is unchanged. A thesis whose checksum cannot be verified
    (`unsupported_checksum_algorithm`) stays `unverifiable`, that option included, until the
    repository lists another checksum; check it by hand.
- **Bounded.** `--limit N` takes the first N theses in snapshot order, and `--per-program N`
  the first N of each program.
- **Guarded.** Only one run at a time may use a PDF directory. A run stops:
  - after 3 failed theses in a row (`error` or `integrity_error`), since that points at the
    server or the network;
  - after 3 theses in a row that its own downloads found `unverifiable`, since so many stale
    listings in a row more likely mean that the server, or a proxy in front of it, sends
    wrong files. Check what it sends before running the command again. An
    `unverifiable` thesis without a download in this run (an unknown checksum algorithm, or
    one kept from an earlier run) neither counts nor breaks that run of theses; any other
    outcome breaks it, and an `unverifiable` thesis also breaks a run of failures.
- **Exit code.** 0 when no thesis looked at needs anything more; 1 after a fatal error, when
  a thesis failed (run the command again), or when a thesis is `unverifiable` (the closing
  lines then say how many and why, one line for stale listings and one for checksums that
  cannot be verified, for a person to check them); 130 after an interruption.
- **One snapshot.** A PDF manifest that belongs to another snapshot is refused before any
  request.

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

## Fichas dataset

Build one ficha per harvested record (T12): the corpus table that every later step reads. The
build takes about a second; the quality report adds about 9 seconds for the spaCy re-check:

```sh
uv run python -m thematic_redundancy.corpus.build_fichas \
    --report-out results/fichas/<snapshot_id>
```

It reads the snapshot harvested last (`--snapshot-id` picks another one), its PDF manifest, the
page-text manifest and the objectives table, whose SHA-256 must match its manifest. It writes
only when every check passes:

| File | Content | In git |
|---|---|---|
| `data/interim/<snapshot_id>/fichas.parquet` | One ficha per record: identity, title, abstract, objectives, metadata, person codes, PDF status, inclusion and quality notes | No: it holds thesis text |
| `data/interim/<snapshot_id>/exclusions.parquet` | The excluded records: item uuid, doc code, handle, program, type, reason and `duplicate_of`; no text | No |
| `data/interim/<snapshot_id>/fichas_manifest.json` | The SHA-256 of the inputs and outputs, the counts and the seconds | No |
| `results/fichas/<snapshot_id>/quality_report.json` and `.md` | Written by `--report-out`: counts, public handles and lemmas only | Yes |

The two tables are staged, moved into place together, and the manifest is written last, as the
commit point. Later steps read the fichas with `load_fichas` (`corpus/build_fichas.py`), which
refuses tables that differ from the manifest's SHA-256 values, as after a stopped build, and
parses the same bytes that it checked. Ctrl+C before the moves leaves the previous outputs;
once the moves have begun, the `interrupted:` line says that the tables may not match the
manifest and that the command must run again. When `--report-out` fails after the build, the
`error:` line says that the fichas outputs were written and the quality report was not.

| Rule | Decision |
|---|---|
| Inclusion (D01) | The exact `renati.type` fragment `#tesis`. Any other type is excluded as `non_thesis_type` and keeps its raw type |
| Duplicates (D24) | Records with the same title, abstract, authors, advisor and issue date (whitespace collapsed, case folded) are one thesis. The one accessioned first (`dc.date.accessioned`) stays, and a tie goes to the smaller item uuid; the others are excluded as `duplicate`, with `duplicate_of` |
| Objectives (O05) | A thesis with its PDF on disk takes the locator's row: the general objective, a blank line, then the specific objectives. Without a PDF, or with a PDF listed in `fichas.wrong_pdfs`, it is `no_pdf` and represented by its title and abstract |
| People | The advisor code comes from a valid ORCID iD, otherwise from the normalized name; a bad ORCID iD leaves a quality note. Authors get one code each. No name is stored. The key `data/interim/pseudonym.key` is created once and never overwritten: never commit or share it |
| Checks | Unique doc codes, no PDF shared by two included theses, and the counts declared in `fichas.expected`. Any failure is one `error:` line, and nothing is written |

The report of snapshot `20261002T224412Z` is `results/fichas/20261002T224412Z/quality_report.md`.
It also re-checks the domain stopwords against the objectives text: how many theses hold each
listed lemma, and the frequent lemmas that the list lacks. Lemmas shaped like an infinitive are
flagged as candidates only; `config/stopwords_domain_es.txt` changes only by hand.

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
  [smoke test](#smoke-test) fetches that model once. When the tokenizer of the other configured
  model, `paraphrase-multilingual-mpnet-base-v2`, is in the cache too, it counts every text
  again, and `summary.md` says whether the counts hold for both models (D09). A cached
  tokenizer with another vocabulary size, as a partly downloaded cache gives, is reported as
  such, and its counts are not compared.
- **Aggregates only.** No title, abstract or person's name appears in the outputs. Before
  writing, the script searches every output for each title, abstract, abstract sentence,
  author, advisor and juror name, and ORCID of the snapshot. A single hit stops it with nothing
  written. The search reads the bytes of each file, so text drawn inside a figure is outside
  it; the figures draw only fixed labels and values that `metadata_profile.json` also holds.
- **Writing.** All outputs are staged and flushed before the first one is renamed into place,
  so a failure while they are staged leaves the previous outputs. A rename cannot be undone: if
  one fails, as on Windows while a viewer holds a figure open, or the run is interrupted
  between two renames, the outputs renamed before it are new and the others are the previous
  ones. The script then says so, and running it again writes them all. Other files in the
  folder, such as `stopword_candidates.json`, stay as they are.

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
such as `tesis` or `arequipa`, picked by hand. The file's header states the criteria. A test
pins the 89 lemmas decided in D26, so any change to a lemma needs a new decision in the plan.
The candidates come from this script, which runs offline:

```sh
uv run python experiments/domain_stopwords.py
```

It writes `results/eda/<snapshot_id>/stopword_candidates.json`: each lemma found in at least 5%
of the theses, with its document count. Like the metadata profile, it writes nothing if the
output holds a title, an abstract, a person's name or an ORCID.

## Gap run

Run the gap rules (a) to (d) and their stability filter (T24a, D32) on the records of the T19
experiment, with the values of `gaps` in `config/default.yaml`:

```sh
uv run python -m thematic_redundancy.gaps.run_gaps --dry-run
uv run python -m thematic_redundancy.gaps.run_gaps
```

It uses the snapshot harvested last (`--snapshot-id` picks another one) and refits nothing. It
reads T19's `manifest.json`, `selection.json`, `topics.parquet` and the gitignored
`labels.parquet`, each checked against the SHA-256 that T19's manifest records; the document
vectors through T19's loader, whose input manifests must be the ones T19 ran on; the fichas, for
each thesis's program and year; `config/career_affinity.yaml` (U4); and the snapshot's metadata,
for the privacy guard. Rule (a)'s size cut and NPMI floor are computed once over the winner's
seed runs, and the stability filter assesses the gaps of T19's medoid run (D30). `--dry-run`
computes and prints everything (the cuts, the gaps per run, the stable gaps and the sensitivity
table), applies the privacy guard and writes nothing; on snapshot `20261002T224412Z` it takes
about 10 seconds. A real run refuses a dirty git tree outside `results/gaps/<snapshot_id>/`,
holds that directory's run lock and writes:

| File | Content | In git |
|---|---|---|
| `results/gaps/<snapshot_id>/cuts.json` | Rule (a)'s size cut and NPMI floor, with the topics behind them | Yes |
| `results/gaps/<snapshot_id>/gaps.parquet` | One row per gap of every run: rule, run, topics as `(run_id, topic)`, programs and years with their counts, the rule's counts and shares, the affinity score, and for the reference run's gaps their support and stability. No doc code, no term | Yes |
| `results/gaps/<snapshot_id>/sensitivity.parquet` | D32's sensitivity table beside the declared values: the thesis plan's literal rules (a) and (b), `rare_max_share` 0.01 and `stability_min_overlap` 0.75. Never used for selection | Yes |
| `results/gaps/<snapshot_id>/summary.md` | The stable gaps with the T19 terms of their topics, the counts per rule and the sensitivity table | Yes |
| `results/gaps/<snapshot_id>/manifest.json` | Written last: snapshot, git commit, the SHA-256 of T19's records and of every input, the values used, T19's `provisional` flag, library versions, CPU, the SHA-256 of every output and the times | Yes |
| `data/processed/<snapshot_id>/gaps/members.parquet` | The doc codes of each gap's theses, joined to `gaps.parquet` by `gap_id` | No |

Before anything is written, every tracked output passes the privacy guard of `corpus/profile.py`:
an output that holds a title, an abstract sentence, a person's name or an ORCID iD of the
snapshot refuses the whole write, and the error names the file, never the string.

## Labeled-set draw

Draw the pairs of the labeled set (T20a, D34) once, with the values of `labeling` in
`config/default.yaml`:

```sh
uv run python -m thematic_redundancy.labeling.draw_pairs --dry-run
uv run python -m thematic_redundancy.labeling.draw_pairs
```

It uses the snapshot harvested last (`--snapshot-id` picks another one). It reads the included
fichas, for each thesis's program; the document vectors of both embedding models and the TF-IDF
matrix (`section_mean`), each checked against its manifest by its loader; and the snapshot's
metadata, for the privacy guard. The universe is every pair of the included theses, within and
across programs, each scored by the cosine under every representation and ranked by its
percentile rank. The pilot (10 pairs, at least 2 per pool) is drawn first, then the labeled set:
20 pairs from each embedding model's top 1 %, 20 from the TF-IDF top 1 %, 30 from the mid-range
(a mean percentile rank of at least 0.90 and below 0.99) and 30 uniformly from the whole
universe; no pair is drawn twice. The labeled set's high and mid-range pools spread their quota
over the strata (each program, and the pairs between programs) in proportion to their
candidates, with at least 2 for each stratum that has any. Every random choice derives from the
seed, so the same configuration and inputs give the same draw. `--dry-run` computes and prints
everything (the cuts, the composition of each pool by stratum, the pilot), applies the privacy
guard and writes nothing; on snapshot `20261002T224412Z` it takes about 20 seconds. A draw happens
once: a real run refuses an earlier draw of the snapshot and a dirty git tree outside
`results/labeled_set/<snapshot_id>/draw/`, holds that directory's run lock and writes:

| File | Content | In git |
|---|---|---|
| `data/labels/labeled_set/<snapshot_id>/pairs.parquet` (`labeling.records_dir`) | One row per drawn pair: its pair id (`P-` for the pilot, `L-` for the labeled set), phase, draw order, pool, source model, stratum, the two doc codes and programs, the score and percentile rank under each representation, and the known-duplicate flag (O09) | No |
| `results/labeled_set/<snapshot_id>/draw/composition.parquet` | Each pool's candidates, quota and drawn pairs per stratum, for the pilot and the labeled set | Yes |
| `results/labeled_set/<snapshot_id>/draw/ranges.parquet` | Each pool's score and percentile-rank ranges under each representation, and of the mean rank | Yes |
| `results/labeled_set/<snapshot_id>/draw/cuts.json` | The pairs per stratum, each representation's top-1 % cut score with its strata, and the mid-range's candidates | Yes |
| `results/labeled_set/<snapshot_id>/draw/summary.md` | The same, readable, with the score ranges and the values used | Yes |
| `results/labeled_set/<snapshot_id>/draw/manifest.json` | Written last: snapshot, git commit, seed and values, the SHA-256 of every input and output, library versions, CPU and the times | Yes |

Before anything is written, every tracked output passes the gap run's privacy guard with the
theses' doc codes added: an output that holds a doc code, a title, an abstract sentence, a
person's name or an ORCID iD of the snapshot refuses the whole write. To draw again, move the
earlier draw's records away first.

## Redundancy threshold and stage 2 of the winner

Once the consensus labels of the labeled set exist, run T22 on them (D15, D30 E3, UD3) with one
command:

```sh
uv run python -m thematic_redundancy.redundancy.run_redundancy --dry-run
uv run python -m thematic_redundancy.redundancy.run_redundancy
```

It uses the snapshot harvested last (`--snapshot-id` picks another one). Its inputs, each
checked before use:

- T20a's draw: `results/labeled_set/<snapshot_id>/draw/manifest.json` and the gitignored
  `pairs.parquet`, which must have the SHA-256 that the manifest records. The labeled set's pairs
  (phase `labeled`) are the pairs of the run; the pilot's are never used.
- The consensus labels, `data/labels/labeled_set/<snapshot_id>/labels/consensus.parquet`
  (`labeling.records_dir`), one row per labeled pair with exactly these columns: `pair_id`
  (T20a's `L-` id), `label_3` (`Redundante`, `Relacionado` or `No relacionado`), `label` (1 for
  `Redundante`, else 0) and `source` (`agreement` or `adjudication`). The contract, its path and
  its row model live in `src/thematic_redundancy/redundancy/labels.py`.
- T19's `manifest.json` and `selection.json`, under the configuration's experiment declaration.
- The document vectors of both models and the TF-IDF matrix (`section_mean`), through T20a's
  loader; their input manifests must be those T20a drew from and T19 ran on, and each labeled
  pair's cosine must equal the score T20a stored, exactly.

The run refuses, writing nothing, missing labels; labels that are not exactly one per labeled
pair (a pair without a label, a pair id given twice, an id that is not of the labeled set, a
pilot pair); a `label` that does not follow `label_3`; a value outside the domains; a labels
file with other columns; scores that differ from T20a's; records that differ from their
manifests; and labels with too few redundant pairs for every training split of the
cross-validation to hold one.

With `redundancy` (5 folds, seed 7, stratified, ties to the highest τ, the final τ from all the
pairs) and `experiment.selection.f1` (2,000 resamples, seed 7, a 95 % interval), it computes,
for each representation, τ by cross-validation from the precision-recall curve, the
cross-validated F1, and the F1 of all the labeled pairs at τ with its bootstrap interval; the
paired comparison of the two embedding models and of each against TF-IDF; and stage 2 of the
winner by D30's full rule (the gates, the labeled-set F1, NPMI, then MiniLM), which confirms
T19's provisional winner, changes it, or is undefined when the F1 step names no higher F1 (a
decision for the user). It also reports acceptance criterion 1 (F1 part, and T19's NPMI part
while the winner stands) and criterion 2 (the winner's cross-validated F1 against 0.70).
`--dry-run` computes and prints everything, applies the privacy guard and writes nothing;
`--labels PATH` reads another labels file, in a dry run only, such as a check with synthetic
labels. A run happens once: a real run refuses an earlier run of the snapshot and a dirty git
tree outside `results/experiments/<snapshot_id>/t22/`, holds that directory's run lock and
writes:

| File | Content | In git |
|---|---|---|
| `results/experiments/<snapshot_id>/t22/representations.parquet` | One row per representation: τ, the counts, precision, recall and F1 at τ with its bootstrap interval, and the cross-validated counts and F1 | Yes |
| `results/experiments/<snapshot_id>/t22/folds.parquet` | Each fold of each representation: its τ and its held-out counts and F1 | Yes |
| `results/experiments/<snapshot_id>/t22/curves.parquet` | Each representation's precision-recall curve over the labeled pairs, the selected τ flagged | Yes |
| `results/experiments/<snapshot_id>/t22/comparisons.parquet` | ΔF1 with its paired interval and verdict: the two models (stage 2) and each against TF-IDF (criterion 1) | Yes |
| `results/experiments/<snapshot_id>/t22/selection.json` | Stage 2 step by step against T19's provisional winner, the sensitivity, criteria 1 and 2, the findings and what follows from the outcome | Yes |
| `results/experiments/<snapshot_id>/t22/summary.md` | The same, readable | Yes |
| `results/experiments/<snapshot_id>/t22/manifest.json` | Written last: snapshot, git commit, the values used, the SHA-256 of T20a's and T19's records, the labels and every input, the outcome, library versions, CPU, the SHA-256 of every output and the times | Yes |
| `data/processed/<snapshot_id>/experiments/t22/pair_scores.parquet` | One row per labeled pair: pair id, labels, fold, the score and prediction under each representation, and the known-duplicate flag | No |

Before anything is written, every tracked output passes the gap run's privacy guard with the
theses' doc codes and the drawn pairs' ids added. To run again, move the earlier run's records
away first.

## Streamlit app

Browse the topic map and the research gaps of a snapshot, and query a thesis proposal against
them (T28), in Spanish (O01). Start it from the project root:

```sh
uv run streamlit run src/thematic_redundancy/app/main.py
uv run streamlit run src/thematic_redundancy/app/main.py -- --snapshot-id 20261002T224412Z
```

The arguments after `--` are the app's own: `--project-root` (the folder that holds
`config/default.yaml`; default: the current directory) and `--snapshot-id` (default: the snapshot
harvested last). Streamlit reads `.streamlit/config.toml` from the folder it starts in, so start
it from the project root. The app serves `http://localhost:8501/` (the map),
`http://localhost:8501/consulta` (the query) and `http://localhost:8501/vacios` (the gaps);
`--server.port` picks another port.

| Page | What it shows |
|---|---|
| Mapa temático | T19's 2-D UMAP map of the winner's representative run (`map.parquet`), each thesis colored by topic, with filters by topic and by program and a switch for the theses without a topic (HDBSCAN's noise); the hover gives a thesis's title, program and year. Below, each topic's size and top terms, and the theses shown with their repository links |
| Consulta de una propuesta | A form for one proposal: its title (required), and its abstract, its objectives or both, the sections that represent a thesis (D09). On submission: its assigned topic (T23, nearest centroid) with its top terms, whether it is atypical for that topic (O06: below the 5th percentile of the map's member-to-own-centroid similarities) and the topic's saturation quartile; its 10 most similar theses by cosine (T22), with no redundancy verdict and a "pendiente de calibración" note; its 5 antecedents by MMR (T26, λ 0.7); and its alternative lines (T26): the 3 neighbor topics and the 3 nearest stable gaps of the gaps page. Each thesis links to the repository |
| Vacíos de investigación | T24a's stable gaps of the reference run, filtered by rule (a) to (d), by program (the programs of a gap's theses, and rule (c)'s source and target) and by rule (c)'s high-confidence group (affinity 2); each with its topics and their terms, its counts and shares, programs, years and stability. Below, the theses of the chosen gap with their repository links |

It reads, and never writes, the saved artifacts of the snapshot, each checked against the SHA-256
that its manifest records: T19's `manifest.json`, `selection.json`, `topics.parquet` and
`data/processed/<snapshot_id>/experiments/t19/map.parquet`; the gap run's `manifest.json`,
`gaps.parquet` and `data/processed/<snapshot_id>/gaps/members.parquet`, which must name the T19
manifest read; and the fichas, whose manifest must be the one T19 ran on. A missing, changed or
inconsistent artifact shows a message in Spanish that names the file and the command that writes
it again, and the page stops there. A notice says that the results are provisional while T19's
winner waits for T22's labeled-set F1. The results are cached in the server's memory only; the
sidebar's button reloads them after a new run.

The query page also reads the winner's document vectors (`load_embeddings`; their manifest must be
the one T19 ran on) and the chunk manifest, and loads the winner's model and every configured
tokenizer from the local Hugging Face cache only (`HF_HUB_OFFLINE=1`), once per server; the model
must be the revision that embedded the corpus, and each tokenizer the one that chunked it, or the
page says so in Spanish. A proposal is represented exactly as a thesis: light cleaning, the
title's place and year suffix removed, sentence-aligned chunks (T13), the model's chunk vectors
pooled into the `section_mean` document vector (T14). The centroids, O06's threshold and the
saturation quartiles are computed from the saved vectors and labels with T23's functions, and
the stable gaps that T26 needs are rebuilt from the gap run's records and checked again with
T24's stability filter, so the query page needs no artifact of its own.

What the app does not do:

- It keeps no query and writes no file: a proposal is used for its answer only, is never
  cached, written or logged, and what a user types or selects lives only in their session; the
  chunk vectors of a proposal are not added to the vector cache.
- It shows no author or advisor: of each thesis it loads only the doc code, title, program, year
  and repository link, never the person codes, abstract, objectives or keywords.
- It sends no usage statistics (`browser.gatherUsageStats = false`), and the server runs headless
  and listens on `localhost` only; pass `--server.address` to serve another interface.
- It shows no redundancy verdict yet: the query page ranks the theses by similarity, marked
  "pendiente de calibración", until T22 calibrates the threshold τ on the labeled set.

## Temporal split

Evaluate the topic map on a temporal split (T27, O04): T19's winner is fitted on the theses of
2021 to 2024 alone and the theses of 2025 to 2026 are assigned to it, with the values of
`temporal` and `proposal` in `config/default.yaml`:

```sh
uv run python -m thematic_redundancy.topic_map.run_temporal --dry-run
uv run python -m thematic_redundancy.topic_map.run_temporal
```

It uses the snapshot harvested last (`--snapshot-id` picks another one). It reads T19's
`manifest.json`, `selection.json`, `topics.parquet` and the gitignored `labels.parquet`, each
checked against the SHA-256 that T19's manifest records; T19's inputs through T19's loader,
whose input manifests must be the ones T19 ran on; the fichas, for each thesis's program, issue
year and title; and the snapshot's metadata, for the privacy guard. A thesis belongs to the
earlier period when its issue year is below `temporal.split_year` (2025); the two theses issued
after the snapshot date (T06) are assigned by their issue year, as every other one.

For each of T19's five seed runs of the winner it fits the winner's configuration, with no new
sweep (D30 E3), on the earlier theses: one UMAP fit with the run's seed, the run's clustering,
the c-TF-IDF terms over the earlier theses' share of T15's vocabulary, and the centroids in the
original space. Each later thesis goes to the topic whose centroid is nearest by cosine (D14) and
is flagged atypical below the P5 of the map's member-to-own-centroid similarities (O06, scope
and method from `proposal`). Both periods are then compared with T19's full-map labels of the
same seed, under D30 E4's two noise handlings (`own_label`, which counts the full map's noise as
a label, and `exclude_either`, which leaves out the theses that are noise in either map):

- for the later theses, the ARI, the NMI and the **counterpart agreement**: the share whose
  full-map label is their topic's counterpart, the full-map label of most of the topic's own
  earlier members;
- for the earlier theses, the ARI and the NMI of their labels on both maps: how far the map
  itself moves when the later theses join it.

No pass threshold is applied: O04 declares none. The expert sample draws, from the map of T19's
medoid run, `temporal.sample.size` later theses (40): `atypical` of them (10) from the theses
flagged atypical, or all of those when fewer, and the rest from the others, each pool spread over
the programs in proportion to its theses with at least `program_floor` (2) per program that has
any, in a seeded presentation order. `--dry-run` computes and prints everything, applies the
privacy guard and writes nothing; on snapshot `20261002T224412Z` it takes about 2 minutes, most of
it the first UMAP fit and BERTopic's import. The sample is a one-time draw: a real run refuses an
earlier temporal split of the snapshot and a dirty git tree outside
`results/temporal/<snapshot_id>/`, holds that directory's run lock and writes:

| File | Content | In git |
|---|---|---|
| `results/temporal/<snapshot_id>/runs.parquet` | One row per seed run: the earlier map's theses, topics, noise and terms, the later theses, how many are atypical and how many are noise in the full map, O06's threshold and the times | Yes |
| `results/temporal/<snapshot_id>/consistency.parquet` | One row per seed run, period and noise handling: the points compared, ARI, NMI and, for the later theses, the counterpart agreement | Yes |
| `results/temporal/<snapshot_id>/topics.parquet` | One row per topic of each earlier map: its size, top terms and scores, its counterpart under each noise handling, and the later theses assigned to it and flagged atypical | Yes |
| `results/temporal/<snapshot_id>/composition.parquet` | The expert sample's candidates and drawn theses per pool and program | Yes |
| `results/temporal/<snapshot_id>/summary.md` | The periods by year and program, the earlier maps, the metrics per seed with their mean and SD, and the sample's composition | Yes |
| `results/temporal/<snapshot_id>/manifest.json` | Written last: snapshot, git commit, the SHA-256 of T19's records and of every input, the values used, T19's `provisional` flag, the counts, library versions, CPU, the SHA-256 of every output and the times | Yes |
| `data/processed/<snapshot_id>/temporal/assignments.parquet` | Every thesis on every seed run's maps: its period, its label on the earlier map, its full-map label and, for a later thesis, its similarity, runner-up, margin, threshold and atypical flag | No |
| `data/processed/<snapshot_id>/temporal/sample.parquet` | The expert sample in presentation order: doc code, title, program, year, pool, assigned topic with its terms, similarity, threshold, atypical flag and full-map label | No |

While T19's winner is provisional, every row and the summary say so; if T22 changes the winner,
move the records away and run the split again. Before anything is written, every tracked output
passes the gap run's privacy guard with the theses' doc codes added.

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## Smoke test

A separate suite checks that the locked stack works together on this machine. It covers spaCy,
the embedding model, UMAP with K-means/HDBSCAN inside BERTopic, the metrics, plotly/Streamlit,
the stats and file libraries, and OCR of a page without a text layer. It also checks that
BERTopic keeps accents and ñ in its topic words (D22). The OCR check needs the files from
[OCR assets](#ocr-assets) and fails with the fetch command when they are missing.

A test run leaves the suite out, whatever other `-m` filter it uses (`tests/conftest.py`).
Run it with a `-m` expression that names the `smoke` marker, read as pytest reads marker
names (`pre-smoke` or `smoke_fast` is another marker):

```sh
uv run pytest -m smoke
```

`--smoke` only stops the run from leaving the suite out; `-m` and `-k` still filter it like
any other test. `uv run pytest --smoke -k ocr` runs the OCR tests of both suites, while
`uv run pytest --smoke -m unit` selects no smoke test, since none is marked `unit`.

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
  Without it, the codes can be neither re-derived nor reversed. Creating it needs no hard links,
  so `data/` may sit on a FAT or exFAT volume, a network share or a container mount.
- The gap run's tracked records name topics by run and id and hold no doc code; only its
  `summary.md` shows the topics' terms, which T19's records already hold. Which theses form each
  gap lives under the gitignored `data/processed/`. The run refuses to write any tracked
  output that holds a title, an abstract, a person's name or an ORCID iD from the snapshot.
- The labeled-set draw's tracked records hold counts and scores only; which theses form each
  drawn pair lives under the gitignored `data/labels/`, and the configuration refuses a records
  directory outside `paths.data_dir`. The draw refuses to write any tracked output that holds a
  doc code, a title, an abstract, a person's name or an ORCID iD from the snapshot.
- The redundancy run's tracked records hold counts, thresholds and F1 values only; each labeled
  pair's scores and labels live under the gitignored `data/processed/`. The run refuses to
  write any tracked output that holds a doc code, a pair id, a title, an abstract, a person's
  name or an ORCID iD from the snapshot.
- The temporal split's tracked records hold counts, metrics and the topics' terms only; which
  thesis went to which topic, and the expert sample with its doc codes and titles, live under
  the gitignored `data/processed/`. The split refuses to write any tracked output that holds a
  doc code, a title, an abstract, a person's name or an ORCID iD from the snapshot.
- `results/` holds aggregates only. The metadata profile refuses to write an output that holds a
  title, an abstract, a person's name or an ORCID from the snapshot.
- Extracted page texts live only under the gitignored `data/interim/`. The extraction manifest
  and its summary hold counts only.
- The objectives table and the verification workbook hold thesis text, so they live only under
  the gitignored `data/`; the configuration refuses a workbook directory outside
  `paths.data_dir`. The workbook replaces every author, advisor and juror name of the snapshot
  with `[nombre omitido]`. The objectives summary and the accuracy report hold numbers only.
- The fichas table holds thesis text, so it lives only under the gitignored `data/interim/`. The
  fichas quality report refuses to write a file that holds a title, an abstract, a person's
  name or an ORCID iD from the snapshot.

## Plan

Scope, decisions, and the task checklist live in
[`odd/tasks/thematic-redundancy-system.md`](odd/tasks/thematic-redundancy-system.md).
Processing times, request counts, data sizes and costs are recorded in
[`docs/processing_costs.md`](docs/processing_costs.md), one row per long or measured operation.
The expert-session kit (the consent template, the session script, the questionnaire, the judges'
form, the samples' specification and the data protocol, in Spanish) is in
[`docs/expert_session/`](docs/expert_session/README.md).
