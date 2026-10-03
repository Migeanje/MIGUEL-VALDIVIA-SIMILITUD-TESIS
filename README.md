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

## Plan

Scope, decisions, and the task checklist live in
[`odd/tasks/thematic-redundancy-system.md`](odd/tasks/thematic-redundancy-system.md).
