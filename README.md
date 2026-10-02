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
- It reads metadata only, never PDFs.

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
- DNI fields (every metadata key ending in `.dni`, such as `renati.author.dni` and
  `renati.advisor.dni`) are dropped at ingestion, before anything is written. The snapshot
  manifest lists the dropped keys.
- Author names are never shown in the app or in reports, and user queries are never persisted.

## Plan

Scope, decisions, and the task checklist live in
[`odd/tasks/thematic-redundancy-system.md`](odd/tasks/thematic-redundancy-system.md).
