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

## Checks

```sh
uv run ruff check .
uv run ruff format --check .
uv run pytest -q
```

## Smoke test

A separate suite checks that the locked stack works together on this machine. It covers spaCy,
the embedding model, UMAP with K-means/HDBSCAN inside BERTopic, the metrics, plotly/Streamlit,
and the stats and file libraries. The default test run excludes it. Run it with:

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
- DNI fields (`renati.author.dni`, `renati.advisor.dni`) are dropped at ingestion.
- Author names are never shown in the app or in reports, and user queries are never persisted.

## Plan

Scope, decisions, and the task checklist live in
[`odd/tasks/thematic-redundancy-system.md`](odd/tasks/thematic-redundancy-system.md).
