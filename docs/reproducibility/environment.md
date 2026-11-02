# Environment

Every recorded run used CPython 3.13.2 on one Windows 11 laptop, CPU only, with the packages
that `uv.lock` pins. To reproduce a run, install from that lock, put the two embedding models
in the Hugging Face cache and the two Tesseract language files in `tessdata/`, and run the
commands offline.

Marks: (M) measured from a file or a command, (D) derived. Every value names its source.

## Quick path

```sh
uv sync
uv run python -m thematic_redundancy.extraction.ocr_assets
uv run pytest -m smoke
```

1. `uv sync` installs the locked environment into `.venv/` (online; over 1 GB, README
   "Setup").
2. `ocr_assets` fetches the two tessdata files once per machine (online).
3. The first smoke run downloads the MiniLM model into the Hugging Face cache (online, about
   0.5 GB); later runs work offline with `HF_HUB_OFFLINE=1` (README "Smoke test"). The mpnet
   model has no download command; see [Embedding models](#embedding-models).

## Python and uv

| Item | Value | Source |
|---|---|---|
| Python | `>=3.13,<3.14`; `.python-version` holds `3.13` | `pyproject.toml` |
| Python of the recorded runs | CPython 3.13.2, 64-bit | `versions.python` of the run manifests (M); ledger §2 |
| Why not 3.14 | gensim 4.4.0 ships no cp314 Windows wheel | plan D06 |
| uv | 0.12.22 | ledger §2 |
| Python downloads | Run uv with `UV_PYTHON_DOWNLOADS=never`, so that it uses the local CPython 3.13.2 | plan T01 |

## The lock file

`uv.lock` (M, read at `2e8f9df`; it, `pyproject.toml` and `.python-version` are unchanged since
`90fedf2`): lock version 1, revision 5, 192 packages,
`requires-python = "==3.13.*"`, resolved for two environments, `sys_platform == 'win32'` and
`sys_platform == 'linux'` (`[tool.uv] environments` in `pyproject.toml`).

| Task | Command | Network |
|---|---|---|
| Install online | `uv sync` | PyPI, the PyTorch CPU index and GitHub (the spaCy model) |
| Check that the lock matches `pyproject.toml` | `uv lock --check` | — |
| Run a command without the network | `uv run --offline <command>`, with `.venv/` already synced | None |
| Install without the network | `uv sync --offline`: uv's flag, which needs every locked package already in uv's own cache. Not exercised in the recorded runs | None |

- `uv run --offline` is how the recorded checks ran in the worktrees (plan T12c, T21, T22,
  T23 and T26 evidence).
- The `dev` dependency group holds pytest, ruff and jupyterlab (`pyproject.toml`).
- Commit a change to `uv.lock` on its own (`build(deps): ...`), since the generated file is too
  large to review beside code (plan T01, "Lesson").

## CPU-only PyTorch

`pyproject.toml` declares the index `pytorch-cpu` (`https://download.pytorch.org/whl/cpu`,
`explicit = true`) and takes `torch` only from it (`[tool.uv.sources]`). The lock pins
`torch` 2.14.1+cpu from that index (M). At T01 the import check printed `2.14.1+cpu False`:
`torch.cuda.is_available()` is `False`. The laptop's GPU is never used (ledger §2).

The spaCy model `es_core_news_md` 3.8.0 is a URL dependency, a wheel from the GitHub releases
of `explosion/spacy-models` (`pyproject.toml`).

## Embedding models

Both models are read from the local Hugging Face cache only (`local_files_only=True` in
`representation/st_encoder.py` and `representation/hf_tokenizers.py`), so every command runs
with `HF_HUB_OFFLINE=1`.

| Model (`embedding.models`) | Dimensions | Cache size (M) | Cache revision | How it got there |
|---|---:|---:|---|---|
| `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 479,729,050 B | `e8f8c211…` | The first smoke run (README "Smoke test") |
| `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` | 768 | 1,126,358,977 B | `4328cf26…` | One download on 2026-10-07 under U8: 287.6 s with its first load, anonymous GET |

- Both read at most 128 tokens (`max_seq_length` 128) with tokenizers of 250,002 entries; the
  two `tokenizer.json` files are byte-identical (plan D09, D27 C6 and ledger, 2026-10-11).
- The cache is `%USERPROFILE%\.cache\huggingface\hub` on Windows (ledger §4). The revisions
  are those of plan D27 C6; the chunk cache key includes them.
- The repository has no command that downloads mpnet. On a new machine, put it in the cache
  once, with authorization for the download (U8), before the chunk build (T13), which counts
  tokens under both tokenizers.

## Tesseract language files

```sh
uv run python -m thematic_redundancy.extraction.ocr_assets
```

PyMuPDF reads pages without a text layer with the Tesseract engine it bundles, so no Tesseract
installation is needed (D07). The command fetches `spa.traineddata` (13,570,187 B) and
`eng.traineddata` (15,400,601 B) from `tessdata_best` into `tessdata/` (`paths.tessdata_dir`,
gitignored) and records each file in `tessdata/manifest.json`. Both files must equal the
SHA-256 values pinned in `PINNED_TRAINEDDATA` (`extraction/ocr_assets.py`), the files the OCR
was validated with on 2026-10-02 (README "OCR assets").

## Hardware of the recorded runs

Read on 2026-10-05 (ledger §2):

| Component | Value |
|---|---|
| Machine | MSI Cyborg 15 A12VF laptop |
| CPU | 12th Gen Intel Core i7-12650H: 10 cores, 16 threads |
| RAM | 16 GiB |
| GPU | Not used: torch is the CPU build |
| Storage | 1 TB NVMe SSD |
| OS | Windows 11 Home 10.0.26200, 64-bit |
| Power | Windows plan "Balanced", on AC power, during the T09 run |

The run manifests record the CPU as `cpu_record()` returns it (`topic_map/results_store.py`):
processor `Intel64 Family 6 Model 154 Stepping 3, GenuineIntel`, machine `AMD64`, system
`Windows 11`, 16 logical CPUs (M, the T19, gap-run, draw and temporal-split manifests; the
pilot step's manifest records no CPU).

Throughput depends on the machine state: the same 103 OCR pages took 4.6 s per page in one
session and 11.9 s in another (ledger §2). Compare times across sessions with care.

## Library versions in the run manifests

`library_versions()` (`topic_map/results_store.py`) returns the Python version and the
versions of the libraries whose bytes the results depend on (`LIBRARIES`). The five tracked
manifests that record them, those of T19, the gap run, the draw, the temporal split and the
pilot step, record the same values (M, read from the five files under `results/`):

| Library | Version |
|---|---|
| python | 3.13.2 |
| numpy | 2.5.3 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.1 |
| umap-learn | 0.5.12 |
| numba | 0.68.0 |
| pynndescent | 0.6.0 |
| hdbscan | 0.8.44 |
| bertopic | 0.17.4 |
| pyarrow | 25.0.1 |

## Other locked versions

From `uv.lock` (M):

| Library | Version | Used for |
|---|---|---|
| torch | 2.14.1+cpu | The embedding models |
| sentence-transformers | 6.1.0 | The embedding models |
| transformers | 5.18.0 | The tokenizers |
| pymupdf | 1.28.2 | PDF text and OCR |
| spacy | 3.8.16, model `es_core_news_md` 3.8.0 | The full cleaner |
| gensim | 4.4.0 | The smoke suite only (`tests/smoke/test_stack.py`) |
| pandas | 3.0.6 | Tables |
| httpx | 0.28.1 | The harvest and the download |
| openpyxl | 3.1.5 | The verification workbook and the labeled set's workbooks (T20b, T20c) |
| streamlit | 1.64.0 | The app (T28) and the smoke suite |
| plotly | 7.1.0 | The app's charts (T28) and the smoke suite |
| matplotlib | 3.11.2 | The metadata profile's figures |
| pytest | 9.1.1 | Tests |
| ruff | 0.16.10 | Lint and format |

BERTopic 0.17.4 predates sentence-transformers 6 and transformers 5, so it receives
precomputed embeddings (D11), and it is always built with `language="spanish"`, which keeps
accents and ñ in its topic words (D22).

## The app's settings

Streamlit reads `.streamlit/config.toml` from the folder it starts in, so the app (T28) starts
from the project root ([how to run](how_to_run.md#the-app)). The file turns off the usage
statistics (`browser.gatherUsageStats = false`), runs the server headless on `localhost`
only, with no file watcher and no rerun on save, and shows an error's type, never its
traceback, in the browser. The app serves port 8501 unless `--server.port` picks another.

## Docker

There is no Docker image yet: the project runs natively, and the Dockerfile is planned for T29
(D08, O07).
