# Data card

The corpus is one frozen snapshot, `20261002T224412Z`, of the public UCSM institutional
repository: 766 records of the five FCIFF program collections, issued 2021 to 2026, of which
744 distinct theses enter every analysis. The texts stay on the local machine under the
gitignored `data/`; git tracks aggregates only.

This card covers the corpus as the pipeline uses it. The snapshot's own card,
[`results/snapshot/20261002T224412Z/data_card.md`](../../results/snapshot/20261002T224412Z/data_card.md),
holds the freeze digests, the counts by program and year, the embargo list, the PDF statuses
and the text-layer profile; this card does not repeat them.

Marks: (M) measured from a file or a command, (D) derived from measured values. Every value
names its source.

## At a glance

| Measure | Value | Source |
|---|---|---|
| Source | UCSM repository, `https://repositorio.ucsm.edu.pe`, DSpace 7.6.1 REST API, anonymous read-only access | `repository.base_url`; snapshot card §1 |
| Scope | The FCIFF faculty community, its 5 program collections, `dc.date.issued` 2021 to 2026 | `repository.faculty_community_uuid`, `snapshot` block |
| Snapshot | `20261002T224412Z`, harvested 2026-10-02T22:44:12Z, frozen on 2026-10-03 | snapshot manifest `harvested_at`; snapshot card §2 |
| Records | 766, equal to the faculty-wide total of the same query | snapshot manifest `total_items`, `faculty_total` (M) |
| Distinct theses used | 744 | `fichas.expected.included`; fichas quality report (M) |
| License | CC BY-NC-ND 4.0 on every record | `dc.rights.uri` of the 766 records (M) |
| Raw snapshot on disk | 11.41 GB: 731 PDFs (11,401,121,329 B) plus metadata and manifests | `docs/processing_costs.md` §4 |
| Backup | None (D23, risk R8 accepted by the user) | plan D23 |

## 1. Source and scope

The five programs are five DSpace collections of the faculty community
`3e706f8d-cba0-4d4a-8d51-dd317c3e7652`, as `snapshot.programs` declares them (uuids resolved
on 2026-10-02).

| Program key | Program | Records (M) | Included theses (M) |
|---|---|---:|---:|
| `sistemas` | Ingeniería de Sistemas | 67 | 65 |
| `industrial` | Ingeniería Industrial | 349 | 339 |
| `electronica` | Ingeniería Electrónica | 55 | 46 |
| `mecanica` | Ingeniería Mecánica, Mecánica-Eléctrica y Mecatrónica | 192 | 191 |
| `minas` | Ingeniería de Minas | 103 | 103 |
| **Total** | | **766** | **744** |

Records: the snapshot manifest's `programs.<key>.items`. Included theses: the fichas quality
report's `counts.included_by_program`
([`results/fichas/20261002T224412Z/quality_report.json`](../../results/fichas/20261002T224412Z/quality_report.json)).

- The corpus holds public defended theses only; thesis plans are not public (D01).
- It is a census of every eligible thesis, with no sampling (D02).
- Titles, abstracts, keywords, dates and advisors come from the API metadata; the program
  comes from the collection; the objectives are parsed from the PDF (D03).

## 2. How the snapshot was taken

| Step | Command | When | Cost (M, ledger) |
|---|---|---|---|
| Metadata harvest (T04, U8) | `uv run python -m thematic_redundancy.corpus.harvest` | 2026-10-02T22:44:12Z | 47 s, 14 requests |
| PDF download (T05, U8) | `uv run python -m thematic_redundancy.corpus.download_pdfs` | 2026-10-02 to 2026-10-03 | 12 h 29 min elapsed, about 3 h 9 min active (D); about 1,463 requests (estimate) |
| Freeze and snapshot card (T06) | none: the card was written from the files | 2026-10-03 | — |

The query, as the snapshot manifest records it: `server/api/discover/search/objects` with
`dsoType=ITEM`, `size=100`, `sort=dc.date.issued,ASC` and `f.dateIssued=[2021 TO 2026],equals`,
one listing per collection. Every item type was kept; the theses are selected later, in the
fichas build.

- **Reconciled.** The five listings add up to 766 distinct item uuids, equal to the faculty
  total (snapshot manifest). The harvest refuses a listing that repeats an item or misses the
  reported total, and a program sum that differs from the faculty total (README, "Corpus
  snapshot").
- **Polite.** One request at a time, at least `repository.request_interval_seconds` (1.0 s)
  apart, with at most `repository.max_retries` (3) retries.
- **PDFs.** 731 of the 746 `#tesis` records have their PDF on disk; 14 are closed by their
  `dc.rights` and were never requested, and 1 has no thesis file (snapshot card §7). Each file
  is stored as `<item uuid>.pdf`, verified against the size and MD5 that the repository lists,
  and its SHA-256 recorded.
- **Frozen.** The snapshot is not modified or downloaded again. A re-harvest is a new
  snapshot; its differences can be documented through the item uuids, bitstream uuids and
  SHA-256 values that the manifests record (D23, snapshot card §2).

## 3. Record counts

The fichas build (T12) checks these counts and writes nothing when one is missed. Each row
is a key of `fichas.expected` in `config/default.yaml`, with the value the build reached in its
quality report.

| `fichas.expected` key | Value | Quality report field (M) | Meaning |
|---|---:|---|---|
| `records` | 766 | `counts.records` 766 | Every harvested record gets one ficha |
| `included` | 744 | `counts.include.included` 744 | Distinct theses, the census |
| `duplicates` | 2 | `counts.exclusion_reason.duplicate` 2 | Exact duplicate records, excluded (D24) |
| `non_thesis` | 20 | `counts.exclusion_reason.non_thesis_type` 20 | Other `renati.type` values: 19 suficiencia profesional, 1 trabajo académico (D01) |
| `objectives_extracted` | 728 | `counts.objectives_status.extracted` 728 | Included theses whose objectives come from their own PDF |
| `metadata_only` | 16 | `counts.objectives_status.no_pdf` 16 | Included theses represented by title and abstract only (O05) |

The counts add up (D): 744 + 2 + 20 = 766, and 728 + 16 = 744. The 16 metadata-only theses
are the 14 restricted ones, the one without a file and the one whose PDF is another item's
(snapshot card §10); by program, industrial 10, sistemas 3 and mecanica 3 (plan D27 C4).

The later builds enforce their own counts on the same 744 theses:

| Configuration key | Expected for `20261002T224412Z` | Task |
|---|---|---|
| `chunking.expected` | 744 documents, 5,217 chunks: 744 title chunks and 4,473 body chunks | T13 (D27) |
| `embedding.expected` | 744 documents, 5,217 chunks, per model | T14 |
| `tfidf.expected` | 744 documents, 2,216 sections, 14,308 terms, 98,282 nonzeros | T15 (D28) |

## 4. Fields kept and dropped

**In the snapshot.** Each line of `data/raw/20261002T224412Z/metadata.jsonl` holds `uuid`,
`handle`, `program_key`, `collection_uuid`, `harvested_at` and `metadata` (M). Across the 766
records, `metadata` holds 32 distinct keys (M, counted on 2026-10-27):

`dc.contributor.advisor`, `dc.contributor.author`, `dc.date.accessioned`,
`dc.date.available`, `dc.date.embargoEnd`, `dc.date.issued`, `dc.description.abstract`,
`dc.description.sponsorship`, `dc.description.uri`, `dc.format`, `dc.identifier.doi`,
`dc.identifier.uri`, `dc.language.iso`, `dc.publisher`, `dc.publisher.country`, `dc.rights`,
`dc.rights.uri`, `dc.source`, `dc.subject`, `dc.subject.ocde`, `dc.title`, `dc.type`,
`dc.type.version`, `renati.advisor.orcid`, `renati.discipline`, `renati.juror`, `renati.level`,
`renati.type`, `thesis.degree.discipline`, `thesis.degree.grantor`, `thesis.degree.level`,
`thesis.degree.name`.

**Dropped at ingestion.** Every metadata key that ends in `.dni` is removed before anything is
written. The snapshot manifest's `dropped_metadata_keys` lists the two that the repository
sent: `renati.advisor.dni` and `renati.author.dni`.

**In the fichas.** A ficha (`corpus/ficha.py`, class `Ficha`) keeps the identity (`item_uuid`,
`doc_code`, `handle_url`, `snapshot_id`), the type (`document_type`, `renati_type_raw`), the
texts (`title`, `abstract`, `objectives`, `objectives_status`), the program, the issue date
and year with an `issued_after_snapshot` flag, `keywords`, `ocde_codes`, `language`, `rights`,
`embargo_end`, the person codes (`advisor_code`, `author_codes`), the PDF status, the source
format, the section source, the inclusion fields (`include`, `exclusion_reason`,
`duplicate_of`) and `quality_notes`. The schema has no field for a person's name or an ORCID
iD.

## 5. Rights and licenses

As the 766 records state them (M, counted from `metadata.jsonl` on 2026-10-27):

| Field | Value | Records |
|---|---|---:|
| `dc.rights.uri` | `https://creativecommons.org/licenses/by-nc-nd/4.0/` | 766 |
| `dc.rights` | `https://purl.org/coar/access_right/c_abf2` (open) | 752 |
| `dc.rights` | `https://purl.org/coar/access_right/c_f1cf` (embargoed) | 8 |
| `dc.rights` | `https://purl.org/coar/access_right/c_16ec` (restricted) | 6 |
| `dc.language.iso` | `spa` | 766 |

The 752 open records are the 732 open theses and the 20 non-thesis records, all open
(snapshot card §6). Under CC BY-NC-ND 4.0 the theses are used for research only, their texts
are not redistributed, and every displayed thesis links to its repository item (README, "Data
policy").

## 6. Codes instead of identities

| Code | Form | How it is made | Secret |
|---|---|---|---|
| Document | `DOC-` and 8 hex digits | The first 8 hex digits of the SHA-256 of the item uuid (`doc_code_for`, `corpus/ficha.py`) | None: anyone can derive it from the public uuid. It shortens, it does not hide |
| Advisor | `ADV-` and 10 hex digits | HMAC-SHA256 of the advisor's identity under the pseudonym key (`pseudonym`, `corpus/anonymize.py`); the identity is a valid ORCID iD, else the normalized name | The key |
| Author | `AUT-` and 10 hex digits | The same, with its own prefix inside the MAC, so one person's advisor and author codes are unrelated | The key |
| Drawn pair, in the workbooks | Two letters and three digits | Drawn at random (`secrets.choice`, `new_blind_codes` in `labeling/workbook.py`) once per drawn pair by the `pilot` step, unrelated to the pair id, so a pair has the same code in every workbook | The gitignored `blind_codes.parquet`, the only link between a code and its pair |
| Expert-sample case | The sample's letter and two digits | A random permutation of each sample's cases, from `expert_samples.seed` (`evaluation/expert_samples.py`) | The gitignored key, `clave.csv`, beside the samples, which maps each case code to doc codes |

- **The key.** 32 random bytes in `data/interim/pseudonym.key`, created by the first fichas
  build (2026-10-07) and never overwritten; a key file of the wrong size stops the run. It is
  gitignored and must never be committed or shared: whoever holds it can test a guessed name
  or ORCID iD against the codes.
- **Without the key** the person codes can be neither re-derived nor reversed. A new key gives
  every person a new code, unrelated to the stored ones, so the key is part of what makes a
  rerun reproduce the same fichas.
- **Counts (M, fichas quality report `advisor_orcid`).** 691 advisor codes come from an ORCID
  iD and 53 from a name; 21 theses carry a malformed ORCID iD or one that fails its check
  digit, and get a quality note. There are 138 distinct advisor codes and 784 distinct author
  codes.

## 7. What git tracks

| Tracked | Not tracked (gitignored) |
|---|---|
| `src/`, `tests/`, `experiments/`, `config/`, `docs/`, the plan under `odd/` | `data/`: the raw snapshot, the page texts, the fichas, the vectors, the maps, the drawn pairs, the pseudonym key, and the records listed below |
| `results/`: aggregates only; each file is listed in the [results index](results_index.md) | `tessdata/`: the OCR language files |
| `uv.lock`, `pyproject.toml`, `.python-version`, and the app's Streamlit settings, `.streamlit/config.toml` | `.venv/`, and the Hugging Face cache, which lives outside the repository |

The commands added since `90fedf2` write these records under `data/`, never in git, because
they name theses or link codes to them (names and roles only; `<id>` is the snapshot id; the
[results index](results_index.md) lists every file):

| Records | Role | Written by |
|---|---|---|
| `data/labels/labeled_set/<id>/blind_codes.parquet` | The blind code of every drawn pair | `pair_workbooks pilot` (T20b) |
| `data/labels/labeled_set/<id>/workbooks/` | The pilot, annotators' and adjudication workbooks, which show the theses' titles, abstracts, objectives and keywords | `pair_workbooks` (T20b, T20c) |
| `data/labels/labeled_set/<id>/labels/` | Each annotator's labels, the third labels and the consensus labels | `pair_workbooks` (T20c) |
| `data/processed/<id>/experiments/t22/` | Each labeled pair's scores and labels | `run_redundancy` (T22) |
| `data/processed/<id>/temporal/` | Which thesis went to which topic, and the expert sample with doc codes and titles | `run_temporal` (T27) |
| `data/expert_session/<id>/muestras/` | The printed session materials, samples A to D, which show titles and, in sample B, abstracts and objectives; never committed. Their key, `clave.csv`, maps the case codes to doc codes | `run_expert_samples` (T31), run on 2026-10-29 |

`.gitignore` also ignores the run locks that the commands leave beside their output folders
(`results/**/.lock`). `.dockerignore` keeps `data/`, `tessdata/`, `results/` and `odd/` out of
any Docker build context.

## 8. Privacy guards

| Guard | Where | What it refuses or removes |
|---|---|---|
| DNI drop | `corpus/harvest.py` | Every `.dni` metadata key, before anything is written |
| PDF names | `corpus/download_pdfs.py` | Original file names, which may hold author names: PDFs are saved as `<item uuid>.pdf` |
| No names in fichas | `corpus/ficha.py`, `corpus/anonymize.py` | Person names and ORCID iDs: only keyed codes are stored |
| Sensitive strings | `sensitive_strings`, `corpus/profile.py` | Builds the snapshot's titles, abstracts, abstract sentences, author, advisor and juror names and ORCID iDs, for the checks below |
| Person strings | `person_name_forms`, `person_strings`, `corpus/profile.py` | Builds each person's name forms (as stored, as "Given Family", and the family part of two words or more) and the ORCID values, without the titles and abstracts that the workbooks show, for the workbooks' name guard |
| Metadata profile and stopword candidates | `experiments/eda_metadata.py`, `experiments/domain_stopwords.py` | An output that holds one of those strings: nothing is written |
| Fichas quality report | `corpus/fichas_quality.py` | The same |
| Tracked outputs of the later runs | `check_privacy`, `gaps/results_store.py`, called by the gap run, the draw, `pair_workbooks`, `run_redundancy` and `run_temporal` | A tracked output that holds one of those strings: the whole write is refused, and the error names the file, never the string. The draw and the temporal split add the theses' doc codes; `run_redundancy` adds the drawn pairs' ids too; the workbook steps' manifests add the doc codes, pair ids and blind codes |
| Objectives workbook | `labeling/objectives_check.py` | Every author, advisor and juror name, replaced with `[nombre omitido]`; the workbook directory must lie inside `paths.data_dir` |
| Labeled-set records | `labeling.records_dir` | A records directory outside `paths.data_dir` |
| Labeled-set workbooks (T20b, T20c, D34 (1)) | `guard_workbooks`, `labeling/workbook.py`; `pair_workbooks.py` | Names first go through the objectives workbook's redactor; then every text of every workbook (each cell, drop-down list and sheet name, the guide included) is searched without word boundaries for the snapshot's person strings. A hit stops the step before anything is written and names the workbook, sheet, cell and pair code, never the name. No workbook shows a score, pool, stratum, link, doc code, pair id or person |
| Expert samples (T31) | `check_texts`, `evaluation/expert_samples.py`; `evaluation/run_expert_samples.py` | A sample that holds, searched without word boundaries, a person's name form or ORCID iD of the snapshot, or a thesis's doc code, uuid or handle, or a link: the command stops, the dry run included. It writes only under `data/expert_session/`; its `composicion.json` and manifest also pass `check_privacy` |
| The app (T28) | `load_theses`, `app/data.py`; `.streamlit/config.toml` | It loads of each thesis only the doc code, title, program, year and repository link, never a person code, abstract, objectives or keywords; it keeps no query and writes no file (README, "Streamlit app"); it sends no usage statistics and serves `localhost` only |

The T19 topic terms (`results/experiments/20261002T224412Z/t19/topics.parquet`) were checked
by hand before their commit, not by code: of their 532 distinct term tokens, 8 also occur in
contributor names, and none is a full name (plan T19, real-run evidence). The temporal split's
terms (`results/temporal/20261002T224412Z/topics.parquet`) were checked the same way (D30 E6):
of their 133 distinct terms and 122 distinct tokens, one token occurs in a person's name, a
vocabulary lemma already in T19's committed terms, and no two-word term is a name (plan T27,
real-run evidence).

## 9. Known limits

- **Temporal cut.** The corpus ends at the snapshot date, 2026-10-02, so 2026 is a partial year,
  which matters for trends (gap rule (d), T27). Two theses carry an issue date after it
  (snapshot manifest `future_dated_items` 2); the fichas flag them with
  `issued_after_snapshot` (snapshot card §4). The temporal split (T27) places them in the later
  period by their issue year, as every other thesis (temporal manifest
  `assigned_issued_after_snapshot` 2).
- **Expired embargoes.** 7 of the 8 embargoed theses have passed their embargo end, yet the
  repository still lists them as embargoed. Their PDFs were not fetched, since that needs a
  separate authorization (U8); they stay in the census as metadata-only theses (snapshot card
  §6).
- **A known duplicate pair (O09).** Two included `mecanica` theses of 2021 are one thesis
  deposited twice: the same objectives, advisor, authors and keywords, but a different title,
  abstract and PDF, so D24's exact-match rule kept both. The user decided to keep both
  (2026-10-19); `redundancy.known_duplicates` lists the pair, and the results flag it. It is
  the most similar pair under TF-IDF (cosine 0.9958) and mpnet (0.9905) (plan O09). The
  labeled-set draw found it in its universe and did not draw it (draw manifest
  `known_duplicates_in_universe` 1, `known_duplicates_drawn` 0).
- **Exact duplicates (D24).** Two pairs of records are the same thesis under two item uuids;
  the record accessioned first stays (snapshot card §5).
- **A wrong file.** One `sistemas` item carries a byte-identical copy of another item's PDF;
  `fichas.wrong_pdfs` names it, and its ficha is metadata-only (snapshot card §7).
- **One program dominates.** `industrial` holds 339 of the 744 included theses, 45.6 % (D);
  results are reported per program (risk R5).
- **A live repository and no backup.** The repository changes over time, and the snapshot has
  no backup (D23): if it is lost, a new harvest gives a new snapshot that differs from this one.
