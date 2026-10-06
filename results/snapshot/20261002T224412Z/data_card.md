# Data card of snapshot 20261002T224412Z

The frozen corpus of this project: 766 records harvested from the public UCSM repository, of
which 746 are theses (744 distinct), and 731 thesis PDFs. 728 distinct theses have a valid PDF
of their own; the other 16 enter the census through their metadata only.

Aggregates and public repository identifiers only: this card holds no title, abstract,
keyword list of a single thesis, or person's name. The metadata profile in
[`results/eda/20261002T224412Z/summary.md`](../../eda/20261002T224412Z/summary.md) has more detail on
the texts (abstract and title lengths, keywords, OCDE codes, advisors).

## At a glance

| Measure | Value |
|---|---:|
| Records harvested | 766 |
| Theses (`renati.type` `#tesis`) | 746 |
| Distinct theses, after the D24 dedupe | 744 |
| Thesis PDFs on disk | 731 (11.40 GB) |
| Distinct theses with a valid own PDF | 728 |
| Distinct theses with metadata only | 16 |
| Theses issued after the snapshot date | 2 |
| Embargoed / restricted theses | 8 / 6 |
| PDF pages | 141,076 |

## 1. Identity and scope

| Field | Value |
|---|---|
| Snapshot id | `20261002T224412Z` |
| Source | UCSM institutional repository, `https://repositorio.ucsm.edu.pe`, DSpace 7.6.1 REST API; public, anonymous access, no credentials |
| Metadata harvested | 2026-10-02T22:44:12Z (`harvested_at` of every record); snapshot date 2026-10-02 (UTC) |
| PDFs downloaded | 2026-10-02 to 2026-10-03; the PDF manifest was last written at 2026-10-03T12:18:20Z |
| Scope | Faculty FCIFF; the 5 program collections below; `dc.date.issued` from 2021 to 2026; every item type (theses are selected later, in T12) |
| Query | `server/api/discover/search/objects`, `dsoType=ITEM`, `f.dateIssued=[2021 TO 2026]`, sorted by `dc.date.issued` ascending, one listing per collection |
| Reconciliation | 766 records, equal to the faculty-wide total of the same query; 766 unique item uuids |
| License | CC BY-NC-ND 4.0: research use only, no redistribution of the texts |
| Location | `data/raw/20261002T224412Z/`, gitignored, local only |

| Program key | Program | Collection uuid | Records |
|---|---|---|---:|
| `sistemas` | Ingeniería de Sistemas | `695b14ab-e5b2-49b5-9edf-f709e883b73b` | 67 |
| `industrial` | Ingeniería Industrial | `f0217548-ce48-4bae-b163-21b5e2504ef6` | 349 |
| `electronica` | Ingeniería Electrónica | `a8bdd7c0-f1aa-4797-b04e-112e27f60da3` | 55 |
| `mecanica` | Ingeniería Mecánica, Mecánica-Eléctrica y Mecatrónica | `cf0ca97e-9b81-484d-a325-611b8a8d8223` | 192 |
| `minas` | Ingeniería de Minas | `9d9ecbe5-56c5-4cdf-81a3-b9a3736a47b3` | 103 |

## 2. Freeze record

- **Frozen as of 2026-10-03.** The snapshot is not modified and not re-downloaded. Every later
  task reads it as is.
- **No backup** (decision D23; risk R8 accepted by the user). If the snapshot is lost, a
  re-harvest is a new snapshot, and the digests below document how it differs.

| File (under `data/raw/20261002T224412Z/`) | Written by | Bytes | SHA-256 |
|---|---|---:|---|
| `metadata.jsonl` | T04 harvest | 4,744,184 | `d88839012df3e6f3bd7ee08f90c8e1f3b6588cf75386127dde52f429b4d9d932` |
| `manifest.json` (snapshot manifest) | T04 harvest | 1,857 | `1ed0c20590cafb8ae11e36675efbc188c4a91a08148644294272f216891594c8` |
| `pdfs/manifest.json` (PDF manifest) | T05 download | 487,971 | `4a1ecdba8e8a3fada0724da94c23ad87265c74259c8eca346ff254871e74816a` |
| `pdfs/*.pdf`, combined (731 files) | T05 download | 11,401,121,329 | `9c72910d5d0b88bfe2e64f7984d05df1b69ee8237c6c24054f9c167276f74586` |

The snapshot manifest records the same `metadata.jsonl` SHA-256 (`metadata_sha256`). The PDF
directory also holds `.lock`, the empty lock file of the downloader, which is not part of the
data.

**Combined PDF digest, recipe.** For each of the 731 PDFs, one line
`<item_uuid> <pdf_sha256>\n`: the item uuid (the file name without `.pdf`), one space, the
lowercase hex SHA-256 of the file, and a line feed, the last line included. Sort the lines by
item uuid, join them, and take the SHA-256 of the result (ASCII). The file hashes and the
`sha256` values of the PDF manifest give the same digest. From the repository root:

```python
import hashlib
import pathlib

pdfs = pathlib.Path("data/raw/20261002T224412Z/pdfs")
lines = sorted(
    f"{p.stem} {hashlib.sha256(p.read_bytes()).hexdigest()}\n" for p in pdfs.glob("*.pdf")
)
print(hashlib.sha256("".join(lines).encode()).hexdigest())
```

**Diffing a later re-harvest.** The repository is live: its item count changed during the
harvest day (plan, progress log of 2026-10-02). A re-harvest can be compared item by item through
the item uuids (`metadata.jsonl`, PDF manifest keys), the bitstream uuids (`bitstream_uuid`, 731
distinct) and the SHA-256 of each PDF.

## 3. Records

| `renati.type` group | Records | In the census |
|---|---:|---|
| `#tesis` | 746 | yes (D01) |
| Trabajo de suficiencia profesional, 5 spellings | 19 | no; T12 logs the reason |
| `#trabajoAcademico` | 1 | no; T12 logs the reason |
| **Total** | 766 | |

The spellings of the suficiencia type are listed in the
[metadata profile](../../eda/20261002T224412Z/summary.md#corpus).

| Program | Records | `#tesis` | Suficiencia profesional | Trabajo académico |
|---|---:|---:|---:|---:|
| `sistemas` | 67 | 65 | 2 | 0 |
| `industrial` | 349 | 340 | 9 | 0 |
| `electronica` | 55 | 46 | 8 | 1 |
| `mecanica` | 192 | 192 | 0 | 0 |
| `minas` | 103 | 103 | 0 | 0 |
| **Total** | 766 | 746 | 19 | 1 |

Theses (`#tesis`) by program and year of `dc.date.issued`:

| Program | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| `sistemas` | 19 | 18 | 5 | 9 | 6 | 8 | 65 |
| `industrial` | 64 | 39 | 48 | 44 | 90 | 55 | 340 |
| `electronica` | 16 | 8 | 3 | 5 | 4 | 10 | 46 |
| `mecanica` | 37 | 37 | 41 | 28 | 28 | 21 | 192 |
| `minas` | 3 | 14 | 8 | 12 | 34 | 32 | 103 |
| **Total** | 139 | 116 | 105 | 98 | 162 | 126 | 746 |

- Every record has a full `YYYY-MM-DD` issue date, from 2021-01-06 to 2026-12-04.
- 2026 is a partial year: it runs to the snapshot date, plus the 2 future-dated theses below.
- The table counts every record, so the 2 duplicate pairs (section 5) count twice. It matches the
  metadata profile.

## 4. Future-dated records

Two theses carry a `dc.date.issued` after the snapshot date (2026-10-02). No other record does.

| Handle | Program | `renati.type` | `dc.date.issued` | Accessioned | Rights | PDF |
|---|---|---|---|---|---|---|
| `20.500.12920/16433` | `industrial` | `#tesis` | 2026-12-01 | 2026-02-20 | open | on disk |
| `20.500.12920/16319` | `industrial` | `#tesis` | 2026-12-04 | 2026-01-19 | open | on disk |

Both were deposited months before their issue date, so the date is probably a cataloguing
error, but the snapshot keeps it as harvested. **The temporal split (T27) must handle these two
theses explicitly**: decide and document whether they belong to 2026, and never let a date after
the snapshot place them outside the evaluation window by accident. The fichas carry a
future-dated flag (T08) for this.

## 5. Duplicates (D24)

Two pairs of `#tesis` records share the title, abstract, authors, advisor, issue date, keywords
and rights (compared with whitespace collapsed and case folded) under different item uuids.
Counted once each, the theses number **744**.

| Program | Handles | PDFs |
|---|---|---|
| `industrial` | `20.500.12920/15247`, `20.500.12920/15265` | Two different files: 4,546,142 and 4,569,503 bytes (SHA-256 `4fad0eb5…`, `dab1311c…`) |
| `mecanica` | `20.500.12920/16449`, `20.500.12920/16450` | Byte-identical: 12,860,458 bytes each (SHA-256 `aa91d0af…`) |

T12 keeps one canonical record per pair and logs the other as `duplicate_of`. For the
industrial pair the rule must also choose which PDF to keep. Duplicates never enter the labeled
pairs (T20) or the redundancy evaluation.

## 6. Access rights

| Access (`dc.rights`, COAR) | Theses |
|---|---:|
| open (`c_abf2`) | 732 |
| embargoed (`c_f1cf`) | 8 |
| restricted (`c_16ec`) | 6 |
| **Total** | 746 |

The 20 non-thesis records are all open. Each record has exactly one `dc.rights` value.

**Embargo list.** One row per embargoed or restricted thesis. Restricted theses carry no end
date.

| Handle | Program | COAR right | Embargo end | Embargo end passed (as of 2026-10-02) |
|---|---|---|---|---|
| `20.500.12920/10563` | `mecanica` | `c_f1cf` embargoed | 2022-02-19 | yes |
| `20.500.12920/11013` | `industrial` | `c_f1cf` embargoed | 2022-08-19 | yes |
| `20.500.12920/11520` | `sistemas` | `c_f1cf` embargoed | 2022-10-29 | yes |
| `20.500.12920/11333` | `industrial` | `c_f1cf` embargoed | 2022-11-22 | yes |
| `20.500.12920/12374` | `industrial` | `c_f1cf` embargoed | 2024-01-13 | yes |
| `20.500.12920/13616` | `mecanica` | `c_f1cf` embargoed | 2025-05-03 | yes |
| `20.500.12920/14670` | `industrial` | `c_f1cf` embargoed | 2025-12-19 | yes |
| `20.500.12920/16876` | `sistemas` | `c_f1cf` embargoed | 2027-03-18 | no |
| `20.500.12920/13715` | `industrial` | `c_16ec` restricted | | n/a |
| `20.500.12920/15695` | `industrial` | `c_16ec` restricted | | n/a |
| `20.500.12920/16325` | `industrial` | `c_16ec` restricted | | n/a |
| `20.500.12920/16852` | `industrial` | `c_16ec` restricted | | n/a |
| `20.500.12920/17034` | `mecanica` | `c_16ec` restricted | | n/a |
| `20.500.12920/17482` | `industrial` | `c_16ec` restricted | | n/a |

- 7 of the 8 embargoes have already ended, yet the repository still lists the theses as
  embargoed. The downloader skipped all 14 by `dc.rights`, without any request.
- **Recommendation (from T05):** do not fetch the 7 expired-embargo PDFs for now. Fetching them
  needs a separate authorization (U8). These theses stay in the census through their metadata
  (title and abstract).
- One open thesis, `20.500.12920/12050` (`electronica`), also carries a past
  `dc.date.embargoEnd` (2023-10-12). Its PDF was downloaded; it is not on the list.

## 7. PDFs

| Program | Theses | PDF on disk | `restricted` | `no_thesis_file` | Errors |
|---|---:|---:|---:|---:|---:|
| `sistemas` | 65 | 63 | 2 | 0 | 0 |
| `industrial` | 340 | 330 | 9 | 1 | 0 |
| `electronica` | 46 | 46 | 0 | 0 | 0 |
| `mecanica` | 192 | 189 | 3 | 0 | 0 |
| `minas` | 103 | 103 | 0 | 0 | 0 |
| **Total** | 746 | 731 | 14 | 1 | 0 |

- **On disk** joins two manifest statuses: `downloaded` (148) and `already_present` (583,
  fetched by an earlier, interrupted run and re-verified when the run resumed).
- **No file:** `20.500.12920/13760` (`industrial`) has no ORIGINAL bundle.
- **Size:** 11,401,121,329 bytes in all (11.40 GB, 10.62 GiB). Per file: median 6,960,931 bytes
  (6.96 MB), min 1,328,386 bytes, max 412,091,393 bytes (412.1 MB, `20.500.12920/15113`).
- **Selection:** the only PDF of the item's ORIGINAL bundle that is neither a similarity report
  (`*.RT.pdf`) nor an authorization form (`Autoriz*`). Every on-disk file had exactly 1 candidate.

**Verification.** At download time each file had to start with `%PDF` and match the size and
MD5 that the repository lists; its SHA-256 was then recorded. For this card all 731 files were
re-hashed (2026-10-03):

- 731 of 731 start with `%PDF`, match the manifest size, SHA-256 and MD5, and the MD5 equals the
  server checksum.
- Each file is named `<item uuid>.pdf`, and the directory holds nothing else but the manifest and
  `.lock`.
- The manifest's 746 item uuids equal the 746 `#tesis` uuids of `metadata.jsonl`.
- Only two SHA-256 values repeat across the 731 files: the mecanica D24 pair, and the wrong file
  below.

**Wrong file in the repository.** `20.500.12920/11777` (`sistemas`) carries a byte-identical copy
of the PDF of `20.500.12920/11776` (SHA-256 `c3c990c1…`, different bitstream uuids). Their
metadata differ in every compared field (title, abstract, authors, advisor, date, keywords), and
the title check below gives 0.0 for 11777 against 1.0 for 11776. T12 treats 11777 as
metadata-only, with a quality note.

**Title check.** For each PDF, the share of the distinct tokens of the item's own title that
occur among the tokens of its first 4 pages. Tokens: lowercase, accents removed, alphanumeric runs
of 4 or more characters.

- Median coverage 1.0; 718 of 731 PDFs reach 0.9 or more.
- 4 are below 0.5: `20.500.12920/11777` (0.0, the wrong file), and `20.500.12920/10952` (0.15),
  `20.500.12920/11240` (0.33) and `20.500.12920/11177` (0.41).
- Those 3 are the right files under a different cover title: every distinct token of the item's
  own abstract occurs in the first 20 pages of its PDF (coverage 1.0, against 0.25 for 11777).

**Valid own PDF.** Over the 744 distinct theses (one record per D24 pair):

| Program | Distinct theses | Valid own PDF | Restricted | No file | Wrong file |
|---|---:|---:|---:|---:|---:|
| `sistemas` | 65 | 62 | 2 | 0 | 1 |
| `industrial` | 339 | 329 | 9 | 1 | 0 |
| `electronica` | 46 | 46 | 0 | 0 | 0 |
| `mecanica` | 191 | 188 | 3 | 0 | 0 |
| `minas` | 103 | 103 | 0 | 0 | 0 |
| **Total** | 744 | 728 | 14 | 1 | 1 |

The 16 metadata-only theses keep their title and abstract; their objectives fall back to title
and abstract (O05).

## 8. Text layer

Measured over the 731 PDFs with PyMuPDF `page.get_text()`. A page is **low-text** when its text,
stripped of surrounding whitespace, has fewer than 50 characters. A document is **text** when
under 10% of its pages are low-text, **mixed** from 10% to under 80%, and **scanned** at 80% or
more.

- **Pages:** 141,076 in all. Per thesis: min 65, P25 138, median 174, P75 230, P95 343, max 695.
- **Low-text pages:** 8,616 (6.1%); 8,595 of them hold at least one image. 707 PDFs have at least
  one low-text page.
- **Every PDF has a text layer on its first 4 pages:** at least one of them has 50 characters or
  more, in all 731.
- 0 PDFs are encrypted or password-protected.

| Program | PDFs | Text | Mixed | Scanned | Pages | Low-text pages |
|---|---:|---:|---:|---:|---:|---:|
| `sistemas` | 63 | 57 | 6 | 0 | 10,458 | 450 (4.3%) |
| `industrial` | 330 | 293 | 37 | 0 | 71,683 | 3,320 (4.6%) |
| `electronica` | 46 | 34 | 12 | 0 | 9,174 | 562 (6.1%) |
| `mecanica` | 189 | 132 | 56 | 1 | 37,386 | 3,533 (9.5%) |
| `minas` | 103 | 88 | 15 | 0 | 12,375 | 751 (6.1%) |
| **Total** | 731 | 604 | 126 | 1 | 141,076 | 8,616 (6.1%) |

The scanned document is `20.500.12920/15595` (`mecanica`): 86 of its 87 pages are low-text.

**Implication for T09:** OCR selectively. Most pages have a usable text layer, so OCR runs only on
the low-text pages of the sections the pipeline needs (the body for objectives, T10), not on
whole documents. A full pass with `get_text` alone takes about 4 minutes on this machine; OCR runs
are far longer, so they go into resumable CLIs run in the user's own terminal.

## 9. Privacy

- The DNI fields (`renati.author.dni`, `renati.advisor.dni`) were dropped at ingestion (T04),
  before anything was written. The snapshot manifest lists the dropped keys.
- This card holds no names, titles, abstracts, keywords of a single thesis, or ORCIDs. Before
  publishing, it was searched for every title, abstract, abstract sentence of 40 characters or
  more, author, advisor and juror name, and ORCID of the snapshot: 0 hits.
- Handles (`20.500.12920/…`) and uuids are public repository identifiers. PDFs are stored under
  their item uuid, never under their original file name.

## 10. Known issues and caveats

- **Future-dated theses (2):** issue dates after the snapshot date; T27 must place them
  explicitly (section 4).
- **Duplicate pairs (2, D24):** T12 keeps one canonical record per pair, logs the other as
  `duplicate_of`, and picks the industrial pair's PDF (section 5).
- **Wrong file (`20.500.12920/11777`):** T12 treats it as metadata-only with a quality note. T09
  extracts every PDF file, this one included, and T12 discards that text (section 7).
- **Metadata-only theses (16):** 14 restricted, 1 without a file, 1 wrong file; T10 falls back to
  title and abstract for their objectives (O05).
- **Expired embargoes (7):** not fetched; fetching them needs U8. They stay as metadata-only.
- **Mixed and scanned PDFs (126 + 1):** T09 detects low-text pages and OCRs selectively.
- **Cover titles that differ (3):** right files; the T12 quality check must not flag them as
  wrong files.
- **Non-thesis records (20):** kept in the snapshot; T12 excludes them with their reason.
- **Partial year:** 2026 runs only to the snapshot date, which matters for trends (T27, gap
  rule d).
- **Live repository, no backup:** a re-harvest will differ; compare it through the digests and
  uuids of section 2 (D23).
