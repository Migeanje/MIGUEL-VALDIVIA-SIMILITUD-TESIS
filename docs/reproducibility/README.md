# Reproducibility package

What an examiner or a developer needs to rerun this project and check its results: the data,
the environment, the configuration with its seeds, an index of every tracked result, and the
order of the commands. It describes the repository at commit `2e8f9df`, for snapshot
`20261002T224412Z`.

## Quick path

1. Read the [data card](data_card.md): what the corpus is and what may be shared.
2. Set up the [environment](environment.md): `uv sync`, the tessdata files and both models.
3. Follow the [how-to-run guide](how_to_run.md), from the step your data allows: from the
   harvest with network access and authorization (U8), or from step 3 with a copy of the frozen
   snapshot and the pseudonym key. Steps 21, 23, 25 and 28 need people: the calibration
   meeting, the user's choice of 3 or 4 annotators (U3), the annotators and the adjudicators.
4. Compare your outputs with the recorded ones through the [results index](results_index.md):
   each run's manifest records the SHA-256 of its inputs and outputs.

## Contents

| Document | Answers |
|---|---|
| [Data card](data_card.md) | Where the corpus comes from, how the snapshot was taken, its counts, fields, rights and codes, what git tracks, the privacy guards and the known limits |
| [Environment](environment.md) | Python, uv and the lock file, CPU-only PyTorch, the models and tessdata, the hardware, the library versions the manifests record, and the app's settings |
| [Configuration and seeds](configuration.md) | Every block of `config/default.yaml`, the decision that fixed it, the file's SHA-256 at each change, the pinned experiment declaration, and every seed with where it is used |
| [Results index](results_index.md) | Every tracked file under `results/`: its command, commit, provenance and content; the tracked files not written yet; and the gitignored files the same commands write |
| [How to run](how_to_run.md) | The 30 steps in order and the app, with inputs, outputs, nature and measured times; the network and authorization each needs; the checks |

Related sources: the snapshot's own card,
[`results/snapshot/20261002T224412Z/data_card.md`](../../results/snapshot/20261002T224412Z/data_card.md);
the processing-cost ledger, [`docs/processing_costs.md`](../processing_costs.md); the
expert-session kit, [`docs/expert_session/`](../expert_session/README.md); the repository
[README](../../README.md); and the plan,
[`odd/tasks/thematic-redundancy-system.md`](../../odd/tasks/thematic-redundancy-system.md),
which holds every decision (D, O, U ids) and each task's evidence.

`tests/unit/test_reproducibility_docs.py` checks this package against the repository: the
commands and options of the guide, the paths it names, the tracked results, the configuration
blocks and seeds, the declared counts and the recorded library versions.

## In `main` since the package's first version

The first version described `90fedf2`. These items were then added to `main` (M,
`git log --oneline 90fedf2..2e8f9df`), and the documents of this package now cover them:

| Item | Task | Commits |
|---|---|---|
| The expert-session kit: consent template, session script, questionnaire, judges' form, the samples' specification and the data protocol, in Spanish | T31 | `c97fcc7` |
| The Streamlit app: the map, the gaps and the query of a proposal | T28a, T28b | `34fca63`, `53d47ee` |
| T22's run command, with the consensus-labels contract, the closure of stage 2 and four `redundancy` keys, the fold-split seed among them | T22 | `a9b8969`, `1cac319` |
| The temporal split, and its real run on the snapshot | T27 | `ec2abf6`, `b161016`; records `eea584a` |
| The labeled set's workbooks (`pilot`, `import-pilot`, `main`), and the real `pilot` step | T20b | `af46c30`, `b969f0e`, `7a9fe30`; record `58198e5` |
| The workbooks' importer, the annotators' agreement, the adjudication and the consensus labels; the draw's summary now prints integers exactly | T20c | `8aa5a90` |
| The samples builder of the expert session, and its real run on 2026-10-29 at `2e8f9df`, whose records are gitignored (the plan's T31 entry of 2026-10-29) | T31 | `2d41c2c`, `108f6e7` |

## Pending

Later units or people add these; none of them exists at `2e8f9df`, and this package will gain
their rows when they do (M, checked against the plan's task list, "Progress log" and "Next
step").

| Item | Task | Waits for |
|---|---|---|
| The calibration meeting and the pilot's import, then the annotators' workbooks (steps 21 to 24) | T20b's steps | The meeting's date (D34 (7)); U3, `labeling.annotators` |
| The labeling round: two annotators per pair, then the import, Fleiss' κ, the adjudication of the disputed pairs and the consensus labels (steps 25 to 29) | T21, with T20c's steps | The annotators (U3) |
| The T22 real run: τ by 5-fold cross-validation with a bootstrap interval, SBERT against TF-IDF, and stage 2 of the selection, which confirms or replaces the provisional winner C4 (step 30) | T22 | The consensus labels of T21; the user's confirmation of the four calibration values (SR-T22-1) |
| The app's redundancy verdict at τ | T28 | The T22 run |
| A rerun of the T19-based outputs if stage 2 changes the model: the gap run and the temporal split | T22, T24a, T27 | The T22 run |
| The OpenAlex reference and the official-lines coverage view | T25 | U8, U5, U1 |
| The Dockerfile and its run guide | T29 | — |
| The expert validation: Aiken's V with the judges, the session, the questionnaire analysis and the success criteria | T30 to T32 | The expert panel (U6) |
| The ledger's open measurements: labeling hours, the Docker build; optionally, the stopword script's time | T21, T29, T11 | Those tasks |
