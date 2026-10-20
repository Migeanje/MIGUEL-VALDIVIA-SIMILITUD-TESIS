"""The experiment runner of T19 (D30 E1 to E6) on synthetic theses, with recording stand-ins.

UMAP and BERTopic are stand-ins (a seeded random projection, and the BERTopic fakes of
``conftest.py``); K-means is scikit-learn's, wrapped to record its calls; HDBSCAN is hdbscan's,
wrapped to record its calls and to mark two seed-dependent theses as noise, so the two ARI noise
handlings differ. The metrics are T17's, recomputed here from the labels the runner returns.
"""

import json
import math
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from thematic_redundancy.evaluation.clustering_metrics import internal_metrics
from thematic_redundancy.evaluation.coherence import npmi_coherence
from thematic_redundancy.evaluation.stability import seed_stability
from thematic_redundancy.representation.tfidf import TfidfSettings, fit_tfidf
from thematic_redundancy.shared.config import load_config
from thematic_redundancy.topic_map import experiment as experiment_module
from thematic_redundancy.topic_map.bertopic_ctfidf import BertopicCtfidf
from thematic_redundancy.topic_map.build import TopicMapPorts, TopicMapSettings, topic_centroids
from thematic_redundancy.topic_map.clustering import (
    HdbscanSettings,
    KMeansSettings,
    hdbscan_model,
    kmeans_model,
)
from thematic_redundancy.topic_map.experiment import (
    DECLARED_SHA256,
    RUNTIME_FIELDS,
    ExperimentError,
    ExperimentInputs,
    ExperimentSettings,
    NpmiSetting,
    SemanticInput,
    check_declaration,
    matched_source,
    plan_experiment,
    reduction_key,
    require_terms_present,
    run_experiment,
    run_id,
)
from thematic_redundancy.topic_map.reduction import ProjectionSettings, UmapSettings
from thematic_redundancy.topic_map.run_experiment import (
    GitState,
    documents_by_row,
    main,
    read_git_state,
)
from thematic_redundancy.topic_map.selection import (
    SettingSummary,
    choose_k,
    choose_min_cluster_size,
    representative_seed,
    summarize,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"
DECLARED_MODELS = (
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
)
MODELS = ("org/model-a", "org/model-b")
TFIDF_SETTINGS = TfidfSettings(max_ngram=2, min_df=2, max_df=1.0, sublinear_tf=True)
SETTINGS = ExperimentSettings(
    topic_map=TopicMapSettings(
        umap=UmapSettings(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine"),
        projection=ProjectionSettings(method="umap", n_neighbors=15, min_dist=0.0, metric="cosine"),
        kmeans=KMeansSettings(n_init=2, init="k-means++"),
        hdbscan=HdbscanSettings(
            min_samples=None, metric="euclidean", cluster_selection_method="eom"
        ),
        top_n=5,
        max_ngram=2,
        include_noise_class=True,
    ),
    models=MODELS,
    k_values=(3, 4, 5),
    min_cluster_sizes=(3, 5),
    seeds=(7, 13, 21),
    npmi=NpmiSetting(unit="document", window_size=None),
    npmi_sensitivity=NpmiSetting(unit="window", window_size=10),
    ari="own_label",
    ari_sensitivity="exclude_either",
    gates=experiment_module.Gates(min_mean_ari=0.65, max_mean_noise_fraction=0.30),
    fallback_algorithm="kmeans",
    fallback_model=MODELS[0],
    max_cv=0.10,
    matched_k=True,
)


def make_inputs(corpus: Any, models: tuple[str, str] = MODELS) -> ExperimentInputs:
    rng = np.random.default_rng(5)
    rotation, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    second = corpus.vectors @ rotation + rng.normal(0, 0.5, corpus.vectors.shape)
    tfidf = fit_tfidf(corpus.documents, TFIDF_SETTINGS)
    return ExperimentInputs(
        snapshot_id="S1",
        doc_codes=tuple(f"D{index:03d}" for index in range(len(corpus.documents))),
        models=(
            SemanticInput(models[0], "rev-a", corpus.vectors.astype(np.float32)),
            SemanticInput(models[1], "rev-b", second.astype(np.float32)),
        ),
        tfidf=tfidf.section_mean,
        vocabulary=tfidf.terms,
        documents=corpus.documents,
        embedding_variant="section_mean",
        tfidf_variant="section_mean",
        input_sha256={"embeddings_manifest.json": "e" * 64, "tfidf_manifest.json": "t" * 64},
    )


class RecordingUmap:
    """UMAP stand-in: a seeded random projection; records every fit."""

    def __init__(self, drift: bool = False) -> None:
        self.calls: list[SimpleNamespace] = []
        self.drift = drift

    def __call__(self, settings: UmapSettings, seed: int) -> Any:
        owner = self

        class Model:
            def fit_transform(self, X: Any, *args: Any, **kwargs: Any) -> np.ndarray:
                matrix = np.asarray(X, dtype=np.float64)
                call = SimpleNamespace(
                    n=settings.n_components, seed=seed, X=matrix, extra=(args, kwargs)
                )
                owner.calls.append(call)
                rng = np.random.default_rng(seed + (len(owner.calls) if owner.drift else 0))
                call.output = matrix @ rng.normal(size=(matrix.shape[1], settings.n_components))
                return call.output.astype(np.float32)

        return Model()

    def fits(self, n_components: int) -> list[SimpleNamespace]:
        return [call for call in self.calls if call.n == n_components]


class RecordingKMeans:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, bool, tuple[int, int]]] = []

    def __call__(self, n_clusters: int, seed: int, settings: KMeansSettings) -> Any:
        return _Recorded(kmeans_model(n_clusters, seed, settings), self.calls, (n_clusters, seed))


class NoisyHdbscan:
    """hdbscan's HDBSCAN, then the two theses lowest on the first reduced axis marked noise."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, int, bool, tuple[int, int]]] = []

    def __call__(self, size: int, settings: HdbscanSettings) -> Any:
        return _Recorded(hdbscan_model(size, settings), self.calls, (size, -1), noisy=True)


class _Recorded:
    def __init__(self, model: Any, calls: list, key: tuple, noisy: bool = False) -> None:
        self.model, self.calls, self.key, self.noisy = model, calls, key, noisy

    def fit(self, X: Any) -> "_Recorded":
        self.calls.append((*self.key, sparse.issparse(X), X.shape))
        labels = np.array(self.model.fit(X).labels_)
        if self.noisy:
            labels[np.argsort(np.asarray(X)[:, 0], kind="stable")[:2]] = -1
        self.labels_ = labels
        return self


class CountingBackend:
    def __init__(self, inner: Any) -> None:
        self.inner, self.calls = inner, 0

    def class_weights(self, *args: Any) -> Any:
        self.calls += 1
        return self.inner.class_weights(*args)


class MemoryCache:
    def __init__(self) -> None:
        self.saved: dict[Any, np.ndarray] = {}

    def get(self, key: Any) -> np.ndarray | None:
        return self.saved.get(key)

    def put(self, key: Any, embedding: np.ndarray) -> None:
        self.saved[key] = np.array(embedding)


def make_ports(fakes: Any, drift: bool = False) -> tuple[TopicMapPorts, SimpleNamespace]:
    recorders = SimpleNamespace(
        umap=RecordingUmap(drift),
        kmeans=RecordingKMeans(),
        hdbscan=NoisyHdbscan(),
        ctfidf=CountingBackend(BertopicCtfidf(fakes.components())),
    )
    ports = TopicMapPorts(
        umap=recorders.umap,
        kmeans=recorders.kmeans,
        hdbscan=recorders.hdbscan,
        ctfidf=recorders.ctfidf,
    )
    return ports, recorders


def run(inputs: ExperimentInputs, fakes: Any, **kwargs: Any) -> SimpleNamespace:
    ports, recorders = make_ports(fakes, kwargs.pop("drift", False))
    settings = kwargs.pop("settings", SETTINGS)
    result = run_experiment(inputs, settings, ports=ports, **kwargs)
    return SimpleNamespace(result=result, inputs=inputs, settings=settings, **vars(recorders))


@pytest.fixture(scope="module")
def corpus39(corpus_maker: Any) -> Any:
    return corpus_maker(per_theme=12, outliers=3)


@pytest.fixture(scope="module")
def inputs(corpus39: Any) -> ExperimentInputs:
    return make_inputs(corpus39)


@pytest.fixture(scope="module")
def done(inputs: ExperimentInputs, fakes: Any) -> SimpleNamespace:
    return run(inputs, fakes)


@pytest.fixture(scope="module")
def random_first_model(inputs: ExperimentInputs, fakes: Any) -> SimpleNamespace:
    """Model A's vectors are noise, so its HDBSCAN is ineligible and its K-means (K 4) wins
    stage 1, while model B's HDBSCAN (3 clusters) wins its own and stage 2: the two stage-1
    winners differ in algorithm and K, and the winner's two ARI matrices pick different seeds."""
    noise = np.random.default_rng(3).normal(size=inputs.models[0].vectors.shape)
    first = replace(inputs.models[0], vectors=noise.astype(np.float32))
    changed = replace(inputs, models=(first, inputs.models[1]))
    return run(changed, fakes, settings=replace(SETTINGS, k_values=(4, 5, 6)))


@pytest.fixture(scope="module")
def low_noise_cap(inputs: ExperimentInputs, fakes: Any) -> SimpleNamespace:
    """A 5% noise cap leaves every HDBSCAN size ineligible by its noise alone."""
    gates = experiment_module.Gates(min_mean_ari=0.65, max_mean_noise_fraction=0.05)
    return run(inputs, fakes, settings=replace(SETTINGS, gates=gates))


@pytest.fixture(params=["shared", "random_first_model", "low_noise_cap"])
def scenario(request: pytest.FixtureRequest) -> SimpleNamespace:
    return request.getfixturevalue("done" if request.param == "shared" else request.param)


def rows_without_runtime(rows: Any) -> list[dict[str, Any]]:
    return [row.model_dump(exclude=set(RUNTIME_FIELDS)) for row in rows]


def vectors_of(inputs: ExperimentInputs, representation: str) -> Any:
    if representation == "tfidf":
        return inputs.tfidf
    return next(m.vectors for m in inputs.models if m.model_id == representation)


def seed_labels(done: SimpleNamespace, row: Any) -> list[np.ndarray]:
    parameter = row.k if row.algorithm == "kmeans" else row.min_cluster_size
    return [
        done.result.labels[run_id(row.representation, row.algorithm, parameter, seed)]
        for seed in SETTINGS.seeds
    ]


# The declaration and the settings


def test_the_settings_come_from_the_pinned_declaration() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    settings = ExperimentSettings.from_config(config)

    assert check_declaration(config) == DECLARED_SHA256
    assert settings.models == DECLARED_MODELS
    assert settings.k_values == tuple(range(8, 51, 2))
    assert (settings.min_cluster_sizes, settings.seeds) == ((5, 10, 15, 20), (7, 13, 21, 42, 73))
    assert settings.npmi == NpmiSetting(unit="document", window_size=None)
    assert settings.npmi_sensitivity == NpmiSetting(unit="window", window_size=10)
    assert (settings.ari, settings.ari_sensitivity) == ("own_label", "exclude_either")
    assert (settings.gates.min_mean_ari, settings.gates.max_mean_noise_fraction) == (0.65, 0.30)
    assert (settings.fallback_algorithm, settings.fallback_model) == ("kmeans", DECLARED_MODELS[0])
    assert (settings.max_cv, settings.matched_k, settings.topic_map.top_n) == (0.10, True, 10)


def test_a_changed_declaration_is_refused() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    changed = config.model_copy(update={"seeds": (7, 13, 21, 42, 74)})

    with pytest.raises(ExperimentError, match="pin"):
        ExperimentSettings.from_config(changed)


def test_the_fallback_model_must_be_a_compared_model(monkeypatch: pytest.MonkeyPatch) -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    embedding = config.embedding.model_copy(update={"models": ("a/one", "a/two")})
    changed = config.model_copy(update={"embedding": embedding})
    monkeypatch.setattr(
        experiment_module, "DECLARED_SHA256", changed.experiment_declaration_sha256()
    )

    with pytest.raises(ExperimentError, match="fallback"):
        check_declaration(changed)


# The sweeps


def test_one_unsupervised_umap_fit_per_model_and_seed_on_the_original_vectors(
    done: SimpleNamespace,
) -> None:
    reductions = done.umap.fits(5)
    first = done.inputs.models[0]

    # One per (model, seed), then the repeatability check refits the first one.
    assert len(reductions) == len(MODELS) * len(SETTINGS.seeds) + 1
    expected = [(m.model_id, s) for m in done.inputs.models for s in SETTINGS.seeds]
    expected.append((first.model_id, SETTINGS.seeds[0]))
    for call, (model_id, seed) in zip(reductions, expected, strict=True):
        assert call.seed == seed
        assert np.array_equal(call.X, vectors_of(done.inputs, model_id))
        assert call.extra == ((), {})


def test_semantic_runs_cluster_the_reduction_and_tfidf_runs_the_sparse_matrix(
    done: SimpleNamespace,
) -> None:
    n = len(done.inputs.doc_codes)
    semantic = [c for c in done.kmeans.calls if not c[2]]
    tfidf = [c for c in done.kmeans.calls if c[2]]

    assert {c[3] for c in semantic} == {(n, 5)}
    assert {c[3] for c in tfidf} == {done.inputs.tfidf.shape}
    assert {(c[2], c[3]) for c in done.hdbscan.calls} == {(False, (n, 5))}
    # Every K with every seed, the seed forwarded: the sweeps, the matched runs, the rerun.
    grid = sorted((k, s) for k in SETTINGS.k_values for s in SETTINGS.seeds)
    assert sorted(c[:2] for c in semantic) == sorted(grid * 2 + [(3, 7)])
    matched = [c[:2] for c in tfidf if c[0] not in SETTINGS.k_values]
    assert sorted(c[:2] for c in tfidf) == sorted(grid + matched)


def test_metrics_are_on_the_original_vectors_with_noise_left_out(done: SimpleNamespace) -> None:
    rows = done.result.runs
    noisy = [row for row in rows if row.noise_points]

    assert {row.algorithm for row in noisy} == {"hdbscan"}
    for row in rows:
        labels = done.result.labels[row.run_id]
        expected = internal_metrics(vectors_of(done.inputs, row.representation), labels)
        got = (row.silhouette_cosine, row.davies_bouldin_l2, row.calinski_harabasz_l2)
        assert got == (
            expected.silhouette_cosine,
            expected.davies_bouldin_l2,
            expected.calinski_harabasz_l2,
        )
        assert (row.noise_points, row.clusters) == (expected.noise_points, expected.clusters)
        assert row.noise_fraction == expected.noise_fraction


def test_the_reassigned_noise_silhouette_sends_noise_to_the_nearest_centroid(
    done: SimpleNamespace,
) -> None:
    for row in done.result.runs:
        if row.algorithm != "hdbscan":
            assert row.silhouette_noise_reassigned is None
            continue
        vectors = np.asarray(vectors_of(done.inputs, row.representation), dtype=np.float64)
        labels = np.array(done.result.labels[row.run_id])
        topics = sorted(set(labels.tolist()) - {-1})
        centroids = topic_centroids(vectors, labels, topics)
        noise = labels == -1
        nearest = (normalize(vectors[noise]) @ normalize(centroids).T).argmax(axis=1)
        labels[noise] = np.asarray(topics)[nearest]
        assert (
            row.silhouette_noise_reassigned == internal_metrics(vectors, labels).silhouette_cosine
        )


# The final runs: c-TF-IDF and NPMI


def test_ctfidf_and_npmi_run_for_the_final_runs_only(done: SimpleNamespace) -> None:
    finals = {row.run_id for row in done.result.runs if row.final}
    configurations = {
        (r.representation, r.algorithm, r.k, r.min_cluster_size)
        for r in done.result.runs
        if r.final
    }

    assert done.ctfidf.calls == len(finals)
    assert len(configurations) in (5, 6)  # C1 to C4 and B0, and a distinct matched K
    assert {row.run_id for row in done.result.topics} == finals
    for row in done.result.runs:
        assert (row.npmi is None) == (row.ctfidf_seconds is None) == (not row.final)


def test_npmi_scores_the_top_terms_at_both_declared_settings(done: SimpleNamespace) -> None:
    for row in done.result.runs:
        if not row.final:
            continue
        topics = [t for t in done.result.topics if t.run_id == row.run_id]
        terms = [t.terms for t in topics]
        assert all(len(t) == SETTINGS.topic_map.top_n for t in terms)
        documents = npmi_coherence(
            terms, done.inputs.documents, top_n=5, unit="document", window_size=None
        )
        windows = npmi_coherence(
            terms, done.inputs.documents, top_n=5, unit="window", window_size=10
        )
        assert (row.npmi, row.npmi_sensitivity) == (documents.mean, windows.mean)
        assert [t.npmi for t in topics] == [t.npmi for t in documents.topics]
        assert [t.npmi_sensitivity for t in topics] == [t.npmi for t in windows.topics]


def test_an_absent_top_term_stops_the_run() -> None:
    coherence = npmi_coherence([("a", "b")], [[("a",)]], top_n=2, unit="document", window_size=None)

    with pytest.raises(ExperimentError, match="absent"):
        require_terms_present(coherence)


def test_a_vocabulary_that_the_documents_do_not_hold_stops_the_run(
    inputs: ExperimentInputs, fakes: Any
) -> None:
    broken = replace(inputs, vocabulary=(*inputs.vocabulary[:-1], "término ausente"))

    with pytest.raises(ValueError):
        run(broken, fakes)


# ARI, the choices and the selection


def test_ari_is_over_the_seeds_under_both_noise_handlings(done: SimpleNamespace) -> None:
    differ = 0
    for row in done.result.configurations:
        labelings = seed_labels(done, row)
        primary = seed_stability(labelings, noise_handling="own_label")
        sensitivity = seed_stability(labelings, noise_handling="exclude_either")
        assert (row.ari_mean, row.ari_minimum) == (primary.mean, primary.minimum)
        assert (row.ari_sensitivity_mean, row.ari_sensitivity_minimum) == (
            sensitivity.mean,
            sensitivity.minimum,
        )
        assert json.loads(row.ari_matrix) == [list(r) for r in primary.matrix]
        differ += primary.mean != sensitivity.mean
    selected = done.result.selection["configurations"]
    for name, record in selected.items():
        row = next(r for r in done.result.configurations if r.configuration == name and r.chosen)
        assert record["ari_mean"] == row.ari_mean
    assert differ


def test_k_and_min_cluster_size_are_chosen_on_seed_means(scenario: SimpleNamespace) -> None:
    def summary(rows: list[Any], parameter: int) -> SettingSummary:
        return SettingSummary(
            parameter=parameter,
            silhouette=summarize([r.silhouette_cosine for r in rows]),
            noise_fraction=summarize([r.noise_fraction for r in rows]),
            metrics_defined=all(r.undefined_reason is None for r in rows),
        )

    runs, settings = scenario.result.runs, scenario.settings
    choices = scenario.result.selection["choices"]
    for representation in (*MODELS, "tfidf"):
        k_rows = [
            summary([r for r in runs if (r.representation, r.k) == (representation, k)], k)
            for k in settings.k_values
        ]
        assert choices[representation]["kmeans"]["parameter"] == choose_k(k_rows).parameter
    for model in MODELS:
        sizes = [
            summary([r for r in runs if (r.representation, r.min_cluster_size) == (model, s)], s)
            for s in settings.min_cluster_sizes
        ]
        cap = settings.gates.max_mean_noise_fraction
        chosen = choose_min_cluster_size(sizes, max_mean_noise_fraction=cap)
        got = choices[model]["hdbscan"]
        assert (got["parameter"], got["eligible"]) == (chosen.parameter, chosen.eligible)
    chosen_rows = [r for r in scenario.result.configurations if r.chosen]
    assert len(chosen_rows) == 5


def test_the_selection_is_stage_one_per_model_then_a_provisional_stage_two(
    done: SimpleNamespace,
) -> None:
    selection = done.result.selection
    names = {MODELS[0]: ("C1", "C2"), MODELS[1]: ("C3", "C4")}

    assert selection["provisional"] is True
    for model, pair in names.items():
        assert selection["stage_one"][model]["decision"]["winner"] in pair
        assert {s["name"] for s in selection["stage_one"][model]["sensitivities"]} >= {
            "npmi_sensitivity",
            "ari_sensitivity",
        }
    stage_two = selection["stage_two"]["decision"]
    assert stage_two["winner"] in MODELS
    assert [s["criterion"] for s in stage_two["steps"]][:2] == ["gates", "f1"]
    winner = selection["winner"]
    assert (
        winner["configuration"] == selection["stage_one"][stage_two["winner"]]["decision"]["winner"]
    )
    assert winner["provisional"] is True


def test_the_matched_k_and_criterion_one(scenario: SimpleNamespace) -> None:
    done, settings = scenario, scenario.settings
    selection = done.result.selection
    winner = selection["winner"]
    row = next(
        r
        for r in done.result.configurations
        if r.configuration == winner["configuration"] and r.chosen
    )
    expected = (
        winner["parameter"]
        if winner["algorithm"] == "kmeans"
        else math.floor(row.clusters_mean + 0.5)
    )
    b0_k = selection["choices"]["tfidf"]["kmeans"]["parameter"]

    assert selection["matched_k"]["k"] == expected
    assert (
        selection["matched_k"]["source"]
        == done.result.matched
        == matched_source(expected, b0_k, settings.k_values, True)
    )
    matched_runs = [r for r in done.result.runs if r.matched_k]
    baseline = done.result.matched == "baseline"
    assert {r.k for r in matched_runs} == (set() if baseline else {expected})
    assert len(matched_runs) == (0 if baseline else len(settings.seeds))
    criterion = selection["criterion_one"]
    assert criterion["f1"].startswith("pending")
    assert criterion["met"] in (None, False)


@pytest.mark.parametrize(
    ("name", "source", "new_runs"),
    [
        # Model B's HDBSCAN wins with 3 clusters, and K 3 lies outside the sweep: new runs.
        ("random_first_model", "new", 3),
        # HDBSCAN is ineligible under a 5% noise cap, so K-means wins at K 5, a sweep value.
        ("low_noise_cap", "sweep", 0),
    ],
)
def test_the_matched_runs_get_their_topics_and_records(
    request: pytest.FixtureRequest, name: str, source: str, new_runs: int
) -> None:
    done = request.getfixturevalue(name)
    k = done.result.selection["matched_k"]["k"]
    rows = [r for r in done.result.configurations if (r.representation, r.k) == ("tfidf", k)]

    assert done.result.matched == source
    assert [(r.configuration, r.chosen, r.final, r.matched_k) for r in rows] == [
        ("B0", False, True, True)
    ]
    matched = [r for r in done.result.runs if r.matched_k]
    assert {(r.k, r.final) for r in matched} == {(k, True)} and len(matched) == 3
    assert len(done.result.runs) == 39 + new_runs
    assert done.result.counts == plan_experiment(done.inputs, done.settings).counts(source)
    assert done.ctfidf.calls == 15 + 3
    assert done.ctfidf.calls == 15 + 3


@pytest.mark.parametrize(
    ("k", "baseline_k", "enabled", "source"),
    [
        (24, 24, True, "baseline"),
        (30, 24, True, "sweep"),
        (31, 24, True, "new"),
        (52, 24, True, "new"),
        (30, 24, False, "off"),
    ],
)
def test_the_matched_runs_come_from_where_the_matched_k_lies(
    k: int, baseline_k: int, enabled: bool, source: str
) -> None:
    assert matched_source(k, baseline_k, tuple(range(8, 51, 2)), enabled) == source


def test_the_representative_run_gives_the_map(scenario: SimpleNamespace) -> None:
    done = scenario
    selection = done.result.selection
    winner = selection["winner"]
    row = next(
        r
        for r in done.result.configurations
        if r.configuration == winner["configuration"] and r.chosen
    )
    seed = representative_seed(json.loads(row.ari_matrix), SETTINGS.seeds).seed
    vectors = vectors_of(done.inputs, winner["representation"])
    maps = done.umap.fits(2)

    assert selection["representative"]["seed"] == seed
    expected_run = run_id(winner["representation"], winner["algorithm"], winner["parameter"], seed)
    assert selection["representative"]["run_id"] == expected_run
    assert len(maps) == 1 and maps[0].seed == seed and np.array_equal(maps[0].X, vectors)
    assert [m.run_id for m in done.result.map] == [expected_run] * len(done.inputs.doc_codes)
    assert [m.label for m in done.result.map] == done.result.labels[expected_run].tolist()
    map_xy = np.array([(m.map_x, m.map_y) for m in done.result.map])
    np.testing.assert_array_equal(map_xy, maps[0].output.astype(np.float32))
    pca = PCA(n_components=2, random_state=seed).fit_transform(
        np.asarray(vectors, dtype=np.float64)
    )
    np.testing.assert_array_equal(np.array([(m.pca_x, m.pca_y) for m in done.result.map]), pca)


def test_criterion_four_and_the_repeatability_check(done: SimpleNamespace) -> None:
    criterion = done.result.selection["criterion_four"]

    assert criterion["max_cv"] == 0.10
    assert done.result.repeatability.umap_identical
    assert done.result.repeatability.kmeans_identical


def test_a_umap_that_does_not_repeat_is_recorded(inputs: ExperimentInputs, fakes: Any) -> None:
    again = run(inputs, fakes, drift=True)

    assert not again.result.repeatability.umap_identical


# The plan, determinism and resuming


def test_the_plan_counts_what_a_run_does(done: SimpleNamespace) -> None:
    plan = plan_experiment(done.inputs, SETTINGS)
    counts = plan.counts(done.result.matched)
    finals = sum(row.final for row in done.result.runs)

    assert done.result.counts == counts
    assert counts.umap_fits == len(done.umap.calls)
    assert counts.clusterings == len(done.kmeans.calls) + len(done.hdbscan.calls)
    assert counts.ctfidf_fits == done.ctfidf.calls == finals
    assert counts.internal_metrics == len(done.result.runs)
    assert counts.npmi_evaluations == 2 * finals
    assert counts.ari_evaluations == 2 * len(done.result.configurations)
    assert (plan.reductions, plan.reductions_cached) == (6, 0)


def test_a_rerun_gives_the_same_records_except_runtime_fields(
    done: SimpleNamespace, inputs: ExperimentInputs, fakes: Any
) -> None:
    again = run(inputs, fakes).result

    assert rows_without_runtime(again.runs) == rows_without_runtime(done.result.runs)
    assert again.configurations == done.result.configurations
    assert again.topics == done.result.topics
    assert again.map == done.result.map
    assert json.dumps(again.selection) == json.dumps(done.result.selection)
    assert all(np.array_equal(again.labels[k], v) for k, v in done.result.labels.items())


def test_a_rerun_resumes_from_the_cached_reductions(
    done: SimpleNamespace, inputs: ExperimentInputs, fakes: Any
) -> None:
    cache = MemoryCache()
    run(inputs, fakes, cache=cache)
    resumed = run(inputs, fakes, cache=cache)

    assert len(cache.saved) == 6
    assert len(resumed.umap.fits(5)) == 1  # the repeatability check only
    assert plan_experiment(inputs, SETTINGS, cache=cache).reductions_cached == 6
    assert resumed.result.reductions_reused == 6
    semantic = [r for r in resumed.result.runs if r.representation != "tfidf"]
    assert all(r.reduction_reused and r.umap_seconds is None for r in semantic)
    assert rows_without_runtime(resumed.result.runs) == rows_without_runtime(done.result.runs)


def test_the_reduction_key_names_the_vectors_settings_and_seed(inputs: ExperimentInputs) -> None:
    model = inputs.models[0]
    umap = SETTINGS.topic_map.umap
    key = reduction_key(model, umap, 7)

    assert key == reduction_key(model, umap, 7)
    assert key.sha256 != reduction_key(model, umap, 13).sha256
    assert key.sha256 != reduction_key(model, replace(umap, n_neighbors=10), 7).sha256
    changed = replace(model, vectors=model.vectors * 2)
    assert key.sha256 != reduction_key(changed, umap, 7).sha256


def test_inconsistent_inputs_are_refused(inputs: ExperimentInputs, fakes: Any) -> None:
    shorter = replace(inputs, documents=inputs.documents[:-1])
    renamed = replace(inputs, models=(inputs.models[1], inputs.models[0]))

    for broken in (shorter, renamed):
        with pytest.raises(ExperimentError):
            plan_experiment(broken, SETTINGS)


def test_run_ids_name_the_representation_algorithm_parameter_and_seed() -> None:
    assert run_id("tfidf", "kmeans", 24, 7) == "tfidf__kmeans_k24__seed7"
    assert run_id("org/model-a", "hdbscan", 10, 73) == "org--model-a__hdbscan_mcs10__seed73"


# The command


@pytest.fixture
def project(tmp_path: Path) -> Path:
    (tmp_path / "config").mkdir()
    shutil.copyfile(DEFAULT_CONFIG_PATH, tmp_path / "config" / "default.yaml")
    return tmp_path


def files_under(root: Path) -> set[str]:
    return {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}


def test_the_dry_run_forecasts_without_fitting_or_writing(
    project: Path, corpus39: Any, fakes: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    ports, recorders = make_ports(fakes)
    declared = make_inputs(corpus39, DECLARED_MODELS)
    git = lambda root, path: GitState(commit="c" * 40, dirty=("odd/notes.md",))  # noqa: E731

    code = main(
        ["--project-root", str(project), "--snapshot-id", "S1", "--dry-run"],
        load=lambda *_: declared,
        git=git,
        ports=ports,
    )

    out = capsys.readouterr().out
    assert code == 0
    assert files_under(project) == {"config/default.yaml"}
    assert not (recorders.umap.calls or recorders.kmeans.calls or recorders.hdbscan.calls)
    for text in (
        "UMAP fits: 12",
        "Clusterings: 371",
        "c-TF-IDF fits: 25",
        "e" * 64,
        DECLARED_SHA256,
        "1 changed path",
    ):
        assert text in out


def test_a_changed_declaration_stops_the_command(
    project: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = project / "config" / "default.yaml"
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            "seeds: [7, 13, 21, 42, 73]", "seeds: [7, 13, 21, 42, 74]"
        ),
        encoding="utf-8",
    )
    loads: list[Any] = []

    code = main(
        ["--project-root", str(project), "--snapshot-id", "S1"], load=lambda *a: loads.append(a)
    )

    assert code == 1
    assert "pin" in capsys.readouterr().err
    assert not loads
    assert files_under(project) == {"config/default.yaml"}


def test_a_dirty_tree_stops_a_real_run(project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    loads: list[Any] = []
    git = lambda root, path: GitState(commit="c" * 40, dirty=("src/x.py", "odd/y.md"))  # noqa: E731

    code = main(
        ["--project-root", str(project), "--snapshot-id", "S1"],
        load=lambda *a: loads.append(a),
        git=git,
    )

    assert code == 1
    assert "2 changed or untracked paths" in capsys.readouterr().err
    assert not loads
    assert files_under(project) == {"config/default.yaml"}


@pytest.mark.parametrize(("error", "code"), [(KeyboardInterrupt(), 130), (OSError("missing"), 1)])
def test_exit_codes_and_the_rerun_hint(
    project: Path, error: BaseException, code: int, capsys: pytest.CaptureFixture[str]
) -> None:
    def load(*_: Any) -> Any:
        raise error

    clean = lambda root, path: GitState(commit="c" * 40, dirty=())  # noqa: E731

    assert (
        main(["--project-root", str(project), "--snapshot-id", "S1"], load=load, git=clean) == code
    )
    err = capsys.readouterr().err
    assert ("run the command again" in err) == (code == 130)


def test_sections_are_grouped_into_theses_in_row_order() -> None:
    def section(code: str, row: int, tokens: tuple[str, ...]) -> Any:
        return SimpleNamespace(doc_code=code, document_row=row, tokens=tokens)

    sections = [section("A", 0, ("x",)), section("B", 1, ("y",)), section("A", 0, ("z",))]

    assert documents_by_row(sections, ("A", "B")) == ((("x",), ("z",)), (("y",),))
    with pytest.raises(ValueError):
        documents_by_row(sections, ("B", "A"))
    with pytest.raises(ValueError):
        documents_by_row(sections[:1], ("A", "B"))


def test_the_git_state_leaves_out_the_results_directory(tmp_path: Path) -> None:
    def git(*arguments: str) -> None:
        settings = ("-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false")
        command = ["git", "-C", str(tmp_path), *settings, *arguments]
        subprocess.run(command, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "a.txt").write_bytes(b"a")
    git("add", "a.txt")
    git("commit", "-q", "--no-verify", "-m", "first")
    results = tmp_path / "results" / "experiments" / "S1" / "t19"
    results.mkdir(parents=True)
    (results / "runs.parquet").write_bytes(b"x")

    assert read_git_state(tmp_path, "results/experiments/S1/t19").dirty == ()
    (tmp_path / "a.txt").write_bytes(b"b")
    (tmp_path / "results" / "other.json").write_bytes(b"{}")
    state = read_git_state(tmp_path, "results/experiments/S1/t19")
    assert sorted(state.dirty) == ["a.txt", "results/other.json"]
    assert len(state.commit) == 40
    with pytest.raises(OSError):
        read_git_state(tmp_path / "missing", "results")
