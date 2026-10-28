"""The query view's data layer (T28): one proposal represented as the corpus is, then its topic
(T23), its most similar theses (T22), its antecedents and alternative lines (T26), all from the
saved artifacts of the synthetic project, with the stand-in encoder."""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from app_artifacts import (
    CONFIG,
    MODEL_REVISION,
    STABLE_GAPS,
    THESES,
    WINNER_MODEL,
    FakeCounter,
    FakeEncoder,
    Project,
    files_under,
    load_fake_encoder,
    sha256_of,
    thesis_vector,
)

from thematic_redundancy.app import query
from thematic_redundancy.app.data import (
    AppSettings,
    ArtifactError,
    load_experiment,
    load_gaps,
    load_theses,
    load_topic_map,
    locate_sources,
)
from thematic_redundancy.app.query import (
    ModelError,
    Proposal,
    answer_query,
    load_proposal_encoder,
    load_query_index,
    proposal_ficha,
    represent,
)
from thematic_redundancy.corpus.ficha import Ficha, doc_code_for
from thematic_redundancy.gaps import results_store as gap_store
from thematic_redundancy.recommendation.alternatives import alternative_lines
from thematic_redundancy.recommendation.antecedents import rank_antecedents
from thematic_redundancy.redundancy.saturation import saturation_quartiles
from thematic_redundancy.redundancy.search import cosine_top_k
from thematic_redundancy.representation.build_chunks import (
    CHUNKS_MANIFEST_FILE_NAME,
    chunk_fichas,
)
from thematic_redundancy.representation.build_embeddings import BUILD_COMMAND as EMBED_COMMAND
from thematic_redundancy.representation.chunker import MaxTokenCounter
from thematic_redundancy.representation.vectors import pool_chunk_vectors
from thematic_redundancy.shared.storage import ParquetTableStore
from thematic_redundancy.topic_map.assignment import ATYPICAL_PERCENTILE, atypical_threshold

MAINTENANCE = Proposal(
    title="Plan de mantenimiento preventivo para una planta de prueba",
    abstract="Se propone un plan de mantenimiento preventivo para cada equipo de la planta.",
    objectives="Reducir cada falla del equipo con mantenimiento preventivo.",
)
"""An invented proposal whose words are those of topic 0."""
UNRELATED = Proposal(title="Un título sin palabras de ningún tema", abstract="Nada en común.")


def loaded(project: Project):
    sources = locate_sources(AppSettings(project.root, "S1"))
    experiment = load_experiment(sources)
    theses = load_theses(sources, experiment)
    topic_map = load_topic_map(sources, experiment, theses)
    records = load_gaps(sources, experiment, theses)
    return sources, experiment, topic_map, records


def index_of(project: Project) -> query.QueryIndex:
    sources, experiment, topic_map, records = loaded(project)
    return load_query_index(sources, experiment, topic_map, records)


def encoder_of(project: Project, fake_models) -> query.ProposalEncoder:
    sources, experiment, topic_map, records = loaded(project)
    return load_proposal_encoder(sources, load_query_index(sources, experiment, topic_map, records))


def refused(problem: str, artifact: str, call) -> ArtifactError:
    with pytest.raises(ArtifactError) as caught:
        call()
    assert (caught.value.problem, caught.value.artifact) == (problem, artifact)
    return caught.value


def corpus_ficha(title: str, abstract: str | None, objectives: str | None) -> Ficha:
    """A ficha of the corpus with these texts, as T12 builds one."""
    uuid = THESES[0].item_uuid
    return Ficha(
        item_uuid=uuid,
        doc_code=doc_code_for(uuid),
        handle_url="https://repo.example.edu/handle/1/1",
        snapshot_id="S1",
        document_type="tesis",
        renati_type_raw=None,
        title=title,
        abstract=abstract,
        objectives=objectives,
        objectives_status="extracted" if objectives else "no_pdf",
        program_key="sistemas",
        program_name="Ingeniería de Sistemas",
        issue_date_raw="2024",
        issued_after_snapshot=False,
        rights="open",
        include=True,
    )


def corpus_vector(ficha: Ficha) -> np.ndarray:
    """The corpus pipeline of T13 and T14 on one ficha, with the stand-ins."""
    counter = MaxTokenCounter(tuple(FakeCounter(model) for model in CONFIG.embedding.models))
    records = chunk_fichas([ficha], counter, CONFIG.chunking.max_tokens)
    vectors = FakeEncoder(WINNER_MODEL).encode([record.text for record in records])
    return pool_chunk_vectors(records, vectors).section_mean[0]


# --- The proposal -----------------------------------------------------------------------------


def test_a_proposal_needs_a_title_and_an_abstract_or_objectives() -> None:
    assert MAINTENANCE.missing() == ()
    assert Proposal(title="Título", abstract="Resumen.").missing() == ()
    assert Proposal(title="Título", objectives="Objetivo.").missing() == ()
    assert Proposal(title="  ", abstract="Resumen.").missing() == ("title",)
    assert Proposal(title="Título", abstract=" \n ", objectives="").missing() == ("body",)
    assert Proposal(title="").missing() == ("title", "body")


def test_the_proposal_ficha_holds_its_texts_and_no_objectives_when_blank() -> None:
    ficha = proposal_ficha(Proposal(title="Un título", abstract="Un resumen.", objectives=" "))

    assert (ficha.title, ficha.abstract, ficha.objectives) == ("Un título", "Un resumen.", None)
    assert ficha.include


@pytest.mark.parametrize(
    ("title", "abstract", "objectives"),
    [
        (MAINTENANCE.title, MAINTENANCE.abstract, MAINTENANCE.objectives),
        ("Plan de mantenimiento de planta, Arequipa 2024", "Un resumen del equipo.", None),
        ("Plan de logística", None, "Objetivo general: reducir el costo del almacén."),
    ],
)
def test_a_proposal_is_represented_as_the_corpus_represents_a_ficha(
    project: Project, fake_models, title: str, abstract: str | None, objectives: str | None
) -> None:
    encoder = encoder_of(project, fake_models)
    expected = corpus_vector(corpus_ficha(title, abstract, objectives))
    encoded_by_the_corpus = list(FakeEncoder.texts)
    FakeEncoder.texts.clear()

    vector = represent(Proposal(title, abstract or "", objectives or ""), encoder)

    np.testing.assert_allclose(vector, expected, rtol=0, atol=1e-12)
    assert FakeEncoder.texts == encoded_by_the_corpus


# --- The index of saved artifacts ---------------------------------------------------------------


def test_the_index_holds_the_winners_vectors_and_the_maps_labels(project: Project) -> None:
    index = index_of(project)

    assert index.model_id == WINNER_MODEL
    assert index.revision == MODEL_REVISION
    assert index.doc_codes == tuple(thesis.doc_code for thesis in THESES)
    expected = np.asarray([thesis_vector(thesis) for thesis in THESES], dtype=np.float32)
    np.testing.assert_array_equal(index.vectors, expected)
    assert list(index.topic_map.labels) == [thesis.topic for thesis in THESES]
    assert [topic.topic for topic in index.topic_map.topics] == [0, 1, 2]
    assert [topic.size for topic in index.topic_map.topics] == [5, 4, 3]


def test_the_atypical_threshold_and_saturation_are_t23s_with_the_confirmed_values(
    project: Project,
) -> None:
    index = index_of(project)
    labels = index.topic_map.labels

    expected = atypical_threshold(
        index.vectors,
        labels,
        index.centroids,
        percentile=ATYPICAL_PERCENTILE,
        method="linear",
        scope="global",
    )

    assert index.threshold == expected
    assert index.saturation == saturation_quartiles(labels, method="linear", boundary="lower")
    assert [entry.level for entry in index.saturation.topics] == ["Q4", "Q2", "Q1"]


def test_the_stable_gaps_are_rebuilt_with_t24s_stability_filter(project: Project) -> None:
    index = index_of(project)

    assert len(index.supports) == len(STABLE_GAPS)
    for support, row in zip(index.supports, STABLE_GAPS, strict=True):
        assert support.stable
        assert support.support == row.support
        assert support.gap.rule == row.rule
        assert support.gap.reason == row.reason


def test_embeddings_other_than_t19s_are_reported_as_stale(project: Project) -> None:
    path = project.embeddings_manifest
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    sources, experiment, topic_map, records = loaded(project)

    refused(
        "stale",
        path.name,
        lambda: load_query_index(sources, experiment, topic_map, records),
    )


def test_missing_embeddings_name_their_command(project: Project) -> None:
    project.embeddings_manifest.unlink()
    sources, experiment, topic_map, records = loaded(project)

    error = refused(
        "missing",
        project.embeddings_manifest.name,
        lambda: load_query_index(sources, experiment, topic_map, records),
    )
    assert error.command == EMBED_COMMAND


def test_a_chunk_size_other_than_the_corpus_chunks_is_reported_as_stale(project: Project) -> None:
    path = project.fichas_dir / CHUNKS_MANIFEST_FILE_NAME
    content = json.loads(path.read_text(encoding="utf-8"))
    content["max_tokens"] = CONFIG.chunking.max_tokens - 1
    path.write_text(json.dumps(content), encoding="utf-8")
    sources, experiment, topic_map, records = loaded(project)

    refused(
        "stale",
        CHUNKS_MANIFEST_FILE_NAME,
        lambda: load_query_index(sources, experiment, topic_map, records),
    )


def test_a_long_section_weighs_as_much_as_a_short_one(project: Project, fake_models) -> None:
    long_abstract = " ".join(["El mantenimiento preventivo de la planta reduce cada falla."] * 40)
    proposal = Proposal(title="Plan de logística", abstract=long_abstract)
    encoder = encoder_of(project, fake_models)
    counter = MaxTokenCounter(tuple(FakeCounter(model) for model in CONFIG.embedding.models))
    records = chunk_fichas([proposal_ficha(proposal)], counter, CONFIG.chunking.max_tokens)
    pooled = pool_chunk_vectors(
        records, FakeEncoder(WINNER_MODEL).encode([record.text for record in records])
    )
    assert sum(record.section == "abstract" for record in records) > 1

    vector = represent(proposal, encoder)

    np.testing.assert_allclose(vector, pooled.section_mean[0], rtol=0, atol=1e-12)
    assert not np.allclose(pooled.section_mean[0], pooled.plain_mean[0])


def test_a_gap_whose_rebuilt_reason_differs_from_its_record_is_refused(project: Project) -> None:
    path = project.gaps_results / gap_store.GAPS_FILE_NAME
    store = ParquetTableStore()
    rows = list(store.read(gap_store.GapRow, path))
    rows[0] = rows[0].model_copy(update={"reason": rows[0].reason.replace("3 theses", "4 theses")})
    store.write(rows, path)
    manifest_path = project.gaps_results / gap_store.MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][gap_store.GAPS_FILE_NAME] = sha256_of(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sources, experiment, topic_map, records = loaded(project)

    refused(
        "inconsistent",
        gap_store.GAPS_FILE_NAME,
        lambda: load_query_index(sources, experiment, topic_map, records),
    )


def test_gap_records_whose_stability_differs_from_t24s_filter_are_refused(
    project: Project,
) -> None:
    path = project.gaps_results / gap_store.GAPS_FILE_NAME
    store = ParquetTableStore()
    rows = list(store.read(gap_store.GapRow, path))
    rows[0] = rows[0].model_copy(update={"support": 3})
    store.write(rows, path)
    manifest_path = project.gaps_results / gap_store.MANIFEST_FILE_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][gap_store.GAPS_FILE_NAME] = sha256_of(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    sources, experiment, topic_map, records = loaded(project)

    refused(
        "inconsistent",
        gap_store.GAPS_FILE_NAME,
        lambda: load_query_index(sources, experiment, topic_map, records),
    )


# --- The model --------------------------------------------------------------------------------


def test_the_proposal_encoder_checks_the_model_and_the_tokenizers(
    project: Project, fake_models
) -> None:
    encoder = encoder_of(project, fake_models)

    assert encoder.encoder.identity.revision == MODEL_REVISION
    assert encoder.max_tokens == CONFIG.chunking.max_tokens
    assert encoder.seconds >= 0.0


def test_a_model_missing_from_the_local_cache_is_reported(project: Project) -> None:
    sources, experiment, topic_map, records = loaded(project)
    index = load_query_index(sources, experiment, topic_map, records)

    def missing(model_id: str, settings) -> FakeEncoder:
        raise OSError(f"modules.json of {model_id} is not in the local cache")

    with pytest.raises(ModelError) as caught:
        load_proposal_encoder(sources, index, load_encoder=missing, load_counter=FakeCounter)
    assert (caught.value.problem, caught.value.model_id) == ("missing", WINNER_MODEL)


def test_another_revision_of_the_model_is_refused(project: Project) -> None:
    sources, experiment, topic_map, records = loaded(project)
    index = load_query_index(sources, experiment, topic_map, records)

    with pytest.raises(ModelError) as caught:
        load_proposal_encoder(
            sources,
            index,
            load_encoder=lambda model_id, settings: FakeEncoder(model_id, revision="other"),
            load_counter=FakeCounter,
        )
    assert caught.value.problem == "mismatch"


def test_another_tokenizer_than_the_one_that_chunked_the_corpus_is_refused(
    project: Project,
) -> None:
    sources, experiment, topic_map, records = loaded(project)
    index = load_query_index(sources, experiment, topic_map, records)

    def other(model_id: str) -> FakeCounter:
        counter = FakeCounter(model_id)
        counter.identity = replace(counter.identity, tokenizer_sha256="0" * 64)
        return counter

    with pytest.raises(ModelError) as caught:
        load_proposal_encoder(sources, index, load_encoder=load_fake_encoder, load_counter=other)
    assert caught.value.problem == "mismatch"


# --- The answer -------------------------------------------------------------------------------


def test_the_answer_assigns_the_topic_with_its_flag_and_saturation(
    project: Project, fake_models
) -> None:
    encoder = encoder_of(project, fake_models)
    index = index_of(project)

    answer = answer_query(represent(MAINTENANCE, encoder), index)

    assert answer.lines.assignment.topic == 0
    assert answer.atypical.topic == 0
    assert answer.atypical.atypical is False
    assert answer.atypical.threshold == index.threshold.value_for(0)
    assert answer.saturation.level == "Q4"


def test_an_unrelated_proposal_is_flagged_atypical(project: Project, fake_models) -> None:
    encoder = encoder_of(project, fake_models)
    index = index_of(project)

    answer = answer_query(represent(UNRELATED, encoder), index)

    assert answer.atypical.atypical is True
    assert answer.atypical.similarity < answer.atypical.threshold


def test_the_answer_ranks_the_similar_theses_antecedents_and_lines_by_t22_and_t26(
    project: Project, fake_models
) -> None:
    encoder = encoder_of(project, fake_models)
    index = index_of(project)
    vector = represent(MAINTENANCE, encoder)
    recommender = CONFIG.recommender

    answer = answer_query(vector, index)

    assert answer.similar == cosine_top_k(
        index.vectors, vector, k=query.SIMILAR_THESES, exclude=None
    )
    assert {index.doc_codes[n.index] for n in answer.similar[:5]} == {
        t.doc_code for t in THESES if t.topic == 0
    }
    assert answer.antecedents == rank_antecedents(
        index.vectors,
        vector,
        k=recommender.top_k,
        mmr_lambda=recommender.mmr_lambda,
        pool_size=None,
        exclude=None,
    )
    expected = alternative_lines(
        vector,
        index.topic_map,
        index.supports,
        neighbor_topics=recommender.neighbor_topics,
        nearby_gaps=recommender.nearby_gaps,
        neighbor_reference="proposal",
        pair_anchor="midpoint",
        max_distance=None,
    )
    assert answer.lines == expected
    assert {n.topic for n in answer.lines.neighbors} == {1, 2}
    assert len(answer.lines.gaps) == recommender.nearby_gaps


def test_each_nearby_gap_is_one_of_the_gap_pages_stable_gaps(project: Project, fake_models) -> None:
    sources, experiment, topic_map, records = loaded(project)
    index = load_query_index(sources, experiment, topic_map, records)
    encoder = load_proposal_encoder(sources, index)

    answer = answer_query(represent(MAINTENANCE, encoder), index)

    found = query.nearby_stable_gaps(answer, index, records)
    assert len(found) == len(answer.lines.gaps)
    for nearby, gap in found:
        assert gap.row.rule == nearby.rule
        assert gap.row.topics == nearby.topics


def test_the_known_duplicates_are_the_configured_ones(project: Project) -> None:
    index = index_of(project)

    configured = {code for pair in CONFIG.redundancy.known_duplicates for code in pair}
    assert index.settings.known_duplicates == frozenset(configured)


def test_answering_a_proposal_writes_no_file(
    project: Project, fake_models, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    before = files_under(project.root)

    encoder = encoder_of(project, fake_models)
    answer_query(represent(MAINTENANCE, encoder), index_of(project))

    assert files_under(project.root) == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["project"]
