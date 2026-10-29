"""A synthetic corpus for the pair sampler's tests (T20a): 130 theses of the five configured
programs, with document vectors of both embedding models and a TF-IDF matrix, written nowhere.

The theses are drawn around one centroid per program, so the pairs within a program score
higher than the pairs between programs, while the noise leaves some pairs between programs in
every top. Industrial holds the most theses, as in the corpus. The theses come in a shuffled
order, and each representation lists them in another one, so a sampler that trusted an input
order would pair the wrong vectors. The pool sizes are the real configuration's: with 8,385
pairs, a top 1 % holds 84.
"""

import hashlib
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

from thematic_redundancy.labeling.draw_pairs import DrawInputs
from thematic_redundancy.labeling.pairs import (
    TFIDF,
    DocumentVectors,
    LabeledSetDraw,
    PairUniverse,
    build_universe,
    draw_labeled_set,
)
from thematic_redundancy.shared.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = load_config(PROJECT_ROOT / "config" / "default.yaml")
MODELS = CONFIG.embedding.models
PROGRAMS = tuple(program.key for program in CONFIG.snapshot.programs)
SIZES = {"sistemas": 20, "industrial": 50, "electronica": 15, "mecanica": 25, "minas": 20}
SNAPSHOT = "S1"
SENSITIVE = frozenset({"un título que no aparece", "inventadoapellido ficticiosegundo"})
INPUT_SHA256 = {
    "embeddings_manifest.json": "e" * 64,
    "tfidf_manifest.json": "t" * 64,
    "fichas_manifest.json": "f" * 64,
    "metadata.jsonl": "m" * 64,
}


def doc_code(index: int) -> str:
    return f"DOC-{hashlib.sha256(f'thesis {index}'.encode()).hexdigest()[:8]}"


@dataclass(frozen=True)
class Corpus:
    theses: tuple[tuple[str, str], ...]
    """``(doc_code, program)``, shuffled."""
    vectors: dict[str, np.ndarray | sparse.csr_matrix]
    """Each representation's vectors, one row per thesis of :attr:`theses`, in that order."""


def make_corpus() -> Corpus:
    rng = np.random.default_rng(34)
    programs = [program for program in PROGRAMS for _ in range(SIZES[program])]
    count = len(programs)
    centroids = {program: rng.normal(size=16) for program in PROGRAMS}
    latent = np.vstack([0.7 * centroids[p] + rng.normal(0, 0.9, 16) for p in programs])
    minilm = latent @ rng.normal(size=(16, 8)) + rng.normal(0, 0.3, (count, 8))
    mpnet = latent @ rng.normal(size=(16, 12)) + rng.normal(0, 0.3, (count, 12))
    terms = np.zeros((count, 40))
    for row, program in enumerate(programs):
        own = 8 * PROGRAMS.index(program)
        terms[row, own + rng.choice(8, size=3, replace=False)] = rng.uniform(0.2, 1.0, 3)
        terms[row, rng.choice(40, size=3, replace=False)] += rng.uniform(0.0, 0.6, 3)
    codes = [doc_code(index) for index in range(count)]
    assert len(set(codes)) == count
    order = list(range(count))
    random.Random(7).shuffle(order)
    return Corpus(
        theses=tuple((codes[i], programs[i]) for i in order),
        vectors={
            MODELS[0]: minilm[order].astype(np.float32),
            MODELS[1]: mpnet[order].astype(np.float32),
            TFIDF: sparse.csr_matrix(terms[order]),
        },
    )


def representation(corpus: Corpus, name: str, *, seed: int) -> DocumentVectors:
    """Return ``name``'s vectors with their theses in an order of their own."""
    order = list(range(len(corpus.theses)))
    random.Random(seed).shuffle(order)
    return DocumentVectors(
        name=name,
        doc_codes=tuple(corpus.theses[i][0] for i in order),
        vectors=corpus.vectors[name][order],
    )


def make_inputs(corpus: Corpus, *, sensitive: frozenset[str] = SENSITIVE) -> DrawInputs:
    return DrawInputs(
        theses=corpus.theses,
        semantic=tuple(
            representation(corpus, model, seed=index) for index, model in enumerate(MODELS)
        ),
        tfidf=representation(corpus, TFIDF, seed=9),
        sensitive=sensitive,
        input_sha256=dict(INPUT_SHA256),
    )


def universe_of(inputs: DrawInputs) -> PairUniverse:
    return build_universe(inputs.theses, inputs.semantic, inputs.tfidf, programs=PROGRAMS)


@pytest.fixture(scope="session")
def corpus() -> Corpus:
    return make_corpus()


@pytest.fixture(scope="session")
def inputs(corpus: Corpus) -> DrawInputs:
    return make_inputs(corpus)


@pytest.fixture(scope="session")
def universe(inputs: DrawInputs) -> PairUniverse:
    return universe_of(inputs)


@pytest.fixture(scope="session")
def draw(universe: PairUniverse) -> LabeledSetDraw:
    return draw_labeled_set(universe, CONFIG.labeling)


@pytest.fixture(scope="session")
def kit() -> SimpleNamespace:
    """The scenario's constants and builders, for the test modules of this folder."""
    return SimpleNamespace(
        config=CONFIG,
        models=MODELS,
        programs=PROGRAMS,
        sizes=SIZES,
        snapshot=SNAPSHOT,
        sensitive=SENSITIVE,
        input_sha256=INPUT_SHA256,
        make_inputs=make_inputs,
        universe_of=universe_of,
    )


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """A project root with the real configuration, in pytest's temporary directory."""
    (tmp_path / "config").mkdir()
    shutil.copyfile(PROJECT_ROOT / "config" / "default.yaml", tmp_path / "config" / "default.yaml")
    return tmp_path
