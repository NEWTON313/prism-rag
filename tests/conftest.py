import os

os.environ["PRISM_FAKE"] = "1"  # avant l'import de prism : config lit l'environnement à l'import

import pytest  # noqa: E402

from prism import corpus, retrieval  # noqa: E402
from prism.embeddings import default_embedder  # noqa: E402


@pytest.fixture(scope="session")
def articles():
    return corpus.load_articles()


@pytest.fixture(scope="session")
def chunks(articles):
    return corpus.chunk_articles(articles)


@pytest.fixture(scope="session")
def dense(chunks):
    return retrieval.DenseRetriever(chunks, default_embedder())


@pytest.fixture(scope="session")
def lexical(chunks):
    return retrieval.BM25Retriever(chunks)


@pytest.fixture(scope="session")
def hybrid(dense, lexical):
    return retrieval.HybridRetriever(dense, lexical)
