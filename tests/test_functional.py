"""Functional test: the docs quickstart flow runs against the wrapper.

Exercises the embedded quickstart end-to-end (create index -> add texts ->
similarity search) with a deterministic local embedding, so no OpenAI key is
needed. cyborgdb-core validates the CyborgDB API key against Cyborg's
backend, so this test requires a real CYBORGDB_API_KEY and is skipped when
the environment does not provide one.
"""

import os
import tempfile

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cyborgdb import CyborgVectorStore, StorageConfig

requires_api_key = pytest.mark.skipif(
    not os.environ.get("CYBORGDB_API_KEY"),
    reason="CYBORGDB_API_KEY not set; functional quickstart needs a real key",
)


@requires_api_key
def test_quickstart_flow():
    with tempfile.TemporaryDirectory() as index_dir:
        store = CyborgVectorStore.from_texts(
            texts=["hello world", "goodbye world"],
            embedding=DeterministicFakeEmbedding(size=128),
            index_key=CyborgVectorStore.generate_key(),
            api_key=os.environ["CYBORGDB_API_KEY"],
            storage_config=StorageConfig.disk(index_dir),
            metric="cosine",
        )

        docs = store.similarity_search("hello", k=2)

        assert len(docs) == 2
        assert {doc.page_content for doc in docs} == {"hello world", "goodbye world"}
