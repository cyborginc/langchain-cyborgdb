"""Functional tests: the re-exported integration works end-to-end.

Adapted from cyborgdb-core's py_langchain_test.py, but importing everything
through ``langchain_cyborgdb`` — proving the wrapper exposes a working
integration, not just importable names.

Uses DeterministicFakeEmbedding (identical text -> identical vector), so
exact-text queries must return that text as the top hit and no embedding
model download or OpenAI key is needed. Semantic-quality tests (synonym /
paraphrase search) stay in cyborgdb-core where the real models live.

cyborgdb-core validates the CyborgDB API key against Cyborg's backend, so
the whole module is skipped unless CYBORGDB_API_KEY is set.
"""

import os
import uuid

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding

from langchain_cyborgdb import CyborgVectorStore, StorageConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("CYBORGDB_API_KEY"),
    reason="CYBORGDB_API_KEY not set; functional tests need a real key",
)

EMBEDDING = DeterministicFakeEmbedding(size=128)


@pytest.fixture
def make_store(tmp_path):
    """Build stores with unique index names; delete their indexes afterward."""
    stores = []

    def _make(texts=None, metadatas=None, ids=None, documents=None, **overrides):
        kwargs = dict(
            embedding=EMBEDDING,
            index_name=f"lc_wrapper_{uuid.uuid4().hex[:12]}",
            index_key=CyborgVectorStore.generate_key(),
            api_key=os.environ["CYBORGDB_API_KEY"],
            storage_config=StorageConfig.disk(str(tmp_path)),
            metric="cosine",
        )
        kwargs.update(overrides)
        if documents is not None:
            store = CyborgVectorStore.from_documents(documents=documents, **kwargs)
        elif texts is not None:
            store = CyborgVectorStore.from_texts(
                texts=texts, metadatas=metadatas, ids=ids, **kwargs
            )
        else:
            store = CyborgVectorStore(**kwargs)
        stores.append(store)
        return store

    yield _make

    for store in stores:
        try:
            store.delete(delete_index=True)
        except Exception:
            pass


def test_quickstart_flow(make_store):
    """The docs quickstart shape: from_texts -> similarity_search."""
    store = make_store(texts=["hello world", "goodbye world"])

    docs = store.similarity_search("hello world", k=2)

    assert len(docs) == 2
    assert docs[0].page_content == "hello world"
    assert {doc.page_content for doc in docs} == {"hello world", "goodbye world"}


def test_search_methods_return_contents_and_metadata(make_store):
    texts = ["cat on a mat", "dog in a fog", "bird on a wire", "fish in a dish"]
    metadatas = [{"animal": text.split()[0], "idx": i} for i, text in enumerate(texts)]
    expected = dict(zip(texts, metadatas))
    store = make_store(texts=texts, metadatas=metadatas)

    query = "cat on a mat"
    searches = {
        "similarity_search": store.similarity_search(query, k=4),
        "similarity_search_with_score": [
            doc for doc, _ in store.similarity_search_with_score(query, k=4)
        ],
        "similarity_search_by_vector": store.similarity_search_by_vector(
            EMBEDDING.embed_query(query), k=4
        ),
    }
    for name, docs in searches.items():
        assert len(docs) == len(texts), name
        for doc in docs:
            assert doc.page_content in expected, name
            assert doc.metadata == expected[doc.page_content], name


def test_metadata_filtering(make_store):
    texts = [f"Doc {i}" for i in range(10)]
    metadatas = [
        {"owner": {"name": "Jane" if i % 2 == 0 else "John"}} for i in range(10)
    ]
    store = make_store(texts=texts, metadatas=metadatas)

    results = store.similarity_search(
        "Doc 0", k=10, filter={"owner.name": {"$eq": "Jane"}}
    )

    assert results
    assert all(doc.metadata["owner"]["name"] == "Jane" for doc in results)


def test_delete_documents(make_store):
    texts = [f"Delete Me {i}" for i in range(5)]
    store = make_store()
    ids = store.add_texts(texts, metadatas=[{"n": i} for i in range(5)])
    assert len(ids) == 5

    store.delete(ids=[ids[1], ids[3]])

    remaining = store.similarity_search("Delete Me 0", k=5)
    remaining_ns = {doc.metadata["n"] for doc in remaining}
    assert remaining_ns == {0, 2, 4}


def test_mmr_search(make_store):
    texts = [f"topic {i} foxes and dogs" for i in range(10)]
    store = make_store(
        texts=texts, metadatas=[{"category": f"group_{i % 3}"} for i in range(10)]
    )

    results = store.max_marginal_relevance_search(
        "topic 1 foxes and dogs", k=3, fetch_k=6, lambda_mult=0.5
    )

    assert len(results) == 3


def test_from_documents(make_store):
    documents = [
        Document(page_content=f"LangChain doc {i}", metadata={"category": "test"})
        for i in range(10)
    ]
    store = make_store(documents=documents)

    results = store.similarity_search("LangChain doc 5", k=2)

    assert results
    assert results[0].page_content == "LangChain doc 5"
    assert results[0].metadata == {"category": "test"}


def test_list_ids_get_and_upsert(make_store):
    texts = ["alpha text", "beta text", "gamma text"]
    ids = ["doc_001", "doc_002", "doc_003"]
    store = make_store()
    store.add_texts(texts, [{"pos": i} for i in range(3)], ids=ids)

    assert sorted(store.list_ids()) == ids

    doc = store.get_document("doc_002")
    assert doc is not None
    assert doc.page_content == "beta text"
    assert doc.metadata == {"pos": 1}

    docs = store.get_documents(["doc_001", "doc_003", "doc_999"])
    assert len(docs) == 3
    assert docs[0] is not None and docs[1] is not None
    assert docs[2] is None

    # Re-adding an existing ID upserts rather than duplicating.
    store.add_texts(["alpha text updated"], ids=["doc_001"])
    assert len(store.list_ids()) == 3
    assert store.get_document("doc_001").page_content == "alpha text updated"


def test_as_retriever(make_store):
    texts = ["red apple", "green pear", "blue sky"]
    store = make_store(texts=texts)

    retriever = store.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke("red apple")

    assert len(docs) == 2
    assert docs[0].page_content == "red apple"
