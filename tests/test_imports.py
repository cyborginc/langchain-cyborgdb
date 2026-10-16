"""Smoke tests: the package installs and the public API is importable.

These mirror the acceptance criteria on cyborginc/cyborgdb-core#2719 —
`from langchain_cyborgdb import CyborgVectorStore, StorageConfig` must work
in a clean environment against cyborgdb-core 0.18.x.
"""

from langchain_core.vectorstores import VectorStore

import langchain_cyborgdb
from langchain_cyborgdb import CyborgEmbeddings, CyborgVectorStore, StorageConfig


def test_cyborg_vector_store_is_a_langchain_vectorstore():
    assert issubclass(CyborgVectorStore, VectorStore)


def test_storage_config_factories_exist():
    assert callable(StorageConfig.disk)
    assert callable(StorageConfig.s3)


def test_public_api_matches_all():
    for name in langchain_cyborgdb.__all__:
        assert getattr(langchain_cyborgdb, name) is not None
    assert CyborgEmbeddings is not None
