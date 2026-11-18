"""LangChain standard vector store test suite for CyborgVectorStore.

Runs the full ``VectorStoreIntegrationTests`` suite from ``langchain-tests``
(sync + async) against CyborgVectorStore, imported through the
``langchain_cyborgdb`` wrapper. Mirrors cyborgdb-core's
langchain_standard_test.py.

Requires: langchain-tests, pytest-asyncio (asyncio_mode=auto is set in
pyproject.toml). Run with:
    pytest tests/test_langchain_standard.py

Runs in cyborgdb-core's free mode (empty API key) by default, so no
CYBORGDB_API_KEY is required; if the env var is set, that key is used.
"""

import os
import uuid

import pytest

# Skip cleanly in envs without the standard-suite dependencies (e.g. a plain
# `pytest tests/` run with only the base test extra) instead of failing
# collection.
pytest.importorskip("langchain_tests", reason="standard suite needs langchain-tests")
pytest.importorskip("pytest_asyncio", reason="the suite's async half needs pytest-asyncio")

from langchain_core.vectorstores import VectorStore
from langchain_tests.integration_tests import VectorStoreIntegrationTests

from langchain_cyborgdb import CyborgVectorStore, StorageConfig


class TestCyborgVectorStoreStandard(VectorStoreIntegrationTests):
    @pytest.fixture()
    def vectorstore(self, tmp_path) -> VectorStore:
        store = CyborgVectorStore(
            index_name=f"std_{uuid.uuid4().hex[:12]}",
            index_key=os.urandom(32),
            # Empty key = cyborgdb-core's free mode; a real key is used if set.
            api_key=os.environ.get("CYBORGDB_API_KEY", ""),
            embedding=self.get_embeddings(),
            storage_config=StorageConfig.disk(str(tmp_path)),
        )
        yield store
        try:
            store.delete(delete_index=True)
        except Exception:
            pass
