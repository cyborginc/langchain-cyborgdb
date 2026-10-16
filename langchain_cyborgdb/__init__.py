"""LangChain integration for CyborgDB.

This package is a thin wrapper around ``cyborgdb_core.integrations.langchain``,
published under the standalone name LangChain's integration directory expects.
All functionality lives in ``cyborgdb-core[langchain]``.
"""

from cyborgdb_core.integrations.langchain import (
    CyborgEmbeddings,
    CyborgVectorStore,
    StorageConfig,
)

__version__ = "0.1.0"

__all__ = ["CyborgVectorStore", "CyborgEmbeddings", "StorageConfig"]
