# langchain-cyborgdb

LangChain integration for [CyborgDB](https://www.cyborg.co), the Confidential Vector Database. Provides `CyborgVectorStore`, a [LangChain](https://www.langchain.com/) `VectorStore` backed by CyborgDB's encrypted vector indexes.

This package is a thin wrapper around the integration that ships in [`cyborgdb-core`](https://pypi.org/project/cyborgdb-core/) — it re-exports `CyborgVectorStore`, `CyborgEmbeddings`, and `StorageConfig` under the standalone package name that LangChain's integration directory expects.

## Installation

```bash
pip install langchain-cyborgdb
```

## Quick Start

The example below uses OpenAI embeddings (`pip install langchain-openai`), but any LangChain `Embeddings` implementation works. You'll need a CyborgDB API key from [cyborg.co](https://www.cyborg.co).

```python
from langchain_cyborgdb import CyborgVectorStore, StorageConfig
from langchain_openai import OpenAIEmbeddings

store = CyborgVectorStore.from_texts(
    texts=["hello world", "goodbye world"],
    embedding=OpenAIEmbeddings(),
    index_key=CyborgVectorStore.generate_key(),
    api_key="your-api-key",
    storage_config=StorageConfig.disk("/tmp/cyborgdb-langchain"),
    metric="cosine"
)

docs = store.similarity_search("hello")
```

## Key Features

- Full VectorStore API compatibility (`add_texts`, `similarity_search`, etc.)
- Async operations support
- Configurable distance metrics (`cosine`, `euclidean`, `squared_euclidean`)
- Document ID tracking and management
- Max marginal relevance search
- Factory methods (`from_texts`, `from_documents`) for quick setup

## Key Classes

### `CyborgVectorStore`

Implements the LangChain-compatible vector store interface. Supports:

- `.add_texts()` / `.add_documents()`
- `.similarity_search()` / `.similarity_search_with_score()`
- `.max_marginal_relevance_search()`
- `.get_document()` / `.get_documents()` / `.list_ids()` / `.delete()`
- `.as_retriever()` for use in LangChain chains
- Async variants (`.aadd_texts()`, `.asimilarity_search()`, etc.)

### `StorageConfig`

Configuration object for defining where index data is stored, built with `StorageConfig.disk(path)` or `StorageConfig.s3(bucket, ...)`.

## Documentation

Full documentation, including metadata filtering and hybrid (BM25) search, lives at [docs.cyborg.co](https://docs.cyborg.co/versions/latest/integrations/langchain/introduction).

Using the CyborgDB client/server deployment instead of the embedded engine? The same integration is available via `pip install cyborgdb[langchain]` — see the [Deployment Models Guide](https://docs.cyborg.co/versions/latest/intro/deployment-models).

## License

This wrapper is MIT licensed. It depends on `cyborgdb-core`, which is distributed under its own [proprietary license](https://www.cyborg.co).
