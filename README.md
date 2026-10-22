# langchain-cyborgdb

LangChain integration for [CyborgDB](https://www.cyborg.co?utm_source=pypi&utm_medium=readme&utm_campaign=2610-db-dev-partner-a), the Confidential Vector Database. Provides `CyborgVectorStore`, a [LangChain](https://www.langchain.com/) `VectorStore` backed by CyborgDB's encrypted vector indexes.

This package is a thin wrapper around the integration that ships in [`cyborgdb-core`](https://pypi.org/project/cyborgdb-core/) — it re-exports `CyborgVectorStore`, `CyborgEmbeddings`, and `StorageConfig` under the standalone package name that LangChain's integration directory expects.

## Installation

```bash
pip install langchain-cyborgdb
```

## Quick Start

The example below runs with no API key and no extra installs: it uses the built-in `all-MiniLM-L6-v2` embedding model and CyborgDB's free tier (`api_key=""`). Any LangChain `Embeddings` implementation also works, for example `OpenAIEmbeddings()` from [`langchain-openai`](https://pypi.org/project/langchain-openai/).

```python
import secrets
from langchain_cyborgdb import CyborgVectorStore, StorageConfig

# Generate an encryption key for the index.
# WARNING: store this key securely; data encrypted with a lost key is unrecoverable.
index_key = secrets.token_bytes(32)

store = CyborgVectorStore.from_texts(
    texts=["hello world", "goodbye world"],
    embedding="all-MiniLM-L6-v2",   # built-in model: no extra install, no paid key
    index_key=index_key,
    api_key="",                     # free tier; pass your key to lift the 1M-item cap
    storage_config=StorageConfig.disk("./cyborgdb-langchain"),
    metric="cosine",
)

for doc in store.similarity_search("hello", k=1):
    print(doc.page_content)  # -> "hello world"

# Delete the index so the example can be run again (the key above is not saved).
store.delete(delete_index=True)
```

Requires Python 3.10 or newer. Full guide: [CyborgDB LangChain docs](https://docs.cyborg.co/versions/latest/integrations/langchain/introduction?utm_source=pypi&utm_medium=readme&utm_campaign=2610-db-dev-partner-a).

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

Full documentation, including metadata filtering and hybrid (BM25) search, lives at [docs.cyborg.co](https://docs.cyborg.co/versions/latest/integrations/langchain/introduction?utm_source=pypi&utm_medium=readme&utm_campaign=2610-db-dev-partner-a).

Using the CyborgDB client/server deployment instead of the embedded engine? The same integration is available via `pip install cyborgdb[langchain]` — see the [Deployment Models Guide](https://docs.cyborg.co/versions/latest/intro/deployment-models?utm_source=pypi&utm_medium=readme&utm_campaign=2610-db-dev-partner-a).

## License

This wrapper is MIT licensed. It depends on `cyborgdb-core`, which is distributed under its own [proprietary license](https://www.cyborg.co?utm_source=pypi&utm_medium=readme&utm_campaign=2610-db-dev-partner-a).
