"""
Ragsdk Document Search Example: Chroma

This example demonstrates how to use the `DocumentSearch` class to search for documents
with a more advanced setup.
We will use the `LiteLLMEmbedder` class to embed the documents and the query,
the `ChromaVectorStore` class to store the embeddings.

The script performs the following steps:

    1. Create a list of documents.
    2. Initialize the `LiteLLMEmbedder` class with the OpenAI
       `text-embedding-3-small` embedding model.
    3. Initialize the `ChromaVectorStore` class with a `EphemeralClient` instance and an index name.
    4. Initialize the `DocumentSearch` class with the embedder and the vector store.
    5. Ingest the documents into the `DocumentSearch` instance.
    6. List all documents in the vector store.
    7. Search for documents using a query.
    8. Print the list of all documents and the search results.

To run the script, execute the following command:

    ```bash
    uv run examples/document-search/chroma.py
    ```
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ragsdk-document-search",
#     "ragsdk-core[chroma]",
# ]
# ///

import asyncio
import os

from chromadb import EphemeralClient  # type: ignore
from dotenv import load_dotenv  # type: ignore

from ragsdk.core.audit import set_trace_handlers  # type: ignore
from ragsdk.core.embeddings.dense import LiteLLMEmbedder, LiteLLMEmbedderOptions  # type: ignore
from ragsdk.core.vector_stores.base import VectorStoreOptions  # type: ignore
from ragsdk.core.vector_stores.chroma import ChromaVectorStore  # type: ignore
from ragsdk.document_search import DocumentSearch, DocumentSearchOptions  # type: ignore
from ragsdk.document_search.documents.document import DocumentMeta  # type: ignore

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

set_trace_handlers("cli")

documents = [
    DocumentMeta.from_literal(
        """
        RIP boiled water. You will be mist.
        """
    ),
    DocumentMeta.from_literal(
        """
        Why doesn't James Bond fart in bed? Because it would blow his cover.
        """
    ),
    DocumentMeta.from_literal(
        """
        Why programmers don't like to swim? Because they're scared of the floating points.
        """
    ),
    DocumentMeta.from_literal(
        """
        This one is completely unrelated.
        """
    ),
]


async def main() -> None:
    """
    Run the example.
    """
    embedder = LiteLLMEmbedder(
        model_name="text-embedding-3-small",
        default_options=LiteLLMEmbedderOptions(
            dimensions=1024,
            timeout=1000,
        ),
    )
    vector_store = ChromaVectorStore(
        client=EphemeralClient(),
        index_name="jokes",
        default_options=VectorStoreOptions(
            k=10,
            score_threshold=0.88,
        ),
        embedder=embedder,
    )
    document_search = DocumentSearch(
        vector_store=vector_store,
    )

    await document_search.ingest(documents)

    all_documents = await vector_store.list()

    print()
    print("All documents:")
    print([doc.metadata["content"] for doc in all_documents])

    query = "I'm boiling my water and I need a joke"
    vector_store_options = VectorStoreOptions(
        k=2,
        score_threshold=0.4,
    )
    options = DocumentSearchOptions(vector_store_options=vector_store_options)
    results = await document_search.search(query, options)

    print()
    print(f"Documents similar to: {query}")
    print([element.text_representation for element in results])


if __name__ == "__main__":
    asyncio.run(main())
