"""
Ragsdk Document Search Example: Basic

This example demonstrates how to use the `DocumentSearch` class to search for documents
with a minimal setup.
We will use the `LiteLLMEmbedder` class to embed the documents and the query
and the `InMemoryVectorStore` class to store the embeddings.

The script performs the following steps:

    1. Create a list of documents.
    2. Initialize the `LiteLLMEmbedder` class with the OpenAI
       `text-embedding-3-small` embedding model.
    3. Initialize the `InMemoryVectorStore` class.
    4. Initialize the `DocumentSearch` class with the embedder and the vector store.
    5. Ingest the documents into the `DocumentSearch` instance.
    6. Search for documents using a query.
    7. Print the search results.

To run the script, execute the following command:

    ```bash
    uv run examples/document-search/basic.py
    ```
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ragsdk-document-search",
#     "ragsdk-core",
# ]
# ///

import asyncio
import os

from dotenv import load_dotenv  # type: ignore

from ragsdk.core.audit import set_trace_handlers  # type: ignore
from ragsdk.core.embeddings.dense import LiteLLMEmbedder  # type: ignore
from ragsdk.core.vector_stores.in_memory import InMemoryVectorStore  # type: ignore
from ragsdk.document_search import DocumentSearch  # type: ignore
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
    )
    vector_store = InMemoryVectorStore(embedder=embedder)
    document_search = DocumentSearch(
        vector_store=vector_store,
    )

    await document_search.ingest(documents)

    results = await document_search.search("I'm boiling my water and I need a joke")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
