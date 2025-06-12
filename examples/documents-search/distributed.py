"""
Ragsdk Document Search Example: Qdrant Distributed Ingest

This example is based on the "Qdrant" example, but it demonstrates how to ingest documents in
a distributed manner.
The distributed ingest is provided by "RayDistributedIngestStrategy" which uses Ray to
parallelize the ingest process.

The script performs the following steps:

    1. Create a list of documents.
    2. Initialize the `LiteLLMEmbedder` class
       with the OpenAI `text-embedding-3-small` embedding model.
    3. Initialize the `QdrantVectorStore` class with a `AsyncQdrantClient` HTTP instance
       and an index name.
    4. Initialize the `RayDistributedIngestStrategy` class with a standard params.
    5. Initialize the `DocumentSearch` class with the embedder and the vector store.
    6. Ingest the documents into the `DocumentSearch` instance using Ray distributed strategy.
    7. Search for documents using a query.
    8. Print the search results.

To run the script, execute the following command:

    ```bash
    uv run examples/document-search/distributed.py
    ```

The script ingests data to the Qdrant instance running on `http://localhost:6333`.
The recommended way to run it is using the official Docker image:

    1. Run Qdrant Docker container:

        ```bash
        docker run -p 6333:6333 qdrant/qdrant
        ```

    2. Open Qdrant dashboard in your browser:

        ```
        http://localhost:6333/dashboard
        ```
"""

# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "ragsdk-document-search[ray]",
#     "ragsdk-core",
# ]
# ///

import asyncio
import os

from dotenv import load_dotenv  # type: ignore
from qdrant_client import AsyncQdrantClient  # type: ignore

from ragsdk.core.embeddings.dense import LiteLLMEmbedder  # type: ignore
from ragsdk.core.vector_stores.qdrant import QdrantVectorStore  # type: ignore
from ragsdk.document_search import DocumentSearch  # type: ignore
from ragsdk.document_search.documents.document import DocumentMeta  # type: ignore
from ragsdk.document_search.ingestion.strategies import RayDistributedIngestStrategy  # type: ignore

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY in environment")

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
    vector_store = QdrantVectorStore(
        client=AsyncQdrantClient(
            host="localhost",
            port=6333,
        ),
        index_name="jokes",
        embedder=embedder,
    )
    ingest_strategy = RayDistributedIngestStrategy(batch_size=1)
    document_search = DocumentSearch(
        vector_store=vector_store,
        ingest_strategy=ingest_strategy,
    )

    await document_search.ingest(documents)

    results = await document_search.search("I'm boiling my water and I need a joke")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
