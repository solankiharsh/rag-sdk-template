import asyncio

from continuous_eval.metrics.retrieval.matching_strategy import RougeChunkMatch  # type: ignore
from datasets import load_dataset  # type: ignore

from ragsdk.core.embeddings.dense import LiteLLMEmbedder  # type: ignore
from ragsdk.core.sources.hf import HuggingFaceSource  # type: ignore
from ragsdk.core.vector_stores.in_memory import InMemoryVectorStore  # type: ignore
from ragsdk.document_search import DocumentSearch  # type: ignore
from ragsdk.document_search.documents.document import DocumentMeta  # type: ignore
from ragsdk.evaluate.dataloaders.document_search import DocumentSearchDataLoader
from ragsdk.evaluate.metrics import MetricSet
from ragsdk.evaluate.metrics.document_search import DocumentSearchPrecisionRecallF1


async def _add_example_documents(document_search: DocumentSearch) -> None:
    dataset = load_dataset(path="deepsense-ai/synthetic-rag-dataset_v1.0", split="train")
    documents = [DocumentMeta.from_literal(doc) for chunks in dataset["chunks"] for doc in chunks]
    await document_search.ingest(documents)


def basic_document_search_factory() -> DocumentSearch:
    """
    Factory for basic example document search instance.
    """
    document_search: DocumentSearch = DocumentSearch(
        vector_store=InMemoryVectorStore(embedder=LiteLLMEmbedder())
    )
    asyncio.run(_add_example_documents(document_search))
    return document_search


def synthetic_rag_dataset() -> DocumentSearchDataLoader:
    """
    Factory for synthetic RAG dataset.
    """
    return DocumentSearchDataLoader(
        source=HuggingFaceSource(path="deepsense-ai/synthetic-rag-dataset_v1.0")
    )


def precision_recall_f1() -> MetricSet:
    """
    Factory of precision recall f1 metric set for retrival evaluation.
    """
    return MetricSet(DocumentSearchPrecisionRecallF1(matching_strategy=RougeChunkMatch()))
