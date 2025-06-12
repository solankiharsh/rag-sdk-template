import asyncio

from continuous_eval.metrics.retrieval.matching_strategy import RougeChunkMatch  # type: ignore
from typer.testing import CliRunner  # type: ignore

from ragsdk.core.embeddings.dense import NoopEmbedder  # type: ignore
from ragsdk.core.sources.hf import HuggingFaceSource  # type: ignore
from ragsdk.core.vector_stores.in_memory import InMemoryVectorStore  # type: ignore
from ragsdk.document_search import DocumentSearch  # type: ignore
from ragsdk.document_search.documents.document import DocumentMeta  # type: ignore
from ragsdk.evaluate.cli import eval_app  # type: ignore
from ragsdk.evaluate.dataloaders.document_search import DocumentSearchDataLoader  # type: ignore
from ragsdk.evaluate.metrics.base import MetricSet  # type: ignore
from ragsdk.evaluate.metrics.document_search import DocumentSearchPrecisionRecallF1  # type: ignore


def document_search_dataloader() -> DocumentSearchDataLoader:
    return DocumentSearchDataLoader(
        source=HuggingFaceSource(path="deepsense-ai/synthetic-rag-dataset_v1.0")
    )


def setup_document_search() -> DocumentSearch:
    documents = [
        DocumentMeta.from_literal("Foo document"),
        DocumentMeta.from_literal("Bar document"),
        DocumentMeta.from_literal("Baz document"),
    ]
    document_search: DocumentSearch = DocumentSearch(
        vector_store=InMemoryVectorStore(embedder=NoopEmbedder())
    )
    asyncio.run(document_search.ingest(documents))
    return document_search


def document_search_metrics() -> MetricSet:
    return MetricSet(
        DocumentSearchPrecisionRecallF1(
            matching_strategy=RougeChunkMatch(threshold=0.5)
        )
    )


def test_run_evaluation() -> None:
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        eval_app,
        [
            "--target-factory-path",
            f"{__name__}:setup_document_search",
            "--dataloader-factory-path",
            f"{__name__}:document_search_dataloader",
            "--metrics-factory-path",
            f"{__name__}:document_search_metrics",
            "run",
        ],
    )
    assert result.exit_code == 0
