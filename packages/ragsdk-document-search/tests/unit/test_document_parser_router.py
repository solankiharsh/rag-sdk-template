import pytest

from ragsdk.core.utils.config_handling import ObjectConstructionConfig  # type: ignore
from ragsdk.document_search.documents.document import DocumentType  # type: ignore
from ragsdk.document_search.ingestion.parsers.base import (  # type: ignore
    ImageDocumentParser,
    TextDocumentParser,
)
from ragsdk.document_search.ingestion.parsers.exceptions import ParserNotFoundError  # type: ignore
from ragsdk.document_search.ingestion.parsers.router import DocumentParserRouter  # type: ignore
from ragsdk.document_search.ingestion.parsers.unstructured import (
    UnstructuredDocumentParser,  # type: ignore
)


def test_parser_router_from_config() -> None:
    config = {
        "txt": ObjectConstructionConfig.model_validate(
            {"type": "ragsdk.document_search.ingestion.parsers.base:TextDocumentParser"}
        ),
        "png": ObjectConstructionConfig.model_validate(
            {"type": "ragsdk.document_search.ingestion.parsers.base:ImageDocumentParser"}
        ),
        "pdf": ObjectConstructionConfig.model_validate(
            {"type": ("ragsdk.document_search.ingestion.parsers.unstructured:"
                      "UnstructuredDocumentParser")}
        ),
    }
    router = DocumentParserRouter.from_config(config)

    assert isinstance(router._parsers[DocumentType.TXT], TextDocumentParser)
    assert isinstance(router._parsers[DocumentType.PNG], ImageDocumentParser)
    assert isinstance(router._parsers[DocumentType.PDF], UnstructuredDocumentParser)


def test_parser_router_get() -> None:
    parser = TextDocumentParser()
    parser_router = DocumentParserRouter({DocumentType.TXT: parser})

    assert parser_router.get(DocumentType.TXT) is parser


def test_parser_router_get_raises_when_no_parser_found() -> None:
    parser_router = DocumentParserRouter()
    parser_router._parsers = {DocumentType.TXT: TextDocumentParser()}

    with pytest.raises(ParserNotFoundError) as exc:
        parser_router.get(DocumentType.PDF)

    assert exc.value.message == f"No parser found for the document type {DocumentType.PDF}"
    assert exc.value.document_type == DocumentType.PDF
