from pathlib import Path

import pytest

from ragsdk.core.llms.litellm import LiteLLM, LiteLLMOptions  # type: ignore
from ragsdk.core.utils.config_handling import ObjectConstructionConfig  # type: ignore
from ragsdk.document_search.documents.document import DocumentMeta  # type: ignore
from ragsdk.document_search.documents.element import (  # type: ignore
    Element,
    ImageElement,
    TextElement,
)
from ragsdk.document_search.ingestion.enrichers.base import ElementEnricher  # type: ignore
from ragsdk.document_search.ingestion.enrichers.exceptions import (
    EnricherElementNotSupportedError,  # type: ignore
)
from ragsdk.document_search.ingestion.enrichers.image import (  # type: ignore
    ImageDescriberPrompt,
    ImageElementEnricher,
)


def test_enricher_validates_supported_element_types() -> None:
    ImageElementEnricher.validate_element_type(ImageElement)

    with pytest.raises(EnricherElementNotSupportedError):
        ImageElementEnricher.validate_element_type(TextElement)


def test_enricher_validates_supported_document_union_types() -> None:
    class CustomElement(Element):
        @property
        def text_representation(self) -> str:
            return ""

    class CustomElementEnricher(ElementEnricher[CustomElement | TextElement]):
        pass

    CustomElementEnricher.validate_element_type(CustomElement)
    CustomElementEnricher.validate_element_type(TextElement)

    with pytest.raises(EnricherElementNotSupportedError):
        CustomElementEnricher.validate_element_type(ImageElement)


@pytest.mark.parametrize(
    ("enricher_type", "expected_enricher"),
    [
        (
            "ragsdk.document_search.ingestion.enrichers.image:ImageElementEnricher",
            ImageElementEnricher
        ),
        ("ImageElementEnricher", ImageElementEnricher),
    ],
)
def test_enricher_subclass_from_config(
    enricher_type: str,
    expected_enricher: type[ImageElementEnricher]
) -> None:
    config = ObjectConstructionConfig.model_validate(
        {
            "type": enricher_type,
            "config": {
                "llm": {
                    "type": "LiteLLM",
                    "prompt": ("ragsdk.document_search.ingestion.enrichers.image:"
                               "ImageDescriberPrompt"),
                },
            },
        }
    )
    enricher = ElementEnricher.subclass_from_config(config)  # type: ignore

    assert isinstance(enricher, expected_enricher)
    assert isinstance(enricher._llm, LiteLLM)
    assert enricher._prompt == ImageDescriberPrompt


async def test_image_enricher_call() -> None:
    default_options = LiteLLMOptions(mock_response='{"description": "response"}')
    llm = LiteLLM(
        model_name="gpt-4o",
        default_options=default_options,
    )
    document_meta = DocumentMeta.from_local_path(
        Path(__file__).parent.parent / "assets" / "img" / "transformers_paper_page.png"
    )
    document = await document_meta.fetch()
    element = ImageElement(
        document_meta=document_meta,
        image_bytes=document.local_path.read_bytes(),
        ocr_extracted_text="ocr text",
    )
    enricher = ImageElementEnricher(llm=llm)

    enriched_elements = await enricher.enrich([element])

    assert len(enriched_elements) == 1
    assert isinstance(enriched_elements[0], ImageElement)
    assert enriched_elements[0].description == "response"
    assert enriched_elements[0].image_bytes == element.image_bytes
    assert enriched_elements[0].ocr_extracted_text == element.ocr_extracted_text


async def test_image_enricher_call_fail() -> None:
    default_options = LiteLLMOptions(mock_response='{"description": "response"}')
    llm = LiteLLM(
        model_name="gpt-4o",
        default_options=default_options,
    )
    document_meta = DocumentMeta.from_local_path(
        Path(__file__).parent.parent / "assets" / "md" / "test_file.md"
    )
    document = await document_meta.fetch()
    element = TextElement(
        document_meta=document_meta,
        content=document.local_path.read_text(),
    )
    enricher = ImageElementEnricher(llm=llm)

    with pytest.raises(EnricherElementNotSupportedError) as exc:
        await enricher.enrich([element])  # type: ignore

    assert exc.value.message == (f"Element type {TextElement} is not supported by the "
                                f"{ImageElementEnricher.__name__}")
    assert exc.value.element_type == TextElement
    assert exc.value.enricher_name == ImageElementEnricher.__name__
