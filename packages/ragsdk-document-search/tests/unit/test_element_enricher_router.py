import pytest

from ragsdk.core.utils.config_handling import ObjectConstructionConfig  # type: ignore
from ragsdk.document_search.documents.element import ImageElement, TextElement  # type: ignore
from ragsdk.document_search.ingestion.enrichers.exceptions import (
    EnricherNotFoundError,  # type: ignore
)
from ragsdk.document_search.ingestion.enrichers.image import ImageElementEnricher  # type: ignore
from ragsdk.document_search.ingestion.enrichers.router import ElementEnricherRouter  # type: ignore


def test_enricher_router_from_config() -> None:
    config = {
        "TextElement": ObjectConstructionConfig.model_validate(
            {"type": "ragsdk.document_search.ingestion.enrichers.image:ImageElementEnricher"}
        ),
        "ImageElement": ObjectConstructionConfig.model_validate(
            {"type": "ragsdk.document_search.ingestion.enrichers.image:ImageElementEnricher"}
        ),
    }
    router = ElementEnricherRouter.from_config(config)

    assert isinstance(router._enrichers[TextElement], ImageElementEnricher)
    assert isinstance(router._enrichers[ImageElement], ImageElementEnricher)


async def test_enricher_router_get() -> None:
    enricher = ImageElementEnricher()
    enricher_router = ElementEnricherRouter({ImageElement: enricher})

    assert enricher_router.get(ImageElement) is enricher


async def test_enricher_router_get_raises_when_no_enricher_found() -> None:
    enricher = ImageElementEnricher()
    enricher_router = ElementEnricherRouter()
    enricher_router._enrichers = {ImageElement: enricher}

    with pytest.raises(EnricherNotFoundError) as exc:
        enricher_router.get(TextElement)

    assert exc.value.message == f"No enricher found for the element type {TextElement}"
    assert exc.value.element_type == TextElement
