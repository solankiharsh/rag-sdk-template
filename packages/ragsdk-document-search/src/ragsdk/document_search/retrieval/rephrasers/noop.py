from collections.abc import Iterable

from ragsdk.core.audit.traces import traceable  # type: ignore
from ragsdk.document_search.retrieval.rephrasers.base import QueryRephraser, QueryRephraserOptions


class NoopQueryRephraser(QueryRephraser[QueryRephraserOptions]):
    """
    A no-op query paraphraser that does not change the query.
    """

    options_cls: type[QueryRephraserOptions] = QueryRephraserOptions

    @traceable
    async def rephrase(  # noqa: PLR6301
        self, query: str, options: QueryRephraserOptions | None = None
    ) -> Iterable[str]:  # noqa: PLR6301
        """
        Mock implementation which outputs the same query as in input.

        Args:
            query: The query to rephrase.
            options: The options for the rephraser.

        Returns:
            The list with non-transformed query.
        """
        return [query]
