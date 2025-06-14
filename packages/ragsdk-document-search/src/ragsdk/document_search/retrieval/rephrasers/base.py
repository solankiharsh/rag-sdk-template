from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import ClassVar, TypeVar

from ragsdk.core.options import Options  # type: ignore
from ragsdk.core.utils.config_handling import ConfigurableComponent  # type: ignore
from ragsdk.document_search.retrieval import rephrasers


class QueryRephraserOptions(Options):
    """
    Object representing the options for the rephraser.
    """


QueryRephraserOptionsT = TypeVar("QueryRephraserOptionsT", bound=QueryRephraserOptions)


class QueryRephraser(ConfigurableComponent[QueryRephraserOptionsT], ABC):
    """
    Rephrases a query. Can provide multiple rephrased queries from one sentence / question.
    """

    options_cls: type[QueryRephraserOptionsT]
    default_module: ClassVar = rephrasers
    configuration_key: ClassVar = "rephraser"

    @abstractmethod
    async def rephrase(
        self, query: str, options: QueryRephraserOptionsT | None = None
    ) -> Iterable[str]:
        """
        Rephrase a query.

        Args:
            query: The query to rephrase.
            options: The options for the rephraser.

        Returns:
            The rephrased queries.
        """
