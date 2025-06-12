"""
Defines the ElasticsearchVectorStore class for interacting with Elasticsearch via ElasticDB.
"""

import builtins
import logging
from collections.abc import Generator
from datetime import datetime, timezone
from typing import Any, Optional, TypeVar
from uuid import UUID

from elasticsearch import AsyncElasticsearch

from ragsdk.core import vector_stores
from ragsdk.core.embeddings import DenseEmbedder
from ragsdk.core.utils.config_handling import ObjectConstructionConfig
from ragsdk.core.utils.dict_transformations import flatten_dict
from ragsdk.core.vector_stores.base import (
    EmbeddingType,
    VectorStoreEntry,
    VectorStoreOptions,
    VectorStoreResult,
    VectorStoreWithDenseEmbedder,
    WhereQuery,
)

# set up module-level logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VectorStoreOptionsT = TypeVar("VectorStoreOptionsT", bound=VectorStoreOptions)

def get_index_settings(
    embedding_dims: int,
    distance_strategy: str,
    **kwargs: Any,  # noqa: ANN401
) -> dict[str, Any]:
    """Builds ES index settings & mappings for dense_vector storage."""
    settings = {
        "settings": {
            "analysis": {
                "normalizer": {
                    "lowercase_normalizer": {
                        "type": "custom",
                        "filter": ["lowercase"],
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "dense_vector",
                    "dims": embedding_dims,
                    "similarity": distance_strategy,
                    "index": True,
                },
                "chunk_content": {"type": "text"},
                "embedding_model": {"type": "keyword"},
                "filename": {"type": "keyword"},
                "score": {"type": "float"},
                "document_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "metadata": {"type": "object"},
                "ingest_date_time": {"type": "date"},
            }
        },
    }
    settings.update(kwargs)
    return settings


def bulk_data_generator(
    index: str, chunk_list: list[dict[str, Any]]
) -> Generator[dict[str, Any], None, None]:
    """Yields bulk‐index actions for Elasticsearch."""
    for chunk in chunk_list:
        yield {"_index": index, "doc": chunk}


class ElasticsearchVectorStore(
    VectorStoreWithDenseEmbedder[VectorStoreOptions]
):
    """
    Vector store implementation using Elasticsearch via AsyncElasticsearch.
    """

    options_cls = VectorStoreOptions
    default_module = vector_stores
    configuration_key = "vector_store"

    def __init__(
        self,
        user: str,
        password: str,
        host: str,
        index_name: str,
        embedder: DenseEmbedder,
        embedding_type: EmbeddingType = EmbeddingType.TEXT,
        distance_strategy: str = "cosine",
        default_options: Optional[VectorStoreOptions] = None,
    ) -> None:
        super().__init__(
            embedder=embedder,
            embedding_type=embedding_type,
            default_options=default_options,
        )
        if embedding_type == EmbeddingType.IMAGE and not embedder.image_support():
            raise ValueError("Embedder does not support image embeddings")

        self._index_name = index_name
        self._distance_strategy = distance_strategy
        self._client = AsyncElasticsearch(
            [host],
            basic_auth=(user, password),
        )
        logger.info("Initialized ElasticsearchVectorStore for index '%s' @ %s", index_name, host)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ElasticsearchVectorStore":
        """
        Create an ElasticsearchVectorStore instance from a configuration dictionary.

        Args:
            config: Dictionary containing configuration parameters like user, password,
                  host, index_name, and embedder configuration.

        Returns:
            An initialized ElasticsearchVectorStore instance.
        """
        # 1) pull off any default_options
        opts = None
        if "default_options" in config:
            opts = cls.options_cls(**config.pop("default_options"))

        # 2) pull and validate the raw embedder config
        raw = config.pop("embedder")
        ocfg = ObjectConstructionConfig.model_validate(raw)

        # 3) lookup the class in ragsdk.core.embeddings by name
        from ragsdk.core.embeddings import DenseEmbedder
        from ragsdk.core.utils.config_handling import import_by_path

        EmbedderCls = import_by_path(ocfg.type, DenseEmbedder.default_module)
        embedder = EmbedderCls(**ocfg.config)

        # 4) return a new instance
        inst = cls(
            user=config["user"],
            password=config["password"],
            host=config["host"],
            index_name=config["index_name"],
            embedder=embedder,
            embedding_type=config.get("embedding_type", EmbeddingType.TEXT),
            distance_strategy=config.get("distance_strategy", "cosine"),
            default_options=opts,
        )
        logger.debug("ElasticsearchVectorStore created from config: %s", config)
        return inst

    async def _create_index(self) -> None:
        # ask the embedder for its vector size
        vec_info = await self._embedder.get_vector_size()
        dims = vec_info.size
        logger.info("Creating Elasticsearch index '%s' with dims=%d, similarity='%s'",
                    self._index_name, dims, self._distance_strategy)
        try:
            await self._client.indices.create(
                index=self._index_name,
                body=get_index_settings(dims, self._distance_strategy),
            )
            logger.info("Index '%s' created successfully", self._index_name)
        except Exception as e:
            logger.error("Failed to create index '%s': %s", self._index_name, e)
            raise RuntimeError(f"Failed to create index {self._index_name}") from e

    async def store(self, entries: list[VectorStoreEntry]) -> None:
        """
        Store the provided entries in the Elasticsearch index.

        Args:
            entries: A list of VectorStoreEntry objects to store in the index.

        Returns:
            None
        """
        logger.info("Storing %d entries into index '%s'", len(entries), self._index_name)
        # 1) generate embeddings
        raw = await self._create_embeddings(entries)
        # 2) prepare chunks
        chunks: list[dict[str, Any]] = []
        now = datetime.now(timezone.utc).isoformat()
        for e in entries:
            vec = raw.get(e.id)
            if vec is None:
                logger.warning("No embedding generated for entry id=%s", e.id)
                continue
            chunks.append({
                "chunk_content": e.text or "",
                "embedding": vec,
                "embedding_model": getattr(self._embedder, "model_name", ""),
                "filename": e.metadata.get("filename"),
                "metadata": e.metadata,
                "ingest_date_time": now,
            })
        logger.debug("Prepared %d chunks for bulk ingest", len(chunks))
        # 3) ensure index exists
        exists = await self._client.indices.exists(index=self._index_name)
        if not exists:
            logger.info("Index '%s' does not exist, creating...", self._index_name)
            await self._create_index()
        else:
            logger.info("Index '%s' already exists", self._index_name)
        # 4) bulk ingest, supplying our UUID as the document _id
        logger.info("Starting bulk ingestion of %d documents", len(chunks))
        from elasticsearch.helpers import async_bulk  # type: ignore
        logger.info("Starting bulk ingestion of %d chunks", len(chunks))
        actions = []
        for entry, chunk in zip(entries, chunks, strict=False):
            # entry.id is a UUID
            actions.append({
                "_op_type": "index",               # index operation
                "_index": self._index_name,
                "_id": str(entry.id),              # force ES to use our UUID
                "_source": chunk,                  # the document body
            })
        success, _ = await async_bulk(self._client, actions)
        logger.info("Bulk ingest complete: %d succeeded", success)

        # make all ingested docs visible to search immediately
        await self._client.indices.refresh(index=self._index_name)
        logger.info("Index '%s' refreshed", self._index_name)

    async def retrieve(
        self,
        text: str,
        options: Optional[VectorStoreOptions] = None,
    ) -> list[VectorStoreResult]:
        """
        Retrieve entries from the Elasticsearch index based on the provided text query.

        Args:
            text: The text to use for similarity search.
            options: Options to configure the retrieval process, including filters and limits.

        Returns:
            A list of VectorStoreResult objects containing the matching entries.
        """
        opts = options or self.default_options or VectorStoreOptions()
        logger.info("Retrieving top %d results (threshold=%s) for query: '%s'",
                    opts.k, opts.score_threshold, text)
        qvec = (await self._embedder.embed_text([text]))[0]
        logger.debug("Query vector generated: %s…", qvec[:5])

        # build KNN query
        filt: list[dict[str, Any]] = []
        if opts.where:
            flat = flatten_dict(opts.where)
            for k, v in flat.items():
                filt.append({"term": {k: v}})
            logger.debug("Applying metadata filters: %s", flat)

        body: dict[str, Any] = {
            "knn": {
                "field": "embedding",
                "query_vector": qvec,
                "k": opts.k,
                "num_candidates": 100,
                "filter": {"bool": {"filter": filt}},
            },
            # Fetch the embedding vector too, so we can return it in VectorStoreResult
            "_source": {"includes": ["chunk_content", "metadata", "embedding"]},
            "size": opts.k,
        }
        if opts.score_threshold is not None:
            body["min_score"] = opts.score_threshold

        logger.debug("Search body: %s", body)
        resp = await self._client.search(index=self._index_name, body=body)

        hits = resp.get("hits", {}).get("hits", [])
        logger.debug("Raw hits _source examples: %r", [hit["_source"] for hit in hits[:2]])

        logger.info("Search returned %d hits", len(hits))


        results: list[VectorStoreResult] = []
        for hit in hits:
            src = hit["_source"]
            entry = VectorStoreEntry(
                id=UUID(hit["_id"]),
                text=src.get("chunk_content"),
                metadata=src.get("metadata", {}),
            )
            results.append(VectorStoreResult(
                entry=entry,
                vector=src["embedding"],  # type: ignore
                score=hit["_score"],
            ))
            logger.debug("Hit: id=%s score=%.4f", hit["_id"], hit["_score"])

        return results

    async def remove(self, ids: list[UUID]) -> None:
        """
        Remove entries from the Elasticsearch index by their IDs.

        Args:
            ids: A list of UUIDs identifying the entries to remove.

        Returns:
            None
        """
        logger.info("Removing %d documents from index '%s'", len(ids), self._index_name)
        for uid in ids:
            resp = await self._client.delete(index=self._index_name, id=str(uid), ignore=[404])
            logger.debug("Delete response for id=%s: %s", uid, resp)

    async def list(
        self,
        where: Optional[WhereQuery] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> list[VectorStoreEntry]:
        """
        List entries from the Elasticsearch index with optional filtering and pagination.

        Args:
            where: Optional query conditions to filter results.
            limit: Optional maximum number of entries to return. Defaults to 10 if None.
            offset: Starting position for pagination. Defaults to 0.

        Returns:
            A list of VectorStoreEntry objects matching the query conditions.
        """
        logger.info("Listing documents from index '%s' (where=%s limit=%s offset=%d)",
                    self._index_name, where, limit, offset)
        query: dict[str, Any] = {"match_all": {}}
        if where:
            flat = flatten_dict(where)
            filters = [{"term": {k: v}} for k, v in flat.items()]
            query = {"bool": {"filter": filters}}
            logger.debug("Applied listing filters: %s", flat)

        resp = await self._client.search(
            index=self._index_name,
            body={"query": query, "from": offset, "size": limit or 10},
        )
        hits = resp.get("hits", {}).get("hits", [])
        logger.info("List query returned %d hits", len(hits))

        out: list[VectorStoreEntry] = []
        for hit in hits:
            raw_id = hit["_id"]
            try:
                uid = UUID(raw_id)
            except ValueError:
                logger.warning("Skipping non-UUID document id=%s", raw_id)
                continue

            src = hit["_source"]
            entry = VectorStoreEntry(
                id=uid,
                text=src.get("chunk_content"),
                metadata=src.get("metadata", {}),
            )
            out.append(entry)
            logger.debug("Listed entry id=%s", raw_id)

        return out

    # -------------------------------------------------------------------------
    # Additional administrative methods matching ElasticDB surface:
    # -------------------------------------------------------------------------

    async def get_embedding_model(self) -> str:
        """
        Retrieve the name of the embedding model used in the Elasticsearch index.

        Returns:
            The name of the embedding model as a string.

        Raises:
            IndexError: If the index is empty or the embedding model field is not present.
        """
        resp = await self._client.search(
            index=self._index_name, body={"size": 1, "query": {"match_all": {}}}
        )
        return resp["hits"]["hits"][0]["_source"]["embedding_model"]

    async def list_index_stats(self) -> tuple[builtins.list[str], dict[str, Any]]:
        """
        Retrieve statistics for the Elasticsearch indices.

        Returns:
            A tuple containing:
            - A list of index names (excluding system indices)
            - A dictionary with index statistics including document count, storage,
              and search metrics
        """
        aliases = await self._client.indices.get_alias(
            index=self._index_name or "*",
            expand_wildcards="open"
        )
        idxs = [i for i in aliases if not i.startswith(".")]
        stats = (await self._client.indices.stats(metric=["docs", "store", "search"]))["indices"]
        return idxs, stats

    async def close(self) -> None:
        """
        Close the connection to the Elasticsearch client.

        This method should be called when the vector store is no longer needed to properly
        clean up resources and close the connection to the Elasticsearch server.

        Returns:
            None
        """
        await self._client.close()
