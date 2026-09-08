"""Semantic graph utilities for Part 2 of the workshop."""

from __future__ import annotations

import asyncio
from typing import Any

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from neo4j import Driver
from neo4j_graphrag.embeddings import Embedder
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import OpenAILLM


class ChromaDefaultEmbedder(Embedder):
    """Adapt Chroma's local default embedding model to Neo4j GraphRAG."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding_function = DefaultEmbeddingFunction()

    def embed_query(self, text: str) -> list[float]:
        """Create one local embedding for a text value."""
        embedding = self.embedding_function([text])[0]
        return embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)


def create_simple_kg_pipeline(
    driver: Driver,
    model_name: str,
    base_url: str,
    api_key: str,
    schema: dict[str, Any],
    neo4j_database: str = "neo4j",
) -> SimpleKGPipeline:
    """Configure Neo4j's official semantic knowledge-graph pipeline."""
    llm = OpenAILLM(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        model_params={
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
    )
    return SimpleKGPipeline(
        llm=llm,
        driver=driver,
        embedder=ChromaDefaultEmbedder(),
        schema=schema,
        from_file=False,
        on_error="IGNORE",
        perform_entity_resolution=True,
        neo4j_database=neo4j_database,
    )


def build_semantic_kg(pipeline: SimpleKGPipeline, text: str):
    """Run the asynchronous SimpleKGPipeline from a normal Python script."""
    return asyncio.run(pipeline.run_async(text=text))
