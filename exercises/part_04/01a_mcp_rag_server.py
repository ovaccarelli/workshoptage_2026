"""Expose the workshop's Wayfarer webpage Graph-RAG as an MCP tool."""

from pathlib import Path

import chromadb
from fastmcp import FastMCP

from graph_rag_workshop.settings import DEFAULT_NB_RETRIEVED_CHUNKS
from graph_rag_workshop.utils.part_02_graph_utils import (
    search_structural_graph_rag_context,
)

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_NAME = "part_01_argo_usecase_webpage"
GRAPH_JSON_PATH = DATA_DIR / "graph_solutions" / "webpage_chunk_graph.json"

VECTORSTORE_DATABASE = chromadb.PersistentClient(
    path=str(VECTORSTORES_DIR)
).get_collection(name=VECTORSTORE_NAME)

# The MCP server makes the existing workshop Graph-RAG available to remote agents.
mcp = FastMCP("Wayfarer Webpage Graph-RAG")


@mcp.tool()
def search_wayfarer_webpage_graph_rag(question: str) -> str:
    """Retrieve Wayfarer webpage chunks and their one-hop graph neighbors."""
    # EXERCISE - MCP Graph-RAG tool:
    # Reuse the structural Graph-RAG function from Parts 02 and 03.
    return search_structural_graph_rag_context(
        vector_database=VECTORSTORE_DATABASE,
        graph_json_path=GRAPH_JSON_PATH,
        question=question,
        top_k=DEFAULT_NB_RETRIEVED_CHUNKS,
    )


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=8001, path="/mcp")
