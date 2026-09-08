"""Agentic webpage Graph-RAG exercise.

The tool uses the structural graph built in part 02: it retrieves the same
top-k webpage chunks as RAG and adds all their one-hop graph neighbors.
"""

from pathlib import Path

import chromadb
import uvicorn
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.settings import DEFAULT_NB_RETRIEVED_CHUNKS
from graph_rag_workshop.utils.console_utils import INFO_STYLE, console, print_step
from graph_rag_workshop.utils.part_02_graph_utils import (
    search_structural_graph_rag_context,
)
from graph_rag_workshop.utils.pydantic_utils import get_llm_model

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_NAME = "part_01_argo_usecase_webpage"
GRAPH_JSON_PATH = DATA_DIR / "graph_solutions" / "webpage_chunk_graph.json"
VECTORSTORE_DATABASE = chromadb.PersistentClient(
    path=str(VECTORSTORES_DIR)
).get_collection(name=VECTORSTORE_NAME)


agent = Agent(
    model=get_llm_model(),
    instructions=(
        "You answer questions about the Wayfarer Hotels website. Use the "
        "Graph-RAG tool for website questions. Answer only from its retrieved "
        "chunks and graph neighbors. Treat Name and Section path as authoritative "
        "metadata when identifying venues, sections, pages, or other entities."
    ),
    model_settings=ModelSettings(thinking="minimal", output_retries=3),
)

# EXERCISE: Register this function as an agent tool.
...


def search_wayfarer_webpage_graph_rag(question: str) -> str:
    """Retrieve webpage chunks and all their one-hop structural neighbors."""
    return search_structural_graph_rag_context(
        vector_database=VECTORSTORE_DATABASE,
        graph_json_path=GRAPH_JSON_PATH,
        question=question,
        top_k=DEFAULT_NB_RETRIEVED_CHUNKS,
    )


if __name__ == "__main__":
    print_step("Graph-RAG Tool Agent")
    app = agent.to_web()
    console.print(
        "Starting Graph-RAG Tool Agent on http://127.0.0.1:8000",
        style=INFO_STYLE,
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)
