"""Agentic workflow with multiple tools.

The agent can choose between:
- listing available source documents,
- searching with Graph-RAG,
- searching the web.
"""

from pathlib import Path

import chromadb
import uvicorn
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.settings import DEFAULT_NB_RETRIEVED_CHUNKS
from graph_rag_workshop.utils.console_utils import INFO_STYLE, console, print_step
from graph_rag_workshop.utils.part_01_document_tools import (
    extract_text_from_image_file,
    extract_text_from_pdf_file,
    list_my_available_documents,
)
from graph_rag_workshop.utils.part_02_graph_utils import (
    search_structural_graph_rag_context,
)
from graph_rag_workshop.utils.part_03_resilience_utils import (
    RetryAndRecoverCapability,
)
from graph_rag_workshop.utils.part_03_web_tools import safe_duckduckgo_search
from graph_rag_workshop.utils.pydantic_utils import get_llm_model

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
VECTORSTORE_NAME = "part_01_argo_usecase_webpage"
GRAPH_JSON_PATH = DATA_DIR / "graph_solutions" / "webpage_chunk_graph.json"


agent = Agent(
    model=get_llm_model(),
    instructions=(
        "You are a helpful assistant. Use the available tools to answer the user "
        "question. For Wayfarer website questions, use the Graph-RAG tool and "
        "answer only from its chunks and graph neighbors. Treat Name and Section "
        "path as authoritative metadata. If a tool returns TOOL_ERROR or "
        "WEB_SEARCH_ERROR, report it briefly and continue with another available "
        "tool or with the context already collected. Never abandon the complete "
        "answer only because one tool failed. Before every tool call, write one "
        "short user-visible sentence explaining which tool you will use and why. "
        "After the tool returns, briefly summarize what useful information it "
        "provided before choosing the next action. Do not reveal private "
        "chain-of-thought."
    ),
    capabilities=[RetryAndRecoverCapability()],
    # EXERCISE - Document tools:
    # Register the three document helpers and safe_duckduckgo_search.
    tools=...,
    model_settings=ModelSettings(
        thinking="minimal",
    ),
    tool_retries=2,
    output_retries=2,
)


VECTORSTORE_DATABASE = chromadb.PersistentClient(
    path=str(VECTORSTORES_DIR)
).get_collection(
    name=VECTORSTORE_NAME,
)

# EXERCISE - Graph-RAG tool:
# Add the structural webpage Graph-RAG tool, as in the previous exercise.
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
    print_step("Document Tools Agent")
    app = agent.to_web()
    console.print(
        "Starting Document Tools Agent on http://127.0.0.1:8000",
        style=INFO_STYLE,
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)
