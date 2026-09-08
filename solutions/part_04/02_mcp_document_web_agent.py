"""Agent combining MCP RAG, document tools, and web search."""

import uvicorn
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.capabilities import MCP, WebSearch

from graph_rag_workshop.utils.console_utils import INFO_STYLE, console, print_step
from graph_rag_workshop.utils.part_01_document_tools import (
    extract_text_from_image_file,
    extract_text_from_pdf_file,
    list_my_available_documents,
)
from graph_rag_workshop.utils.pydantic_utils import get_llm_model

MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"

agent = Agent(
    model=get_llm_model(),
    tools=[
        list_my_available_documents,
        extract_text_from_image_file,
        extract_text_from_pdf_file,
    ],
    instructions=(
        "You are a helpful workshop assistant. For Wayfarer website questions, "
        "use the MCP Graph-RAG tool and answer from its retrieved chunks and "
        "graph neighbors. Use the local document tools for files in the workshop "
        "document folder and web search only when external information is needed."
    ),
    model_settings=ModelSettings(thinking="minimal"),
    # Run MCP and web search locally so both Ollama and HEIA vLLM can use them.
    capabilities=[
        MCP(url=MCP_SERVER_URL, builtin=False),
        WebSearch(builtin=False),
    ],
)

if __name__ == "__main__":
    print_step("MCP Document and Web Agent")
    app = agent.to_web()
    console.print(
        "Starting MCP Document and Web Agent on http://127.0.0.1:8000",
        style=INFO_STYLE,
    )
    uvicorn.run(app, host="127.0.0.1", port=8000)
