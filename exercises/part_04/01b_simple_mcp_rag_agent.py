"""Connect an agent to the Wayfarer Graph-RAG MCP server."""

import uvicorn
from pydantic_ai import Agent, ModelSettings
from pydantic_ai.capabilities import MCP

from graph_rag_workshop.utils.console_utils import INFO_STYLE, console, print_step
from graph_rag_workshop.utils.pydantic_utils import get_llm_model

MCP_SERVER_URL = "http://127.0.0.1:8001/mcp"

agent = Agent(
    model=get_llm_model(),
    instructions=(
        "You answer questions about the Wayfarer Hotels website. Always use "
        "the MCP Graph-RAG tool and answer only from its retrieved webpage "
        "chunks and graph neighbors. Say when the provided context is insufficient."
    ),
    # Discover and call the tools exposed by the local MCP server.
    capabilities=[MCP(url=MCP_SERVER_URL, builtin=False)],
    model_settings=ModelSettings(thinking="minimal"),
)


if __name__ == "__main__":
    print_step("MCP Wayfarer Graph-RAG Agent")
    app = agent.to_web()
    console.print("Starting MCP agent on http://127.0.0.1:8000", style=INFO_STYLE)
    uvicorn.run(app, host="127.0.0.1", port=8000)
