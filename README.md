# Agents Graph RAG Workshop

A hands-on workshop for building AI agents with **Pydantic AI**, vector RAG, knowledge graphs, tool calling, and agents.

The workshop combines two coherent sources: a **Wayfarer Hotels** webpage used
for vector and graph RAG, and local **Rivendell event** documents used for PDF
and image tool calling.

---

## Setup

### 1. Install uv

Install `uv`, the Python project and dependency manager used by this entry
point.

macOS and Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then restart your terminal and check that `uv` is available:

```bash
uv --version
```

### 2. Install the dependencies

From the project root, run:

```bash
uv sync
```

This command creates or updates the local `.venv` environment and installs all
dependencies needed for the entry script from `pyproject.toml` and `uv.lock`.

### 3. Choose an LLM

The workshop supports two OpenAI-compatible model endpoints.

#### Option 1 — Run Ollama on your computer (default)

1. Install Ollama from <https://ollama.com/download>.
2. Start Ollama.
3. Pull the default model:

```bash
ollama pull qwen3:1.7b
```

No additional configuration is required. The default settings are:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

The same provider is configured centrally in `src/graph_rag_workshop/settings.py`:

```python
LLM_PROVIDER = "ollama"
```

Change only this value to `"vllm"` to use the shared HEIA endpoint in every
exercise and solution. You can still override it for a single command with the
`LLM_PROVIDER` environment variable.

#### Option 2 — Use the HEIA vLLM endpoint

HEIA provides a larger vLLM model running in its Kubernetes infrastructure:

```text
LLM_PROVIDER=vllm
VLLM_MODEL=qwen3.8:27b
VLLM_BASE_URL=https://litellm.kube-ext.isc.heia-fr.ch/v1
```

This endpoint requires the secret API key provided by the workshop instructors.
In the project root—the same folder as `pyproject.toml`—create a file named
`.vllm_api_key` and paste only the provided key inside it:

```text
graph_rag/
├── .vllm_api_key
├── pyproject.toml
└── ...
```

Do not add quotes or a variable name around the key. The file is listed in
`.gitignore`, so the secret is not committed to Git.

Select it when running a script:

```bash
LLM_PROVIDER=vllm uv run python exercises/part_01/01_entry_point.py
```

This is a shared workshop service. It can occasionally time out or fail when
many participants send parallel requests to the Kubernetes deployment. If that
happens, wait briefly and retry, or switch back to local Ollama.

### 4. Check that everything works

To verify that your environment is ready for the entry point, run:

```bash
uv run python exercises/part_01/01_entry_point.py
```

Expected result: the script asks the selected model a simple question, prints
the answer, and starts the local web app at <http://127.0.0.1:8000>. Press
`Ctrl+C` to stop it.

Note: the first run can take a few seconds because Ollama has to load the
`qwen3:1.7b` model.

### 5. Run any script with `uv`

   ```bash
   uv run python exercises/part_01/01_entry_point.py
   ```

> **Models:** Local Ollama with `qwen3:1.7b` is the default. Set
> `LLM_PROVIDER=vllm` to use the shared HEIA model.

---

## Project Structure

| Directory | Purpose |
| --- | --- |
| `data/my_documents/` | Rivendell event PDF and image documents |
| `data/webpage_rag/` | Extracted webpage Markdown and saved webpage chunks |
| `data/vectorstores/` | ChromaDB vector store |
| `data/graph_solutions/` | JSON and HTML examples of graphs |
| `exercises/` | Your workspace — fill in the blanks here |
| `presentation/` | PowerPoint presentation for the workshop |
| `solutions/` | Exercises solutions |
| `src/graph_rag_workshop/` | Shared utilities for document tools, RAG, graph construction, and Pydantic AI model setup |

---

## Workshop Exercises

### Part 01 — Agents, Document Tools, and Webpage RAG

| Script | What it does |
| --- | --- |
| [`01_entry_point.py`](exercises/part_01/01_entry_point.py) | Minimal agent and smoke test (CLI and web app on port 8000) |
| [`02_document_tools.py`](exercises/part_01/02_document_tools.py) | Lists workshop documents and extracts text from PDF/image files (CLI) |
| [`03_webpage_rag.py`](exercises/part_01/03_webpage_rag.py) | Extracts, chunks, indexes, retrieves, and answers from a webpage |

```bash
uv run python exercises/part_01/01_entry_point.py
uv run python exercises/part_01/02_document_tools.py
uv run python exercises/part_01/03_webpage_rag.py
```

---

### Part 02 — Structural Graph Construction and Graph-RAG

| Script | What it does |
| --- | --- |
| [`01_graph_construction.py`](exercises/part_02/01_graph_construction.py) | Builds and reviews a structural graph from the saved webpage chunks |
| [`02_graph_rag.py`](exercises/part_02/02_graph_rag.py) | Adds all one-hop graph neighbors to the retrieved RAG chunks |
| [`03_simpleKG_construction.py`](exercises/part_02/03_simpleKG_construction.py) | Bonus: builds a semantic entity graph in Neo4j |
| [`04_semantic_graph_rag.py`](exercises/part_02/04_semantic_graph_rag.py) | Bonus: queries the semantic graph with text-to-Cypher Graph-RAG |

```bash
uv run python exercises/part_02/01_graph_construction.py
uv run python exercises/part_02/02_graph_rag.py
uv run python exercises/part_02/03_simpleKG_construction.py
uv run python exercises/part_02/04_semantic_graph_rag.py
```

The semantic graph exercises require Neo4j. Start it with Docker:

```bash
docker run --name neo4j-workshop -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 neo4j:latest
```

The structural graph writer removes only nodes marked as belonging to this
workshop's structural graph. It does not clear unrelated or semantic graph data.

---

### Part 03 — Tool Calling and Agentic Workflows

This part exposes the workshop capabilities as tools an agent can call.

| Script | What it does |
| --- | --- |
| [`01_simple_tool_call.py`](exercises/part_03/01_simple_tool_call.py) | Agent with a simple date/time tool (web app on port 8000) |
| [`02_graph_rag_tool.py`](exercises/part_03/02_graph_rag_tool.py) | Agent with Graph-RAG exposed as a tool (web app on port 8000) |
| [`03_document_tools_agent.py`](exercises/part_03/03_document_tools_agent.py) | Agent with document, Graph-RAG, and resilient web-search tools |

```bash
uv run python exercises/part_03/01_simple_tool_call.py
uv run python exercises/part_03/02_graph_rag_tool.py
uv run python exercises/part_03/03_document_tools_agent.py
```

---

### Part 04 — MCP and Agent Guardrails

This part exposes the same Wayfarer webpage Graph-RAG from Parts 02 and 03
through an MCP server. Run the server and agent in two separate terminals.

| Script | What it does |
| --- | --- |
| [`01a_mcp_rag_server.py`](exercises/part_04/01a_mcp_rag_server.py) | Serves `search_wayfarer_webpage_graph_rag` over MCP on port 8001 |
| [`01b_simple_mcp_rag_agent.py`](exercises/part_04/01b_simple_mcp_rag_agent.py) | Connects an agent to the Wayfarer Graph-RAG MCP tool |
| [`02_mcp_document_web_agent.py`](exercises/part_04/02_mcp_document_web_agent.py) | Combines MCP Graph-RAG, local document tools, and web search |
| [`03_guardrails_with_hooks.py`](exercises/part_04/03_guardrails_with_hooks.py) | Demonstrates request validation with agent hooks |

Terminal 1:

```bash
uv run python exercises/part_04/01a_mcp_rag_server.py
```

Terminal 2:

```bash
uv run python exercises/part_04/01b_simple_mcp_rag_agent.py
```

Part 04 requires the ChromaDB collection from Part 01 and the structural graph
JSON from Part 02. Run those exercises first if these artifacts do not exist.

Then run the remaining examples with the MCP server still active:

```bash
uv run python exercises/part_04/02_mcp_document_web_agent.py
uv run python exercises/part_04/03_guardrails_with_hooks.py
```

## Development checks

Exercise files intentionally contain `...` placeholders; every corresponding
solution is complete. Validate the shared code, solutions, and saved graph with:

```bash
uv run ruff check src solutions tests
uv run pytest
```
