# Agents Graph RAG Workshop

Entry point for the workshop **Design Thinking Machines**.

The first exercise runs a Pydantic AI agent with either Ollama on your computer
or the shared HEIA vLLM service.

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

#### Option 1 — Run Ollama on your computer (default)

1. Install Ollama from <https://ollama.com/download>.
2. Start Ollama.
3. Pull the default model:

```bash
ollama pull qwen3:1.7b
```

The entry point uses these settings by default:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:1.7b
OLLAMA_BASE_URL=http://localhost:11434/v1
```

The provider is selected centrally in `src/graph_rag_workshop/settings.py`:

```python
LLM_PROVIDER = "ollama"
```

Change it to `"vllm"` to use the shared HEIA endpoint. Alternatively, override
the provider for one command with the `LLM_PROVIDER` environment variable.

#### Option 2 — Use the HEIA vLLM endpoint

HEIA provides a larger vLLM model running in its Kubernetes infrastructure:

```text
LLM_PROVIDER=vllm
VLLM_MODEL=qwen3.8:27b
VLLM_BASE_URL=https://litellm.kube-ext.isc.heia-fr.ch/v1
```

The HEIA endpoint requires a secret API key, which will be given to you by the
workshop instructors. In the project root—the same folder as `pyproject.toml`—
create a file named `.vllm_api_key` and paste only the provided key inside it:

```text
graph_rag/
├── .vllm_api_key
├── pyproject.toml
└── ...
```

Do not add quotes or a variable name around the key. The file is already listed
in `.gitignore`, so it will not be committed to Git.

Run the entry point with this option:

```bash
LLM_PROVIDER=vllm uv run python exercises/part_01/01_entry_point.py
```

The HEIA endpoint is shared. It can occasionally time out or fail when many
participants make parallel calls to the Kubernetes deployment. Retry after a
short wait or use local Ollama if this happens.

## Check That Everything Works

To verify that your environment is ready for the entry point, run:

```bash
uv run python exercises/part_01/01_entry_point.py
```

Expected result: the script asks the selected model a simple question, prints
the answer, and starts the local web app at <http://127.0.0.1:8000>. Press
`Ctrl+C` to stop it.

Note: the first run can take a few seconds because Ollama has to load the
`qwen3:1.7b` model.

### 4. Run Neo4j with Docker (optional bonus)

Neo4j is used only for the bonus section on semantic graph construction and
graph visualization. It is not required for the rest of the workshop. If you
cannot set it up before the workshop, we can do it together or you can skip
this section.

1. Install and start [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Download the `neo4j:latest` image and start a Neo4j container:

```bash
docker run \
  --name neo4j-workshop \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password123 \
  neo4j:latest
```

Docker downloads the image automatically the first time. When Neo4j is ready,
open <http://localhost:7474> and sign in with:

```text
Username: neo4j
Password: password123
```

Keep this terminal running while using Neo4j. Press `Ctrl+C` to stop the
container.
