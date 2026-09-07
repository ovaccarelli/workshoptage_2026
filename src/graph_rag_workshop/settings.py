# Choose "ollama" for a local model or "vllm" for the shared HEIA model.
LLM_PROVIDER = "ollama"

# Option 1: Ollama running on the participant's computer.
DEFAULT_OLLAMA_MODEL = "qwen3:1.7b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"

# Option 2: vLLM running on the HEIA Kubernetes infrastructure.

DEFAULT_VLLM_MODEL = "qwen3.8:27b"
DEFAULT_VLLM_BASE_URL = "https://litellm.kube-ext.isc.heia-fr.ch/v1"

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_CHUNK_OVERLAP = 400
DEFAULT_VECTORSTORE_NAME = "part_01_argo_usecase_webpage"

DEFAULT_NB_RETRIEVED_CHUNKS = 5
