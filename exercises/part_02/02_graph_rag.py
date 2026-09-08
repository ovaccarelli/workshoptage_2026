"""Graph-RAG using one-hop neighbors from the structural webpage graph.

Flow:
1. Load the structural graph built in the previous exercise.
2. Retrieve the chunks most relevant to the user's question from ChromaDB.
3. Add the incoming and outgoing one-hop graph neighbors of those chunks.
4. Ask the model using the retrieved chunks plus their neighbors.

The graph is never extended or modified in this exercise.
"""

from pathlib import Path

import chromadb
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.settings import DEFAULT_NB_RETRIEVED_CHUNKS
from graph_rag_workshop.utils.console_utils import (
    INFO_STYLE,
    console,
    print_result,
    print_step,
)
from graph_rag_workshop.utils.part_02_graph_utils import (
    StructuralGraph,
    format_neighbor_graph_context,
    graph_context_from_chroma_results,
    load_structural_graph,
)
from graph_rag_workshop.utils.pydantic_utils import get_llm_model

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"
GRAPH_SOLUTIONS_DIR = DATA_DIR / "graph_solutions"

VECTORSTORE_NAME = "part_01_argo_usecase_webpage"
GRAPH_JSON_PATH = GRAPH_SOLUTIONS_DIR / "webpage_chunk_graph.json"

TOP_K = DEFAULT_NB_RETRIEVED_CHUNKS
USER_QUERY = "What venues are available and what are their characteristics?"


#################################################################
# STEP 1 - Load the existing structural graph
#################################################################

print_step("STEP 1 - Load the existing structural graph")

structural_graph = load_structural_graph(GRAPH_JSON_PATH)

console.print("Loaded the structural webpage graph.", style=INFO_STYLE)
print_result(
    f"Nodes: {len(structural_graph.nodes)}\n"
    f"Relationships: {len(structural_graph.edges)}\n"
    f"Source: {GRAPH_JSON_PATH}"
)


#################################################################
# STEP 2 - Retrieve the most relevant RAG chunks
#################################################################

print_step("STEP 2 - Retrieve the most relevant RAG chunks")

vector_database = chromadb.PersistentClient(path=str(VECTORSTORES_DIR)).get_collection(
    name=VECTORSTORE_NAME
)

# EXERCISE - Run vector retrieval:
# Use the ChromaDB collection's query method to retrieve the TOP_K chunks most
# semantically similar to the user question.
results = ...(
    query_texts=[USER_QUERY],
    n_results=TOP_K,
    include=["documents"],
)

# Match the retrieved Chroma chunks to graph nodes and format their content.
retrieved_node_id_set, retrieved_context = graph_context_from_chroma_results(
    graph=structural_graph,
    results=results,
)

console.print(f"Question: {USER_QUERY}", style=INFO_STYLE)
console.print("Retrieved RAG chunks:", style=INFO_STYLE)
print_result(retrieved_context)


#################################################################
# STEP 3 - Add one-hop graph neighbors
#################################################################

print_step("STEP 3 - Add one-hop graph neighbors")


def get_neighbor_node_ids(
    graph: StructuralGraph,
    node_ids: set[str],
) -> set[str]:
    """Return the incoming and outgoing one-hop neighbors of selected nodes."""
    neighbors = set()
    for edge in graph.edges:
        if edge.source in node_ids:
            neighbors.add(edge.target)
        if edge.target in node_ids:
            neighbors.add(edge.source)

    return neighbors - node_ids


# EXERCISE - Find the first-hop neighbors:
# Pass the structural graph and the node IDs returned by vector retrieval.
first_hop = get_neighbor_node_ids(..., ...)

# To retrieve up to two hops, uncomment these lines and pass
# two_hop_neighbors to format_neighbor_graph_context below.
# second_hop = get_neighbor_node_ids(structural_graph, first_hop)
# two_hop_neighbors = (first_hop | second_hop) - retrieved_node_id_set

neighbor_context, neighbor_link_count, neighbor_chunk_count = (
    format_neighbor_graph_context(
        graph=structural_graph,
        retrieved_node_ids=retrieved_node_id_set,
        neighbor_node_ids=first_hop,
    )
)

console.print(
    f"Followed {neighbor_link_count} graph links and added "
    f"{neighbor_chunk_count} unique one-hop neighboring chunks.",
    style=INFO_STYLE,
)
print_result(neighbor_context or "No additional graph neighbors were found.")


#################################################################
# STEP 4 - Ask the model with RAG chunks and graph neighbors
#################################################################

print_step("STEP 4 - Ask the model")

agent = Agent(
    model=get_llm_model(),
    instructions=(
        "Answer only from the retrieved webpage chunks and their one-hop graph "
        "neighbors. Use the neighbors to complete context that semantic retrieval "
        "may have missed. The Name and Section path fields are authoritative metadata; "
        "use them when identifying pages, sections, venues, products, or other entities. "
        "If the answer is absent, say what information is missing."
    ),
    model_settings=ModelSettings(thinking="minimal"),
)

prompt = f"""
Question:
{USER_QUERY}

Retrieved RAG chunks:
{retrieved_context}

One-hop graph neighbors:
{neighbor_context or "No additional graph neighbors were found."}
"""

result = agent.run_sync(prompt)

print_result(result.output)
