"""Build a structural Neo4j graph from a webpage.

The graph is deterministic: it reuses the chunks created by the Part 01
webpage-RAG exercise and creates exactly one node for each saved RAG chunk.
Relationships between chunk nodes preserve the website structure.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic_ai import Agent, ModelSettings
from pydantic_ai.output import NativeOutput

from graph_rag_workshop.utils.console_utils import (
    INFO_STYLE,
    console,
    print_result,
    print_step,
)
from graph_rag_workshop.utils.part_02_graph_utils import (
    RelationshipReview,
    StructuralGraph,
    add_edge,
    add_referenced_section_edges,
    apply_relationship_corrections,
    create_chunk_nodes_and_hierarchy_links,
    format_relationship_corrections,
    graph_to_review_json,
    load_webpage_chunks,
    save_graph_cypher,
    save_graph_html,
    save_graph_json,
    write_structural_graph_to_neo4j,
)
from graph_rag_workshop.utils.pydantic_utils import get_llm_model

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
WEBPAGE_RAG_DIR = DATA_DIR / "webpage_rag"
GRAPH_SOLUTIONS_DIR = DATA_DIR / "graph_solutions"

WEBPAGE_URL = os.getenv(
    "WEBPAGE_URL",
    "https://federicab03.github.io/ARGO_Usecase/",
)
WEBPAGE_CHUNKS_PATH = WEBPAGE_RAG_DIR / "ARGO_Usecase_chunks.md"

GRAPH_JSON_PATH = GRAPH_SOLUTIONS_DIR / "webpage_chunk_graph.json"
GRAPH_CYPHER_PATH = GRAPH_SOLUTIONS_DIR / "webpage_chunk_graph.cypher"
GRAPH_HTML_PATH = GRAPH_SOLUTIONS_DIR / "webpage_chunk_graph.html"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

#################################################################
# STEP 1 - Load chunks created by the webpage-RAG exercise
#################################################################

print_step("STEP 1 - Load existing webpage chunks")
chunk_records = load_webpage_chunks(WEBPAGE_CHUNKS_PATH)
console.print("Loaded existing webpage chunks.", style=INFO_STYLE)
print_result(f"Chunks: {len(chunk_records)}\nSource: {WEBPAGE_CHUNKS_PATH}")

#################################################################
# STEP 2 - Convert chunks to a structural graph
#################################################################

print_step("STEP 2 - Build structural graph")


def build_structural_graph(chunks: list[dict]) -> StructuralGraph:
    """Create chunk nodes and connect them by webpage structure."""
    graph = StructuralGraph(nodes={}, edges=[])

    # 1. Create the chunk nodes and identify their hierarchy.
    section_links, subsection_links, chunk_ids_by_section = (
        create_chunk_nodes_and_hierarchy_links(graph, chunks)
    )

    # 2. SOLUTION - Connect sections:
    # Iterate over the prepared section links to connect each page to its section.
    for parent_id, child_id, properties in section_links:
        add_edge(graph, parent_id, "SECTION", child_id, properties)

    # 3. SOLUTION - Connect subsections:
    # Iterate over the prepared subsection links to connect each section to its
    # child subsection.
    for parent_id, child_id, properties in subsection_links:
        add_edge(graph, parent_id, "SUBSECTION", child_id, properties)

    # 4. Connect consecutive chunks that belong to the same section.
    for section_titles, chunk_ids in chunk_ids_by_section.items():
        for current_id, next_id in zip(chunk_ids, chunk_ids[1:]):
            add_edge(
                graph,
                current_id,
                "NEXT_PART",
                next_id,
                {"section": " > ".join(section_titles)},
            )

    # 5. Add cross-links when a chunk mentions another named website section.
    add_referenced_section_edges(graph)

    return graph


structural_graph = build_structural_graph(chunk_records)
console.print("Converted the webpage chunks into a structural graph.", style=INFO_STYLE)
print_result(
    f"Nodes: {len(structural_graph.nodes)}\n"
    f"Relationships: {len(structural_graph.edges)}"
)


#################################################################
# STEP 3 - Review relationships with an LLM
#################################################################

print_step("STEP 3 - Review graph relationships with an LLM")


def review_graph_relationships(
    graph: StructuralGraph,
) -> tuple[StructuralGraph, list[str]]:
    """Review explicit and implicit website links without changing any nodes."""
    review_agent = Agent(
        model=get_llm_model(),
        output_type=NativeOutput(RelationshipReview, strict=True),
        instructions=(
            "Every graph node represents exactly one webpage chunk. "
            "Review both incorrect existing relationships and important missing "
            "relationships. Do not assume that equal Markdown heading levels mean "
            "that chunks are unrelated. Website extraction often flattens pages that "
            "are semantically hierarchical. "
            "Work systematically: first inspect every node that acts as an overview, "
            "hub, index, category, list, menu, or collection. Then identify detail "
            "nodes that elaborate the items, entities, services, locations, people, "
            "products, or topics introduced by that hub. Add a directed SUBSECTION "
            "relationship from the hub chunk to each matching detail chunk, even when "
            "both chunks have the same heading level or appear far apart. Use SECTION "
            "for broader landing-page-to-major-section links. Use NEXT_PART only for "
            "consecutive chunks belonging to the exact same section. "
            "Treat explicit hyperlinks, navigation labels, repeated entity names, "
            "overview-to-detail wording, and clear category membership as evidence. "
            "Do not add links based only on weak topical similarity. Review all nodes "
            "before returning, so every clear hub-to-detail flow is represented. "
            "Never create, remove, rename, or modify nodes. Every source and target "
            "must be an existing node ID. Make only corrections supported by concrete "
            "evidence and assign an honest confidence score. "
            "Use SECTION for section-level structural links, SUBSECTION for "
            "subsection-level structural links, and NEXT_PART for consecutive "
            "parts of the same section. Do not create any other relationship type. "
            "Return a result matching the required output schema. Inspect every "
            "overview-to-detail relationship "
            "before deciding that no correction is needed. Return "
            "an empty corrections list only when the existing relationships are "
            "already correct and complete."
        ),
        # SOLUTION - Configure deterministic review:
        # Temperature 0.0 keeps the proposed corrections stable and repeatable.
        model_settings=ModelSettings(
            thinking="minimal",
            temperature=0.0,
        ),
        output_retries=2,
    )

    # Convert the Python graph objects into a JSON string the model can read.
    graph_json = graph_to_review_json(graph, WEBPAGE_URL)
    review = review_agent.run_sync(graph_json).output
    return apply_relationship_corrections(graph, review.corrections)


structural_graph, relationship_corrections = review_graph_relationships(
    structural_graph
)

console.print("Completed the LLM relationship review.", style=INFO_STYLE)
print_result(format_relationship_corrections(relationship_corrections))

#################################################################
# STEP 4 - Save and publish the graph
#################################################################

print_step("STEP 4 - Save and publish the graph")

# Save the reviewed graph as JSON, Cypher, and an interactive HTML page.
json_path = save_graph_json(structural_graph, GRAPH_JSON_PATH)
cypher_path = save_graph_cypher(structural_graph, GRAPH_CYPHER_PATH)
html_path = save_graph_html(structural_graph, GRAPH_HTML_PATH)

wrote_to_neo4j = write_structural_graph_to_neo4j(
    graph=structural_graph,
    uri=NEO4J_URI,
    user=NEO4J_USER,
    password=NEO4J_PASSWORD,
    database=NEO4J_DATABASE,
)
console.print("Finished the Neo4j write attempt.", style=INFO_STYLE)
print_result(
    f"Neo4j: {NEO4J_URI}\nStatus: graph written successfully"
    if wrote_to_neo4j
    else f"Neo4j: {NEO4J_URI}\nStatus: unavailable\nFallback: {cypher_path}"
)

console.print("Saved and published the graph outputs.", style=INFO_STYLE)
print_result(
    f"HTML visualization: {html_path}\n"
    f"JSON artifact: {json_path}\n"
    f"Cypher artifact: {cypher_path}"
)
