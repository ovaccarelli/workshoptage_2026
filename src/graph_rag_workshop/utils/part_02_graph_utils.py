"""Structural graph utilities for Part 2 of the workshop."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable
from pydantic import BaseModel, Field

from graph_rag_workshop.settings import DEFAULT_NB_RETRIEVED_CHUNKS


@dataclass
class StructuralGraphNode:
    """A node in a webpage structural graph."""

    id: str
    labels: list[str]
    properties: dict[str, str | int]


@dataclass
class StructuralGraphEdge:
    """An edge in a webpage structural graph."""

    source: str
    relation: str
    target: str
    properties: dict[str, str | int]


@dataclass
class StructuralGraph:
    """A graph representing webpage sections and chunks."""

    nodes: dict[str, StructuralGraphNode]
    edges: list[StructuralGraphEdge]


HierarchyLink = tuple[str, str, dict[str, str]]


class RelationshipCorrection(BaseModel):
    """A proposed relationship correction using existing graph nodes."""

    action: Literal["add", "remove", "replace"]
    source: str = Field(description="ID of an existing source node.")
    relation: str = Field(description="Current relation, or relation to add.")
    target: str = Field(description="ID of an existing target node.")
    new_source: str | None = Field(default=None)
    new_relation: str | None = Field(default=None)
    new_target: str | None = Field(default=None)
    evidence: str = Field(
        description="Short content or navigation evidence supporting the correction."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that the correction reflects the website flow.",
    )
    reason: str = Field(description="Short explanation for the correction.")


class RelationshipReview(BaseModel):
    """Corrections proposed by the relationship-review agent."""

    corrections: list[RelationshipCorrection] = Field(default_factory=list)


def graph_to_review_json(graph: StructuralGraph, website: str) -> str:
    """Convert a graph into the JSON string read by the review agent."""
    return json.dumps(
        {
            "website": website,
            "nodes": [
                {
                    "id": node.id,
                    "labels": node.labels,
                    "properties": node.properties,
                }
                for node in graph.nodes.values()
            ],
            "relationships": [
                {
                    "source": edge.source,
                    "relation": edge.relation,
                    "target": edge.target,
                }
                for edge in graph.edges
            ],
        },
        ensure_ascii=False,
    )


def apply_relationship_corrections(
    graph: StructuralGraph,
    corrections: list[RelationshipCorrection],
) -> tuple[StructuralGraph, list[str]]:
    """Validate and apply relationship corrections proposed by the review agent."""
    node_ids = set(graph.nodes)
    allowed_relations = {"SECTION", "SUBSECTION", "NEXT_PART"}
    applied_corrections = []

    for correction in corrections:
        source = correction.source
        relation = correction.relation.upper()
        target = correction.target

        if source not in node_ids or target not in node_ids or source == target:
            continue
        if correction.confidence < 0.7 or relation not in allowed_relations:
            continue

        matching_edge = next(
            (
                edge
                for edge in graph.edges
                if edge.source == source
                and edge.relation == relation
                and edge.target == target
            ),
            None,
        )

        if correction.action == "remove" and matching_edge:
            graph.edges.remove(matching_edge)
        elif correction.action == "add" and matching_edge is None:
            add_edge(
                graph,
                source,
                relation,
                target,
                {
                    "review_evidence": correction.evidence,
                    "review_confidence_percent": int(correction.confidence * 100),
                },
            )
        elif correction.action == "replace" and matching_edge:
            new_source = correction.new_source or source
            new_relation = (correction.new_relation or relation).upper()
            new_target = correction.new_target or target
            if (
                new_source not in node_ids
                or new_target not in node_ids
                or new_source == new_target
                or new_relation not in allowed_relations
            ):
                continue
            graph.edges.remove(matching_edge)
            add_edge(
                graph,
                new_source,
                new_relation,
                new_target,
                matching_edge.properties,
            )
        else:
            continue

        applied_corrections.append(
            f"{correction.action}: {source} --{relation}--> {target}\n"
            f"Reason: {correction.reason}\n"
            f"Evidence: {correction.evidence}\n"
            f"Confidence: {correction.confidence:.0%}"
        )

    return graph, applied_corrections


def format_relationship_corrections(corrections: list[str]) -> str:
    """Format applied corrections for workshop output."""
    return (
        "\n\n".join(corrections)
        if corrections
        else "No relationship corrections were needed."
    )


def load_webpage_chunks(chunks_path: Path) -> list[dict]:
    """Load the documented webpage chunks saved by Part 1."""
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Chunk file not found: {chunks_path}. Run the Part 1 webpage RAG "
            "exercise first."
        )

    saved_chunks = chunks_path.read_text(encoding="utf-8").split("\n\n---\n\n")
    chunks = []
    for saved_chunk in saved_chunks:
        chunk_record = saved_chunk.split("\n\n", 1)[1]
        chunk_metadata, content = chunk_record.split("Content:\n", 1)
        fields = dict(
            line.split(": ", 1) for line in chunk_metadata.splitlines() if ": " in line
        )
        section_titles = fields.get("Section", "").split(" > ")
        metadata = {
            f"h{level}": title
            for level, title in enumerate(section_titles, start=1)
            if title and title != "No header"
        }
        chunks.append(
            {
                "content": content,
                "metadata": metadata,
                "source": fields.get("Source", "Unknown"),
                "start_index": int(fields.get("Start index", 0)),
            }
        )

    section_counts = {}
    for chunk in chunks:
        section = tuple(chunk["metadata"].items())
        section_counts[section] = section_counts.get(section, 0) + 1

    section_positions = {}
    for chunk in chunks:
        section = tuple(chunk["metadata"].items())
        section_positions[section] = section_positions.get(section, 0) + 1
        chunk["part_index"] = section_positions[section]
        chunk["part_count"] = section_counts[section]

    return chunks


def section_path(metadata: dict[str, str]) -> list[tuple[int, str]]:
    """Return ordered heading levels from LangChain metadata."""
    return [
        (level, metadata[key])
        for level, key in [(1, "h1"), (2, "h2"), (3, "h3")]
        if metadata.get(key)
    ]


def prepare_chunk_node(
    chunk: dict,
    chunk_index: int,
) -> tuple[str, tuple[str, ...], list[str], dict[str, str | int]]:
    """Prepare the ID, section path, labels, and properties for a chunk node."""
    path = section_path(chunk["metadata"])
    section_titles = tuple(title for _, title in path)
    section_name = " > ".join(section_titles) or "No section"
    node_name = section_titles[-1] if section_titles else "Untitled section"
    heading_level = path[-1][0] if path else 0
    part_index = chunk["part_index"]

    if part_index > 1:
        node_type = "NextPart"
        node_name = f"{node_name} — Part {part_index}"
    elif heading_level >= 3:
        node_type = "Subsection"
    else:
        node_type = "Section"

    return (
        f"chunk:{chunk_index}",
        section_titles,
        [node_type, "Chunk"],
        {
            "index": chunk_index,
            "name": node_name,
            "content": chunk["content"],
            "section": section_name,
            "heading_level": heading_level,
            "part_index": part_index,
            "part_count": chunk["part_count"],
            "source": chunk["source"],
        },
    )


def add_node(
    graph: StructuralGraph,
    node_id: str,
    labels: list[str],
    properties: dict[str, str | int],
) -> None:
    """Add a structural graph node once."""
    graph.nodes.setdefault(
        node_id,
        StructuralGraphNode(node_id, labels, properties),
    )


def add_edge(
    graph: StructuralGraph,
    source: str,
    relation: str,
    target: str,
    properties: dict[str, str | int],
) -> None:
    """Add a structural graph relationship once."""
    edge_key = (source, relation, target, tuple(sorted(properties.items())))
    existing_keys = {
        (
            edge.source,
            edge.relation,
            edge.target,
            tuple(sorted(edge.properties.items())),
        )
        for edge in graph.edges
    }
    if edge_key not in existing_keys:
        graph.edges.append(StructuralGraphEdge(source, relation, target, properties))


def create_chunk_nodes_and_hierarchy_links(
    graph: StructuralGraph,
    chunks: list[dict],
) -> tuple[
    list[HierarchyLink],
    list[HierarchyLink],
    dict[tuple[str, ...], list[str]],
]:
    """Create chunk nodes and prepare section, subsection, and part links."""
    first_chunk_by_section = {}
    chunk_ids_by_section = {}

    for chunk_index, chunk in enumerate(chunks):
        chunk_id, section_titles, labels, properties = prepare_chunk_node(
            chunk,
            chunk_index,
        )
        add_node(graph, chunk_id, labels, properties)
        first_chunk_by_section.setdefault(section_titles, chunk_id)
        chunk_ids_by_section.setdefault(section_titles, []).append(chunk_id)

    section_links = []
    subsection_links = []
    for child_titles, child_chunk_id in first_chunk_by_section.items():
        parent_titles = child_titles[:-1]
        while parent_titles and parent_titles not in first_chunk_by_section:
            parent_titles = parent_titles[:-1]

        if not parent_titles:
            continue

        link = (
            first_chunk_by_section[parent_titles],
            child_chunk_id,
            {
                "parent_section": " > ".join(parent_titles),
                "child_section": " > ".join(child_titles),
            },
        )
        if len(child_titles) >= 3:
            subsection_links.append(link)
        else:
            section_links.append(link)

    return section_links, subsection_links, chunk_ids_by_section


def add_referenced_section_edges(graph: StructuralGraph) -> None:
    """Link chunks whose text refers to another named website section."""
    ignored_words = {"and", "our", "the", "with", "from", "your"}

    def topic_terms(text: str) -> set[str]:
        return {
            word[:-1] if word.endswith("s") else word
            for word in re.findall(r"[a-z0-9]+", text.lower())
            if word not in ignored_words and len(word) >= 5
        }

    for source_node in graph.nodes.values():
        source_terms = topic_terms(str(source_node.properties["content"]))
        for target_node in graph.nodes.values():
            if source_node.id == target_node.id:
                continue
            if target_node.properties["heading_level"] <= 1:
                continue
            if target_node.properties["part_index"] > 1:
                continue

            source_path = str(source_node.properties["section"]).split(" > ")
            target_path = str(target_node.properties["section"]).split(" > ")
            if source_path[: len(target_path)] == target_path:
                continue

            target_terms = topic_terms(str(target_node.properties["name"]))
            if not target_terms or not target_terms.issubset(source_terms):
                continue

            relation = (
                "SUBSECTION" if target_node.labels[0] == "Subsection" else "SECTION"
            )
            add_edge(
                graph,
                source_node.id,
                relation,
                target_node.id,
                {
                    "evidence": "Referenced section topic in source chunk",
                    "matched_terms": ", ".join(sorted(target_terms)),
                },
            )


def save_graph_json(graph: StructuralGraph, output_path: Path) -> Path:
    """Save structural graph nodes and edges as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "nodes": [asdict(node) for node in graph.nodes.values()],
                "edges": [asdict(edge) for edge in graph.edges],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path


def load_structural_graph(graph_path: Path) -> StructuralGraph:
    """Load a structural graph previously saved as JSON."""
    if not graph_path.is_file():
        raise FileNotFoundError(
            f"Structural graph not found: {graph_path}. Run the graph construction "
            "exercise first."
        )

    data = json.loads(graph_path.read_text(encoding="utf-8"))
    return StructuralGraph(
        nodes={node["id"]: StructuralGraphNode(**node) for node in data["nodes"]},
        edges=[StructuralGraphEdge(**edge) for edge in data["edges"]],
    )


def get_neighbor_node_ids(
    graph: StructuralGraph,
    node_ids: set[str],
) -> set[str]:
    """Return incoming and outgoing one-hop neighbors of the selected nodes."""
    neighbors = set()
    for edge in graph.edges:
        if edge.source in node_ids and edge.target not in node_ids:
            neighbors.add(edge.target)
        if edge.target in node_ids and edge.source not in node_ids:
            neighbors.add(edge.source)
    return neighbors


def format_structural_graph_chunk(
    graph: StructuralGraph,
    node_id: str,
    connection: str | None = None,
) -> str:
    """Format one webpage chunk with its title, section path, and graph link."""
    node = graph.nodes[node_id]
    properties = node.properties
    connection_text = f"Graph connection: {connection}\n" if connection else ""
    return (
        f"Node: {node_id}\n"
        f"Name: {properties.get('name', 'Untitled section')}\n"
        f"Section path: {properties.get('section', 'No section')}\n"
        f"Source: {properties.get('source', 'Unknown')}\n"
        f"Node type: {node.labels[0]}\n"
        f"{connection_text}"
        f"Content:\n{properties['content']}"
    )


def chroma_ids_to_graph_node_ids(chroma_ids: list[str]) -> list[str]:
    """Convert Chroma IDs to unique graph-node IDs while preserving rank."""
    node_ids = []
    for chunk_id in chroma_ids:
        node_id = chunk_id.replace("webpage_chunk_", "chunk:").replace(
            "chunk_", "chunk:"
        )
        if node_id not in node_ids:
            node_ids.append(node_id)
    return node_ids


def format_structural_graph_chunks(
    graph: StructuralGraph,
    node_ids: list[str],
) -> str:
    """Format and join the graph chunks matching the supplied node IDs."""
    chunks = [
        format_structural_graph_chunk(graph, node_id)
        for node_id in node_ids
        if node_id in graph.nodes
    ]
    return "\n\n------------\n\n".join(chunks)


def graph_context_from_chroma_results(
    graph: StructuralGraph,
    results: dict,
) -> tuple[set[str], str]:
    """Convert Chroma query results into graph node IDs and formatted context."""
    chroma_ids = (results.get("ids") or [[]])[0]
    node_ids = chroma_ids_to_graph_node_ids(chroma_ids)
    existing_node_ids = [node_id for node_id in node_ids if node_id in graph.nodes]
    context = format_structural_graph_chunks(graph, existing_node_ids)
    return set(existing_node_ids), context


def format_neighbor_graph_context(
    graph: StructuralGraph,
    retrieved_node_ids: set[str],
    neighbor_node_ids: set[str],
) -> tuple[str, int, int]:
    """Format one-hop neighbor chunks and count the followed graph links."""
    neighbor_chunks = []
    link_count = 0

    for neighbor_id in sorted(neighbor_node_ids):
        if neighbor_id not in graph.nodes:
            continue

        connections = []
        for edge in graph.edges:
            if edge.source in retrieved_node_ids and edge.target == neighbor_id:
                link_count += 1
                source_name = graph.nodes[edge.source].properties.get(
                    "name", edge.source
                )
                connections.append(f"{source_name} --{edge.relation}--> neighbor")
            elif edge.target in retrieved_node_ids and edge.source == neighbor_id:
                link_count += 1
                target_name = graph.nodes[edge.target].properties.get(
                    "name", edge.target
                )
                connections.append(f"neighbor --{edge.relation}--> {target_name}")

        neighbor_chunks.append(
            format_structural_graph_chunk(
                graph,
                neighbor_id,
                "; ".join(connections),
            )
        )

    context = "\n\n------------\n\n".join(neighbor_chunks)
    return context, link_count, len(neighbor_chunks)


def search_structural_graph_rag_context(
    vector_database,
    graph_json_path: Path,
    question: str,
    top_k: int = DEFAULT_NB_RETRIEVED_CHUNKS,
) -> str:
    """Retrieve top-k webpage chunks and add all their one-hop graph neighbors."""
    graph = load_structural_graph(graph_json_path)
    results = vector_database.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents"],
    )
    retrieved_node_id_set, retrieved_context = graph_context_from_chroma_results(
        graph, results
    )
    neighbor_node_ids = get_neighbor_node_ids(graph, retrieved_node_id_set)
    neighbor_context, _, _ = format_neighbor_graph_context(
        graph, retrieved_node_id_set, neighbor_node_ids
    )
    return (
        "Retrieved RAG chunks:\n"
        f"{retrieved_context or 'No matching chunks were found.'}\n\n"
        "One-hop graph neighbors:\n"
        f"{neighbor_context or 'No additional graph neighbors were found.'}"
    )


def _cypher_literal(value: str | int) -> str:
    """Return a JSON-escaped value that can be used as a Cypher literal."""
    return json.dumps(value)


def save_graph_cypher(graph: StructuralGraph, output_path: Path) -> Path:
    """Save a Cypher script that can recreate a structural graph in Neo4j."""
    lines = [
        "CREATE CONSTRAINT webpage_id IF NOT EXISTS FOR (n:Webpage) REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE;",
        "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (n:Chunk) REQUIRE n.id IS UNIQUE;",
        "MATCH (n {graph_rag_workshop_graph: 'structural'}) DETACH DELETE n;",
    ]

    for node in graph.nodes.values():
        labels = ":".join(node.labels)
        properties = {
            "id": node.id,
            "graph_rag_workshop_graph": "structural",
            **node.properties,
        }
        property_text = ", ".join(
            f"{key}: {_cypher_literal(value)}" for key, value in properties.items()
        )
        lines.append(
            f"MERGE (n:{labels} {{id: {_cypher_literal(node.id)}}}) "
            f"SET n += {{{property_text}}};"
        )

    for edge in graph.edges:
        property_text = ", ".join(
            f"{key}: {_cypher_literal(value)}" for key, value in edge.properties.items()
        )
        properties = f" {{{property_text}}}" if property_text else ""
        lines.append(
            f"MATCH (a {{id: {_cypher_literal(edge.source)}}}), "
            f"(b {{id: {_cypher_literal(edge.target)}}}) "
            f"MERGE (a)-[r:{edge.relation}{properties}]->(b);"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def write_structural_graph_to_neo4j(
    graph: StructuralGraph,
    uri: str,
    user: str,
    password: str,
    database: str = "neo4j",
) -> bool:
    """Write a structural graph to Neo4j, returning False when unavailable."""
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        with driver.session(database=database) as session:
            session.run(
                "MATCH (n {graph_rag_workshop_graph: 'structural'}) DETACH DELETE n"
            )
            session.run(
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS "
                "FOR (n:Chunk) REQUIRE n.id IS UNIQUE"
            )

            for node in graph.nodes.values():
                labels = ":".join(node.labels)
                session.run(
                    f"MERGE (n:{labels} {{id: $id}}) SET n += $properties",
                    id=node.id,
                    properties={
                        "id": node.id,
                        "graph_rag_workshop_graph": "structural",
                        **node.properties,
                    },
                )

            for edge in graph.edges:
                session.run(
                    f"""
                    MATCH (source {{id: $source_id}})
                    MATCH (target {{id: $target_id}})
                    MERGE (source)-[r:{edge.relation}]->(target)
                    SET r += $properties
                    """,
                    source_id=edge.source,
                    target_id=edge.target,
                    properties=edge.properties,
                )

        return True
    except (ServiceUnavailable, Neo4jError, OSError):
        return False
    finally:
        if "driver" in locals():
            driver.close()


def save_graph_html(
    graph: StructuralGraph,
    output_path: Path,
) -> Path:
    """Create an interactive HTML visualization of a structural graph."""
    from pyvis.network import Network

    network = Network(
        height="760px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111827",
        cdn_resources="remote",
    )
    node_colors = {
        "Section": "#2563eb",
        "Subsection": "#16a34a",
        "NextPart": "#f97316",
    }
    edge_colors = {
        "SECTION": "#2563eb",
        "SUBSECTION": "#16a34a",
        "NEXT_PART": "#f97316",
    }

    for node in graph.nodes.values():
        primary_label = node.labels[0]
        label = node.properties.get("name") or "Untitled section"
        title = "<br>".join(
            f"{key}: {str(value)[:500]}" for key, value in node.properties.items()
        )
        network.add_node(
            node.id,
            label=str(label),
            title=title,
            color=node_colors.get(primary_label, "#64748b"),
        )

    for edge in graph.edges:
        network.add_edge(
            edge.source,
            edge.target,
            label=edge.relation,
            color=edge_colors.get(edge.relation, "#64748b"),
            title="<br>".join(
                f"{key}: {value}" for key, value in edge.properties.items()
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    network.write_html(str(output_path), open_browser=False)
    return output_path
