"""Graph-RAG over the semantic knowledge graph built by SimpleKGPipeline.

Run ``03_simpleKG_construction.py`` first. This example uses an LLM to turn a
question into read-only Cypher, retrieves matching facts from Neo4j, and asks
the LLM to answer using only those facts.
"""

import os
import sys
import time

from neo4j import GraphDatabase
from neo4j_graphrag.generation import GraphRAG
from neo4j_graphrag.generation.prompts import RagTemplate
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.retrievers import Text2CypherRetriever

from graph_rag_workshop.utils.console_utils import (
    INFO_STYLE,
    console,
    print_result,
    print_step,
)
from graph_rag_workshop.utils.pydantic_utils import get_llm_settings

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
LLM_MODEL, LLM_BASE_URL, LLM_API_KEY = get_llm_settings()

DEFAULT_QUERY = (
    "What venues are available, and which services or activities do they offer?"
)
USER_QUERY = " ".join(sys.argv[1:]).strip() or DEFAULT_QUERY

# Giving Text2Cypher the workshop schema keeps it focused on the semantic
# entities instead of the Document/Chunk nodes also created by the KG builder.
SEMANTIC_GRAPH_SCHEMA = """
Node properties:
- Venue {name: STRING}
- Service {name: STRING}
- Activity {name: STRING}
- Requirement {name: STRING}
- Itinerary {name: STRING}

Relationships:
- (:Venue)-[:OFFERS]->(:Service)
- (:Venue)-[:SUPPORTS]->(:Activity)
- (:Venue)-[:SATISFIES]->(:Requirement)
- (:Service)-[:SATISFIES]->(:Requirement)
- (:Itinerary)-[:INCLUDES]->(:Activity)
- (:Itinerary)-[:INCLUDES]->(:Service)
"""

CYPHER_EXAMPLES = [
    "Question: What services does each venue offer?\n"
    "Cypher: MATCH (v:Venue)-[:OFFERS]->(s:Service) "
    "RETURN v.name AS venue, collect(DISTINCT s.name) AS services",
    "Question: Which venues support activities?\n"
    "Cypher: MATCH (v:Venue)-[:SUPPORTS]->(a:Activity) "
    "RETURN v.name AS venue, collect(DISTINCT a.name) AS activities",
]


#################################################################
# STEP 1 - Connect to the semantic graph
#################################################################

print_step("STEP 1 - Connect to the semantic knowledge graph")

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
)
driver.verify_connectivity()

console.print("Connected to Neo4j.", style=INFO_STYLE)
print_result(f"Database: {NEO4J_DATABASE}")


#################################################################
# STEP 2 - Configure Text-to-Cypher GraphRAG
#################################################################

print_step("STEP 2 - Configure semantic GraphRAG")

llm = OpenAILLM(
    model_name=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    model_params={"temperature": 0.0},
)

# Here we use Neo4j GraphRAG's imported Text2CypherRetriever, which translates a
# natural-language question into Cypher using the schema and examples, executes
# the query in Neo4j, and returns the resulting records as context.
retriever = Text2CypherRetriever(
    driver=driver,
    llm=llm,
    neo4j_schema=SEMANTIC_GRAPH_SCHEMA,
    examples=CYPHER_EXAMPLES,
    neo4j_database=NEO4J_DATABASE,
)

# EXERCISE - Configure GraphRAG:
# Here we use Neo4j GraphRAG's imported GraphRAG class, which asks the retriever
# for relevant Neo4j records and passes that context to the LLM to generate a
# grounded answer using the instructions below.
graph_rag = ...(
    retriever=retriever,
    llm=llm,
    prompt_template=RagTemplate(
        system_instructions=(
            "Answer only from the Neo4j records supplied as context. "
            "Clearly distinguish venues, services, activities, requirements, "
            "and itineraries. If the records do not contain the answer, say so."
        )
    ),
)

console.print(f"Question: {USER_QUERY}", style=INFO_STYLE)


#################################################################
# STEP 3 - Retrieve graph facts and generate the answer
#################################################################

print_step("STEP 3 - Query the graph and generate an answer")

start = time.time()
try:
    result = graph_rag.search(
        query_text=USER_QUERY,
        return_context=True,
        response_fallback="No matching facts were found in the semantic graph.",
    )
finally:
    driver.close()
elapsed = time.time() - start

retriever_result = result.retriever_result
if retriever_result:
    console.print("Generated Cypher:", style=INFO_STYLE)
    print_result(retriever_result.metadata.get("cypher", "Unavailable"))

console.print(f"Answer generated in {elapsed:.2f} seconds.", style=INFO_STYLE)
print_result(result.answer)
