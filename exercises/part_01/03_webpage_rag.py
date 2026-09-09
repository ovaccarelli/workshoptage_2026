"""RAG exercise from a webpage

This script extracts a JavaScript-rendered website as Markdown, stores chunks in
ChromaDB, retrieves relevant chunks for a question, and asks the model to answer
using only the retrieved context.
"""

import os
from pathlib import Path

import chromadb
from chromadb.errors import NotFoundError
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from pydantic_ai import Agent, ModelSettings

from graph_rag_workshop.settings import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_NB_RETRIEVED_CHUNKS,
    DEFAULT_VECTORSTORE_NAME,
)
from graph_rag_workshop.utils.console_utils import (
    INFO_STYLE,
    console,
    print_result,
    print_step,
)
from graph_rag_workshop.utils.part_01_document_tools import extract_webpage_to_markdown
from graph_rag_workshop.utils.part_01_rag_utils import (
    format_markdown_chunks,
    save_chunks_to_markdown,
)
from graph_rag_workshop.utils.pydantic_utils import get_llm_model

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
WEBPAGE_RAG_DIR = DATA_DIR / "webpage_rag"
VECTORSTORES_DIR = DATA_DIR / "vectorstores"

WEBPAGE_URL = os.getenv(
    "WEBPAGE_URL",
    "https://federicab03.github.io/ARGO_Usecase/",
)
WEBPAGE_MARKDOWN_PATH = WEBPAGE_RAG_DIR / "ARGO_Usecase.md"
CHUNKS_MARKDOWN_PATH = WEBPAGE_RAG_DIR / "ARGO_Usecase_chunks.md"
COLLECTION_NAME = DEFAULT_VECTORSTORE_NAME

USER_QUESTION = "What venues are available and what are their characteristics?"


#################################################################
# STEP 1 - Extract webpage as Markdown
#################################################################

print_step("STEP 1 - Extract webpage as Markdown")

webpage_markdown = extract_webpage_to_markdown(
    url=WEBPAGE_URL,
    output_path=WEBPAGE_MARKDOWN_PATH,
)

#################################################################
# STEP 2 - Split Markdown into overlapping chunks
#################################################################

print_step("STEP 2 - Split Markdown into overlapping chunks")

# EXERCISE - Configure chunking:
# Choose the maximum chunk size and the number of characters shared by
# consecutive chunks. The overlap must be smaller than the chunk size.
CHUNK_SIZE = ...
CHUNK_OVERLAP = ...


def build_chunks(text: str) -> list[str]:
    """Split Markdown by headings, then recursively with overlap."""

    # EXERCISE - Split by Markdown headings:
    # Use the heading-aware splitter so each section retains its heading path.
    header_splitter = ...(
        headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ],
        strip_headers=False,
    )
    markdown_sections = header_splitter.split_text(text)

    # EXERCISE - Split large sections recursively:
    # Use the configured size and overlap, and record each chunk's start index.
    text_splitter = ...(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    split_documents = text_splitter.split_documents(markdown_sections)

    return format_markdown_chunks(text, split_documents)


webpage_chunks = build_chunks(webpage_markdown)
console.print(
    f"Created {len(webpage_chunks)} chunks "
    f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).",
    style=INFO_STYLE,
)
console.print("Preview of first chunk:", style=INFO_STYLE)
print_result(webpage_chunks[0][:700])

save_chunks_to_markdown(webpage_chunks, CHUNKS_MARKDOWN_PATH)
console.print(f"Saved chunks to '{CHUNKS_MARKDOWN_PATH}'.", style=INFO_STYLE)

#################################################################
# STEP 3 - Store chunks in ChromaDB
#################################################################

print_step("STEP 3 - Store chunks in ChromaDB")


def create_vector_collection(chunks: list[str]) -> chromadb.Collection:
    """Create a ChromaDB collection and store the webpage chunks."""
    client = chromadb.PersistentClient(path=str(VECTORSTORES_DIR))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except NotFoundError:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    # ChromaDB automatically embeds the documents and associates each one with
    # the unique ID at the same list position.
    collection.upsert(
        documents=chunks,
        ids=[f"chunk_{index}" for index in range(len(chunks))],
    )

    return collection


# EXERCISE - Build the vector collection:
# Pass the webpage chunks to the function defined above.
collection = create_vector_collection(...)

console.print(
    f"Stored chunks in ChromaDB collection '{COLLECTION_NAME}', "
    f"with {collection.count()} chunks",
    style=INFO_STYLE,
)

#################################################################
# STEP 4 - Retrieve context for a question
#################################################################

print_step("STEP 4 - Retrieve context for a question")

# EXERCISE - Configure retrieval:
# Choose the maximum number of relevant chunks returned for each question.
TOP_K = ...


def retrieve_context(
    collection: chromadb.Collection,
    question: str,
    top_k: int,
) -> str:
    """Retrieve the most relevant chunks for a user question.

    Args:
        collection: The ChromaDB collection containing webpage chunks.
        question: The user's question.
        top_k: The maximum number of relevant chunks to retrieve.

    Returns:
        A formatted string containing the retrieved chunks.
    """
    # EXERCISE - Retrieve the most relevant chunks:
    # Use the function argument to control the number of ChromaDB results.
    results = collection.query(
        query_texts=[question],
        n_results=...,
        include=["documents"],
    )

    retrieved_chunks = results.get("documents", [[]])

    if not retrieved_chunks:
        return "No relevant webpage chunks were found."

    return "\n\n------------\n\n".join(retrieved_chunks[0])


retrieved_context = retrieve_context(
    collection=collection,
    question=USER_QUESTION,
    top_k=TOP_K,
)
console.print(f"Question: {USER_QUESTION}", style=INFO_STYLE)
console.print("Preview of retrieved context:", style=INFO_STYLE)
print_result(retrieved_context[:1200])

#################################################################
# STEP 5 - Generate an answer from the retrieved context
#################################################################

print_step("STEP 5 - Generate an answer from the retrieved context")


def answer_with_context(question: str, context: str) -> str:
    """Ask the model to answer using retrieved webpage context.

    Args:
        question: The user's question.
        context: The retrieved context from ChromaDB.

    Returns:
        The model answer based on the retrieved context.
    """
    agent = Agent(
        model=get_llm_model(),
        instructions=(
            "You are a helpful assistant. Use the retrieved webpage context to "
            "answer the user's question. If the answer is not in the context, "
            "say you don't know. Cite the webpage source, chunk, and section."
        ),
        model_settings=ModelSettings(thinking="minimal"),
    )

    # Keep the question and retrieved evidence clearly separated in the prompt.
    prompt = f"Question: {question}\n\nRetrieved context:\n{context}"

    result = agent.run_sync(prompt)
    return result.output


# EXERCISE - Generate a grounded answer:
# Pass the original user question and the context retrieved from ChromaDB.
answer = answer_with_context(
    question=...,
    context=...,
)

console.print("Final answer:", style=INFO_STYLE)
print_result(answer)
