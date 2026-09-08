"""RAG preparation helpers for the Part 1 webpage exercise."""

import re
from pathlib import Path


def format_markdown_chunks(text: str, documents: list) -> list[str]:
    """Format split Markdown documents with source and structural metadata."""
    source_by_page_heading = dict(
        re.findall(
            r"^##[ \t]+(.+?)\s*\n\s*Source:[ \t]*(\S+)",
            text,
            flags=re.MULTILINE,
        )
    )
    root_source_match = re.search(
        r"^#[ \t]+.+?\s*\n\s*Source:[ \t]*(\S+)",
        text,
        flags=re.MULTILINE,
    )
    root_source = root_source_match.group(1) if root_source_match else ""

    chunks = []
    for index, document in enumerate(documents):
        section = (
            " > ".join(
                document.metadata[key]
                for key in ("Header 1", "Header 2", "Header 3")
                if key in document.metadata
            )
            or "No header"
        )
        source = source_by_page_heading.get(
            document.metadata.get("Header 2", ""),
            root_source,
        )
        content = re.sub(
            r"^Source:[ \t]*\S+\s*",
            "",
            document.page_content,
            flags=re.MULTILINE,
        ).strip()
        chunks.append(
            f"Source: {source or 'Unknown'}\n"
            f"Chunk: {index}\n"
            f"Section: {section}\n"
            f"Start index: {document.metadata.get('start_index', 0)}\n"
            f"Content:\n{content}"
        )

    if not chunks:
        raise ValueError("The webpage extraction produced no text chunks.")

    return chunks


def save_chunks_to_markdown(chunks: list[str], output_path: Path) -> None:
    """Save documented chunks to a Markdown file."""
    chunks_markdown = "\n\n---\n\n".join(
        f"# Chunk {index}\n\n{chunk}" for index, chunk in enumerate(chunks, start=1)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(chunks_markdown, encoding="utf-8")
