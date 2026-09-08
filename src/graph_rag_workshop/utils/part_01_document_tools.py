"""Document handling and text extraction tools for a Pydantic AI agent."""

import json
import re
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from loguru import logger
from markdownify import markdownify
from pdfminer.high_level import extract_text
from rapidocr import RapidOCR

# Document paths
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"


# -----------------------------------------------------------------------------
# Document discovery and path resolution
# -----------------------------------------------------------------------------


def resolve_document_path(file_path: str) -> Path:
    """Resolve document names relative to the configured document folder.

    Relative paths are resolved from the workshop document directory.

    Args:
        file_path: The filename or path of the document to resolve.

    Returns:
        A Path object representing the resolved absolute path to the document.
    """
    document_root = MY_DOCUMENTS.resolve()
    path = Path(file_path).expanduser()
    if not path.is_absolute():
        path = document_root / path

    path = path.resolve()

    if not path.is_relative_to(document_root):
        raise ValueError(
            f"Document paths must stay inside the workshop document directory: "
            f"{document_root}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Document not found: {path.name}")

    return path


def list_my_available_documents() -> list[str]:
    """List the filenames available in the workshop document directory.

    Returns:
        A sorted list of visible document filenames.
    """
    return sorted(
        path.name
        for path in MY_DOCUMENTS.iterdir()
        if path.is_file() and not path.name.startswith(".")
    )


# -----------------------------------------------------------------------------
# Local document extraction
# -----------------------------------------------------------------------------


def extract_text_from_pdf_file(file_path: str) -> str:
    """Extract text from a PDF file.

    Args:
        file_path: The filename or path of the PDF to extract text from.

    Returns:
        The extracted PDF text.
    """
    path = resolve_document_path(file_path)
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, received: {path.name}")
    logger.info(f"Extracting PDF text from {path.name}")
    return extract_text(path)


def extract_text_from_image_file(file_path: str) -> str:
    """Extract text from an image using RapidOCR.

    Args:
        file_path: The filename or path of the image to extract text from.

    Returns:
        The extracted text content of the image.
    """
    path = resolve_document_path(file_path)
    if path.suffix.lower() not in {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}:
        raise ValueError(f"Expected a supported image file, received: {path.name}")

    engine = RapidOCR(params={"Global.log_level": "error"})
    result = engine(path)
    return (
        "\n".join(result.txts or []) if result else "Image text could not be extracted."
    )


# -----------------------------------------------------------------------------
# Website extraction
# -----------------------------------------------------------------------------


def fetch_url(url: str) -> str:
    """Fetch a URL and decode its response body as text."""
    request = Request(
        url,
        headers={"User-Agent": "graph-rag-workshop/0.1"},
    )

    with urlopen(request, timeout=10) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def extract_script_urls(html: str, base_url: str) -> list[str]:
    """Extract same-site JavaScript URLs from an HTML page."""
    base_domain = urlparse(base_url).netloc
    sources = re.findall(r"<script[^>]+src=['\"]([^'\"]+)['\"]", html, flags=re.I)
    urls = [urljoin(base_url, source) for source in sources]
    return [url for url in urls if urlparse(url).netloc == base_domain]


def html_to_markdown(html: str) -> str:
    """Convert HTML content to Markdown."""
    return markdownify(html, heading_style="ATX", strip=["script", "style"]).strip()


def site_javascript_to_markdown(
    javascript: str,
    base_url: str | None = None,
) -> str:
    """Convert this static site's JavaScript content data to Markdown."""
    match = re.search(r"const site = (.*?);\n\nconst pageKey", javascript, re.S)
    if not match:
        return ""

    site_json = re.sub(
        r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*):",
        r'\1"\2":',
        match.group(1),
    )
    site_json = re.sub(r",\s*([}\]])", r"\1", site_json)
    site = json.loads(site_json)

    def collect_hrefs(value) -> list[str]:
        """Collect relative HTML links from the parsed site data."""
        if isinstance(value, dict):
            hrefs = [value["href"]] if isinstance(value.get("href"), str) else []
            return hrefs + [
                href
                for nested_value in value.values()
                for href in collect_hrefs(nested_value)
            ]
        if isinstance(value, list):
            return [href for item in value for href in collect_hrefs(item)]
        return []

    relative_urls = {
        Path(href).stem: href.removeprefix("../")
        for href in collect_hrefs(site)
        if href.endswith(".html")
    }
    relative_urls["home"] = "index.html"

    if base_url:
        parsed_base_url = urlparse(base_url)
        project_path = parsed_base_url.path
        for folder in ("/pages/", "/venues/"):
            if folder in project_path:
                project_path = project_path.split(folder, 1)[0] + "/"
                break
        if not project_path.endswith("/"):
            project_path = project_path.rsplit("/", 1)[0] + "/"
        project_url = parsed_base_url._replace(
            path=project_path,
            params="",
            query="",
            fragment="",
        ).geturl()
        source_urls = {
            page_key: urljoin(project_url, relative_url)
            for page_key, relative_url in relative_urls.items()
        }
    else:
        source_urls = relative_urls

    lines = [
        f"# {site['brand']}",
        f"Source: {source_urls['home']}",
        site.get("tagline", ""),
    ]

    for page_key, page in site["pages"].items():
        page_source = source_urls.get(page_key, source_urls["home"])
        lines.extend(["", f"## {page['title']}", f"Source: {page_source}"])
        for key in ["eyebrow", "description", "intro"]:
            if page.get(key):
                lines.append(page[key])
        for item in page.get("timeline", []):
            lines.append(f"- **{item[0]}** {item[1]}")
        for item in page.get("stats", []):
            lines.append(f"- **{item['value']}** {item['label']}")
        for item in page.get("highlights", []) + page.get("venues", []):
            lines.append(f"- **{item['title']}** {item['text']}")
        for item in page.get("contacts", []):
            lines.append(f"- **{item['label']}** {item['value']}")
        for question, answer in page.get("faqs", []):
            lines.append(f"- **{question}** {answer}")
        if page.get("meta"):
            lines.extend([f"- {item}" for item in page["meta"]])
        for section in page.get("sections", []):
            lines.extend(["", f"### {section['title']}"])
            lines.extend(section.get("body", []))
            for card in section.get("cards", []):
                lines.append(f"- **{card['title']}** {card['text']}")

    return "\n\n".join(line for line in lines if line)


def extract_text_from_website(url: str) -> str:
    """Fetch a web page and extract its content as structured Markdown.

    Args:
        url: The HTTP or HTTPS URL to scrape.

    Returns:
        The extracted page content as Markdown.
    """
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs are supported.")

    logger.info(f"Extracting website text from {url}")
    html = fetch_url(url)
    html_markdown = html_to_markdown(html)
    js_markdown_parts = []

    # Some static sites render their visible text from JavaScript data.
    for script_url in extract_script_urls(html, url):
        javascript = fetch_url(script_url)
        js_markdown_parts.append(site_javascript_to_markdown(javascript, url))

    return (
        "\n\n".join(markdown for markdown in js_markdown_parts if markdown)
        or html_markdown
    )


def extract_webpage_to_markdown(url: str, output_path: Path) -> str:
    """Extract a webpage to Markdown and save it locally.

    Args:
        url: The webpage URL to extract.
        output_path: The Markdown file where the extracted page content is saved.

    Returns:
        The extracted Markdown content.
    """
    markdown = extract_text_from_website(url)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")

    return markdown
