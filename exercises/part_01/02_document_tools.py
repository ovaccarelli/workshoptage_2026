"""Document Tools Exercise

In this exercise, we explore how to extract text from different document formats.

1. List all available documents in the configured document directory.
2. Extract text from a PDF document using pdfminer.
3. Extract text from an image file using RapidOCR.
4. Extract a webpage to Markdown and save it locally.

"""

import time
from pathlib import Path

from pdfminer.high_level import extract_text
from rapidocr import RapidOCR

from graph_rag_workshop.utils.console_utils import (
    INFO_STYLE,
    console,
    print_result,
    print_step,
)
from graph_rag_workshop.utils.part_01_document_tools import extract_webpage_to_markdown

# Define file paths and constants
HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "data"
MY_DOCUMENTS = DATA_DIR / "my_documents"
WEBPAGE_URL = "https://federicab03.github.io/ARGO_Usecase/"
WEBPAGE_MARKDOWN_PATH = DATA_DIR / "webpage_rag" / "ARGO_Usecase.md"

#################################################################
# STEP 1 - List all available documents in the directory
#################################################################

print_step("Step 1 - List Available Documents")

# List all files in the MY_DOCUMENTS directory to see what documents are available for preprocessing.
available_documents = sorted(
    path.name
    for path in MY_DOCUMENTS.iterdir()
    if path.is_file() and not path.name.startswith(".")
)
console.print(f"Available documents: {available_documents}", style=INFO_STYLE)

#################################################################
# STEP 2 - Extract PDF text
#################################################################

print_step("Step 2 - Extract PDF Text")

pdf_path = MY_DOCUMENTS / "Rivendell_Event_Email_Meeting_Minutes.pdf"

# Extract the text from the PDF using pdfminer (returns a single string with the full text)
start = time.time()

# EXERCISE - PDF extraction:
# The PDF is not directly useful for the model,
# so we first extract its text content into a normal Python string.
# Complete this line with pdfminer's extract_text function.
doc_pdfminer = ...(str(pdf_path))

end = time.time()

console.print(f"Using file: {pdf_path.name}", style=INFO_STYLE)
console.print(
    f"pdfminer extracted {len(doc_pdfminer)} characters in {end - start:.2f} seconds",
    style=INFO_STYLE,
)
print_result(doc_pdfminer[:500])

#################################################################
# STEP 3 - Extract text from image file
#################################################################

print_step("Step 3 - Extract Image Text with RapidOCR")

image_path = MY_DOCUMENTS / "Fondue_Recipe.png"

# Extract the text from the image using RapidOCR (returns a list of recognized text segments in result.txts)

# EXERCISE - OCR engine:
# Images also need to be converted to text before the model can use them.
# Create a quiet RapidOCR engine, then the next line will run it on the image file.
engine = ...(params={"Global.log_level": "error"})

result = engine(str(image_path))
doc_ocr = "\n".join(result.txts or [])
print_result(doc_ocr[:500])

#################################################################
# STEP 4 - Extract webpage to Markdown
#################################################################

print_step("Step 4 - Extract Webpage to Markdown")

# EXERCISE - Webpage extraction:
# Use extract_webpage_to_markdown to fetch the webpage, convert its content to
# Markdown, and save it at WEBPAGE_MARKDOWN_PATH.
webpage_markdown = ...(
    url=WEBPAGE_URL,
    output_path=WEBPAGE_MARKDOWN_PATH,
)

console.print(
    f"Extracted {len(webpage_markdown)} characters and saved them to "
    f"'{WEBPAGE_MARKDOWN_PATH}'.",
    style=INFO_STYLE,
)
print_result(webpage_markdown[:500])
