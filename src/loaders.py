from pathlib import Path

import pymupdf
from PIL import Image

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    PDF_DIR,
    IMAGE_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


# ============================================================
# PDF LOADER
# ============================================================

def load_pdf(pdf_path):
    """
    Load a PDF and return its pages as dictionaries.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document):

        text = page.get_text()

        pages.append({
            "text": text,
            "page": page_number,
            "page_label": page_number + 1,
            "source": str(pdf_path)
        })

    document.close()

    return pages


# ============================================================
# PDF CHUNKING
# ============================================================

def chunk_pdf_pages(pages):
    """
    Split PDF page text into smaller chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = []

    for page in pages:

        text = page["text"]

        if not text.strip():
            continue

        split_texts = text_splitter.split_text(text)

        for chunk_text in split_texts:

            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "page_label": page["page_label"],
                "source": page["source"],
                "type": "pdf_text"
            })

    return chunks


# ============================================================
# FIND PDF FILES
# ============================================================

def find_pdf_files():
    """
    Find all PDF files inside the PDF directory.
    """

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    return pdf_files


# ============================================================
# IMAGE LOADER
# ============================================================

def load_image(image_path):
    """
    Load an image using Pillow.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image file not found: {image_path}"
        )

    image = Image.open(image_path)

    return image


# ============================================================
# FIND IMAGE FILES
# ============================================================

def find_image_files():
    """
    Find supported image files inside the image directory.
    """

    extensions = [
        "*.jpg",
        "*.jpeg",
        "*.png",
        "*.webp"
    ]

    image_files = []

    for extension in extensions:
        image_files.extend(IMAGE_DIR.glob(extension))

    return image_files