"""
pdf_reader.py — PDF text extraction using PyMuPDF (fitz).
"""
import pymupdf as fitz  # PyMuPDF v1.24+ (formerly import fitz)


def read_pdf_bytes(data: bytes) -> str:
    """Extract text from raw PDF bytes (in-memory upload)."""
    doc = fitz.open(stream=data, filetype="pdf")
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text.strip())
    doc.close()
    return "\n".join(pages)


def read_pdf_path(path: str) -> str:
    """Extract text from a PDF file on disk."""
    doc = fitz.open(path)
    pages: list[str] = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages.append(text.strip())
    doc.close()
    return "\n".join(pages)
