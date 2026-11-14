"""Reads the text out of a document so it can be split into chunks.

Plain text, Markdown and code are read directly. PDF needs `pypdf` and Word
(.docx) needs `python-docx`; both are in requirements.txt. Each reader returns
a list of Sections. A text file is one section; a PDF has one per page, so
answers can point to the page a passage came from.
"""

import logging
from dataclasses import dataclass
from pathlib import Path


class ReadError(Exception):
    """The file can't be indexed. The message says why, for the index report."""


@dataclass
class Section:
    text: str
    page: int | None = None  # PDF page number, starting at 1
    unit: str = "lines"      # what the chunker's line numbers count: "lines" or "paragraphs"


TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst", ".log", ".csv",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".c", ".h", ".cpp", ".hpp", ".cs",
    ".go", ".rs", ".rb", ".php", ".sql", ".sh", ".ps1", ".bat",
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".xml", ".html", ".css",
}
DOCUMENT_EXTENSIONS = {".pdf", ".docx"}


def read_document(path: Path) -> list[Section]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return read_pdf(path)
    if suffix == ".docx":
        return read_docx(path)
    return read_text(path)


def read_text(path: Path) -> list[Section]:
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ReadError("looks like a binary file")
    # utf-8-sig drops the byte-order mark some Windows editors add.
    return [Section(data.decode("utf-8-sig", errors="replace"))]


def read_pdf(path: Path) -> list[Section]:
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ReadError("needs pypdf (pip install -r requirements.txt)") from None
    # pypdf logs a warning for every small flaw it works around; the index
    # report already says when a file couldn't be read.
    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        reader = PdfReader(path)
        if reader.is_encrypted:
            raise ReadError("the PDF is password protected")
        sections = []
        for number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                sections.append(Section(text, page=number))
    except ReadError:
        raise
    except Exception as e:  # pypdf raises many kinds for a damaged file
        raise ReadError(f"couldn't read the PDF ({e})") from e
    if not sections:
        # Usually a scanned document: pages are pictures, with no text to extract.
        raise ReadError("no text found (a scanned PDF needs OCR, which isn't supported yet)")
    return sections


def read_docx(path: Path) -> list[Section]:
    try:
        import docx
    except ImportError:
        raise ReadError("needs python-docx (pip install -r requirements.txt)") from None
    try:
        document = docx.Document(str(path))
    except Exception as e:  # python-docx raises several kinds for a damaged file
        raise ReadError(f"couldn't read the Word file ({e})") from e
    # One line per paragraph, so chunk locations can say "paragraphs 4-9".
    # Table rows become lines too, with cells separated by " | ".
    lines = []
    for block in document.iter_inner_content():
        if isinstance(block, docx.table.Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells]
                lines.append(" | ".join(cells))
            lines.append("")
        else:
            lines.append(block.text)
    return [Section("\n".join(lines), unit="paragraphs")]
