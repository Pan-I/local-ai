"""Splits document text into chunks small enough to embed and to paste into a prompt.

Chunks are built from whole paragraphs (blocks separated by blank lines) where
possible, so a passage isn't cut mid-thought. A paragraph that is too long is
split by lines, and a single very long line by characters. Each chunk starts
with the last small piece of the one before it (the "overlap"), so a sentence
near a boundary still has some context. Every chunk remembers its first and
last line number, which become the source reference shown with an answer.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start_line: int  # 1-based, inclusive
    end_line: int


@dataclass
class _Piece:
    text: str
    start_line: int
    end_line: int
    paragraph: int  # pieces from the same paragraph are joined with one newline


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> list[Chunk]:
    pieces = _pieces(text, size)
    chunks: list[Chunk] = []
    current: list[_Piece] = []
    length = 0
    for piece in pieces:
        extra = len(piece.text) + (2 if current else 0)
        if current and length + extra > size:
            chunks.append(_join(current))
            last = current[-1]
            # Carry the last piece over if it's small enough to be overlap.
            if len(last.text) <= overlap and len(last.text) + 2 + len(piece.text) <= size:
                current, length = [last], len(last.text)
            else:
                current, length = [], 0
            extra = len(piece.text) + (2 if current else 0)
        current.append(piece)
        length += extra
    if current:
        chunks.append(_join(current))
    return chunks


def _pieces(text: str, size: int) -> list[_Piece]:
    """Paragraphs no longer than `size`, splitting long ones into runs of lines."""
    pieces: list[_Piece] = []
    paragraph = 0
    run: list[tuple[int, str]] = []  # (line number, line) of the current run of lines

    def flush():
        if run:
            pieces.append(_Piece("\n".join(t for _, t in run), run[0][0], run[-1][0], paragraph))
            run.clear()

    for number, line in enumerate(text.splitlines(), start=1):
        line = line.rstrip()
        if not line.strip():
            flush()
            paragraph += 1
            continue
        # A single line longer than a chunk is cut into slices.
        while len(line) > size:
            flush()
            pieces.append(_Piece(line[:size], number, number, paragraph))
            line = line[size:]
        if run and sum(len(t) + 1 for _, t in run) + len(line) > size:
            flush()
        run.append((number, line))
    flush()
    return pieces


def _join(pieces: list[_Piece]) -> Chunk:
    parts = [pieces[0].text]
    for before, piece in zip(pieces, pieces[1:]):
        parts.append(("\n" if piece.paragraph == before.paragraph else "\n\n") + piece.text)
    return Chunk("".join(parts), pieces[0].start_line, pieces[-1].end_line)
