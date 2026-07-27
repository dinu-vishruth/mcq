"""
Document chunking for RAG. Pure Python, no dependencies.

Strategy: paragraph-aware sliding window. We split on blank lines / sentence
boundaries first, then pack pieces into ~CHUNK_SIZE-char windows with
CHUNK_OVERLAP-char overlap so a concept spanning a boundary is still retrievable
from at least one chunk. Each chunk records its char offsets in the cleaned
document so retrieval can report provenance.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import config

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    index: int
    content: str
    char_start: int
    char_end: int

    @property
    def token_estimate(self) -> int:
        # Rough heuristic: ~4 chars/token. Good enough for budgeting.
        return max(1, len(self.content) // 4)


def _segments(text: str, size: int) -> list[str]:
    """Break text into small units (paragraphs, then sentences) to pack.

    `size` bounds a single segment so downstream packing can respect the
    requested chunk size rather than the global config default.
    """
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= size:
            parts.append(para)
        else:
            # Paragraph too big: fall back to sentence-level pieces.
            buf = ""
            for sent in _SENT_SPLIT.split(para):
                # A single sentence longer than `size` is hard-split so no
                # segment ever exceeds the requested size.
                while len(sent) > size:
                    if buf:
                        parts.append(buf)
                        buf = ""
                    parts.append(sent[:size])
                    sent = sent[size:]
                if len(buf) + len(sent) + 1 <= size:
                    buf = f"{buf} {sent}".strip()
                else:
                    if buf:
                        parts.append(buf)
                    buf = sent
            if buf:
                parts.append(buf)
    return parts


def chunk_text(text: str, chunk_size: int | None = None, overlap: int | None = None) -> list[Chunk]:
    """Return ordered Chunks covering the text with overlap.

    Works on a single-line cleaned string (the app's clean_text collapses
    whitespace) as well as multi-paragraph text.
    """
    text = (text or "").strip()
    if not text:
        return []

    size = chunk_size or config.CHUNK_SIZE
    ov = overlap if overlap is not None else config.CHUNK_OVERLAP
    ov = min(ov, max(0, size - 1))

    # If the cleaned text has no paragraph structure (common: clean_text collapses
    # everything to one line), do a direct char-window sweep with word-boundary snap.
    segs = _segments(text, size)
    if len(segs) <= 1:
        return _window_sweep(text, size, ov)

    chunks: list[Chunk] = []
    idx = 0
    cursor = 0  # char offset tracker into the original text
    buf = ""
    buf_start = 0

    def flush(end_pos):
        nonlocal buf, idx, buf_start
        if buf.strip():
            chunks.append(Chunk(idx, buf.strip(), buf_start, end_pos))
            idx += 1

    for seg in segs:
        seg_pos = text.find(seg, cursor)
        if seg_pos == -1:
            seg_pos = cursor
        if not buf:
            buf_start = seg_pos
        if len(buf) + len(seg) + 1 <= size:
            buf = f"{buf}\n\n{seg}".strip() if buf else seg
        else:
            flush(seg_pos)
            # Start new buffer with tail overlap from the previous one.
            tail = buf[-ov:] if ov and buf else ""
            buf = f"{tail}\n\n{seg}".strip() if tail else seg
            buf_start = max(0, seg_pos - len(tail))
        cursor = seg_pos + len(seg)
    flush(len(text))
    return chunks


def _window_sweep(text: str, size: int, ov: int) -> list[Chunk]:
    """Char-window sweep for structureless text, snapping to word boundaries."""
    chunks: list[Chunk] = []
    step = max(1, size - ov)
    idx = 0
    pos = 0
    n = len(text)
    while pos < n:
        end = min(pos + size, n)
        # Snap end to the nearest space to avoid cutting words, unless at EOF.
        if end < n:
            space = text.rfind(" ", pos + step, end)
            if space != -1 and space > pos:
                end = space
        piece = text[pos:end].strip()
        if piece:
            chunks.append(Chunk(idx, piece, pos, end))
            idx += 1
        if end >= n:
            break
        pos = max(end - ov, pos + 1)
    return chunks
