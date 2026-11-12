"""Cut a long text into pieces the speech model reads in one pass."""

from __future__ import annotations

# The model truncates at 256 tokens and caps a clip at 30 s; 120 characters stays inside both.
SEGMENT_CHARS = 120

SENTENCE_ENDS = "。！？!?…\n"
CLAUSE_ENDS = "、，,"
# A closing quote or bracket belongs to the sentence it ends.
CLOSERS = "」』）)”’】"


def cut_after(text: str, ends: str, closers: str = "") -> list[str]:
    """Pieces of text, each ending after a run of `ends` (and any closers that follow)."""
    pieces = []
    start = 0
    index = 0
    while index < len(text):
        if text[index] in ends:
            index += 1
            while index < len(text) and (text[index] in ends or text[index] in closers):
                index += 1
            pieces.append(text[start:index])
            start = index
        else:
            index += 1
    if start < len(text):
        pieces.append(text[start:])
    return pieces


def merge(pieces: list[str], limit: int) -> list[str]:
    """Join neighbours greedily while the result stays within the limit."""
    merged: list[str] = []
    for piece in pieces:
        if merged and len(merged[-1]) + len(piece) <= limit:
            merged[-1] += piece
        else:
            merged.append(piece)
    return merged


def fit(sentence: str, limit: int) -> list[str]:
    """A sentence as pieces within the limit: at commas first, then anywhere."""
    if len(sentence) <= limit:
        return [sentence]
    result = []
    for clause in merge(cut_after(sentence, CLAUSE_ENDS), limit):
        result.extend(clause[i : i + limit] for i in range(0, len(clause), limit))
    return result


def split_text(text: str, limit: int = SEGMENT_CHARS) -> list[str]:
    """The text as pieces of at most `limit` characters, cut after sentence ends where it can.

    Whitespace around a sentence is dropped; everything else is kept, in order."""
    sentences = []
    for sentence in cut_after(text, SENTENCE_ENDS, CLOSERS):
        sentence = sentence.strip()
        if sentence:
            sentences.extend(fit(sentence, limit))
    return [piece for piece in merge(sentences, limit) if piece.strip()]
