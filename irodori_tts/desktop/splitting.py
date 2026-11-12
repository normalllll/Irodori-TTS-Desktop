"""Cut long recordings into training-sized segments at silences."""

from __future__ import annotations

import numpy as np

MAX_SEGMENT = 12.0
# A segment shorter than this is not worth training on, so no cut leaves one behind.
MIN_SEGMENT = 1.0
HOP = 0.01
# Silences are searched from the longest tier down: a clear pause first, then a short gap.
SILENCE_TIERS = (0.25, 0.12)
SILENCE_BELOW_PEAK_DB = 30.0


def level_db(mono: np.ndarray, rate: int) -> np.ndarray:
    """Short-time level in dB, one value per 10 ms."""
    hop = max(1, round(rate * HOP))
    frames = len(mono) // hop
    if frames == 0:
        return np.zeros(0)
    energy = np.square(mono[: frames * hop].astype(np.float64)).reshape(frames, hop).mean(axis=1)
    return 10 * np.log10(energy + 1e-12)


def silent_runs(level: np.ndarray, minimum: float) -> list[tuple[int, int]]:
    """Frame ranges [start, end) below the silence threshold that last at least `minimum` s."""
    if len(level) == 0:
        return []
    quiet = level < np.percentile(level, 90) - SILENCE_BELOW_PEAK_DB
    edges = np.flatnonzero(np.diff(np.concatenate(([0], quiet.astype(np.int8), [0]))))
    wanted = round(minimum / HOP)
    return [(a, b) for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= wanted]


def choose_cut(level: np.ndarray, low: float, high: float) -> float:
    """Where to cut within [low, high] seconds: the latest clear silence (its middle), else
    the latest short gap, else the quietest point. Never inside speech when a gap exists."""
    # Whole frames inside the range, so the cut can never push a segment past its limit.
    first, last = int(np.ceil(low / HOP)), min(len(level), int(high // HOP))
    for tier in SILENCE_TIERS:
        found = [(a, b) for a, b in silent_runs(level, tier) if b > first and a < last]
        if found:
            a, b = found[-1]
            a, b = max(a, first), min(b, last)
            return (a + b) / 2 * HOP
    window = max(1, round(0.05 / HOP))
    smooth = np.convolve(level[first:last], np.ones(window) / window, mode="same")
    # Latest minimum: longer segments mean fewer cuts.
    return (first + len(smooth) - 1 - int(np.argmin(smooth[::-1]))) * HOP + HOP / 2


def split_points(
    mono: np.ndarray, rate: int, *, max_length: float = MAX_SEGMENT
) -> list[tuple[float, float]]:
    """Contiguous (start, end) seconds covering the recording, each at most `max_length`."""
    total = len(mono) / rate
    if total <= max_length:
        return [(0.0, total)]
    level = level_db(mono, rate)
    segments, start = [], 0.0
    while total - start > max_length:
        cut = choose_cut(level, start + MIN_SEGMENT, min(start + max_length, total - MIN_SEGMENT))
        segments.append((start, cut))
        start = cut
    segments.append((start, total))
    return segments
