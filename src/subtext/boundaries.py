"""Refine Mexican-document boundaries by shrinking windows with decreasing thresholds.

The detector scores the corpus in fixed 1000-line blocks, so a document's edge is only
known to +/-1000 lines — and 523 of 697 documents were a *single* block, meaning both
their edges were unknown. 84.7% of the old corpus sat in such an edge zone, so this is a
correction, not a polish.

The method (Bruno's): once a block passes, re-test its neighbourhood at half the window
and a lower threshold, and repeat. Coarse levels move the edge in big steps; fine levels
place it precisely. Marker density cannot survive bisection all the way down — 1000 lines
hold ~45 markers, 50 lines hold ~2 — so the threshold has to fall with the window.

The chain below was chosen by sweeping 156 monotone chains (see `sweep_chains`). Its
boundaries are identical, for the median document, to those of the next nine best chains:
the edges are a property of the corpus, not of the thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

#: Bucket resolution the scores are aggregated at, in lines.
RESOLUTION = 25

#: (window in lines, minimum Mexican markers in that window). Coarse to fine.
CHAIN: tuple[tuple[int, int], ...] = ((1000, 10), (500, 7), (250, 3), (125, 2), (50, 1))

#: A window also has to be more Mexican than peninsular, at every level.
MX_OVER_ES = 2.0

#: Refinement extends as well as trims; without a stop it would walk a whole run of
#: consecutive Mexican productions. Six documents legitimately reach this.
MAX_LINES = 6000


@dataclass(frozen=True)
class Scores:
    """Marker counts per `RESOLUTION`-line bucket, as prefix sums for O(1) windows."""

    mx: Sequence[int]
    es: Sequence[int]

    def window(self, b0: int, b1: int) -> tuple[int, int]:
        b0 = max(b0, 0)
        b1 = min(b1, len(self.mx) - 2)
        if b1 < b0:
            return 0, 0
        return self.mx[b1 + 1] - self.mx[b0], self.es[b1 + 1] - self.es[b0]

    def passes(self, b0: int, b1: int, threshold: int) -> bool:
        mx, es = self.window(b0, b1)
        return mx >= threshold and mx >= MX_OVER_ES * es


def refine(lo: int, hi: int, scores: Scores, chain=CHAIN) -> tuple[int, int]:
    """Walk one document's edges outward then inward, at each level of the chain."""
    a, z = lo // RESOLUTION, hi // RESOLUTION
    for window, threshold in chain:
        step = window // RESOLUTION
        while (z - a + 1) * RESOLUTION < MAX_LINES and scores.passes(a - step, a - 1, threshold):
            a -= step
        while (z - a + 1) * RESOLUTION < MAX_LINES and scores.passes(z + 1, z + step, threshold):
            z += step
        while (z - a + 1) > step and not scores.passes(a, a + step - 1, threshold):
            a += step
        while (z - a + 1) > step and not scores.passes(z - step + 1, z, threshold):
            z -= step
    return a * RESOLUTION, z * RESOLUTION + RESOLUTION - 1


def merge(ranges: Iterable[tuple[int, int]]) -> list[list[int]]:
    """Union refined ranges that now touch. Two seeds can refine onto the same film."""
    out: list[list[int]] = []
    for lo, hi in sorted(ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out
