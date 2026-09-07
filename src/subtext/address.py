"""The vocabulary for talking about who a Spanish line addresses.

This module used to *decide* the form as well, from Spanish morphology: pronouns and
clitics, the written accent that marks an enclitic imperative (`súbete`, `súbase`,
`súbanse`), and 117 second-person verb forms derived from the corpus by co-occurrence with
tú/te/ti. That reader was measured against a model on 585 retrieved lines on 2026-09-07 and
removed the same day. The result was not close:

    agreement 65.8%
    of 200 disagreements: 198 were the rule missing a form the model read correctly
                            2 were the rule being wrong outright
                            0 went the other way

The misses were systematic, not incidental. The rules handled the enclitics `-te`, `-se`
and `-nse` but not `-me`, `-lo`, `-la` (`Tráeme dinero.`, `Muéstramelo.`, `Hazlo.`); they
needed a written accent, so unaccented bare imperatives were invisible (`Dale.`,
`Entra al auto.`, `Procede.`); and the `ustedes` rule required the verb to open the line,
so `Buenas noches, pasen.` read as nothing. Coverage was 22% of the corpus against the
model's ~71%.

The underlying reason is not fixable by more rules: Spanish marks the person
morphologically only about half the time, and `usted`/`ustedes` take *third*-person forms
that are identical to él/ella/ellos. `Quiere café` is `usted` or `she` and the word does
not say. The rest is carried by sense, and sense is what a model reads.

What remains is the enum, which is vocabulary rather than logic. The tagger now lives in
`address_llm.py`, behind a ClickHouse cache: a line's form is a fact about the line and not
about the query, so it is paid for once.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Address"]


class Address(str, Enum):
    """The form a line addresses its listener in."""

    TU = "tú"
    USTED = "usted"
    USTEDES = "ustedes"
    VOSOTROS = "vosotros"       # Spain only; its presence is a contamination signal
    UNMARKED = "unmarked"       # nothing in the line says. The common case.
