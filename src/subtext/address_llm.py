"""The other tagger: the model reads the form of address, not the morphology.

`address.py` decides tú/usted/ustedes from Spanish spelling — pronouns, enclitic accents,
a corpus-derived list of second-person verb forms. It is deterministic, free, and it has a
known blind spot: `usted` and `ustedes` take *third*-person verb morphology, identical to
él/ella/ellos, so `Quiere café` is unreadable without context and `Sube al auto` could be a
command or a statement about somebody else. It answers UNMARKED for 78% of lines.

This module asks a model instead. Bruno's argument for it is the blind spot: no rule can
read what the morphology does not encode, and a reader who understands the sentence can.
The argument against it is the project's own measurement — reasoning walked away from the
evidence on translation — plus determinism, cost, and the fact that a grouping used to
justify a citation ought to be reproducible.

Neither argument settles it, so both taggers exist and are measured against each other on
the same lines. Whichever wins, the other is deleted; keeping two is the worst outcome.

Lines are labelled in **batches**, because the unit of the decision is one line but the
unit of the cost is one call. A cue retrieves a few hundred precedents; tagging them one
at a time would be a few hundred calls per cue and the comparison would be unaffordable
rather than merely expensive.
"""

from __future__ import annotations

import json
import re
from typing import Sequence

from .address import Address

#: Lines per call. Large enough that a cue costs one or two calls; small enough that the
#: model does not lose the numbering, which is the only thing tying answers back to lines.
BATCH = 20

TAG_INSTRUCTION = """\
Each numbered line below is Spanish subtitle dialogue. For each one, say how it addresses
the person being spoken to:

  "tu"       — informal singular (tú): tú forms, te/ti, -te enclitics
  "usted"    — formal singular (usted): usted forms, -se enclitics on commands
  "ustedes"  — plural (ustedes): addressing two or more people
  "vosotros" — Spain's plural. Rare here and always a contamination signal.
  "none"     — the line does not address a listener at all, OR nothing in it reveals which
               form is used. Spanish drops the subject pronoun, so this is common and it
               is the correct answer whenever the line genuinely does not say.

Answer "none" rather than guessing. These labels are used to group evidence for a
citation, so a wrong label is worse than a missing one.

Return ONLY a JSON object mapping each number to its label, and nothing else:
{"1": "tu", "2": "none", "3": "usted"}
"""

_LABELS = {
    "tu": Address.TU, "tú": Address.TU,
    "usted": Address.USTED,
    "ustedes": Address.USTEDES,
    "vosotros": Address.VOSOTROS,
    "none": Address.UNMARKED, "unmarked": Address.UNMARKED, "": Address.UNMARKED,
}


def _parse(reply: str, count: int) -> list[Address]:
    """Read the label map. Anything missing or unrecognised comes back UNMARKED.

    A tagger that raises on bad output would take down a run of a hundred cues over one
    malformed reply, and the honest fallback for "the model did not say" is exactly the
    same as for "the model said it cannot tell".
    """
    out = [Address.UNMARKED] * count
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return out
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out
    for key, value in data.items():
        try:
            index = int(str(key).strip()) - 1
        except ValueError:
            continue
        if 0 <= index < count:
            out[index] = _LABELS.get(str(value).strip().lower(), Address.UNMARKED)
    return out


def tag_batch(lines: Sequence[str], *, ask, batch: int = BATCH) -> list[Address]:
    """Label every line with the form of address it uses. One call per `batch` lines.

    The batches run at once. A new cue retrieves a few hundred precedents the cache has
    not seen, and one call over sixty of them was measured at 7 s -- the longest single
    step of a request -- because the model writes sixty answers in series. Three calls
    over twenty each write them in parallel and the order of `lines` is kept.
    """
    from concurrent.futures import ThreadPoolExecutor

    chunks = [lines[start:start + batch] for start in range(0, len(lines), batch)]

    def one(chunk):
        numbered = "\n".join(f"{i + 1}. {line}" for i, line in enumerate(chunk))
        return _parse(ask(f"{TAG_INSTRUCTION}\n{numbered}\n"), len(chunk))

    if len(chunks) <= 1:
        return one(chunks[0]) if chunks else []
    out: list[Address] = []
    with ThreadPoolExecutor(max_workers=min(len(chunks), 6)) as pool:
        for labels in pool.map(one, chunks):
            out.extend(labels)
    return out


def llm_tagger(ask, *, batch: int = BATCH):
    """A drop-in for `group_by_address(tag_forms=...)` backed by the model."""

    def tag_forms(lines: Sequence[str]) -> list[Address]:
        return tag_batch(lines, ask=ask, batch=batch)

    return tag_forms


# --------------------------------------------------------------------------- cache


#: Tagging is the expensive half of grouping: a cue retrieves a few hundred precedents and
#: each has to be read. But the corpus does not change, so a line's form of address is a
#: fact about that line, not about the query — the same `Súbase.` is `usted` for every cue
#: that ever retrieves it. Tagging it twice is paying twice for one answer.
#:
#: So tags are written to ClickHouse and looked up first. The first run over the AFI quotes
#: pays for what it uses; the second pays nothing. Over a film of 1,500 cues the saving is
#: the difference between ~3,000 extra calls and a few hundred, and it grows with every run
#: because popular lines are retrieved again and again.
TAG_TABLE = "address_tags"

TAG_TABLE_DDL = f"""
CREATE TABLE IF NOT EXISTS {TAG_TABLE} (
    es      String,
    model   LowCardinality(String),
    form    LowCardinality(String),
    tagged  DateTime DEFAULT now()
) ENGINE = ReplacingMergeTree(tagged)
ORDER BY (es, model)
"""


def ensure_tag_table(ch=None) -> None:
    from .db import client

    (ch or client()).command(TAG_TABLE_DDL)


def load_tags(lines: Sequence[str], model: str, ch=None) -> dict[str, Address]:
    """Whatever of `lines` has already been tagged by this model."""
    from .db import client

    if not lines:
        return {}
    ch = ch or client()
    rows = ch.query(
        f"SELECT es, argMax(form, tagged) FROM {TAG_TABLE} "
        "WHERE model = {m:String} AND es IN {l:Array(String)} GROUP BY es",
        parameters={"m": model, "l": list(dict.fromkeys(lines))},
    ).result_rows
    return {es: _LABELS.get(form, Address.UNMARKED) for es, form in rows}


def save_tags(pairs: Sequence[tuple[str, Address]], model: str, ch=None) -> None:
    from .db import client

    if not pairs:
        return
    (ch or client()).insert(
        TAG_TABLE, [[es, model, form.value] for es, form in pairs],
        column_names=["es", "model", "form"],
    )


def cached_tagger(ask, *, model: str, batch: int = BATCH, ch=None):
    """The model tagger, with a ClickHouse cache in front of it.

    Only lines never seen before reach the model, and the answer order is preserved so the
    caller cannot tell a hit from a miss. A cache miss and a cache hit produce the same
    label for the same line, because the label does not depend on the cue.
    """
    ensure_tag_table(ch)

    def tag_forms(lines: Sequence[str]) -> list[Address]:
        known = load_tags(lines, model, ch)
        missing = [s for s in dict.fromkeys(lines) if s not in known]
        if missing:
            fresh = tag_batch(missing, ask=ask, batch=batch)
            save_tags(list(zip(missing, fresh)), model, ch)
            known.update(zip(missing, fresh))
        return [known.get(s, Address.UNMARKED) for s in lines]

    return tag_forms
