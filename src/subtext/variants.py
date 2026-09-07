"""One English line in, every Mexican Spanish reading the corpus attests out.

A famous quote arrives with no scene attached. `Get in the car.` is a command to one
friend, to one stranger, or to three people, and English does not say which — `you` is all
three. A system that returns one Spanish line has silently chosen, and on this corpus it
would choose by counting translators, which is to say by accident.

So it does not choose. It returns the readings the corpus can support, each grounded in
its own evidence and each citable:

    Get in the car.
      tú       Súbete al coche.      pair_id 82349552, 87883008
      usted    Súbase.               pair_id 83693863
      ustedes  Entren al carro.      pair_id 38819324

That is the honest answer to a question that has no single answer, and it is the corpus
doing work a prompt cannot: the three groups are *found*, not invented. A model asked for
"three levels of formality" would produce three plausible lines with nothing behind them.

**One call per reading, each seeing only its own evidence.** Handing all three groups to
one call invites blending — the `usted` line coming back with `tú` precedent's wording —
and it makes the citations unverifiable, because nothing then ties an output line to the
evidence that produced it.

Readings with no attested precedent are not offered. If nothing in 718,925 lines addresses
this cue as `ustedes`, that variant does not exist and is not manufactured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import re

from .address import Address, read_address
from .localise import MIN_SIMILARITY, check_register, gather_phrases

#: How much precedent a reading needs before it is offered. A *marked* reading makes a
#: claim -- "Mexican subtitlers address this line as usted" -- and one line is not evidence
#: for it: at the corpus's rate of machine-translated documents a singleton is as likely to
#: be that as to be usage. Measured on the AFI quotes, the threshold is what separates a
#: real second reading from a bad neighbour: `I'll be back` picked up `usted` from a single
#: `Regresé.`, and `Rosebud` picked up `tú` from `¡Maldita seas, Rose!`. Both are noise and
#: both disappear at 3. The unmarked reading needs only one, because it claims nothing.
MIN_EVIDENCE_UNMARKED = 1
MIN_EVIDENCE_MARKED = 3

#: The order readings are presented in: the unmarked reading first when it exists, then
#: singular before plural, familiar before formal. Stable, so a table stays comparable.
ORDER = (Address.UNMARKED, Address.TU, Address.USTED, Address.USTEDES)

#: English words that put a listener in the line.
_SECOND_PERSON = re.compile(
    r"\b(you|your|yours|yourself|yourselves|y'all|ya'll)\b", re.IGNORECASE)

#: Base-form verbs that begin an imperative. Derived 2026-09-07 from `mx_corpus`: the word
#: following `Don't` at the head of a line is a base-form verb by construction, so 5,593
#: negative imperatives yield the vocabulary of the positive ones for free. Pronouns and
#: adverbs that the frame also admits (`you`, `just`, `even`, `ever`) are removed.
_IMPERATIVE_VERBS: frozenset[str] = frozenset("""
worry be let tell get move do forget say touch go look make talk leave take call play
think cry give know come try start ask listen shoot bother mess lie put lose kill mention
laugh mind run act speak stop pretend blame waste hurt push miss feel open want believe
turn screw thank fight pay judge bring pull eat show wait shut hold keep sit stand watch
follow help remember hurry drop stay send buy read write walk drive close answer
""".split())


def addresses_listener(cue: str) -> bool:
    """Does this English line speak to somebody?

    Only lines that do can have a tú/usted/ustedes reading, and this is the bound that
    keeps the variants honest. Without it the grouping offered a second reading for
    `Frankly, my dear, I don't give a damn` — a line with no addressee at all — because
    some neighbours happened to contain `te`. The two outputs then differed only in word
    choice while being labelled as different forms of address, which is a claim the corpus
    never made.
    """
    if _SECOND_PERSON.search(cue):
        return True
    first = re.match(r"[^A-Za-z]*([A-Za-z']+)", cue)
    return bool(first and first.group(1).lower() in _IMPERATIVE_VERBS)


@dataclass(frozen=True)
class VariantEvidence:
    """The precedent supporting one reading of the cue.

    `shared` is precedent whose Spanish does not reveal a form of address. Spanish drops
    the subject pronoun, so this is most of the corpus, and it is real evidence about
    *wording* for every reading -- it just says nothing about *grammar*. Keeping it in its
    own field is what lets the prompt say "these eight all address the listener as usted"
    without that being a lie about the other twenty.
    """

    form: Address
    precedents: tuple = ()      # tuple[Precedent, ...]
    shared: tuple = ()          # tuple[Precedent, ...]; form-neutral wording evidence

    @property
    def pair_ids(self) -> tuple[int, ...]:
        return tuple(p.pair_id for p in self.precedents)


@dataclass
class Variant:
    """One reading, translated."""

    form: Address
    spanish: str = ""
    pair_ids: tuple[int, ...] = ()
    register: str = ""
    not_mexican: list[str] = field(default_factory=list)
    #: Whether the output actually uses the form it was asked for. Checked with the same
    #: tagger that grouped the evidence, so the claim in the table is verified, not stated.
    form_confirmed: bool = False
    #: False when nothing in the corpus came close enough to this cue. The line is still
    #: translated -- and must be marked, or a guess is indistinguishable from a citation.
    grounded: bool = True


def _looks_untranslated(english: str, spanish: str) -> bool:
    """True when the Spanish side is really the English line, code-switched or copied."""
    def words(text: str) -> set[str]:
        return set(re.findall(r"[a-z']+", text.lower()))
    en, es = words(english), words(spanish)
    return bool(en) and len(en & es) / len(en) >= 0.6


def group_by_address(cue: str, *, pool: int = 400, depth: int = 300, tag_forms=None) -> list[VariantEvidence]:
    """Split retrieved precedent into the readings it attests.

    `depth` is how many embeddings the vector search itself examines, and it is the number
    that matters. At the default 40 the AFI 100 produced a second reading for **zero** of
    the hundred quotes: usted and ustedes are rare in the corpus, so the forty nearest
    neighbours of any line are almost all tú, and the rarer readings never appear to be
    grouped. At 300 the same quotes surface them -- `You talkin' to me?` goes from
    `tú 19` to `tú 111, usted 9, ustedes 2` -- and it costs nothing measurable (0.05s),
    because the work is a brute-force scan that was already paid for. Above ~300 the
    similarity floor binds and nothing further arrives.

    `pool` then caps how many of those survive to be grouped, and is deliberately loose.
    """
    from .retrieval import find_precedent

    addressed = addresses_listener(cue)

    candidates = [p for p in find_precedent(cue, limit=pool, neighbours=depth)
                  if p.similarity >= MIN_SIMILARITY
                  and not _looks_untranslated(p.english, p.spanish)]
    # `tag_forms` labels every candidate at once. The default is the morphological reader
    # in `address.py`; `address_llm.llm_tagger(ask)` is the other arm of that comparison.
    # It is injected rather than imported so this function does not decide which tagger the
    # project uses -- the measurement does.
    forms = (tag_forms([p.spanish for p in candidates]) if tag_forms
             else [read_address(p.spanish).form for p in candidates])

    grouped: dict[Address, list] = {}
    for precedent, form in zip(candidates, forms):
        if not addressed and form is not Address.UNMARKED:
            # The line has no listener, so a marked precedent is a coincidence of its own
            # wording. Fold it into the unmarked reading rather than offering it as one.
            form = Address.UNMARKED
        grouped.setdefault(form, []).append(precedent)

    neutral = tuple(grouped.get(Address.UNMARKED, ()))

    if not addressed:
        # Nobody is being spoken to, so there is one reading and it is not a choice.
        # An empty `neutral` is still returned as a reading: six of the AFI quotes are long
        # or singular enough that nothing in the corpus comes within the similarity floor
        # ("Soylent Green is people!", "Open the pod bay doors, HAL"), and dropping them
        # would leave blank rows in the review table. They are translated ungrounded and
        # marked as such, which is the same contract v1 has always had for a cue with no
        # precedent -- the failure is worth showing, not hiding.
        return [VariantEvidence(form=Address.UNMARKED, precedents=neutral)]

    marked = [
        VariantEvidence(form=f, precedents=tuple(grouped[f]), shared=neutral)
        for f in ORDER
        if f not in (Address.UNMARKED, Address.VOSOTROS)   # Spain's forms are contamination
        and len(grouped.get(f, ())) >= MIN_EVIDENCE_MARKED
    ]
    if marked:
        return marked
    # The cue speaks to someone but no form is attested often enough to claim. Offering
    # the neutral reading alone is the honest fallback -- and it must NOT be told to avoid
    # second-person forms, which is what left `Show me the money!` untranslated: the line
    # is an imperative, so "use no second person" and "translate this" contradict, and the
    # model resolved the contradiction by returning the English.
    return [VariantEvidence(form=Address.UNMARKED, precedents=neutral)]


_ASKED = {
    Address.TU: ("tú", "one person, informally — a friend, family, someone your own age"),
    Address.USTED: ("usted", "one person, formally — a stranger, an elder, someone senior"),
    Address.USTEDES: ("ustedes", "two or more people"),
    Address.UNMARKED: ("", "the corpus does not attest one form over another for this "
                          "line — render it the way the precedent does, without going out "
                          "of your way to state or avoid a pronoun"),
}

VARIANT_INSTRUCTION = """\
You localise English film subtitles into MEXICAN Spanish.

English `you` is tú, usted and ustedes at once. This line is being rendered for ONE of
them, and the evidence below is only the evidence for that one. Do not hedge across the
others.

THIS RENDERING ADDRESSES: {who}

Below are lines from a corpus of 718,925 lines written by Mexican subtitlers, all of which
address their listener the same way. They show both the wording and the grammar to use.

Rules:
- Output ONLY the Spanish line. No quotes, no explanation, no alternatives.
- Subtitle length. If the English is short, the Spanish is short.
- {grammar}
- Take wording from the evidence. Do not add slang it does not show — reaching for `güey`
  or `órale` with nothing behind it is the failure this system exists to prevent.
- Never use `vosotros`/`vuestro` or their verb forms.
- Preserve names, numbers and proper nouns exactly.
"""

_GRAMMAR = {
    Address.TU: "Use tú forms. State the pronoun only if the evidence does; Spanish drops it.",
    Address.USTED: "Use usted forms (third-person verbs). State `usted` only if the "
                   "evidence does.",
    Address.USTEDES: "Use ustedes forms (third-person plural verbs). Do NOT force the "
                     "pronoun: only 21.3% of plural-you lines in this corpus state it.",
    Address.UNMARKED: "Follow the evidence on grammar. Spanish drops the subject pronoun; "
                      "do not insert one, and do not avoid second-person verbs if the line "
                      "is a command.",
}


def format_variant_prompt(cue: str, evidence: VariantEvidence, *,
                          limit: int = 8, phrases: Sequence = ()) -> str:
    """The prompt for one reading: its own evidence and nothing else."""
    who, description = _ASKED[evidence.form]
    header = VARIANT_INSTRUCTION.format(
        who=f"{who} — {description}" if who else description,
        grammar=_GRAMMAR[evidence.form],
    )
    lines = [header, f"\nENGLISH CUE:\n{cue}\n"]
    if phrases:
        # The phrase channel is form-neutral: a rendering of `"the truth"` is the same
        # whether the scene is tú or usted. It is fetched once and shown to every reading.
        lines.append("ATTESTED PHRASES (exact wording, from the corpus):")
        for hit in phrases:
            lines.append(f'  "{hit.phrase}" — in {hit.support} lines')
            for r in hit.renderings:
                lines.append(f"      {r.count:>3}x  {r.spanish}")
                lines.append(f"           from: {r.english}   (pair_id {r.pair_id})")
        lines.append("")
    def render(items, header: str) -> None:
        if not items:
            return
        lines.append(header)
        for precedent in items:
            agree = (f"{precedent.times}/{precedent.english_times}"
                     if precedent.english_times > 1 else "1")
            lines.append(f"  [{precedent.similarity:.2f}] {precedent.english}")
            lines.append(f"          -> {precedent.spanish}   "
                         f"({agree} translators, pair_id {precedent.pair_id})")
        lines.append("")

    render(evidence.precedents[:limit],
           "ATTESTED PRECEDENT — every line below addresses its listener this same way:")
    render(evidence.shared[:limit],
           "WORDING PRECEDENT — same meaning, but the Spanish does not state a form of\n"
           "address. Take vocabulary from these, not grammar:")
    lines.append("MEXICAN SPANISH:")
    return "\n".join(lines)


def translate_variants(cue: str, *, ask, limit: int = 8, tag_forms=None,
                       model: str | None = None) -> list[Variant]:
    """Every attested reading of `cue`, translated. One model call per reading.

    `ask` takes a prompt and returns text, so the caller owns the model configuration and
    this is testable without a key.

    Grouping uses the model tagger with its ClickHouse cache, decided by measurement on
    2026-09-07: against the morphological reader it agreed on 65.8% of 585 retrieved lines,
    and of the 200 disagreements **198 were the rule missing a form the model read
    correctly** and 2 were the rule being wrong outright. Nothing went the other way. The
    misses were systematic rather than incidental -- enclitics beyond `-te`/`-se`
    (`Tráeme`, `Hazlo`), unaccented bare imperatives (`Dale`, `Entra`), and any `ustedes`
    command that did not open the line (`Buenas noches, pasen`). Spanish marks the person
    morphologically only about half the time; the rest is carried by sense, which a rule
    cannot read. The cache is what makes it affordable: the label belongs to the line, not
    to the query, so it is paid for once.
    """
    out: list[Variant] = []
    phrases = gather_phrases(cue)
    if tag_forms is None:
        from .address_llm import cached_tagger
        from .config import settings
        tag_forms = cached_tagger(ask, model=model or settings().gemini_model)
    for evidence in group_by_address(cue, tag_forms=tag_forms):
        spanish = (ask(format_variant_prompt(cue, evidence, limit=limit,
                                             phrases=phrases)) or "").strip()
        spanish = spanish.strip('"').splitlines()[0].strip() if spanish else ""
        report = check_register(spanish)
        grounded = bool(evidence.precedents or evidence.shared)
        reading = read_address(spanish)
        out.append(Variant(
            form=evidence.form,
            spanish=spanish,
            pair_ids=evidence.pair_ids[:limit],
            register=report.summary,
            not_mexican=list(report.not_mexican),
            # An unmarked reading is confirmed by *not* addressing anyone; a marked one by
            # using the form asked for. A marked cue coming back unmarked is not a failure:
            # Spanish drops the pronoun, so many correct lines state nothing.
            form_confirmed=(reading.form is evidence.form
                            or reading.form is Address.UNMARKED),
            grounded=grounded,
        ))
    return out
