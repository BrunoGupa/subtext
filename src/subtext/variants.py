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

import json
import re

from .address import Address
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

#: How much of the retrieved precedent has to address *somebody* before this cue is
#: treated as addressing somebody. Not a rule about English -- a measurement of what the
#: corpus does with lines like this one.
#:
#: The first version of this asked the question of the English instead: does the cue
#: contain `you`, or begin with a verb from a list of imperatives derived from the corpus's
#: `Don't X` frame? That list was closed and the input is not. It had neither `have` nor
#: `calm`, so `Have a seat.` and `Calm down.` -- two lines the corpus renders in all three
#: forms -- were judged to address nobody, and a judge typing `Buckle up.` would have hit
#: the same wall. No amount of adding verbs fixes the shape of that mistake.
#:
#: The Spanish side already answers it, in labels we are paying for anyway. Lines close in
#: meaning to `Have a seat.` come back as `Siéntate` / `Siéntese` / `Siéntense`; lines close
#: to `Frankly, my dear, I don't give a damn` come back unmarked. So the signal is the
#: *proportion* of retrieved precedent that states a form, and it works for any input in
#: any wording.
ADDRESSED_SHARE = 0.35

_CHECK_LABELS = {"tu": Address.TU, "tú": Address.TU, "usted": Address.USTED,
                 "ustedes": Address.USTEDES, "vosotros": Address.VOSOTROS,
                 "none": Address.UNMARKED, "": Address.UNMARKED}

#: The model's way of saying a reading does not apply to this line.
REFUSAL = "NONE"


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
    #: False when the grammar check still failed after the retry. The line is returned
    #: anyway, flagged, rather than dropped: a reviewer needs to see what came out.
    well_formed: bool = True


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

    candidates = [p for p in find_precedent(cue, limit=pool, neighbours=depth)
                  if p.similarity >= MIN_SIMILARITY
                  and not _looks_untranslated(p.english, p.spanish)]
    # `tag_forms` labels every candidate at once. The default is the morphological reader
    # in `address.py`; `address_llm.llm_tagger(ask)` is the other arm of that comparison.
    # It is injected rather than imported so this function does not decide which tagger the
    # project uses -- the measurement does.
    if tag_forms is None:
        raise ValueError(
            "group_by_address needs a tagger. The morphological one was removed on "
            "2026-09-07 after losing to the model 198-2; use "
            "`address_llm.cached_tagger(ask, model=...)`."
        )
    forms = tag_forms([p.spanish for p in candidates])

    marked = sum(1 for f in forms if f not in (Address.UNMARKED, Address.VOSOTROS))
    addressed = bool(forms) and marked / len(forms) >= ADDRESSED_SHARE

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

CHECK_INSTRUCTION = """\
Check one Spanish subtitle line. Answer ONLY with JSON, no prose:

{"addresses": "tu" | "usted" | "ustedes" | "none",
 "well_formed": true | false,
 "fix": "<the corrected line, or an empty string if well_formed is true>"}

`addresses` — how the line addresses its listener, or "none" if it does not say.
`well_formed` — is every word real, correctly conjugated Mexican Spanish? Judge the
  grammar only. Do NOT judge word choice, register, punctuation or style, and do not
  rewrite a line that is merely different from what you would have written.

LINE: {line}
"""

VARIANT_INSTRUCTION = """\
You localise English film subtitles into MEXICAN Spanish.

English `you` is tú, usted and ustedes at once. This line is being rendered for ONE of
them, and the evidence below is only the evidence for that one. Do not hedge across the
others.

THIS RENDERING ADDRESSES: {who}

Below are lines from a corpus of 718,925 lines written by Mexican subtitlers, all of which
address their listener the same way. They show both the wording and the grammar to use.

If this form of address is WRONG for this line, output exactly NONE and nothing else.
The line itself can rule it out: `Here's looking at you, kid` cannot be usted, because
nobody addresses a child as usted; a line calling the listener `sir` cannot be tú. Refusing
is correct and costs nothing. Producing a reading the line forbids is the worse error --
it was `Brindo por usted, niña.`, which is wrong Spanish manners, not wrong grammar.

Rules:
- Output ONLY the Spanish line, or NONE. No quotes, no explanation, no alternatives.
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


def check_output(line: str, *, ask) -> tuple[Address, bool, str]:
    """Two narrow questions about one output line: which form, and is it real Spanish.

    Deliberately not "is this a good translation?" -- a model grading a model on quality is
    the pattern this project has no way to trust, and the register gate exists precisely so
    that judgement is made by counting against the corpus. These two questions have
    checkable answers, and Python decides what to do with them.

    The grammar question earns its call. The register gate counts words against lexicons,
    so `Adelante, alégranme el día.` passed it clean: every word looked Mexican and the
    conjugation was invented. Nothing else in the pipeline reads morphology any more, and
    the corpus cannot fill in either -- `alégrame`, `cuídese` and `cállense` occur **zero**
    times in 718,925 lines and are all perfectly good Spanish, so "not attested" is not
    evidence of "not a word".
    """
    reply = ask(CHECK_INSTRUCTION.replace("{line}", line)) or ""
    match = re.search(r"\{.*\}", reply, re.DOTALL)
    if not match:
        return Address.UNMARKED, True, ""      # unreadable check never fails a line
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return Address.UNMARKED, True, ""
    if not isinstance(data, dict):
        return Address.UNMARKED, True, ""
    form = _CHECK_LABELS.get(str(data.get("addresses", "")).strip().lower(),
                             Address.UNMARKED)
    return form, bool(data.get("well_formed", True)), str(data.get("fix", "")).strip()


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
        prompt = format_variant_prompt(cue, evidence, limit=limit, phrases=phrases)
        spanish = _first_line(ask(prompt))

        # The model was given the right to refuse a reading the line rules out -- usted to
        # somebody the line calls `kid`. A refused reading is not offered at all, which is
        # the point: `Brindo por usted, niña.` was not bad grammar, it was a reading that
        # should never have been generated.
        if spanish.upper().rstrip(".!") == REFUSAL:
            continue

        form, well_formed, fix = check_output(spanish, ask=ask)
        wrong_form = (evidence.form is not Address.UNMARKED
                      and form is not Address.UNMARKED
                      and form is not evidence.form)
        # The register check has to gate the retry, not merely be recorded next to the
        # output. v1's `RegisterGate` sent a line back on this and the variants path
        # dropped it, so `Jesus fucking Christ` -> `La concha de Dios.` was reported as a
        # register failure and returned anyway.
        report = check_register(spanish)

        # One retry, and **Python decides whether to take it** -- from two yes/no answers,
        # never from the model judging its own work. The retry says exactly what was wrong
        # and forbids anything else changing, because a free rewrite tends to drift off the
        # evidence, which is the failure this system exists to prevent.
        if wrong_form or not well_formed or not report.ok:
            note = []
            if not report.ok:
                wrong_country = list(report.not_mexican) + list(report.other_latam)
                note.append(f"It used forms Mexican subtitlers do not write: "
                            f"{', '.join(wrong_country)}. Rewrite it in Mexican Spanish.")
            if wrong_form:
                note.append(f"It addressed the listener as {form.value}, but this rendering "
                            f"must address them as {evidence.form.value}.")
            if not well_formed:
                note.append("It is not well-formed Mexican Spanish"
                            + (f"; the conjugation should be: {fix}" if fix else "."))
            retry = (f"{prompt}\n\nYour previous attempt was rejected. "
                     f"{' '.join(note)}\nKeep the same meaning, wording and length; change "
                     f"nothing else.\nPrevious attempt: {spanish}\n")
            candidate = _first_line(ask(retry))
            if candidate and candidate.upper().rstrip(".!") != REFUSAL:
                spanish = candidate
                form, well_formed, _ = check_output(spanish, ask=ask)
                report = check_register(spanish)

        out.append(Variant(
            form=evidence.form,
            spanish=spanish,
            pair_ids=evidence.pair_ids[:limit],
            register=report.summary,
            not_mexican=list(report.not_mexican) + list(report.other_latam),
            # A marked reading is confirmed by using the form asked for. Coming back
            # UNMARKED is not a failure: Spanish drops the subject pronoun, so plenty of
            # correct lines state nothing -- only stating the *wrong* one is.
            form_confirmed=(form is evidence.form or form is Address.UNMARKED),
            well_formed=well_formed,
            grounded=bool(evidence.precedents or evidence.shared),
        ))
    return out


def _first_line(reply: str | None) -> str:
    """The model's answer, reduced to the one line it was asked for."""
    text = (reply or "").strip().strip('"')
    return text.splitlines()[0].strip() if text else ""
