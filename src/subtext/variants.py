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

Readings with no attested precedent are not offered. If nothing in the corpus addresses
this cue as `ustedes`, that variant does not exist and is not manufactured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import json
import re

from .address import Address
from .guard import FENCE_NOTE, fence, looks_like_leak
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

#: What an asker returns when the model produced no text at all, followed by the reason it
#: gave. A blocked answer and a refused reading are different events and the difference is
#: the one a reader needs: a refusal is this pipeline working -- it declines a reading the
#: line rules out, which is why `Brindo por usted, niña.` does not happen -- while a block
#: is the provider declining to answer, and only one of the two is honest to describe as a
#: policy. Left undistinguished they both arrive as an empty string and the row silently
#: shows nothing.
BLOCKED = "BLOCKED>"

#: Below this, the nearest thing the corpus offered is not close enough to have taught the
#: model anything about the cue, and `grounded` becomes a claim the evidence cannot support.
#:
#: `Helps fellatio.` is the case that produced it. Every neighbour scored 0.47-0.55 and all
#: of them matched on *helps*: `It helps.` -> `Eso ayuda.`, `It gives you the squirts.` ->
#: `Te dan diarrea.` Nothing about the difficult half of the line came back at all, so the
#: model wrote `felación` unaided -- a word occurring **zero** times in 335,800 lines of
#: Mexican subtitling, where `mamada` occurs 206 times at 21x enrichment.
#:
#: The line was still reported as grounded, because `grounded` only asked whether anything
#: came back. It now also asks whether anything came back *close*.
#: Re-measured 2026-09-08 against ten cues, six with a rendering in the corpus and four
#: with none. The two bands do not overlap -- grounded 0.939 to 1.000, ungrounded 0.829 to
#: 0.889 -- and 0.91 sits in the gap with margin on both sides. Left at MiniLM's 0.65 the
#: mark would never have fired again: `Helps fellatio.` now scores 0.884.
WEAK_SIMILARITY = 0.91


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
    #: False when the model returned no text for this reading. `block_reason` carries what
    #: it said about why -- a safety category, a token limit, or nothing at all.
    answered: bool = True
    block_reason: str = ""
    #: The closest precedent behind this reading. Below `WEAK_SIMILARITY` the evidence is
    #: too far away to have grounded anything, whatever `grounded` says.
    top_similarity: float = 0.0
    #: An agreed rendering stood behind this line. Recorded because `top_similarity` reads
    #: the neighbour channel alone and therefore cannot see the stronger evidence: `up his
    #: ass` has six corpus lines agreeing on `por el culo` and a nearest neighbour of 0.56,
    #: so without this it is flagged exactly like `Helps fellatio.`, which has nothing at
    #: all. A warning that fires on the grounded case and the unfounded one alike tells a
    #: reviewer nothing.
    agreed: tuple[str, ...] = ()
    #: The gender this line's agreement is in, and the same line in the other one. Both are
    #: None when the line marks nobody's gender, which is most of them -- and that absence
    #: is the signal the page needs: no gender marked, no choice to offer.
    gender: "Gender | None" = None
    other_gender: str = ""
    #: Attested renderings of this cue, in this form of address, that were not chosen.
    #: They are quotations, so they are shown as their translator wrote them: regendering
    #: one to match a reader's preference would keep the `pair_id` while changing the line
    #: it points at, which is the one thing this project must not do.
    alternatives: tuple = ()

    @property
    def weakly_grounded(self) -> bool:
        """Precedent came back, but none of it close. Worse than none, because it reads
        as support while the model was in fact writing unaided."""
        if self.agreed:
            return False
        return self.grounded and self.top_similarity < WEAK_SIMILARITY


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

Below are lines from a corpus of 335,800 lines written by Mexican subtitlers, all of which
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
- An AGREED RENDERING is measured across every corpus line carrying that phrase, so it is
  the phrase's own Spanish. Use its wording, and do not soften it: where the corpus agrees
  on a coarse word, a politer synonym is a word the corpus does not support. But it ranks
  BELOW a precedent that renders your line itself: where the evidence shows your exact line
  already translated, that translation wins and the agreed rendering is only a check on it.
  And it is a claim about WORDING, never about grammar — it carries no pronoun and no form
  of address, so never insert a subject pronoun to accommodate it.
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
    lines = [header, "", FENCE_NOTE, f"\nENGLISH CUE:\n{fence(cue)}\n"]
    if phrases:
        # The phrase channel is form-neutral: a rendering of `"the truth"` is the same
        # whether the scene is tú or usted. It is fetched once and shown to every reading.
        lines.append("ATTESTED PHRASES (exact wording, from the corpus):")
        for hit in phrases:
            lines.append(f'  {fence(hit.phrase)} — in {hit.support} lines')
            # The agreed rendering is the only evidence here that is about the PHRASE
            # rather than about a line that happened to contain it, so it leads. Leaving
            # it out of this prompt while `--evidence-only` printed it is what let
            # `up his ass` reach the model with three phrase headings and no Spanish
            # under any of them.
            for c in hit.consensus:
                lines.append(f"      AGREED RENDERING: {fence(c.spanish)}   "
                             f"({c.lines} of {c.of_lines} lines carrying this phrase, "
                             f"{c.enrichment:.0f}x the corpus rate) — use this wording")
            for r in hit.renderings:
                lines.append(f"      {r.count:>3}x  {fence(r.spanish)}")
                lines.append(f"           from: {fence(r.english)}   (pair_id {r.pair_id})")
        lines.append("")
    def render(items, header: str) -> None:
        if not items:
            return
        lines.append(header)
        for precedent in items:
            agree = (f"{precedent.times}/{precedent.english_times}"
                     if precedent.english_times > 1 else "1")
            lines.append(f"  [{precedent.similarity:.2f}] {fence(precedent.english)}")
            lines.append(f"          -> {fence(precedent.spanish)}   "
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
    times in the corpus and are all perfectly good Spanish, so "not attested" is not
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


#: An attested rendering has to be this close to the cue before it is offered as an
#: alternative. Below it the line is precedent for the register and nothing more -- it
#: answers a different sentence, and putting it under the chosen line as a second option
#: would invite a reader to pick it.
#:
#: 0.80 under MiniLM, 0.95 here, and again read rather than counted: at 0.95 `Get in the
#: car.` offers `Get in the car already.` -> `Ay, ya, sube, por favor.` and `- Get into
#: the car!` -> `Súbete al coche, Rebeca.`, while 0.93 would add `Take the car.` ->
#: `Llévate el carro.`, which is a different sentence.
ALTERNATIVE_SIMILARITY = 0.95

#: How many alternatives are shown. Two, because the point is that the choice is real, not
#: to hand a reviewer a list to grade.
ALTERNATIVES = 2


#: The other axis English leaves open. `you` marks no gender, so `I don't wanna kill you`
#: is `matarlo` or `matarla` and the source cannot settle it; neither can the corpus, which
#: is why this is a preference the reader sets rather than something retrieved. It is read
#: the same way the person is: by declining the decided line and seeing whether anything
#: moves. A line that marks no gender comes back NONE, and then there is no choice to offer.
class Gender(str, Enum):
    MASCULINE = "masculino"
    FEMININE = "femenino"

    @property
    def other(self) -> "Gender":
        return Gender.FEMININE if self is Gender.MASCULINE else Gender.MASCULINE


REGENDER_INSTRUCTION = """\
You are given one Mexican Spanish subtitle line that is already decided, and one job:
flip the gender of the PERSON it agrees with. This is a change of AGREEMENT, not of wording.

THE ENGLISH IT COMES FROM:
{cue}

THE SPANISH LINE, as written:
{spanish}

Rules:
- Keep every word the change of gender does not force you to touch. Same verbs, same nouns,
  same profanity, same punctuation, same order, same length.
- Change only what agreement forces: adjective and participle endings, articles and object
  pronouns that refer to the person, `-o`/`-a` on words that describe them.
- Do NOT change the gender of a thing. `el coche` stays `el coche`; only the gender of a
  PERSON — whoever is spoken to, or whoever is speaking — moves.
- If the line marks nobody's gender, there is nothing to flip. Output NONE.
- If the ENGLISH already settles that person's gender, there is no choice to offer and you
  must output NONE. `he`, `she`, `him`, `her`, `guy`, `bitch`, `sir`, `ma'am`, `brother`
  and a person's name all settle it. Only flip what English left open — `you`, `I`, `we`,
  `they` and anyone the English does not gender.
- Otherwise output `M>` if the line AS GIVEN is masculine, or `F>` if it is feminine,
  and then the flipped line. Nothing else — no quotes, no explanation.

Examples of the shape, not of the wording:
    You're so tired.        / Estás muy cansado.   ->  M> Estás muy cansada.
    I don't wanna kill you  / No lo quiero matar.  ->  M> No la quiero matar.
    Get in the car.         / Súbete al coche.     ->  NONE   (marks nobody)
    Tell that bitch to be cool / Dile a esa perra… ->  NONE   (English said `bitch`)

ANSWER:
"""


DECLINE_INSTRUCTION = """\
You are given one Mexican Spanish subtitle line that is already decided, and one job:
put it into a different form of address. This is a change of GRAMMAR, not of wording.

THE LINE, as written:
{spanish}

IT MUST NOW ADDRESS: {who}
{grammar}

Rules:
- Keep every word that the change of address does not force you to touch. Same verbs, same
  nouns, same profanity, same punctuation, same order, same length. If a word can stay, it
  stays.
- Change only what the form forces: pronouns, clitics, possessives, and the verb endings
  that carry the person.
- If the line addresses nobody — an exclamation, a statement about the speaker — there is
  nothing to decline. Output NONE.
- Never use `vosotros`/`vuestro` or their verb forms.
- Output ONLY the Spanish line, or NONE. No quotes, no explanation.

MEXICAN SPANISH:
"""


def _regender(spanish: str, cue: str, *, ask) -> tuple["Gender | None", str]:
    """The gender this line agrees with, and the same line in the other one.

    One call, because the two facts arrive together: asked separately, a line already in
    the feminine and a line with no gender at all both answer "unchanged" to "make it
    feminine", and telling them apart cost a second question on every genderless line --
    which is most of them. The `M>` / `F>` prefix carries the direction instead.
    """
    reply = _first_line(ask(REGENDER_INSTRUCTION.format(
        spanish=fence(spanish), cue=fence(cue))))
    if not reply or reply.upper().rstrip(".!") == REFUSAL:
        return None, ""
    marker, _, flipped = reply.partition(">")
    flipped = flipped.strip()
    was = {"M": Gender.MASCULINE, "F": Gender.FEMININE}.get(marker.strip().upper())
    if not was or not flipped or flipped == spanish:
        return None, ""
    return was, flipped


def alternatives_for(evidence, chosen: str, *, limit: int = ALTERNATIVES) -> tuple:
    """The attested renderings this reading did not use, best first.

    Ranked by how close the English is and then by how many translators agreed, which is
    the same order `find_precedent` defends: a misaligned row is almost always a lone
    reading of a line several others agree on.
    """
    seen = {_key(chosen)}
    out = []
    for p in sorted(tuple(evidence.precedents) + tuple(evidence.shared),
                    key=lambda p: (-p.similarity, -p.consensus, -p.times)):
        if p.similarity < ALTERNATIVE_SIMILARITY or _key(p.spanish) in seen:
            continue
        seen.add(_key(p.spanish))
        out.append(p)
        if len(out) == limit:
            break
    return tuple(out)


def _key(text: str) -> str:
    """Two renderings that differ only in punctuation or a leading dash are one option."""
    return re.sub(r"[^a-záéíóúüñ ]", "", text.lower()).strip()


def translate_variants(cue: str, *, ask, limit: int = 8, tag_forms=None,
                       model: str | None = None, phrases: Sequence | None = None) -> list[Variant]:
    """Every attested reading of `cue`, translated. One model call per reading.

    `ask` takes a prompt and returns text, so the caller owns the model configuration and
    this is testable without a key.

    `phrases` is the output of `gather_phrases`, accepted rather than always fetched
    because a caller that shows the evidence beside the readings needs the same object and
    would otherwise ask for it twice. It is ten queries -- measured at 4.5 s against
    ClickHouse Cloud, a sixth of a whole request -- and the web endpoint was paying it
    twice for one line. Left at None it is fetched here, so the CLI and the tests are
    unaffected.

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
    if phrases is None:
        phrases = gather_phrases(cue)
    # Form-neutral, like the phrase channel it comes from: what `up his ass` agrees on is
    # the same whether the scene is tu or usted.
    # De-duplicated: two phrases of the same cue can agree on the same word, and
    # `Get in the car.` reported `coche, coche` because `the car` and `in the car` both did.
    agreed = tuple(dict.fromkeys(c.spanish for hit in phrases for c in hit.consensus))
    if tag_forms is None:
        from .address_llm import cached_tagger
        from .config import settings
        tag_forms = cached_tagger(ask, model=model or settings().gemini_model)

    # The readings used to be N independent translations, one per form, each shown only
    # the precedent that addresses its own listener. That is not one line in three forms,
    # it is three unrelated lines: `Oh, fuck me!` gave tú 117 precedents headed by
    # `Fuck me!` -> `¡Cógeme!` at 0.84, and usted **three**, none of them about fuck me,
    # so usted came back `¡La puta madre!`. Switching the form rewrote the sentence.
    #
    # So the wording is decided once, on the reading the corpus attests best, and every
    # other form is that same line declined. One call per reading either way.
    readings = group_by_address(cue, tag_forms=tag_forms)
    base = max(readings, key=lambda e: (len(e.precedents),
                                        max((p.similarity for p in e.precedents), default=0.0)))
    decided = ""
    order = sorted(readings, key=lambda e: e is not base)

    def render(evidence, prompt: str) -> "Variant | None":
        """One reading, start to finish: ask, gate, retry if Python says so, gender it.

        Pulled out of the loop so the readings that do not decide the wording can run at
        the same time as each other. Nothing in here touches anything shared: it reads its
        own evidence and the already-decided string, and returns one `Variant`. `None`
        means the model refused this reading, which is not a failure -- see REFUSAL.
        """
        reply = ask(prompt)

        # No text came back. That is not a refusal and must not be reported as one: the
        # reading is offered as unanswered, with whatever reason the provider gave, so a
        # blank row can be explained instead of guessed at.
        if not reply.strip() or reply.startswith(BLOCKED):
            return Variant(
                form=evidence.form, spanish="", answered=False,
                block_reason=reply[len(BLOCKED):].strip() or "no reason given",
                pair_ids=evidence.pair_ids[:limit], agreed=agreed,
                grounded=bool(evidence.precedents or evidence.shared or agreed),
            )

        spanish = _first_line(reply)
        # The gates below read register, form and grammar. None of them can see the one
        # thing an injection produces: an answer that is the prompt talking rather than a
        # subtitle. That is checked here, before anything downstream trusts the string.
        if looks_like_leak(spanish, cue=cue):
            return Variant(
                form=evidence.form, spanish="", answered=False,
                block_reason="the answer did not look like a subtitle line",
                pair_ids=evidence.pair_ids[:limit], agreed=agreed,
                grounded=bool(evidence.precedents or evidence.shared or agreed),
            )

        # The model was given the right to refuse a reading the line rules out -- usted to
        # somebody the line calls `kid`. A refused reading is not offered at all, which is
        # the point: `Brindo por usted, niña.` was not bad grammar, it was a reading that
        # should never have been generated.
        if spanish.upper().rstrip(".!") == REFUSAL:
            return None

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

        # Asking for the other gender is also how we find out whether there is one: a line
        # that marks nobody's gender has nothing to change and comes back NONE. Most lines
        # are that, so the question is asked once on the decided wording and only pursued
        # for the remaining readings when the first answer says there is something to move.
        # Asked per reading and not once per cue: gender can appear in the declension even
        # when the decided wording has none. `No te quiero matar.` marks nobody -- `te` is
        # the same for either -- while its usted reading has to choose `lo` or `la`.
        gender, other = _regender(spanish, cue, ask=ask) if spanish else (None, "")

        return Variant(
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
            top_similarity=max((p.similarity for p in
                                tuple(evidence.precedents) + tuple(evidence.shared)),
                               default=0.0),
            grounded=bool(evidence.precedents or evidence.shared or agreed),
            agreed=agreed,
            gender=gender,
            other_gender=other,
            alternatives=alternatives_for(evidence, spanish),
        )

    # The base decides the wording, so it goes first and alone.
    results: dict[int, "Variant | None"] = {}
    results[0] = render(base, format_variant_prompt(cue, base, limit=limit, phrases=phrases))
    first = results[0]
    if first is not None and first.answered:
        decided = first.spanish

    rest = order[1:]

    def decline_prompt(evidence) -> str:
        who, description = _ASKED[evidence.form]
        return DECLINE_INSTRUCTION.format(
            spanish=fence(decided),
            who=f"{who} — {description}" if who else description,
            grammar=_GRAMMAR[evidence.form])

    if decided and rest:
        # Every remaining reading is the decided line declined, so none of them depends on
        # another: they only read `decided`, which will not change again. Run them at once.
        # Measured on `Are you sure?`: nine Gemini calls in series were 17.7 s of a 24 s
        # request, and the readings after the base are most of them.
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=len(rest)) as pool:
            futures = {i: pool.submit(render, e, decline_prompt(e))
                       for i, e in enumerate(rest, start=1)}
        for i, future in futures.items():
            results[i] = future.result()      # exceptions surface here, never swallowed
    else:
        # The base did not answer, so nothing has been decided and the relay is still open:
        # the next reading that answers gets the full prompt and becomes the decider. That
        # is a chain, and a chain cannot be run in parallel without turning these back into
        # independent translations -- which is the failure `DECLINE_INSTRUCTION` exists to
        # prevent. Rare path, kept sequential and identical to what it always was.
        for i, evidence in enumerate(rest, start=1):
            prompt = (decline_prompt(evidence) if decided
                      else format_variant_prompt(cue, evidence, limit=limit, phrases=phrases))
            variant = render(evidence, prompt)
            results[i] = variant
            if not decided and variant is not None and variant.answered:
                decided = variant.spanish

    out = [results[i] for i in range(len(order)) if results.get(i) is not None]
    return out


def _first_line(reply: str | None) -> str:
    """The model's answer, reduced to the one line it was asked for."""
    text = (reply or "").strip().strip('"')
    return text.splitlines()[0].strip() if text else ""
