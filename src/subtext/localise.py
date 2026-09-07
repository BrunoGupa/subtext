"""Localisation: an English subtitle cue in, Mexican Spanish out, grounded in precedent.

The shape of this pipeline is a deliberate choice, and it is the opposite of the
question-answering agent next door. That agent loops, because it cannot know in advance
which tool answers "how many times does Vale break a promise?" — deciding that *is* the
work. Here nothing is decided: every cue takes the same three steps, in the same order.

    gather evidence  ->  translate  ->  check the register
       (Python)          (one call)        (Python)

So only the middle step is a model. Retrieval makes no creative decision — it tokenizes,
looks up phrases, finds neighbours — and register checking is counting words against the
same lexicons that built the corpus. Handing either to an LLM would buy latency, tokens
and non-determinism, and would let the model decide whether to bother retrieving at all.

The check can send a cue back for one retry, so a cue costs one Gemini call, or two if
the first attempt drifts into peninsular Spanish. The retry is decided by Python, from a
word count — never by the model judging its own output.

Two kinds of evidence go into the prompt, and they are labelled differently on purpose:

* **phrase precedent** — exact n-grams from `phrase_index` with occurrence counts. Strong:
  somebody really wrote this, and it can be cited by `pair_id`.
* **neighbour precedent** — lines close in meaning from the vector index. Weaker: they
  prove a Mexican translator said something *like* this, not this.

Presenting both flat would invite the model to treat a 0.6-similarity neighbour as
attested fact. They are separated in the prompt, and the instruction says which to prefer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .address import Address, read_address
from .frame import Frame
from .ingest.mexican import MEXICAN, PENINSULAR
from .tokenizer import MAX_N, clickhouse_haystack_sql, ngrams, tokens

#: A neighbour below this cosine similarity is noise dressed as evidence.
MIN_SIMILARITY = 0.55

#: Phrases seen fewer than this are too thin to quote as precedent.
MIN_PHRASE_SUPPORT = 3

#: A phrase must account for at least this much of the line it was found in before that
#: line's Spanish is worth showing. Below it the phrase is a fragment and the Spanish is
#: about something else -- this is the bound that stops `"you can't handle"` from being
#: answered with `Nada más.`
MIN_COVERAGE = 0.6

#: Same bound `find_precedent` uses: a Spanish line this far out of proportion to its
#: English is a misalignment, not a translation.
MAX_LENGTH_RATIO = 2.2


@dataclass(frozen=True)
class Rendering:
    """One Spanish rendering of a phrase, with what it is a rendering *of*.

    `english` is the whole corpus line the Spanish came from, and it is carried because
    the corpus offers no word alignment: a phrase lookup can only return the Spanish of
    the *line* that contained the phrase, never the Spanish of the phrase alone. Hiding
    that produced evidence like `"you can't handle"` -> `Nada más.`, presented to the
    model as attested fact. Showing the English line next to the Spanish makes the
    mismatch visible instead of laundering it, and `pair_id` makes it checkable.
    """

    spanish: str
    count: int
    pair_id: int
    doc_id: int
    english: str
    #: How much of `english` the phrase accounts for, 0..1. At 1.0 the line *is* the
    #: phrase and the Spanish really is the phrase's rendering; well below that, the
    #: Spanish is the whole line's and the phrase is a fragment inside it.
    coverage: float = 0.0


@dataclass(frozen=True)
class PhraseHit:
    phrase: str
    words: int
    support: int
    renderings: tuple[Rendering, ...] = ()


@dataclass(frozen=True)
class Evidence:
    """Everything retrieved for one cue. Built without a model."""

    cue: str
    phrases: tuple[PhraseHit, ...] = ()
    neighbours: tuple = ()  # tuple[Precedent, ...]; typed loosely to avoid a cycle
    #: The scene this cue sits in, when v2 supplied one. `None` is v1: no scene, no
    #: reranking, precedent ordered by similarity and agreement alone.
    frame: Frame | None = None

    @property
    def is_empty(self) -> bool:
        return not self.phrases and not self.neighbours

    @property
    def strongest_phrase(self) -> PhraseHit | None:
        """The longest phrase with real support — short ones carry no information.

        `my` occurs 35,178 times and answers "Dios mío"; `my car` occurs 106 times and
        answers with a car. Length beats frequency here, every time.
        """
        usable = [p for p in self.phrases if p.support >= MIN_PHRASE_SUPPORT and p.words >= 2]
        return max(usable, key=lambda p: (p.words, p.support)) if usable else None


#: Forms a Mexican subtitler does not write. This list is SHORT on purpose, and it is
#: derived from the corpus rather than from intuition -- see the long note below.
#:
#: Two kinds of thing qualify:
#:
#: 1. Spain's second-person-plural morphology. `vosotros`/`vuestro` is grammar, not word
#:    choice, and Mexico does not have it. The 59 lines carrying it in `mx_corpus` are
#:    demonstrably the documented contamination, not usage -- they come with Spain verb
#:    forms attached ("llegáis tarde", "¿qué esperáis?", "me hacéis falta").
#: 2. Vocabulary attested at most twice in 718,925 lines of Mexican Spanish. At that rate
#:    the occurrences are more likely to BE the 0.46% peninsular leak than evidence of
#:    Mexican usage, so rejecting them is safe in both directions.
#:
#: Measured 2026-09-04 against mx_corpus. Re-derive after any corpus rebuild with
#: `sql/peninsular_rates.sql`.
NOT_MEXICAN: tuple[str, ...] = (
    # Spain's 2nd person plural -- grammar, not vocabulary
    "vosotros", "vuestro", "vuestra", "vuestros", "vuestras",
    # 0 occurrences in 718,925 lines
    "chorrada", "currar", "curro", "flipante", "flipar", "flipas", "mogollon",
    # 1-2 occurrences
    "cabreado", "cabrear", "cutre", "fontanero", "gilipolleces", "majo", "maja",
    "molar", "mola", "pijo", "aparcar", "chavales",
    # `pajita` removed 2026-09-04 at Bruno's call: it is ordinary vocabulary rather than
    # Spain-only slang, and a single occurrence is not grounds for a blocklist entry.
)

#: Peninsular-*leaning* vocabulary that Mexicans nonetheless write. Reported, NEVER
#: rejected. Getting this distinction wrong is the single easiest way to make this system
#: worse than no system, so the reasoning is recorded here rather than in a commit message.
#:
#: The detector lexicon in `ingest/mexican.py` marks 66 words as peninsular. That list is
#: sound for what it does: score a *1,000-line window* and compare totals with a 2:1 ratio.
#: In bulk, over that much text, `coche` really is 2.6x rarer in Mexican productions.
#:
#: It is wrong as a rule about a *single line*, and the corpus says so plainly: **59 of
#: those 66 words appear in Mexican Spanish**. Only 7 never do.
#:
#:     vale   924 occurrences      tio   450      coche  373      piso  297
#:
#: Worse, two of them are more common in Mexican productions than in the corpus at large
#: (`vales` 2.03x, `cazadora` 2.11x), and `vale` is not even one word: of its 924
#: occurrences, at least 374 are the verb *valer* or the Mexican idiom -- **156 are
#: "me vale"**, as in *me vale madre*. A gate built on the detector list rejects
#: "Eso a mí me vale madre." and "Súbete al coche, güey." Both are real corpus lines.
#: Both are unmistakably Mexican. That gate would have been worse than none.
WATCH: tuple[str, ...] = tuple(
    w for w in PENINSULAR if w not in set(NOT_MEXICAN)
)


@dataclass(frozen=True)
class RegisterReport:
    """The deterministic gate. No model judges the output; measured usage does."""

    not_mexican: tuple[str, ...] = ()
    watch: tuple[str, ...] = ()
    mexican: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Only forms Mexicans genuinely do not write can fail a line."""
        return not self.not_mexican

    @property
    def summary(self) -> str:
        parts = []
        if self.not_mexican:
            parts.append(f"NOT Mexican: {', '.join(self.not_mexican)}")
        if self.mexican:
            parts.append(f"Mexican markers: {', '.join(self.mexican)}")
        if self.watch:
            parts.append(f"peninsular-leaning (allowed): {', '.join(self.watch)}")
        return " · ".join(parts) if parts else "clean, neutral Spanish"


def _padded(text: str) -> str:
    """Mirror the corpus normalisation so the gate counts what the detector counted."""
    import re
    import unicodedata

    folded = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in folded if unicodedata.category(c) != "Mn")
    return " " + re.sub(r"[^a-z0-9]+", " ", stripped).strip() + " "


def check_register(spanish: str) -> RegisterReport:
    """Which register markers does this Spanish contain, and which of them disqualify it?

    Only `NOT_MEXICAN` disqualifies. `WATCH` words are counted and shown so a human can
    see them, but they never fail a line, because Mexicans write them.
    """
    haystack = _padded(spanish)
    hit = lambda words: tuple(w for w in words if f" {w} " in haystack)
    return RegisterReport(
        not_mexican=hit(NOT_MEXICAN),
        watch=hit(WATCH),
        mexican=hit(MEXICAN),
    )


def candidate_phrases(cue: str, max_n: int = MAX_N) -> list[str]:
    """Every 2..max_n word phrase in the cue, longest first — the lookup order."""
    words = tokens(cue)
    out: list[str] = []
    for n in range(min(max_n, len(words)), 1, -1):
        out.extend(ngrams(words, n))
    return out


def _rank_key(form: Address, frame: Frame | None):
    """How well a precedent's form fits the scene: 0 agrees, 1 says nothing, 2 contradicts.

    Contradicting precedent is ranked down, never dropped. `Súbete al coche.` is a correct
    Mexican line and stays visible in a `usted` scene; it just stops being the first thing
    the model reads. Deleting it would hide the corpus behind a frame that one model call
    produced, which is a worse failure than showing it in the wrong order.
    """
    if frame is None or not frame.constrains:
        return 0
    if form is Address.UNMARKED:
        return 1
    return 0 if form is frame.address else 2


def gather_phrases(cue: str, *, renderings: int = 3) -> tuple[PhraseHit, ...]:
    """The attested-phrase half of the evidence, on its own.

    Split out because the phrase channel says nothing about who is being addressed -- a
    rendering of `"the truth"` is the same rendering whether the scene is tú or usted -- so
    every reading of a cue wants the same phrase evidence, and fetching it once per reading
    would ask the same question three times.
    """
    from .db import client

    ch = client()
    phrases = candidate_phrases(cue)
    hits: list[PhraseHit] = []
    if phrases:
        rows = ch.query(
            "SELECT ng, n, support FROM phrase_index "
            "WHERE ng IN {p:Array(String)} AND support >= {m:UInt32} "
            "ORDER BY n DESC, support DESC LIMIT 3",
            parameters={"p": phrases, "m": MIN_PHRASE_SUPPORT},
        ).result_rows
        # Longest first: a 2-word phrase is usually too generic to carry a translation.
        for ng, n, support in rows:
            # Short source lines only. In a long line the phrase is a fragment and its
            # Spanish is buried in unrelated words -- "we're going to be" pulled back a
            # sentence about a spine operation. Bounding the line keeps the rendering
            # close to the phrase itself.
            # `pair_id` and the source line come back with every rendering. Without them
            # this block asserted things it could not support: it grouped by `es` alone,
            # so nothing was citable, and it showed the Spanish of a whole line as if it
            # were the Spanish of the phrase. The length-ratio pair drops the crudest
            # misalignments the same way `find_precedent` does.
            rend = ch.query(
                f"SELECT es, count() AS c, min(pair_id) AS pid, any(doc_id) AS did, "
                f"       any(en) AS src "
                f"FROM mx_corpus "
                f"WHERE {clickhouse_haystack_sql('en')} LIKE {{pat:String}} "
                f"  AND length(en) <= {{cap:UInt32}} "
                f"  AND length(es) <= length(en) * {{ratio:Float64}} "
                f"  AND length(en) <= length(es) * {{ratio:Float64}} "
                f"GROUP BY es ORDER BY c DESC, length(es) ASC LIMIT {{k:UInt32}}",
                parameters={"pat": f"% {ng} %", "k": renderings * 3,
                            "cap": max(int(len(ng) / MIN_COVERAGE) + 4, 24),
                            "ratio": MAX_LENGTH_RATIO},
            ).result_rows
            # A rendering seen once in a corpus with ~0.5% machine-translated documents is
            # as likely to be that as to be usage. Agreed readings win; singletons are kept
            # only to fill the slots nothing better claimed.
            scored = [
                Rendering(spanish=r[0], count=int(r[1]), pair_id=int(r[2]),
                          doc_id=int(r[3]), english=r[4],
                          coverage=round(len(ng) / max(len(r[4]), 1), 2))
                for r in rend
            ]
            scored.sort(key=lambda x: (-min(x.count, 3), -x.coverage, len(x.spanish)))
            hits.append(PhraseHit(phrase=ng, words=int(n), support=int(support),
                                  renderings=tuple(scored[:renderings])))

    return tuple(hits)


def gather_evidence(cue: str, *, neighbours: int = 8, renderings: int = 3,
                    frame: Frame | None = None) -> Evidence:
    """Retrieve precedent for one cue. Deterministic: same cue, same evidence.

    Both indexes are consulted every time. They fail in opposite directions — phrases are
    silent on new wording, neighbours always answer something — so asking only one leaves
    a hole that the other covers.

    With a `frame`, precedent that matches the scene's form of address is ranked first.
    This is the whole of v2's retrieval difference, and it is deliberately small: the same
    evidence, in a different order. `Get in the car.` has six renderings in this corpus
    across three forms of address; without a frame the top four are chosen by how many
    translators agreed, which puts `Súbase.` and `- Entren al carro.` — the only two that
    state a form — below the cut. The scene is what says which of the three is right.
    """
    from .retrieval import find_precedent

    hits = list(gather_phrases(cue, renderings=renderings))

    # Over-fetch, then rank: reordering only helps if the candidates it needs survived the
    # cut. Fetching `neighbours` and sorting them is the bug this replaces -- the forms
    # worth promoting were the rare ones, and the cut removed them first.
    pool = [p for p in find_precedent(cue, limit=neighbours * 3)
            if p.similarity >= MIN_SIMILARITY]
    if frame is not None and frame.constrains:
        pool.sort(key=lambda p: _rank_key(read_address(p.spanish).form, frame))
    near = tuple(pool[:neighbours])
    return Evidence(cue=cue, phrases=tuple(hits), neighbours=near, frame=frame)


INSTRUCTION = """\
You localise English film subtitles into MEXICAN Spanish.

You are given evidence retrieved from a corpus of 718,925 lines written by Mexican
subtitlers. Your job is to choose what a Mexican translator would actually have written —
not to invent something that sounds Mexican.

The evidence comes in two kinds and they are not equal:

* ATTESTED PHRASES are exact phrases from the corpus. Under each one are Spanish lines
  that contained it, with how often that Spanish was used and the English line it came
  from. Read the English line before you trust the Spanish: the corpus has no word-level
  alignment, so when the phrase covers only part of its line, the Spanish translates the
  whole line and not your phrase. Use a rendering when its English line is close to your
  cue; ignore it when it is not, however many times it occurs.
* SIMILAR LINES are lines close in meaning, with a similarity score. They show the register
  a Mexican translator reaches for. They are weaker: nobody wrote your exact line this way.
  Treat them as a guide to tone, not as words to copy.

Rules:
- Output ONLY the Spanish line. No quotes, no explanation, no alternatives, no notes.
- Keep it the length of a subtitle. If the English is short, the Spanish is short.
- Never use `vosotros`/`vuestro` or their verb forms. Mexico does not have Spain's second
  person plural. Do NOT compensate by inserting `ustedes`: Spanish drops the subject
  pronoun, and 79% of plural-you lines in this corpus do exactly that — "You guys want one
  of these?" is "¿Quieren una de estas?", not "¿Ustedes quieren...?". Let the evidence
  decide when the pronoun is stated.
- Avoid vocabulary Mexican subtitlers do not write: currar, flipar, mola, cutre, chorrada,
  mogollón, aparcar, majo, pijo, cabrear, fontanero, chavales.
- Do NOT avoid a word merely because Spain also uses it. `coche` appears 373 times in this
  corpus and `vale` 924. Mexicans write them. Follow the evidence, not a blocklist.
- Do not add slang the evidence does not support. A neutral line translated neutrally is
  correct. Reaching for `güey` or `órale` where no evidence shows them is the failure mode
  this whole system exists to prevent.
- Preserve names, numbers and proper nouns exactly.
- If the evidence is empty, still translate as a Mexican subtitler would, but add nothing
  you cannot support: no slang, no regional vocabulary, no invention. Plain, correct
  Mexican Spanish. The line will be flagged as ungrounded for a human to check.
"""


def format_evidence(evidence: Evidence) -> str:
    """The evidence block, as the model sees it."""
    lines: list[str] = [f"ENGLISH CUE:\n{evidence.cue}\n"]

    frame = evidence.frame
    if frame is not None and frame.constrains:
        listeners = "one listener" if frame.listeners <= 1 else f"{frame.listeners} listeners"
        lines.append(
            f"SCENE: this cue is addressed as **{frame.address.value}** ({listeners}).\n"
            f"  Reason: {frame.why}\n"
            f"  Use this form. Precedent below is ordered to put it first; where a\n"
            f"  precedent uses a different form, take its wording and not its grammar.\n"
        )

    if evidence.phrases:
        lines.append("ATTESTED PHRASES (exact, from the corpus):")
        for hit in evidence.phrases:
            lines.append(f'  "{hit.phrase}" — appears in {hit.support} lines')
            for r in hit.renderings:
                lines.append(f"      {r.count:>3}x  {r.spanish}")
                lines.append(f"           from: {r.english}   "
                             f"(pair_id {r.pair_id}, covers {r.coverage:.0%} of the line)")
        lines.append("")

    if evidence.neighbours:
        lines.append("SIMILAR LINES (close in meaning — a guide to register, not to words):")
        for n in evidence.neighbours:
            agree = f"{n.times}/{n.english_times}" if n.english_times > 1 else "1"
            form = read_address(n.spanish).form
            tag = "" if form is Address.UNMARKED else f" [{form.value}]"
            lines.append(f"  [{n.similarity:.2f}] {n.english}")
            lines.append(f"          -> {n.spanish}{tag}   "
                         f"({agree} translators, pair_id {n.pair_id})")
        lines.append("")

    if evidence.is_empty:
        lines.append(
            "NO PRECEDENT FOUND for this line. Translate it plainly and correctly, and add\n"
            "nothing the corpus has not shown you. This output will be marked ungrounded.\n"
        )

    lines.append("MEXICAN SPANISH:")
    return "\n".join(lines)


RETRY_SUFFIX = """\

Your previous attempt was rejected. It used forms Mexican subtitlers do not write:
{markers}. Rewrite the line in Mexican Spanish, keeping the same meaning and length.
Do not over-correct: only those forms are the problem.
Previous attempt: {previous}
"""
