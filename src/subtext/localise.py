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

from .ingest.mexican import MEXICAN, PENINSULAR
from .tokenizer import MAX_N, clickhouse_haystack_sql, ngrams, tokens

#: A neighbour below this cosine similarity is noise dressed as evidence.
MIN_SIMILARITY = 0.55

#: Phrases seen fewer than this are too thin to quote as precedent.
MIN_PHRASE_SUPPORT = 3


@dataclass(frozen=True)
class PhraseHit:
    phrase: str
    words: int
    support: int
    renderings: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class Evidence:
    """Everything retrieved for one cue. Built without a model."""

    cue: str
    phrases: tuple[PhraseHit, ...] = ()
    neighbours: tuple = ()  # tuple[Precedent, ...]; typed loosely to avoid a cycle

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
    "molar", "mola", "pajita", "pijo", "aparcar", "chavales",
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


def gather_evidence(cue: str, *, neighbours: int = 4, renderings: int = 3) -> Evidence:
    """Retrieve precedent for one cue. Deterministic: same cue, same evidence.

    Both indexes are consulted every time. They fail in opposite directions — phrases are
    silent on new wording, neighbours always answer something — so asking only one leaves
    a hole that the other covers.
    """
    from .db import client
    from .retrieval import find_precedent

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
            rend = ch.query(
                f"SELECT es, count() AS c FROM mx_corpus "
                f"WHERE {clickhouse_haystack_sql('en')} LIKE {{pat:String}} "
                f"  AND length(en) <= {{cap:UInt32}} "
                f"GROUP BY es ORDER BY c DESC, length(es) ASC LIMIT {{k:UInt32}}",
                parameters={"pat": f"% {ng} %", "k": renderings,
                            "cap": max(len(ng) * 2 + 12, 30)},
            ).result_rows
            hits.append(PhraseHit(phrase=ng, words=int(n), support=int(support),
                                  renderings=tuple((r[0], int(r[1])) for r in rend)))

    near = tuple(p for p in find_precedent(cue, limit=neighbours)
                 if p.similarity >= MIN_SIMILARITY)
    return Evidence(cue=cue, phrases=tuple(hits), neighbours=near)


INSTRUCTION = """\
You localise English film subtitles into MEXICAN Spanish.

You are given evidence retrieved from a corpus of 718,925 lines written by Mexican
subtitlers. Your job is to choose what a Mexican translator would actually have written —
not to invent something that sounds Mexican.

The evidence comes in two kinds and they are not equal:

* ATTESTED PHRASES are exact phrases from the corpus with the number of times each Spanish
  rendering was used. This is fact. Prefer it. When a rendering is listed, use it or a
  close variant of it, unless the sentence genuinely will not take it.
* SIMILAR LINES are lines close in meaning, with a similarity score. They show the register
  a Mexican translator reaches for. They are weaker: nobody wrote your exact line this way.
  Treat them as a guide to tone, not as words to copy.

Rules:
- Output ONLY the Spanish line. No quotes, no explanation, no alternatives, no notes.
- Keep it the length of a subtitle. If the English is short, the Spanish is short.
- Use `ustedes` and its verb forms, never `vosotros`/`vuestro`. This is grammar: Mexico
  does not have Spain's second person plural at all.
- Avoid vocabulary Mexican subtitlers do not write: currar, flipar, mola, cutre, chorrada,
  mogollón, aparcar, majo, pijo, cabrear, fontanero, pajita, chavales.
- Do NOT avoid a word merely because Spain also uses it. `coche` appears 373 times in this
  corpus and `vale` 924. Mexicans write them. Follow the evidence, not a blocklist.
- Do not add slang the evidence does not support. A neutral line translated neutrally is
  correct. Reaching for `güey` or `órale` where no evidence shows them is the failure mode
  this whole system exists to prevent.
- Preserve names, numbers and proper nouns exactly.
- If the evidence is empty, translate plainly into neutral Latin-American Spanish.
"""


def format_evidence(evidence: Evidence) -> str:
    """The evidence block, as the model sees it."""
    lines: list[str] = [f"ENGLISH CUE:\n{evidence.cue}\n"]

    if evidence.phrases:
        lines.append("ATTESTED PHRASES (exact, from the corpus — prefer these):")
        for hit in evidence.phrases:
            lines.append(f'  "{hit.phrase}" — appears in {hit.support} lines')
            for spanish, count in hit.renderings:
                lines.append(f"      {count:>4}x  {spanish}")
        lines.append("")

    if evidence.neighbours:
        lines.append("SIMILAR LINES (close in meaning — a guide to register, not to words):")
        for n in evidence.neighbours:
            agree = f"{n.times}/{n.english_times}" if n.english_times > 1 else "1"
            lines.append(f"  [{n.similarity:.2f}] {n.english}")
            lines.append(f"          -> {n.spanish}   ({agree} translators, pair_id {n.pair_id})")
        lines.append("")

    if evidence.is_empty:
        lines.append("NO EVIDENCE FOUND. Translate plainly into neutral Latin-American Spanish.\n")

    lines.append("MEXICAN SPANISH:")
    return "\n".join(lines)


RETRY_SUFFIX = """\

Your previous attempt was rejected. It used forms Mexican subtitlers do not write:
{markers}. Rewrite the line in Mexican Spanish, keeping the same meaning and length.
Do not over-correct: only those forms are the problem.
Previous attempt: {previous}
"""
