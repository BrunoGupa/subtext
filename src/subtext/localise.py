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
from .tokenizer import MAX_N, clickhouse_haystack_sql, es_tokens, ngrams, tokens

#: A neighbour below this cosine similarity is noise dressed as evidence.
#:
#: Re-measured 2026-09-08 for `gemini-embedding-001`, which does not use MiniLM's scale:
#: an exact line match came back 1.00 either way, but an unsupported cue bottomed out at
#: 0.55 under MiniLM and at 0.88 here. Every bound below moved with it.
#:
#: 0.85 was chosen by reading the band rather than counting it. Between 0.90 and 0.96 for
#: `Get in the car.` sit `Get in.` -> `Súbete` / `Sube` / `Súbanse` / `Trépate` and
#: `- Get into the car!` -> `Súbete al coche, Rebeca.` -- real evidence, and a floor at
#: 0.92 would start dropping them. Below 0.85 the neighbours answer other sentences.
MIN_SIMILARITY = 0.85

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

#: `MIN_COVERAGE` above is a bound on *whole-line* evidence, and it is right there: if the
#: phrase is a quarter of the line, the line's Spanish is not the phrase's Spanish. But
#: applied as the only rule it also silences every short idiom, because a short idiom only
#: ever occurs inside long lines. `up his ass` is 10 characters; the seven corpus lines
#: containing it run 42 to 124, so all seven fail coverage and the channel goes quiet --
#: while six of those seven say `culo`. The constants below drive the second channel that
#: reads that agreement, without touching the first one.
#:
#: How many of a phrase's lines must contain a candidate Spanish n-gram before it counts
#: as agreement rather than coincidence.
MIN_CONSENSUS_SHARE = 0.25

#: ...and by how much that must beat the n-gram's corpus-wide rate. This is the bound that
#: does the real work, and frequency could not do it: `el` appears in 5 of the 7 `up his
#: ass` lines and `por el culo` in only 3, so by share alone the function word wins. By
#: enrichment `el` scores 4x and `por el culo` 4962x. Ranking by enrichment rather than by
#: count is the same choice made for the register lexicon, for the same reason.
MIN_CONSENSUS_ENRICHMENT = 50.0

#: A phrase attested in fewer lines than this cannot show agreement, only coincidence.
#: Measured, not guessed: at three lines the channel produced `desde` as the rendering of
#: `been screwing` -- 3 of 3 lines, 290x, and meaningless. Six lines is what `up his ass`
#: has, so this is set below that and no lower.
MIN_CONSENSUS_LINES = 5

#: Lines read per phrase to measure that agreement. A sample, ordered by `pair_id` so the
#: evidence stays deterministic, as the module promises.
CONSENSUS_SAMPLE = 400

#: Candidate n-grams carried to the (single, batched) baseline query per phrase.
CONSENSUS_CANDIDATES = 20


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
class Consensus:
    """The Spanish a phrase's lines AGREE on -- a claim about the phrase, not the line.

    `Rendering` can only ever say "a line containing this phrase was translated so". This
    says "the lines containing this phrase all put this in their Spanish", which is a
    different and stronger statement, and it is reached without a word aligner: unrelated
    lines share function words and nothing else, so whatever survives across them at a rate
    the corpus at large cannot explain is the phrase's own rendering.

    It abstains where it should. `his ass` spans 102 lines that mostly mean `kick his ass`,
    the readings disagree, and the best candidate is `este` at 18x -- under the bound, so
    nothing is claimed. Silence there is the feature; a wrong rendering asserted as
    agreement is worse than none.
    """

    spanish: str
    #: Lines containing this Spanish, out of `of_lines` read for the phrase.
    lines: int
    of_lines: int
    #: How many times more often this appears in the phrase's lines than corpus-wide.
    enrichment: float

    @property
    def share(self) -> float:
        return self.lines / self.of_lines if self.of_lines else 0.0


@dataclass(frozen=True)
class PhraseHit:
    phrase: str
    words: int
    support: int
    renderings: tuple[Rendering, ...] = ()
    #: What this phrase's lines agree on. Independent of `renderings`:
    #: a phrase can have consensus and no whole-line evidence, which is
    #: exactly the case the coverage bound leaves uncovered.
    consensus: tuple[Consensus, ...] = ()


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
#: 2. Vocabulary attested at most twice in the Mexican corpus. At that rate
#:    the occurrences are more likely to BE the 0.46% peninsular leak than evidence of
#:    Mexican usage, so rejecting them is safe in both directions.
#:
#: Measured 2026-09-04 against mx_corpus. Re-derive after any corpus rebuild with
#: `sql/peninsular_rates.sql`.
NOT_MEXICAN: tuple[str, ...] = (
    # Spain's 2nd person plural -- grammar, not vocabulary
    "vosotros", "vuestro", "vuestra", "vuestros", "vuestras",
    # 0 occurrences in the Mexican corpus
    "chorrada", "currar", "curro", "flipante", "flipar", "flipas", "mogollon",
    # 1-2 occurrences
    "cabreado", "cabrear", "cutre", "fontanero", "gilipolleces", "majo", "maja",
    "molar", "mola", "pijo", "aparcar", "chavales",
    # `pajita` removed 2026-09-04 at Bruno's call: it is ordinary vocabulary rather than
    # Spain-only slang, and a single occurrence is not grounds for a blocklist entry.
    #
    # The reflexive imperatives of *joder*, added 2026-09-07 at Bruno's call and measured
    # before acting on it. The family splits, so the entries are the forms and not the verb:
    #
    #     joder    0.4x     jodete   0.9x     jodanse  1.6x (3 lines)     jodase  0x
    #     no jodas 5.7x  <- genuinely Mexican, deliberately NOT listed
    #     chinga tu madre 188x · chingue a su madre 176x · chinguen a su madre 120x
    #
    # `Fuck you!` was returning `¡Chinga tu madre!` / `¡Chingue a su madre!` / **`¡Jódanse!`**,
    # so the plural broke a set the other two got right. `jódete` at 0.9x is *less* common in
    # Mexican productions than in Spanish subtitles at large; the chingar forms are two orders
    # of magnitude more Mexican. Listing the verb would have taken `no jodas` with it.
    "jodete", "jodanse", "jodase", "jodeos",
)

#: Forms from *other* Latin American varieties. The gate had no defence against these at
#: all -- it checked Spain and nothing else -- and on 2026-09-07 that produced
#: `Jesus fucking Christ` -> **`La concha de Dios.`**, which is Rioplatense, and which passed
#: every check clean: grounded, right form of address, well-formed grammar.
#:
#: It passed because it was *true*: the corpus really contains it. Other-Latin-American
#: contamination is only 0.120% of `mx_corpus` (402 lines of 335,800), and that is exactly
#: why it is dangerous rather than harmless. A rate that low is invisible to any aggregate
#: and decisive in retrieval, because retrieval returns the *nearest* neighbour, not the
#: average one. `Jesus fucking Christ` is a rare cue, so one contaminated line won it.
#:
#: The list follows the discipline the lexicon audit taught: only forms that are not also
#: Mexican. Voseo morphology (`sos`, `tenés`, `querés`, `podés`, `vos`) is grammar rather
#: than vocabulary, like Spain's `vosotros`, so it cannot be anything else. The vocabulary
#: entries are the ones with no Mexican reading. Deliberately EXCLUDED for being ordinary in
#: Mexico or ambiguous: `mina`, `plata`, `flaco`, `chorro`, `man`, `tinto`, `forro`, `che`,
#: and bare `concha` -- which in Mexico is a pastry. Only the phrase `concha de` is listed.
NOT_MEXICAN_LATAM: tuple[str, ...] = (
    # voseo: grammar, not word choice
    "vos", "sos", "tenes", "queres", "podes", "vení", "veni", "andá", "anda vos",
    # Rioplatense
    "boludo", "boluda", "boludos", "pelotudo", "pelotuda", "quilombo", "laburo",
    "laburar", "pibe", "piba", "pibes", "chabon", "bondi", "concha de",
    # Chilean
    "cachai", "weon", "weona", "po",
    # Colombian / Venezuelan
    "parcero", "chimba", "berraco", "chamo", "chevere", "bacano",
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
    #: Forms from other Latin American varieties. Fails a line for the same reason Spain's
    #: forms do: it is the wrong country, and being the wrong country is the one thing this
    #: system exists to prevent.
    other_latam: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        """Only forms Mexicans genuinely do not write can fail a line."""
        return not self.not_mexican and not self.other_latam

    @property
    def summary(self) -> str:
        parts = []
        if self.not_mexican:
            parts.append(f"NOT Mexican: {', '.join(self.not_mexican)}")
        if self.other_latam:
            parts.append(f"NOT Mexican (other Latin American): "
                         f"{', '.join(self.other_latam)}")
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
        other_latam=hit(NOT_MEXICAN_LATAM),
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


def phrase_consensus(phrase: str, *, keep: int = 2) -> tuple[Consensus, ...]:
    """What the lines containing `phrase` agree their Spanish contains.

    Two queries, whatever the phrase: one for the lines, one that measures every candidate
    against the whole corpus at once. The second is the expensive half and batching it is
    the difference between a lookup and a scan per candidate. Both go through the MCP
    server -- see `rows`.
    """
    from collections import Counter

    from . import mcp_sql

    lines = rows(
        f"SELECT es FROM mx_corpus "
        f"WHERE {clickhouse_haystack_sql('en')} LIKE {mcp_sql.quote(f'% {phrase} %')} "
        f"ORDER BY pair_id LIMIT {CONSENSUS_SAMPLE}"
    )
    total_lines = len(lines)
    if total_lines < MIN_PHRASE_SUPPORT:
        return ()

    # Document frequency, not raw count: a word repeated inside one line is one line's
    # worth of evidence, not two.
    seen_in = Counter()
    for (es,) in lines:
        words = es_tokens(es)
        here = set()
        for n in range(1, MAX_N + 1):
            here.update(ngrams(words, n))
        seen_in.update(here)

    floor = max(MIN_PHRASE_SUPPORT, total_lines * MIN_CONSENSUS_SHARE)
    candidates = [ng for ng, d in seen_in.most_common() if d >= floor][:CONSENSUS_CANDIDATES]
    if not candidates:
        return ()

    # One pass over the corpus answering every candidate, instead of one pass each.
    counts = ", ".join(f"countIf(es ILIKE {mcp_sql.quote(f'%{ng}%')})" for ng in candidates)
    baseline = rows(f"SELECT count(), {counts} FROM mx_corpus")[0]
    corpus_lines = int(baseline[0]) or 1

    return rank_consensus(measured_consensus(candidates, seen_in, baseline),
                          total_lines=total_lines, corpus_lines=corpus_lines, keep=keep)


def measured_consensus(candidates, seen_in, baseline) -> list[tuple[str, int, int]]:
    """(n-gram, lines carrying the phrase, lines corpus-wide) -- the raw counts, before
    any bound is applied to them. Separated so a sweep can measure once and threshold many
    times instead of re-reading the corpus per candidate bound."""
    return [(ng, seen_in[ng], int(baseline[i + 1])) for i, ng in enumerate(candidates)]


def rank_consensus(measured: Sequence[tuple[str, int, int]], *, total_lines: int,
                   corpus_lines: int, keep: int = 2,
                   min_enrichment: float = MIN_CONSENSUS_ENRICHMENT,
                   min_lines: int = MIN_CONSENSUS_LINES) -> tuple[Consensus, ...]:
    """Turn counted candidates into the ones worth showing. Pure, so it can be pinned.

    `measured` is (spanish n-gram, lines here, lines corpus-wide). `min_enrichment` is a
    parameter and not just the constant so the bound can be swept over the evaluation set
    without the sweep having to reach in and reassign a module global.
    """
    scored: list[Consensus] = []
    if total_lines < min_lines:
        return ()
    for ng, here, base in measured:
        if not base or not total_lines:
            continue
        enrichment = (here / total_lines) / (base / corpus_lines)
        if enrichment >= min_enrichment:
            scored.append(Consensus(spanish=ng, lines=here, of_lines=total_lines,
                                    enrichment=round(enrichment, 1)))

    scored.sort(key=lambda c: -c.enrichment)
    # `por el culo` (4962x), `el culo` (2200x) and `culo` (507x) are one finding stated
    # three times. Keeping the nested ones would pad the prompt with its own echo.
    out: list[Consensus] = []
    for c in scored:
        if any(c.spanish in kept.spanish for kept in out):
            continue
        out.append(c)
        if len(out) == keep:
            break
    return tuple(out)


def rows(sql: str) -> list[tuple]:
    """Run one read-only statement, through the MCP server when it is available.

    The ClickHouse track asks for the partner's product to be used at runtime *through
    the official MCP server*, and the product is this path -- not the question-answering
    agent, which is where the MCP toolset already lived. So the phrase channel's SQL goes
    through `mcp-clickhouse`, and the driver stays the fallback for a machine that has no
    server on PATH.

    The statements are built with literals rather than bound parameters because the MCP
    tool takes SQL text and nothing else. `mcp_sql.quote` does the escaping, and the only
    values inlined are phrases cut from the cue by the shared tokenizer.
    """
    from . import mcp_sql

    if mcp_sql.available():
        return mcp_sql.query_values(sql)
    from .db import client

    return client().query(sql).result_rows


def gather_phrases(cue: str, *, renderings: int = 3) -> tuple[PhraseHit, ...]:
    """The attested-phrase half of the evidence, on its own.

    Split out because the phrase channel says nothing about who is being addressed -- a
    rendering of `"the truth"` is the same rendering whether the scene is tú or usted -- so
    every reading of a cue wants the same phrase evidence, and fetching it once per reading
    would ask the same question three times.
    """
    from . import mcp_sql

    phrases = candidate_phrases(cue)
    hits: list[PhraseHit] = []
    if phrases:
        wanted = ", ".join(mcp_sql.quote(p) for p in phrases)
        found = rows(
            f"SELECT ng, n, support FROM phrase_index "
            f"WHERE ng IN ({wanted}) AND support >= {MIN_PHRASE_SUPPORT} "
            f"ORDER BY n DESC, support DESC LIMIT 3"
        )
        # Longest first: a 2-word phrase is usually too generic to carry a translation.
        # Each phrase is three round trips to ClickHouse Cloud and none depends on
        # another, so the phrases run at once and the order of `found` is kept.
        def one(item):
            ng, n, support = item
            # Short source lines only. In a long line the phrase is a fragment and its
            # Spanish is buried in unrelated words -- "we're going to be" pulled back a
            # sentence about a spine operation. Bounding the line keeps the rendering
            # close to the phrase itself.
            # `pair_id` and the source line come back with every rendering. Without them
            # this block asserted things it could not support: it grouped by `es` alone,
            # so nothing was citable, and it showed the Spanish of a whole line as if it
            # were the Spanish of the phrase. The length-ratio pair drops the crudest
            # misalignments the same way `find_precedent` does.
            rend = rows(
                f"SELECT es, count() AS c, min(pair_id) AS pid, any(doc_id) AS did, "
                f"       any(en) AS src "
                f"FROM mx_corpus "
                f"WHERE {clickhouse_haystack_sql('en')} LIKE {mcp_sql.quote(f"% {ng} %")} "
                f"  AND length(en) <= {int(len(ng) / MIN_COVERAGE) + 8} "
                f"  AND length(es) <= length(en) * {MAX_LENGTH_RATIO} "
                f"  AND length(en) <= length(es) * {MAX_LENGTH_RATIO} "
                f"GROUP BY es ORDER BY c DESC, length(es) ASC LIMIT {renderings * 3}"
            )
            # A rendering seen once in a corpus with ~0.5% machine-translated documents is
            # as likely to be that as to be usage. Agreed readings win; singletons are kept
            # only to fill the slots nothing better claimed.
            scored = [
                Rendering(spanish=r[0], count=int(r[1]), pair_id=int(r[2]),
                          doc_id=int(r[3]), english=r[4],
                          coverage=round(len(ng) / max(len(r[4]), 1), 2))
                for r in rend
            ]
            # Coverage was computed, printed, and never actually used -- the bound existed
            # only as a length cap with a 24-character floor, which lets a short phrase match
            # a line it barely occupies. `He wore this watch up his ass` is what that cost:
            # the specific phrase `up his ass` (6 lines) returned nothing because no line
            # containing it is under 24 characters, while the generic `his ass` returned
            # `I could kick his ass.` -> `Podría patearle su trasero.` at 33% coverage. So
            # the model was handed `trasero` from an unrelated idiom, and used it, and the
            # corpus's own `culo` -- 165 lines, 94 of them rendering `ass` in exactly this
            # crude sense -- never reached the prompt.
            #
            # Filtering on coverage rather than merely sorting by it is the fix: below the
            # bound the Spanish is not about the phrase, it is about the rest of the line.
            scored = [x for x in scored if x.coverage >= MIN_COVERAGE]
            scored.sort(key=lambda x: (-min(x.count, 3), -x.coverage, len(x.spanish)))
            return PhraseHit(phrase=ng, words=int(n), support=int(support),
                             renderings=tuple(scored[:renderings]),
                             consensus=phrase_consensus(ng))

        if found:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(found)) as pool:
                hits.extend(pool.map(one, found))

    return tuple(hits)


def gather_evidence(cue: str, *, neighbours: int = 8, renderings: int = 3) -> Evidence:
    """Retrieve precedent for one cue. Deterministic: same cue, same evidence.

    Both indexes are consulted every time. They fail in opposite directions — phrases are
    silent on new wording, neighbours always answer something — so asking only one leaves
    a hole that the other covers.

    This is the single-answer path: one cue, one Spanish line. `variants.py` is the other
    one, and it exists because for a cue that addresses somebody this function has to
    choose a form of address it cannot know. `Get in the car.` has renderings here as tú,
    usted and ustedes, and the ordering below picks between them by how many translators
    agreed — which is to say by accident.
    """
    from .retrieval import find_precedent

    hits = list(gather_phrases(cue, renderings=renderings))

    near = tuple(p for p in find_precedent(cue, limit=neighbours * 3)
                 if p.similarity >= MIN_SIMILARITY)[:neighbours]
    return Evidence(cue=cue, phrases=tuple(hits), neighbours=near)


INSTRUCTION = """\
You localise English film subtitles into MEXICAN Spanish.

You are given evidence retrieved from a corpus of 335,800 lines written by Mexican
subtitlers. Your job is to choose what a Mexican translator would actually have written —
not to invent something that sounds Mexican.

The evidence comes in two kinds and they are not equal:

* AGREED RENDERINGS are the strongest evidence here. When many corpus lines contain the
  same English phrase, their Spanish sides share almost nothing except that phrase's own
  translation, so what they agree on IS the phrase's Spanish — measured, not aligned. The
  multiplier says how much more often it appears in these lines than in the corpus at
  large. Prefer an agreed rendering over anything below, and use the word it gives you
  even when it is coarse — where the corpus agrees on a blunt word, a politer synonym is
  a translation it does not support. If no agreed rendering is shown, the phrase had no
  agreement and you get the weaker evidence only.
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
    """The evidence block, as the model sees it.

    Every untrusted string is fenced -- the cue, and every corpus line. The corpus is
    third-party subtitle text, so it is no more trusted than the cue is; see `guard`.
    """
    from .guard import FENCE_NOTE, fence

    lines: list[str] = [FENCE_NOTE, "", f"ENGLISH CUE:\n{fence(evidence.cue)}\n"]


    if evidence.phrases:
        lines.append("ATTESTED PHRASES (exact, from the corpus):")
        for hit in evidence.phrases:
            lines.append(f'  {fence(hit.phrase)} — appears in {hit.support} lines')
            for c in hit.consensus:
                lines.append(f"      AGREED RENDERING: {fence(c.spanish)}   "
                             f"({c.lines} of {c.of_lines} lines carrying this phrase, "
                             f"{c.enrichment:.0f}x the corpus rate)")
            for r in hit.renderings:
                lines.append(f"      {r.count:>3}x  {fence(r.spanish)}")
                lines.append(f"           from: {fence(r.english)}   "
                             f"(pair_id {r.pair_id}, covers {r.coverage:.0%} of the line)")
        lines.append("")

    if evidence.neighbours:
        lines.append("SIMILAR LINES (close in meaning — a guide to register, not to words):")
        for n in evidence.neighbours:
            agree = f"{n.times}/{n.english_times}" if n.english_times > 1 else "1"
            lines.append(f"  [{n.similarity:.2f}] {fence(n.english)}")
            lines.append(f"          -> {fence(n.spanish)}   "
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
