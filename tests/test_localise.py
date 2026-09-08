"""The deterministic halves of the localisation pipeline.

Everything here runs without ClickHouse, without ADK and without an API key — which is
the point: the two steps that bracket the model call are pure functions, so they can be
pinned by tests rather than inspected by eye in a demo.
"""

import pytest

from subtext.localise import (
    Consensus,
    Rendering,
    INSTRUCTION,
    MIN_PHRASE_SUPPORT,
    NOT_MEXICAN,
    WATCH,
    Evidence,
    PhraseHit,
    candidate_phrases,
    check_register,
    rank_consensus,
    format_evidence,
)


# ---- the register gate --------------------------------------------------------------
#
# Bruno caught the original version of this gate rejecting "Eso a mí me vale madre." and
# "Súbete al coche, güey." Both are real corpus lines and both are unmistakably Mexican.
# The gate had been built on the DETECTOR's 66-word peninsular list, which is sound for
# scoring a 1,000-line window in bulk and wrong as a rule about one line: 59 of those 66
# words appear in Mexican Spanish. These tests exist so that never comes back.

def test_spain_second_person_plural_is_rejected():
    """Grammar, not vocabulary. Mexico does not have vosotros at all."""
    r = check_register("Vosotros llegáis tarde.")
    assert not r.ok and "vosotros" in r.not_mexican


def test_words_absent_from_the_corpus_are_rejected():
    """`currar` occurs 0 times in 718,925 lines of Mexican Spanish."""
    assert not check_register("Vamos a currar un poco.").ok


@pytest.mark.parametrize("line,word", [
    ("Eso a mí me vale madre.", "vale"),          # 924 occurrences; 156 are "me vale"
    ("Súbete al coche, güey.", "coche"),          # 373 occurrences
    ("Vale la pena intentarlo.", "vale"),         # the verb valer, not Spain's "ok"
    ("Mi tío vive en el piso de arriba.", "tio"), # tío 450, piso 297
])
def test_peninsular_leaning_words_mexicans_actually_write_are_never_rejected(line, word):
    """The failure Bruno caught. A false rejection breaks the system's core promise --
    that its output is grounded in attested Mexican usage -- so it is the worse error."""
    r = check_register(line)
    assert r.ok, f"{line!r} was rejected"
    assert word in r.watch, f"{word} should still be reported, just not fatal"


def test_a_watched_word_is_reported_not_hidden():
    """Allowed is not the same as invisible: a human should still see it."""
    assert "coche" in check_register("Súbete al coche.").watch


def test_mexican_spanish_passes_and_is_credited():
    r = check_register("Órale güey, ahorita nos vemos.")
    assert r.ok
    assert "orale" in r.mexican and "guey" in r.mexican


def test_neutral_spanish_passes_with_no_markers():
    """Not every line should be slang. Neutral is a correct answer, not a failure."""
    r = check_register("No te preocupes, ya vamos.")
    assert r.ok and not r.mexican and not r.watch


def test_the_gate_is_accent_insensitive():
    """The corpus is accent-stripped; the gate must be too, or a form walks straight past."""
    assert not check_register("Vosotros sabéis.").ok
    assert not check_register("Vosotros sabeis.").ok


def test_markers_must_be_whole_words():
    """'mola' inside 'molalidad' is not Spain, and must not fail an innocent line."""
    assert check_register("Midieron la molalidad de la muestra.").ok


def test_the_hard_list_stays_small():
    """It is a blocklist against a living language. Growth should require evidence, and a
    passing test is not evidence -- re-derive from the corpus before adding anything."""
    assert len(NOT_MEXICAN) <= 30
    assert set(NOT_MEXICAN).isdisjoint(WATCH)


def test_the_instruction_does_not_ban_words_mexicans_use():
    """The prompt used to tell the model 'never use coche'. It is in the corpus 373 times."""
    banned_line = [l for l in INSTRUCTION.splitlines() if "Avoid vocabulary" in l]
    assert banned_line, "the instruction should name what to avoid"
    assert "coche" not in "".join(banned_line)
    assert "vale," not in "".join(banned_line)


# ---- phrase selection ---------------------------------------------------------------

def test_candidate_phrases_are_longest_first():
    got = candidate_phrases("Where is my car?")
    assert got[0] == "where is my car"
    assert got[-1] == "my car"
    assert all(len(p.split()) >= 2 for p in got)


def test_single_words_are_never_candidates():
    """`my` occurs 35,178 times and answers 'Dios mío'. One-word phrases are noise."""
    assert all(" " in p for p in candidate_phrases("Hurry up now"))


def test_strongest_phrase_prefers_length_over_frequency():
    ev = Evidence(cue="x", phrases=(
        PhraseHit("my", 1, 35178),
        PhraseHit("my car", 2, 106),
        PhraseHit("where is my car", 4, 5),
    ))
    assert ev.strongest_phrase.phrase == "where is my car"


def test_thinly_supported_phrases_are_not_quoted_as_precedent():
    ev = Evidence(cue="x", phrases=(PhraseHit("a rare phrase", 3, MIN_PHRASE_SUPPORT - 1),))
    assert ev.strongest_phrase is None


# ---- the prompt ---------------------------------------------------------------------

def test_prompt_separates_attested_phrases_from_mere_neighbours():
    """Flattened together, a 0.6 neighbour reads as fact. The labels carry the weight."""
    ev = Evidence(cue="Hurry up!", phrases=(
        PhraseHit("hurry up", 2, 374,
                  (Rendering("¡Apúrate!", 11, 4210099, 77, "Hurry up.", 0.89),)),))
    text = format_evidence(ev)
    assert "ATTESTED PHRASES" in text
    assert "¡Apúrate!" in text and "11x" in text
    assert text.rstrip().endswith("MEXICAN SPANISH:")


def test_a_rendering_shows_the_line_it_came_from_and_its_citation():
    """The corpus has no word alignment, so a phrase lookup returns the Spanish of the
    *line* that contained the phrase. Presented bare, that produced `"you can't handle"`
    -> `Nada más.` labelled as attested fact. The source line and the `pair_id` are what
    let a reader -- or the model -- see that the Spanish is about something else."""
    ev = Evidence(cue="You can't handle the truth!", phrases=(
        PhraseHit("you can't handle", 3, 10,
                  (Rendering("No podrían soportarlo.", 1, 24408638, 141,
                             "You can't handle mine.", 0.73),)),))
    text = format_evidence(ev)
    assert "You can't handle mine." in text      # what it is a rendering OF
    assert "pair_id 24408638" in text            # checkable
    assert "73%" in text                         # how much of the line the phrase covers


def test_empty_evidence_says_so_rather_than_going_silent():
    text = format_evidence(Evidence(cue="For Frodo."))
    assert "NO PRECEDENT FOUND" in text
    assert "ungrounded" in text


def test_empty_evidence_does_not_ask_for_neutral_spanish():
    """It used to say "translate into neutral Latin-American Spanish", which is wrong twice:
    neutral-vs-Mexican arbitration is explicitly out of scope for this project, and a
    neutral line is exactly what the ungrounded baseline produces. The output would then
    pass the register gate and be indistinguishable from a grounded result."""
    text = format_evidence(Evidence(cue="Recalibrate the tachyon manifold."))
    assert "neutral" not in text.lower()


def test_evidence_knows_when_it_is_empty():
    assert Evidence(cue="x").is_empty
    assert not Evidence(cue="x", phrases=(PhraseHit("a b", 2, 9),)).is_empty


def test_the_instruction_does_not_mandate_ustedes():
    """Spanish drops the subject pronoun: only 21.3% of plural-you lines in the corpus
    state `ustedes`. Mandating it forces a construction real translators omit 4 times in 5."""
    line = [l for l in INSTRUCTION.splitlines() if "vosotros" in l or "ustedes" in l]
    joined = " ".join(line)
    assert "vosotros" in joined, "the vosotros prohibition is real and must stay"
    assert "Do NOT compensate by inserting `ustedes`" in joined


def test_the_cue_always_reaches_the_prompt():
    assert "For Frodo." in format_evidence(Evidence(cue="For Frodo."))


@pytest.mark.parametrize("spanish,marker", [
    ("La concha de Dios.", "concha de"),
    ("¿Vos sabés?", "vos"),
    ("No seas boludo.", "boludo"),
    ("Qué quilombo.", "quilombo"),
    ("¿Cachai?", "cachai"),
])
def test_other_latin_american_forms_fail_a_line(spanish, marker):
    """The gate checked Spain and nothing else, so `Jesus fucking Christ` came back as
    `La concha de Dios.` -- Rioplatense -- having passed every check clean.

    It passed because it was true: the corpus contains it. Other-Latin-American forms are
    0.120% of `mx_corpus`, 402 lines of 335,800, and that rate is what makes them dangerous
    rather than harmless. It is invisible to any aggregate and decisive in retrieval, which
    returns the nearest neighbour and not the average one.
    """
    report = check_register(spanish)
    assert not report.ok
    assert marker in report.other_latam


@pytest.mark.parametrize("spanish", [
    "Me compré una concha en la panadería.",   # in Mexico a concha is a pastry
    "Tráeme la plata.",
    "Ese flaco es mi cuate.",
    "Oye, man, ¿qué onda?",
])
def test_words_that_are_ordinary_in_mexico_are_not_rejected(spanish):
    """The lexicon audit's lesson applied to a second list: a marker has to be a string that
    is not also Mexican. `concha` alone is bread here, so only the phrase `concha de` is
    listed; `plata`, `flaco` and `man` are excluded outright."""
    assert check_register(spanish).ok


# --- Agreed renderings ------------------------------------------------------------
#
# The channel that exists because `MIN_COVERAGE` cannot cover short idioms: `up his ass`
# is 10 characters and its seven corpus lines run 42 to 124, so every one of them fails
# coverage while six of them say `culo`. The numbers below are the measured ones.

def test_enrichment_decides_and_not_how_many_lines_agree():
    """`el` is in 5 of the 7 `up his ass` lines and `por el culo` in only 3. Ranked by
    share the function word wins and the finding is lost; ranked against what the corpus
    does anyway, `el` is ordinary and `por el culo` is not."""
    got = rank_consensus([("el", 5, 120_000), ("por el culo", 3, 26)],
                         total_lines=7, corpus_lines=337_225, keep=2)
    assert [c.spanish for c in got] == ["por el culo"]


def test_a_phrase_whose_lines_disagree_claims_nothing():
    """`his ass` spans 102 lines that mostly mean `kick his ass`. The readings scatter and
    the best candidate is `este` at 18x. Silence is the right answer: an asserted
    consensus that is not one is worse evidence than none."""
    assert rank_consensus([("este", 29, 5_700), ("su", 29, 19_000)],
                          total_lines=102, corpus_lines=337_225) == ()


def test_nested_agreements_are_one_finding_not_three():
    got = rank_consensus([("por el culo", 3, 26), ("el culo", 5, 98), ("culo", 6, 507)],
                         total_lines=7, corpus_lines=337_225, keep=2)
    assert [c.spanish for c in got] == ["por el culo"]


def test_an_agreed_rendering_reaches_the_prompt_with_its_evidence():
    ev = Evidence(cue="He wore this watch up his ass", phrases=(
        PhraseHit("up his ass", 3, 6, (), (Consensus("por el culo", 3, 6, 5790.0),)),))
    text = format_evidence(ev)
    assert "AGREED RENDERING: por el culo" in text
    assert "3 of 6 lines" in text and "5790x" in text


def test_the_instruction_ranks_agreement_above_the_weaker_channels():
    assert "AGREED RENDERINGS" in INSTRUCTION
    assert INSTRUCTION.index("AGREED RENDERINGS") < INSTRUCTION.index("ATTESTED PHRASES")


def test_consensus_share_is_reported_honestly():
    assert Consensus("por el culo", 3, 6, 5790.0).share == 0.5
