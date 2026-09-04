"""The deterministic halves of the localisation pipeline.

Everything here runs without ClickHouse, without ADK and without an API key — which is
the point: the two steps that bracket the model call are pure functions, so they can be
pinned by tests rather than inspected by eye in a demo.
"""

import pytest

from subtext.localise import (
    INSTRUCTION,
    MIN_PHRASE_SUPPORT,
    NOT_MEXICAN,
    WATCH,
    Evidence,
    PhraseHit,
    candidate_phrases,
    check_register,
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
    ev = Evidence(cue="Hurry up!", phrases=(PhraseHit("hurry up", 2, 374,
                                                      (("¡Apúrate!", 11),)),))
    text = format_evidence(ev)
    assert "ATTESTED PHRASES" in text
    assert "¡Apúrate!" in text and "11x" in text
    assert text.rstrip().endswith("MEXICAN SPANISH:")


def test_empty_evidence_says_so_rather_than_going_silent():
    text = format_evidence(Evidence(cue="For Frodo."))
    assert "NO EVIDENCE FOUND" in text


def test_the_cue_always_reaches_the_prompt():
    assert "For Frodo." in format_evidence(Evidence(cue="For Frodo."))
