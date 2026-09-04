"""The deterministic halves of the localisation pipeline.

Everything here runs without ClickHouse, without ADK and without an API key — which is
the point: the two steps that bracket the model call are pure functions, so they can be
pinned by tests rather than inspected by eye in a demo.
"""

import pytest

from subtext.localise import (
    MIN_PHRASE_SUPPORT,
    Evidence,
    PhraseHit,
    candidate_phrases,
    check_register,
    format_evidence,
)


# ---- the register gate --------------------------------------------------------------

def test_peninsular_spanish_is_rejected():
    r = check_register("Vale tío, qué guay, cojo el coche.")
    assert not r.ok
    assert "vale" in r.peninsular and "tio" in r.peninsular and "guay" in r.peninsular


def test_mexican_spanish_passes():
    r = check_register("Órale güey, ahorita nos vemos.")
    assert r.ok
    assert "orale" in r.mexican and "guey" in r.mexican


def test_neutral_spanish_passes_with_no_markers():
    """Not every line should be slang. Neutral is a correct answer, not a failure."""
    r = check_register("No te preocupes, ya vamos.")
    assert r.ok
    assert not r.mexican and not r.peninsular


def test_the_gate_is_accent_insensitive():
    """The corpus is accent-stripped; the gate must be too, or 'tío' walks straight past."""
    assert not check_register("Qué pasa, tío.").ok
    assert not check_register("Que pasa, tio.").ok


def test_markers_must_be_whole_words():
    """'vale' inside 'equivale' is not Spain, and must not fail an innocent line."""
    assert check_register("Eso equivale a lo mismo.").ok


def test_a_single_peninsular_marker_is_enough_to_fail():
    """The one promise this system makes is no peninsular leakage."""
    assert not check_register("Ahorita agarro el coche, güey.").ok


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
