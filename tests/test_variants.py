"""One cue in, every reading the corpus attests out.

Everything here runs offline: the model call is injected, so what is tested is the logic
around it — what gets asked, what gets believed, and what happens when the answer is junk.
"""

import pytest

from subtext.address import Address
from subtext.variants import (
    ADDRESSED_SHARE,
    REFUSAL,
    VariantEvidence,
    _first_line,
    _looks_untranslated,
    check_output,
    format_variant_prompt,
    translate_variants,
)


class _Precedent:
    """The fields `variants` actually reads off a retrieval hit."""

    def __init__(self, english, spanish, pair_id=1, times=2, english_times=4, similarity=0.8):
        self.english, self.spanish, self.pair_id = english, spanish, pair_id
        self.times, self.english_times, self.similarity = times, english_times, similarity


def test_a_prompt_offers_the_model_a_way_out():
    """Refusing has to be an option, or the model produces the reading anyway.

    `Here's looking at you, kid` came back as `Brindo por usted, niña.` — grammatical, and
    wrong, because nobody addresses a child as usted. That is not caught by checking the
    output: the reading should never have been generated.
    """
    prompt = format_variant_prompt(
        "Here's looking at you, kid",
        VariantEvidence(Address.USTED, (_Precedent("Look at you.", "Mírese."),)))
    assert REFUSAL in prompt
    assert "kid" in prompt


def test_a_refused_reading_is_not_offered():
    calls = []

    def ask(prompt):
        calls.append(prompt)
        return "NONE"

    evidence = VariantEvidence(Address.USTED, (_Precedent("Look at you.", "Mírese."),))
    variants = translate_variants("Here's looking at you, kid", ask=ask,
                                  tag_forms=lambda lines: [Address.USTED] * len(lines))
    assert variants == [] or all(v.spanish != "NONE" for v in variants)


@pytest.mark.parametrize("reply,expected", [
    ('{"addresses": "usted", "well_formed": true, "fix": ""}',
     (Address.USTED, True, "")),
    ('{"addresses": "ustedes", "well_formed": false, "fix": "Alégrenme el día."}',
     (Address.USTEDES, False, "Alégrenme el día.")),
])
def test_the_output_check_reads_both_answers(reply, expected):
    assert check_output("x", ask=lambda _p: reply) == expected


@pytest.mark.parametrize("reply", ["no puedo", "", "{", "[]", "null"])
def test_an_unreadable_check_never_fails_a_line(reply):
    """The check gates a retry. A malformed reply must not send a good line back — that
    would spend a second call and invite the rewrite to drift off the evidence."""
    form, well_formed, fix = check_output("Cuídese.", ask=lambda _p: reply)
    assert well_formed and fix == "" and form is Address.UNMARKED


def test_a_bad_conjugation_is_sent_back_once_and_python_decides():
    """`Adelante, alégranme el día.` passed the register gate clean: every word looked
    Mexican and the conjugation was invented. Nothing else in the pipeline reads morphology,
    and the corpus cannot stand in — `alégrame`, `cuídese` and `cállense` occur zero times
    in 718,925 lines and are all correct Spanish."""
    replies = iter([
        "Adelante, alégranme el día.",                                  # first attempt
        '{"addresses": "ustedes", "well_formed": false, "fix": "alégrenme"}',
        "Adelante, alégrenme el día.",                                  # retry
        '{"addresses": "ustedes", "well_formed": true, "fix": ""}',
    ])
    variants = translate_variants(
        "Go ahead, make my day", ask=lambda _p: next(replies),
        tag_forms=lambda lines: [Address.USTEDES] * len(lines))
    assert variants and variants[0].spanish == "Adelante, alégrenme el día."
    assert variants[0].well_formed


def test_a_line_still_wrong_after_the_retry_is_returned_flagged():
    """Dropping it would hide the failure from the reviewer, who is the point."""
    replies = iter([
        "Adelante, alégranme el día.",
        '{"addresses": "ustedes", "well_formed": false, "fix": ""}',
        "Adelante, alégranme el día.",
        '{"addresses": "ustedes", "well_formed": false, "fix": ""}',
    ])
    variants = translate_variants(
        "Go ahead, make my day", ask=lambda _p: next(replies),
        tag_forms=lambda lines: [Address.USTEDES] * len(lines))
    assert variants and not variants[0].well_formed


@pytest.mark.parametrize("english,spanish,expected", [
    ("Show me the money.", "Güey, show me the money.", True),
    ("Show me the money.", "¡Muéstrame el dinero!", False),
    ("Rosebud", "Rosebud", True),
])
def test_code_switched_rows_are_not_precedent(english, spanish, expected):
    """The corpus carries them honestly — Mexican subtitles do leave English in — but as
    precedent they teach the model to return the cue untranslated, which is what happened
    to `Show me the money!` on the first run."""
    assert _looks_untranslated(english, spanish) is expected


def test_the_addressed_share_is_a_proportion_not_a_word_list():
    """The first version asked whether the *English* contained `you` or began with a verb
    from a derived list. The list had neither `have` nor `calm`, so `Have a seat.` and
    `Calm down.` — which the corpus renders in all three forms — were judged to address
    nobody. A closed list cannot cover an open input; the tagged Spanish can."""
    assert 0 < ADDRESSED_SHARE < 1


@pytest.mark.parametrize("reply,expected", [
    ('  "¡Cállense!"  ', "¡Cállense!"),
    ("Vengan aquí.\nOtra opción: Vengan.", "Vengan aquí."),
    (None, ""),
])
def test_only_the_first_line_of_an_answer_is_taken(reply, expected):
    assert _first_line(reply) == expected
