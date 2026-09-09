"""One cue in, every reading the corpus attests out.

Everything here runs offline: the model call is injected, so what is tested is the logic
around it — what gets asked, what gets believed, and what happens when the answer is junk.
"""

import pytest

from subtext.address import Address
from subtext.variants import (
    ADDRESSED_SHARE,
    REFUSAL,
    Variant,
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


def scripted(*, translations, checks, gender="NONE"):
    """A fake model that answers by what it is asked, not by turn order.

    The grammar check and the gender question run concurrently since 2026-09-09, so a
    reply script in a fixed order would be answered in whichever order the threads
    arrive. Each kind of prompt has its own queue instead; the translation queue serves
    the first attempt and then the retry.
    """
    translations, checks = iter(translations), iter(checks)

    def ask(prompt: str) -> str:
        head = prompt.strip().splitlines()[0]
        if head.startswith("Check one Spanish subtitle line"):
            return next(checks)
        if head.startswith("You are given one Mexican Spanish subtitle line"):
            return gender
        return next(translations)

    return ask


def test_a_bad_conjugation_is_sent_back_once_and_python_decides():
    """`Adelante, alégranme el día.` passed the register gate clean: every word looked
    Mexican and the conjugation was invented. Nothing else in the pipeline reads morphology,
    and the corpus cannot stand in — `alégrame`, `cuídese` and `cállense` occur zero times
    in 718,925 lines and are all correct Spanish."""
    ask = scripted(
        translations=["Adelante, alégranme el día.",      # first attempt
                      "Adelante, alégrenme el día."],     # retry
        checks=['{"addresses": "ustedes", "well_formed": false, "fix": "alégrenme"}',
                '{"addresses": "ustedes", "well_formed": true, "fix": ""}'],
    )
    variants = translate_variants(
        "Go ahead, make my day", ask=ask,
        tag_forms=lambda lines: [Address.USTEDES] * len(lines))
    assert variants and variants[0].spanish == "Adelante, alégrenme el día."
    assert variants[0].well_formed


def test_a_line_still_wrong_after_the_retry_is_returned_flagged():
    """Dropping it would hide the failure from the reviewer, who is the point."""
    ask = scripted(
        translations=["Adelante, alégranme el día.", "Adelante, alégranme el día."],
        checks=['{"addresses": "ustedes", "well_formed": false, "fix": ""}',
                '{"addresses": "ustedes", "well_formed": false, "fix": ""}'],
    )
    variants = translate_variants(
        "Go ahead, make my day", ask=ask,
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


# --- Agreed renderings reaching the reading that uses them -------------------------

def test_the_agreed_rendering_reaches_the_translation_prompt():
    """`--evidence-only` printed it while this prompt did not, so `up his ass` arrived at
    the model as three phrase headings with no Spanish under any of them, and the `culo`
    that came back was the model guessing, not the corpus."""
    from subtext.localise import Consensus, PhraseHit

    text = format_variant_prompt(
        "He wore this watch up his ass",
        VariantEvidence(form=Address.UNMARKED),
        phrases=(PhraseHit("up his ass", 3, 6, (), (Consensus("por el culo", 3, 6, 5790.0),)),))
    assert "AGREED RENDERING: <<<por el culo>>>" in text


def test_an_exact_line_precedent_outranks_an_agreed_phrase():
    """`what the fuck` agrees on `qué chingados`, and the corpus also renders the whole
    line as `¿Qué pedo?` 18 times. Told the agreed rendering wins outright, the model
    dropped the better evidence for the weaker one."""
    from subtext.variants import VARIANT_INSTRUCTION

    flat = " ".join(VARIANT_INSTRUCTION.split())
    assert "ranks BELOW a precedent that renders your line itself" in flat
    assert "never insert a subject pronoun" in flat


def test_no_test_case_answer_is_written_into_the_static_prompt():
    """An instruction sent on every call naming `up his ass` -> `por el culo` hands the
    model the answer to the case the channel is being judged on, and the retrieval can
    then be broken without the output changing. The rule stays; the example goes."""
    from subtext.localise import INSTRUCTION
    from subtext.variants import VARIANT_INSTRUCTION

    for prompt in (INSTRUCTION, VARIANT_INSTRUCTION):
        low = prompt.lower()
        for leak in ("up his ass", "culo", "trasero", "felaci", "pedo", "chingados"):
            assert leak not in low, f"{leak!r} is baked into a static prompt"


def test_agreement_is_grounding_even_with_no_close_neighbour():
    """Six lines agreeing on `por el culo` is stronger evidence than a 0.56 neighbour, and
    the flag read only the neighbour -- so the grounded line and `Helps fellatio.`, which
    has nothing behind it at all, carried the same warning."""
    backed = Variant(form=Address.UNMARKED, spanish="Llevaba este reloj por el culo",
                     top_similarity=0.56, agreed=("por el culo",))
    assert backed.grounded and not backed.weakly_grounded

    unbacked = Variant(form=Address.UNMARKED, spanish="Ayuda a la felación.",
                       top_similarity=0.55)
    assert unbacked.weakly_grounded


# --- One decision, declined --------------------------------------------------------

def test_the_second_reading_declines_the_first_instead_of_retranslating():
    """Readings were N independent translations, one per form, each shown only the
    precedent addressing its own listener -- which is three unrelated lines, not one line
    in three forms. `Oh, fuck me!` gave tú 117 precedents headed by `Fuck me!` ->
    `¡Cógeme!` at 0.84 and usted three, none of them about fuck me, so usted came back
    `¡La puta madre!`. Switching the form rewrote the sentence."""
    from subtext.variants import DECLINE_INSTRUCTION

    flat = " ".join(DECLINE_INSTRUCTION.split())
    assert "change of GRAMMAR, not of wording" in flat
    assert "If a word can stay, it stays." in flat
    # A line that addresses nobody has nothing to decline, and saying so is how the false
    # ambiguity collapses instead of producing a second sentence.
    assert "addresses nobody" in flat and "Output NONE" in flat


def test_declension_prompt_carries_the_decided_line_and_the_target_grammar():
    from subtext.address import Address
    from subtext.variants import DECLINE_INSTRUCTION, _ASKED, _GRAMMAR

    who, description = _ASKED[Address.USTED]
    text = DECLINE_INSTRUCTION.format(spanish="Súbete al coche.",
                                      who=f"{who} — {description}",
                                      grammar=_GRAMMAR[Address.USTED])
    assert "Súbete al coche." in text
    assert _GRAMMAR[Address.USTED] in text


# --- Gender, the other axis English leaves open ------------------------------------

def test_a_line_that_marks_nobody_gender_offers_no_choice():
    """Most lines are this, which is why the question is asked once per cue and not once
    per reading: `Cállate.` cannot be feminine, so there is nothing for a button to do."""
    from subtext.variants import _regender

    gender, other = _regender("Cállate.", "Shut up!", ask=lambda _p: "NONE")
    assert gender is None and other == ""


def test_the_direction_comes_back_with_the_flip():
    """One call carries both facts. Asked separately, a line already feminine and a line
    with no gender both answer "unchanged" to "make it feminine", and telling them apart
    cost a second question on every genderless line -- which is most of them."""
    from subtext.variants import Gender, _regender

    was, other = _regender("No quiero matarlo.", "I don't wanna kill you",
                          ask=lambda _p: "M> No quiero matarla.")
    assert was is Gender.MASCULINE and other == "No quiero matarla."

    was, other = _regender("Estoy cansada.", "I'm tired.", ask=lambda _p: "F> Estoy cansado.")
    assert was is Gender.FEMININE and other == "Estoy cansado."


def test_an_answer_without_a_direction_is_not_trusted():
    """A bare line back means the model ignored the protocol, and guessing which gender it
    started in would put a wrong label on a button."""
    from subtext.variants import _regender

    assert _regender("Estás cansado.", "You're tired.",
                     ask=lambda _p: "Estás cansada.") == (None, "")


def test_the_regender_prompt_moves_people_and_not_things():
    from subtext.variants import REGENDER_INSTRUCTION

    flat = " ".join(REGENDER_INSTRUCTION.split())
    assert "change of AGREEMENT, not of wording" in flat
    assert "`el coche` stays `el coche`" in flat
    # Same rule the person axis follows: the source settles it, or there is no choice.
    # Without this the flip fired on `Tell that bitch to be cool` -> `esa perra` / `ese
    # perro`, offering a reader a gender the English had already stated.
    assert "If the ENGLISH already settles that person's gender" in flat


# --- Three outcomes, told apart ----------------------------------------------------

def test_a_blocked_answer_is_not_reported_as_a_refusal():
    """A refusal is this pipeline working -- it declines a reading the line rules out.
    A block is the provider declining to answer. Both used to arrive as an empty string
    and produce the same blank row, which is what would have let a blocked line be
    described as a policy when it was our own NONE, or the reverse."""
    from subtext.variants import BLOCKED

    variants = translate_variants(
        "Am I a nigger?", ask=lambda _p: f"{BLOCKED} SAFETY",
        tag_forms=lambda lines: [Address.UNMARKED] * len(lines))
    assert len(variants) == 1
    assert not variants[0].answered
    assert variants[0].block_reason == "SAFETY"
    assert variants[0].spanish == ""


def test_an_empty_reply_is_also_unanswered_and_says_so():
    variants = translate_variants(
        "Shut up!", ask=lambda _p: "   ",
        tag_forms=lambda lines: [Address.TU] * len(lines))
    assert variants and not variants[0].answered
    assert variants[0].block_reason == "no reason given"


def test_every_reading_refused_returns_nothing_at_all():
    """Distinct from unanswered: there is no reading to show, and the caller can tell the
    two apart by whether it got a Variant back."""
    variants = translate_variants(
        "And I will strike down upon thee", ask=lambda _p: "NONE",
        tag_forms=lambda lines: [Address.TU] * len(lines))
    assert variants == []
