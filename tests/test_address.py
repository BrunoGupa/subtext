"""The form-of-address reader.

The cases here are the ones that were wrong at some point on 2026-09-07, kept as tests
because each was a different way of being wrong, not a typo.
"""

import pytest

from subtext.address import Address, read_address


@pytest.mark.parametrize("spanish,expected", [
    # The three readings of one English cue: `Get in the car.` in three corpus documents.
    ("Súbete al coche.", Address.TU),
    ("Súbase.", Address.USTED),
    ("Cállense, por favor.", Address.USTEDES),
    # Third-person-plural morphology, read as a command by its position in the line.
    ("- Entren al carro.", Address.USTEDES),
    ("¡Suban al auto!", Address.USTEDES),
    # Explicit pronouns and clitics.
    ("¿Usted qué opina?", Address.USTED),
    ("Ustedes dos, vengan.", Address.USTEDES),
    ("No te preocupes.", Address.TU),
    ("Esto es para ti.", Address.TU),
    # Verb morphology with no pronoun anywhere.
    ("No puedes manejar la verdad.", Address.TU),
    ("¿Qué hiciste?", Address.TU),
    ("Tendrás que esperar.", Address.TU),
    # Spain's second person plural, by pronoun and by morphology.
    ("¿Vosotros qué queréis?", Address.VOSOTROS),
    ("¿Estáis bien?", Address.VOSOTROS),
])
def test_forms_that_the_line_states(spanish, expected):
    assert read_address(spanish).form is expected


@pytest.mark.parametrize("spanish", [
    "Es la verdad.",
    "Volveré.",
    "Sube al auto.",        # tú imperative or `él sube` -- the line cannot say which
    "Que entren todos.",    # `entren` is a command only at the head of the line
])
def test_silence_is_returned_rather_than_a_guess(spanish):
    """`UNMARKED` is the right answer for ~78% of lines, and it has to stay right.

    The reading is used to *group* precedent by form. A precedent filed under the wrong
    form is worse than one left ungrouped, so a rule that guesses buys coverage with
    errors in the only place they are expensive.
    """
    assert read_address(spanish).form is Address.UNMARKED


@pytest.mark.parametrize("spanish", [
    "Ya vámonos.",          # first person plural: the speaker is going too
    "Sentémonos aquí.",
    "Adiós.",               # accented plural noun
    "¿Cuántos años tienes?",
    "Los subtítulos están mal.",
    "Déjalos en la mesa.",  # `-los` is an object pronoun, not Spain's `-os`
])
def test_accented_os_endings_are_not_spains_enclitic(spanish):
    """An `-os` enclitic rule tagged 0.69% of the corpus as `vosotros` against a true rate
    of 0.01% -- wrong roughly fifty times out of fifty-one -- because Spain's `sentaos`
    carries no written accent while `adiós` and `números` do. The rule was removed; these
    cases keep it from coming back."""
    assert read_address(spanish).form is not Address.VOSOTROS


@pytest.mark.parametrize("spanish", ["Quizás mañana.", "Además, no.", "Estas cosas pasan."])
def test_adverbs_and_demonstratives_are_not_verbs(spanish):
    """`estás` is second person; `estas` is a demonstrative. Only the written accent
    separates them, so the rule reads the accent and not the folded form."""
    assert read_address(spanish).form is Address.UNMARKED


def test_a_reading_carries_what_produced_it():
    reading = read_address("¿No te acuerdas de mí?")
    assert reading.form is Address.TU
    assert "te" in reading.evidence and reading.marked


def test_plural_wins_over_singular_when_both_are_marked():
    """`te` inside an `ustedes` line singles one listener out of a group; the line is
    still addressed to the group, and precedent should be filed that way."""
    reading = read_address("Ustedes esperen, y a ti te llamo después.")
    assert reading.form is Address.USTEDES
    assert reading.conflicted
