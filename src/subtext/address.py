"""Which form of address does a Spanish line use — tú, usted, ustedes?

English `you` is all three at once. Spanish has to choose, and the choice is not free:
it encodes who is speaking to whom. The same English cue takes three different Spanish
lines depending on the scene, and the corpus shows exactly that:

    Get in the car.  ->  Súbete al coche.   (tú,      doc 496)
                     ->  Súbase.            (usted,   doc 511)
                     ->  - Entren al carro. (ustedes, doc 258)

So a retriever that ranks precedent without knowing which of the three the scene calls
for is choosing between them by frequency, which is to say at random. This module is the
part that reads the form off a Spanish line, so precedent can be grouped by it.

**Only 10% of the corpus marks the form explicitly** (72,247 of 718,925 lines carry a
second-person pronoun or clitic). That is the argument for doing this at the scene level
rather than the line: nine lines in ten say nothing, and the one that does fixes the
register for the rest.

Three signals, in order of how much they can be trusted:

1. **Pronouns and clitics** — `usted`, `ustedes`, `tú`, `ti`, `contigo`, `te`, `os`.
   Unambiguous where present.
2. **Enclitic imperatives** — `súbete` / `súbase` / `súbanse`. Spanish spelling settles
   these: a verb with a pronoun stuck on the end is an esdrújula and *must* carry a
   written accent, so an accented word ending in `-te`/`-se`/`-nse` is an imperative and
   not a noun. `clase` and `base` are unaccented and never match; `cállese` always does.
3. **Second-person singular verb forms** — derived from the corpus rather than written by
   hand, by taking words that co-occur with `tú`/`te`/`ti` far above chance (117 forms,
   filtered for the `-mos` first-person-plural endings that the raw lift let through).

`usted` and `ustedes` have no equivalent of signal 3, and this is a real limitation rather
than an oversight: they take *third*-person verb forms, identical to the ones used for
*él/ella/ellos*. `Quiere café` is `usted` or `she` with nothing in the morphology to say
which. So this module is strong on tú and thin on usted/ustedes, and `UNMARKED` is the
honest answer far more often than a guess would be.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

__all__ = ["Address", "AddressReading", "read_address", "TU_FORMS", "TU_FORMS_CORE", "USTEDES_IMPERATIVES"]


class Address(str, Enum):
    """The form a line addresses its listener in."""

    TU = "tú"
    USTED = "usted"
    USTEDES = "ustedes"
    VOSOTROS = "vosotros"       # Spain only; its presence is a contamination signal
    UNMARKED = "unmarked"       # nothing in the line says. The common case.


#: Explicit pronouns and clitics, accent-folded. `te` is second-person singular and
#: nothing else, which makes it the single most productive marker in the corpus.
_PRONOUNS: tuple[tuple[str, Address], ...] = (
    ("ustedes", Address.USTEDES),
    ("usted", Address.USTED),
    ("vosotros", Address.VOSOTROS), ("vuestro", Address.VOSOTROS),
    ("vuestra", Address.VOSOTROS), ("vuestros", Address.VOSOTROS),
    ("vuestras", Address.VOSOTROS),
    ("os", Address.VOSOTROS),
    ("tu", Address.TU), ("ti", Address.TU), ("te", Address.TU),
    ("contigo", Address.TU), ("tuyo", Address.TU), ("tuya", Address.TU),
    ("tuyos", Address.TU), ("tuyas", Address.TU), ("tus", Address.TU),
)

#: Second-person-singular verb forms, derived 2026-09-07 from `mx_corpus` by co-occurrence
#: with `tú`/`te`/`ti` (lift >= 3.0, >= 20 occurrences, `-s` ending, `-mos` excluded).
#: Re-derive after a corpus rebuild; the query is in `sql/address_forms.sql`.
TU_FORMS: frozenset[str] = frozenset("""
acerques acuerdas agarras amas apures asustes atrevas atreves caes callas chingas coges
comas comes confias consigues creas crees cuidas das decides dedicas dejaras dejas dejes
des dices diras echas empiezas encuentras enojes entras equivocas eras eres estabas
estarias estuvieras fueras gustas habias habrias hagas has hayas hicieras hubieras ibas
imaginas iras juegas llamas llegas llevas matas mereces metas metes miras molestes mueras
mueres muevas mueves olvides pagas pasas pases pides pierdas pierdes podias podras pones
pongas portas preocupas preocupes puedas quedaras quedas quedes quieras quisieras quitas
quites refieres regresas regreses rias ries sacas sales salgas sentias sepas seras serias
sientas sientes sigues subes tendrias tomes traes tuvieras vas vayas veas vengas ves
vienes vinieras vives vuelvas vuelves
""".split())

#: The forms the derivation above *missed*, and the reason it missed them matters: lift
#: measures how much more often a word appears near `tú`/`te` than elsewhere, and the most
#: frequent second-person verbs are common everywhere, so they score low. `puedes` occurs
#: constantly and scored under the threshold; `atreves` is rare and scored 683. A purely
#: statistical lexicon is therefore blind to exactly the forms that matter most. These are
#: added by hand, each one checked for collision with a noun or a name (`tomas`/`Tomás`,
#: `bebes`/`bebés`, `ayudas` and `estas` were rejected for that reason -- `estás` is caught
#: by the accent rule instead).
TU_FORMS_CORE: frozenset[str] = frozenset("""
tienes quieres puedes sabes haces dices entiendes necesitas debes piensas hablas esperas
buscas oyes duermes corres escribes lees conoces recuerdas prometes mientes trabajas
seas estes tengas digas esperes hables entiendas vengas hagas puedas quieras sepas
""".split())

#: `ustedes` imperatives, and the one place a rule can catch them. Their morphology is
#: third-person plural — `entren` is both "(you all) come in" and "(that they) come in" —
#: so nothing inside the word decides. **Position does**: at the start of a subtitle line
#: these are commands, and the subjunctive reading needs a `que` or a main clause in front
#: of it. Taken from the most frequent line-initial `-en`/`-an` words in `mx_corpus`, with
#: the names (`Sebastián`, `Adrián`) and adverbs (`bien`, `también`, `alguien`) that share
#: the ending removed by hand.
#:
#: This rule exists because without it the frame could not act on `ustedes` at all: with a
#: three-listener scene, `- Entren al carro.` and `¡Suban al auto!` both read as UNMARKED
#: and were ranked below singular precedent — the exact failure the frame was built to fix.
USTEDES_IMPERATIVES: frozenset[str] = frozenset("""
miren oigan esperen vengan disculpen escuchen vayan sigan pasen dejen recuerden tomen
hagan vean abran disfruten tengan suban entren salgan corran callen muevan traigan
pongan digan quiten bajen agarren perdonen permitan ayuden busquen aguanten piensen
olviden prueben coman beban duerman levanten paren cierren cuiden empiecen terminen
contesten respondan hablen griten rian sonrian sientense quedense acerquense
""".split())

#: Words ending in `-ste`/`-aste`/`-iste` that are not second-person preterites. Without
#: these the rule below reads `existe` and `triste` as verbs addressed to someone.
_NOT_PRETERITE: frozenset[str] = frozenset("""
este oeste chiste triste peste poste coste baste contraste desgaste existe consiste
insiste resiste asiste persiste desiste embiste reviste viste alpiste batiste
""".split())

_ACCENTED = re.compile(r"[áéíóúÁÉÍÓÚ]")

#: A written `-ás` ending is second-person singular (`estás`, `serás`, `tendrás`) — the
#: accent is what separates it from `estas`, the demonstrative. Only adverbs collide.
_AS_ENDING = re.compile(r"ás$", re.IGNORECASE)
_NOT_SECOND_PERSON_AS: frozenset[str] = frozenset(
    "quizas ademas jamas detras atras compas mas demas".split()
)

#: Spain's second-person-plural verb morphology. Unlike `vosotros` itself this cannot be
#: anything else, so it catches `¿qué esperáis?` in a line that never names the pronoun.
_VOSOTROS_ENDING = re.compile(r"(áis|éis|ais|eis)$", re.IGNORECASE)
#: French and Spanish words that end the same way. `seis` is excluded by the length bound.
_NOT_VOSOTROS: frozenset[str] = frozenset("palais mais jamais relais".split())
_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+")


def _fold(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text.lower())
    return "".join(c for c in decomposed if unicodedata.category(c) != "Mn")


@dataclass(frozen=True)
class AddressReading:
    """What the line says about its listener, and what said it."""

    form: Address
    evidence: tuple[str, ...] = ()
    #: True when more than one form is marked -- a line addressing two people, or a
    #: misalignment. The caller decides what to do; this module does not pick a winner.
    conflicted: bool = False

    @property
    def marked(self) -> bool:
        return self.form is not Address.UNMARKED


def _enclitic(raw_word: str, folded: str) -> Address | None:
    """An imperative with a pronoun welded on: `súbete`, `súbase`, `súbanse`.

    The written accent does the work. Spanish requires it on these forms because they are
    esdrújulas, and ordinary words ending in `-se`/`-te` (`clase`, `gente`, `parte`) do
    not carry one. That single condition separates them without a verb list.
    """
    if not _ACCENTED.search(raw_word) or len(folded) < 5:
        return None
    if folded.endswith("nse"):
        return Address.USTEDES
    if folded.endswith("se") and not folded.endswith("rse"):
        return Address.USTED
    if folded.endswith("te") and not folded.endswith("rte"):
        return Address.TU
    # There is deliberately no `-os` rule here. One was tried and removed the same day:
    # Spain's enclitic `sentaos` carries no written accent, so an accented word ending in
    # `-os` is never that -- it is a plural noun (`adiós`, `números`, `subtítulos`) or an
    # object pronoun (`déjalos`). The rule tagged 0.69% of the corpus as `vosotros` where
    # the true rate is 0.01%, i.e. it was wrong roughly fifty times out of fifty-one.
    # Spain's forms are caught by the pronouns and by the `-áis`/`-éis` endings instead.
    return None


def read_address(spanish: str) -> AddressReading:
    """Read the form of address off one Spanish line.

    Returns `UNMARKED` when the line says nothing, which is the majority of lines. A guess
    would be worse than silence here: the caller uses this to *group* precedent, and a
    wrongly grouped precedent is more damaging than an ungrouped one.
    """
    found: dict[Address, list[str]] = {}

    def note(form: Address, why: str) -> None:
        found.setdefault(form, []).append(why)

    for position, raw in enumerate(_WORD.findall(spanish)):
        folded = _fold(raw)
        if position == 0 and folded in USTEDES_IMPERATIVES:
            note(Address.USTEDES, folded)
            continue
        for token, form in _PRONOUNS:
            if folded == token:
                note(form, folded)
                break
        else:
            if folded in TU_FORMS or folded in TU_FORMS_CORE:
                note(Address.TU, folded)
            elif (folded.endswith(("aste", "iste")) and len(folded) > 4
                  and folded not in _NOT_PRETERITE):
                note(Address.TU, folded)
            elif (_AS_ENDING.search(raw) and len(folded) > 3
                  and folded not in _NOT_SECOND_PERSON_AS):
                note(Address.TU, folded)
            elif (_VOSOTROS_ENDING.search(folded) and len(folded) > 4
                  and folded not in _NOT_VOSOTROS):
                note(Address.VOSOTROS, folded)
            else:
                enclitic = _enclitic(raw, folded)
                if enclitic is not None:
                    note(enclitic, folded)

    if not found:
        return AddressReading(Address.UNMARKED)

    # Plural and Spain's forms win over tú when both appear: `te` inside an `ustedes` line
    # is one listener singled out of a group, and the line is still addressed to the group.
    for form in (Address.VOSOTROS, Address.USTEDES, Address.USTED, Address.TU):
        if form in found:
            return AddressReading(
                form=form,
                evidence=tuple(dict.fromkeys(found[form])),
                conflicted=len(found) > 1,
            )
    return AddressReading(Address.UNMARKED)
