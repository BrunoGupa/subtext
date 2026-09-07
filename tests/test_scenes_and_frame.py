"""Scene segmentation and the discourse frame — v2's two model-backed steps.

Both take the model call as an injected `ask`, so everything here runs offline with a
stub. What is being tested is the plumbing around the model: the clock, the parsing, and
above all what happens when the model says something useless.
"""

from subtext.address import Address
from subtext.frame import Frame, parse_frame, read_frame
from subtext.scenes import GAP_MS, Scene, segment

CUES = ["Ma'am, do you need a ride?", "My son!", "I wouldn't know.",
        "Get in the car.", "How are the kids?", "Fine."]


def test_a_long_silence_cuts_the_scene_without_asking_the_model():
    """The clock is free and settles most boundaries. Spending a call on a 16-second gap
    would be paying for an answer the file already gives."""
    timings = [(0, 2000), (2200, 3000), (3200, 4000), (4200, 5000),
               (4200 + GAP_MS * 4, 21000), (21200, 22000)]
    scenes = segment(CUES, timings)
    assert [s.start for s in scenes] == [0, 4]
    assert not any(s.inferred for s in scenes)


def test_without_timings_or_a_model_it_under_segments_rather_than_guessing():
    """`mx_corpus` has no timestamps at all. One long scene degrades v2 toward v1; a
    guessed partition would build frames out of unrelated lines, which is worse."""
    scenes = segment(CUES)
    assert scenes == [Scene(start=0, cues=tuple(CUES))]


def test_the_model_only_decides_boundaries_the_clock_left_open():
    asked = []

    def ask(prompt: str) -> str:
        asked.append(prompt)
        return "[4]"

    timings = [(i * 1000, i * 1000 + 900) for i in range(len(CUES))]   # no real gaps
    scenes = segment(CUES, timings, ask=ask)
    assert asked == []                      # every gap under SAME_SCENE_MS
    assert len(scenes) == 1


def test_a_model_answer_that_is_not_a_list_changes_nothing():
    """Segmentation runs unattended over a whole file. A refusal, an apology or a
    truncated reply must leave the partition it started with, not raise."""
    for reply in ("I'm sorry, I can't help with that.", "", "[", "{}"):
        assert segment(CUES, ask=lambda _p, r=reply: r) == [Scene(0, tuple(CUES))]


def test_cuts_outside_the_window_are_ignored():
    """A model that answers with indices from a window it was not shown would re-cut a
    part of the file it never read."""
    assert segment(CUES, ask=lambda _p: "[0, 99, -3]") == [Scene(0, tuple(CUES))]


def test_frame_reads_the_form_and_its_confidence():
    frame = parse_frame(
        '{"address": "usted", "confidence": "high", "listeners": 1, "why": "Ma\'am"}')
    assert frame.address is Address.USTED and frame.constrains and frame.listeners == 1


def test_a_low_confidence_frame_does_not_constrain():
    """The frame reorders evidence. Ranking precedent by a guess would make the one
    decision v2 exists to get right noisier than v1's ordering, which is at least
    unbiased."""
    frame = parse_frame('{"address": "tu", "confidence": "low", "listeners": 1}')
    assert frame.address is Address.TU
    assert not frame.constrains


def test_unparseable_frames_are_empty_rather_than_assumed():
    for reply in ("no puedo", "", "{", '{"address": "vosotros"}', "[]"):
        assert not parse_frame(reply).constrains


def test_no_model_means_no_frame():
    """Without a key there is nothing in an English scene that says tú from usted, so v2
    falls back to v1's ordering instead of inventing one."""
    assert read_frame(CUES) == Frame()
