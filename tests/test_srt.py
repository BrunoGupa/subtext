from pathlib import Path

from reel_query.ingest.srt import parse_season_episode, parse_srt

SAMPLE = """1
00:00:01,000 --> 00:00:03,500
VALE: We're taking it.

2
00:00:04,000 --> 00:00:06,200
<i>[engine noise]</i>
- MARISOL: We are not.

3
00:00:07,000 --> 00:00:09,000
"""


def test_parse_season_episode_handles_common_naming():
    assert parse_season_episode("Show.S01E02.720p") == (1, 2)
    assert parse_season_episode("show 3x11 title") == (3, 11)
    assert parse_season_episode("some_movie_2019") == (0, 0)


def test_parse_srt_reads_cues_and_skips_empty_ones(tmp_path: Path):
    path = tmp_path / "Show.S01E02.srt"
    path.write_text(SAMPLE, encoding="utf-8")
    cues = list(parse_srt(path))
    assert len(cues) == 2
    assert cues[0] == (1000, 3500, "VALE: We're taking it.")
    # tags and bracketed stage directions are stripped
    assert "engine noise" not in cues[1][2]
    assert "<i>" not in cues[1][2]
