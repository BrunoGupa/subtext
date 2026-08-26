from reel_query.chunking import Line, chunk_lines


def make_line(line_id: int, *, season: int = 1, episode: int = 1, character: str = "A") -> Line:
    return Line(
        line_id=line_id, title="T", title_id="tt", season=season, episode=episode,
        episode_title="E", scene=1, line_no=line_id, character=character, speaks_to="",
        timecode="00:00:00,000", start_ms=line_id * 1000, end_ms=line_id * 1000 + 500,
        text=f"line {line_id}",
    )


def test_line_strategy_is_one_chunk_per_line():
    lines = [make_line(i) for i in range(1, 6)]
    chunks = list(chunk_lines(lines, strategy="line"))
    assert len(chunks) == 5
    assert [c.line_ids for c in chunks] == [[i] for i in range(1, 6)]
    assert all(c.window_size == 1 for c in chunks)


def test_window_strategy_covers_every_line():
    lines = [make_line(i) for i in range(1, 8)]
    chunks = list(chunk_lines(lines, strategy="window", window_size=3))
    assert len(chunks) == 5  # 7 lines, window 3, stride 1
    covered = {lid for c in chunks for lid in c.line_ids}
    assert covered == {i for i in range(1, 8)}
    assert all(len(c.line_ids) == 3 for c in chunks)


def test_windows_never_cross_an_episode_boundary():
    lines = [make_line(i, episode=1) for i in range(1, 4)]
    lines += [make_line(i, episode=2) for i in range(4, 7)]
    chunks = list(chunk_lines(lines, strategy="window", window_size=3))
    for chunk in chunks:
        episodes = {ln.episode for ln in lines if ln.line_id in chunk.line_ids}
        assert len(episodes) == 1, "a window spanned two episodes"


def test_window_shorter_than_the_episode_still_yields_one_chunk():
    lines = [make_line(i) for i in range(1, 3)]
    chunks = list(chunk_lines(lines, strategy="window", window_size=5))
    assert len(chunks) == 1
    assert chunks[0].line_ids == [1, 2]


def test_unknown_strategy_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        list(chunk_lines([make_line(1)], strategy="semantic"))
