"""Tests for the side-by-side comparison renderer."""

from __future__ import annotations

import numpy as np
import pytest

from vision_nav.viz.comparison import Panel, PanelStyle, compose_frame, load_font, pad_to_length


def blank(h=100, w=120, value=200):
    return np.full((h, w, 3), value, dtype=np.uint8)


# ----------------------------------------------------------------------
# pad_to_length
# ----------------------------------------------------------------------
def test_padding_holds_the_final_frame():
    """Holding, not looping: the slower run must show as still going."""
    frames = [blank(value=v) for v in (10, 20, 30)]
    out = pad_to_length(frames, 6)
    assert len(out) == 6
    assert all(np.array_equal(f, frames[-1]) for f in out[3:])


def test_padding_is_a_no_op_when_already_long_enough():
    frames = [blank() for _ in range(5)]
    assert len(pad_to_length(frames, 3)) == 3
    assert len(pad_to_length(frames, 5)) == 5


def test_padding_an_empty_list_is_an_error():
    with pytest.raises(ValueError, match="empty frame list"):
        pad_to_length([], 4)


# ----------------------------------------------------------------------
# compose_frame
# ----------------------------------------------------------------------
def test_composite_has_room_for_both_panels_and_chrome():
    left = Panel("Classical", blank(100, 120))
    right = Panel("Learned", blank(100, 120))
    out = compose_frame(left, right, caption="world 1")

    s = PanelStyle
    assert out.shape[0] == s.header_h + 100 + s.footer_h
    assert out.shape[1] == s.margin * 2 + 120 * 2 + s.gutter
    assert out.dtype == np.uint8


def test_both_panels_are_actually_pasted():
    left = Panel("L", blank(60, 80, value=17))
    right = Panel("R", blank(60, 80, value=211))
    out = compose_frame(left, right)

    s = PanelStyle
    y = s.header_h + 30
    assert out[y, s.margin + 40].tolist() == [17, 17, 17]
    assert out[y, s.margin + 80 + s.gutter + 40].tolist() == [211, 211, 211]


def test_mismatched_panel_sizes_are_refused():
    with pytest.raises(ValueError, match="same size"):
        compose_frame(Panel("L", blank(100, 120)), Panel("R", blank(90, 120)))


def test_outcome_colour_differs_from_running():
    """The status text must actually change colour, not just wording."""
    base = blank(60, 200, value=40)
    running = compose_frame(Panel("L", base, "running", 5), Panel("R", base, "running", 5))
    crashed = compose_frame(Panel("L", base, "collision", 5), Panel("R", base, "collision", 5))
    header = slice(0, PanelStyle.header_h)
    assert not np.array_equal(running[header], crashed[header])


def test_each_outcome_renders_distinctly():
    base = blank(60, 200, value=40)
    rendered = {}
    for outcome in ("success", "collision", "timeout"):
        f = compose_frame(Panel("L", base, outcome, 9), Panel("R", base, outcome, 9))
        rendered[outcome] = f[: PanelStyle.header_h].tobytes()
    assert len(set(rendered.values())) == 3, "outcomes must be visually distinguishable"


def test_caption_changes_the_footer():
    base = blank(60, 200)
    without = compose_frame(Panel("L", base), Panel("R", base))
    with_cap = compose_frame(Panel("L", base), Panel("R", base), caption="narrow, seed 30001")
    footer = slice(PanelStyle.header_h + 60, None)
    assert not np.array_equal(without[footer], with_cap[footer])


def test_font_loading_always_returns_something():
    """Must degrade to the bitmap default rather than fail at render time."""
    assert load_font(18) is not None
    assert load_font(9) is not None
