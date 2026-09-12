"""Side-by-side comparison video frames.

The demo exists to show the study's actual finding, which is not "the robot
navigates". It is that a classical planner and a learned policy diverge
specifically under clutter, and that the learned policy's failure there is
stalling rather than crashing. So the two runs are rendered on the **same
world, synchronised frame by frame**, with the outcome labelled honestly
including when the learned policy loses.

Text is drawn with PIL rather than matplotlib: the panels are already NumPy
rasters from :mod:`vision_nav.viz.topdown`, and routing them through a
plotting library to add captions would be slower and would resample the
image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = ["PanelStyle", "compose_frame", "pad_to_length", "load_font"]


class PanelStyle:
    """Layout and palette for the composite frame (RGB, 0-255)."""

    header_h: int = 56
    footer_h: int = 46
    #: Wide enough that a right-aligned status on the left panel cannot be
    #: mistaken for belonging to the right panel's title.
    gutter: int = 32
    margin: int = 10

    background = (28, 30, 34)
    header_text = (245, 245, 248)
    subtle_text = (150, 156, 166)

    success = (70, 190, 120)
    collision = (226, 92, 84)
    timeout = (226, 176, 78)
    running = (150, 156, 166)


OUTCOME_COLOUR = {
    "success": PanelStyle.success,
    "collision": PanelStyle.collision,
    "timeout": PanelStyle.timeout,
    "running": PanelStyle.running,
}


def load_font(size: int) -> ImageFont.ImageFont:
    """A readable font, falling back to PIL's bitmap default.

    DejaVuSans ships with matplotlib and Pillow on most installs; the default
    bitmap font is tiny but never missing, so the video still renders on a
    machine without it rather than failing at the last step.
    """
    for name in ("DejaVuSans.ttf", "arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@dataclass
class Panel:
    """One side of the comparison."""

    title: str
    frame: np.ndarray
    outcome: str = "running"
    step: int = 0
    detail: str = ""


def pad_to_length(frames: list[np.ndarray], length: int) -> list[np.ndarray]:
    """Hold the final frame so both panels stay time-aligned.

    The two policies finish at different steps. Truncating to the shorter run
    would hide the slower one's ending; looping would misrepresent it. Holding
    the last frame keeps the clock shared and shows one side finishing while
    the other is still going, which is the point of the comparison.
    """
    if not frames:
        raise ValueError("cannot pad an empty frame list")
    if len(frames) >= length:
        return frames[:length]
    return frames + [frames[-1]] * (length - len(frames))


def compose_frame(
    left: Panel,
    right: Panel,
    caption: str = "",
    style: type[PanelStyle] = PanelStyle,
) -> np.ndarray:
    """Lay two rendered panels side by side with titles and a caption."""
    if left.frame.shape != right.frame.shape:
        raise ValueError(
            f"panels must be the same size, got {left.frame.shape} and {right.frame.shape}"
        )

    ph, pw = left.frame.shape[:2]
    width = style.margin * 2 + pw * 2 + style.gutter
    height = style.header_h + ph + style.footer_h

    canvas = Image.new("RGB", (width, height), style.background)
    canvas.paste(Image.fromarray(left.frame), (style.margin, style.header_h))
    canvas.paste(
        Image.fromarray(right.frame), (style.margin + pw + style.gutter, style.header_h)
    )

    draw = ImageDraw.Draw(canvas)
    title_font = load_font(20)
    small_font = load_font(14)
    caption_font = load_font(16)

    for panel, x0 in (
        (left, style.margin),
        (right, style.margin + pw + style.gutter),
    ):
        draw.text((x0, 12), panel.title, fill=style.header_text, font=title_font)
        status = panel.outcome if panel.outcome != "running" else f"step {panel.step}"
        colour = OUTCOME_COLOUR.get(panel.outcome, style.subtle_text)
        # Right-align the status within its panel.
        w = draw.textlength(status, font=small_font)
        draw.text((x0 + pw - w, 17), status, fill=colour, font=small_font)
        if panel.detail:
            draw.text((x0, 36), panel.detail, fill=style.subtle_text, font=small_font)

    if caption:
        draw.text(
            (style.margin, style.header_h + ph + 14),
            caption,
            fill=style.subtle_text,
            font=caption_font,
        )

    return np.asarray(canvas)
