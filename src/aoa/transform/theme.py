"""Theme definitions for AoA transform outputs.

Themes mirror the CSS custom properties defined in
``frontend/src/style.css``.  They are used to colorise the Graphviz DOT
output and the PlantUML PERT diagram so that server-rendered artifacts
match the theme selected in the web frontend.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Theme:
    """Colors used when rendering AoA graphs.

    Fields mirror the ``--graph-*`` / related CSS custom properties used
    by the web frontend.
    """

    critical: str
    red: str
    orange: str
    green: str
    edge: str
    arrow: str
    node_stroke: str
    node_fill: str
    text: str
    text_muted: str
    border: str
    bgcolor: str = "transparent"


THEMES: dict[str, Theme] = {
    # ---- Light themes ----
    "material-light": Theme(
        critical="#c62828",
        red="#e53935",
        orange="#fb8c00",
        green="#43a047",
        edge="#757575",
        arrow="#424242",
        node_stroke="#424242",
        node_fill="#ffffff",
        text="#212121",
        text_muted="#757575",
        border="#e0e0e0",
    ),
    "catppuccin-latte": Theme(
        critical="#d20f39",
        red="#e64553",
        orange="#fe640b",
        green="#40a02b",
        edge="#8c8fa1",
        arrow="#4c4f69",
        node_stroke="#4c4f69",
        node_fill="#ffffff",
        text="#4c4f69",
        text_muted="#6c6f85",
        border="#ccd0da",
    ),
    "tokyonight-day": Theme(
        critical="#f52a65",
        red="#c64343",
        orange="#8c6c3e",
        green="#587539",
        edge="#6172b0",
        arrow="#3760bf",
        node_stroke="#3760bf",
        node_fill="#d5d6db",
        text="#3760bf",
        text_muted="#6172b0",
        border="#b4b5b9",
    ),
    # ---- Dark themes ----
    "darcula": Theme(
        critical="#ff6b68",
        red="#f44747",
        orange="#d19a66",
        green="#6a8759",
        edge="#808080",
        arrow="#a9b7c6",
        node_stroke="#a9b7c6",
        node_fill="#3c3f41",
        text="#a9b7c6",
        text_muted="#808080",
        border="#555555",
    ),
    "catppuccin-mocha": Theme(
        critical="#f38ba8",
        red="#eba0ac",
        orange="#fab387",
        green="#a6e3a1",
        edge="#a6adc8",
        arrow="#cdd6f4",
        node_stroke="#cdd6f4",
        node_fill="#313244",
        text="#cdd6f4",
        text_muted="#a6adc8",
        border="#45475a",
    ),
    "material-dark": Theme(
        critical="#f07178",
        red="#ff5370",
        orange="#f78c6c",
        green="#c3e88d",
        edge="#b0bec5",
        arrow="#eeffff",
        node_stroke="#eeffff",
        node_fill="#2d2d2d",
        text="#eeffff",
        text_muted="#b0bec5",
        border="#424242",
    ),
    "kanagawa": Theme(
        critical="#e82424",
        red="#ff5d62",
        orange="#ffa066",
        green="#98bb6c",
        edge="#727169",
        arrow="#dcd7ba",
        node_stroke="#dcd7ba",
        node_fill="#2a2a37",
        text="#dcd7ba",
        text_muted="#727169",
        border="#54546d",
    ),
    "sonokai": Theme(
        critical="#fc5d7c",
        red="#f85e84",
        orange="#e7c664",
        green="#9ed072",
        edge="#7f8490",
        arrow="#e2e2e3",
        node_stroke="#e2e2e3",
        node_fill="#33353f",
        text="#e2e2e3",
        text_muted="#7f8490",
        border="#4b4e58",
    ),
    "tokyonight": Theme(
        critical="#f7768e",
        red="#db4b4b",
        orange="#e0af68",
        green="#9ece6a",
        edge="#565f89",
        arrow="#c0caf5",
        node_stroke="#c0caf5",
        node_fill="#24283b",
        text="#c0caf5",
        text_muted="#565f89",
        border="#3b4261",
    ),
}


DEFAULT_THEME_NAME: str = "material-light"


def resolve_theme(name: str | None) -> Theme:
    """Resolve a theme by name, falling back to the default theme.

    ``None`` or an unknown name both return the default theme rather
    than raising, so callers (e.g. HTTP endpoints) can tolerate stale
    or missing client state.
    """
    if not name:
        return THEMES[DEFAULT_THEME_NAME]
    return THEMES.get(name, THEMES[DEFAULT_THEME_NAME])
