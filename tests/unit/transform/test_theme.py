from aoa.transform.theme import DEFAULT_THEME_NAME, THEMES, Theme, resolve_theme


EXPECTED_THEMES = {
    "material-light",
    "catppuccin-latte",
    "tokyonight-day",
    "darcula",
    "catppuccin-mocha",
    "material-dark",
    "kanagawa",
    "sonokai",
    "tokyonight",
}


def test_all_expected_themes_present() -> None:
    assert EXPECTED_THEMES.issubset(THEMES.keys())


def test_default_theme_is_registered() -> None:
    assert DEFAULT_THEME_NAME in THEMES


def test_resolve_theme_none_returns_default() -> None:
    assert resolve_theme(None) is THEMES[DEFAULT_THEME_NAME]


def test_resolve_theme_empty_returns_default() -> None:
    assert resolve_theme("") is THEMES[DEFAULT_THEME_NAME]


def test_resolve_theme_unknown_returns_default() -> None:
    assert resolve_theme("does-not-exist") is THEMES[DEFAULT_THEME_NAME]


def test_resolve_theme_known_returns_that_theme() -> None:
    assert resolve_theme("tokyonight") is THEMES["tokyonight"]


def test_theme_fields_are_populated() -> None:
    for name, theme in THEMES.items():
        assert isinstance(theme, Theme), name
        for field_name in (
            "critical",
            "red",
            "orange",
            "green",
            "edge",
            "arrow",
            "node_stroke",
            "node_fill",
            "text",
            "text_muted",
            "border",
            "bgcolor",
        ):
            value = getattr(theme, field_name)
            assert isinstance(value, str) and value, f"{name}.{field_name}"
