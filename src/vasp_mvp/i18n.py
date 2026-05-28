from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
I18N_DIR = PROJECT_ROOT / "config" / "i18n"
LANGUAGES = ("zh", "en")


def get_available_languages() -> list[str]:
    return list(LANGUAGES)


@lru_cache(maxsize=8)
def load_translations(lang: str) -> dict:
    if lang not in LANGUAGES:
        lang = "zh"
    path = I18N_DIR / f"{lang}.json"
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
    except FileNotFoundError:
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def t(key: str, lang: str = "zh", **kwargs) -> str:
    text = load_translations(lang).get(key)
    if text is None:
        return f"[[missing:{key}]]"
    if not kwargs:
        return str(text)
    try:
        return str(text).format(**kwargs)
    except Exception:
        return str(text)
