from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VASPKIT_OPTIONS_PATH = PROJECT_ROOT / "config" / "vaspkit_options.json"


class VaspkitOptionsError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_vaspkit_options() -> dict[str, Any]:
    with VASPKIT_OPTIONS_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "sections" not in data:
        raise VaspkitOptionsError("Invalid VASPKIT options config: missing sections")
    return data


def get_vaspkit_section(section: str) -> dict[str, Any]:
    sections = load_vaspkit_options()["sections"]
    try:
        return sections[section]
    except KeyError as exc:
        raise VaspkitOptionsError(f"Unknown VASPKIT section: {section}") from exc


def validate_vaspkit_values(section: str, values: dict) -> list[str]:
    section_config = get_vaspkit_section(section)
    errors: list[str] = []
    options = {item["key"]: item for item in section_config.get("options", [])}

    for key, option in options.items():
        value = values.get(key)
        if option.get("required") and _is_empty(value):
            errors.append(f"Missing required option: {key}")
            continue

        if _is_empty(value):
            continue

        option_type = option.get("type")
        choices = option.get("choices", [])
        if option_type == "select" and choices and str(value) not in choices:
            errors.append(f"Invalid choice for {key}: {value}")
        elif option_type == "multiselect":
            selected = value if isinstance(value, list) else [value]
            invalid = [item for item in selected if str(item) not in choices]
            if invalid:
                errors.append(f"Invalid choices for {key}: {', '.join(str(item) for item in invalid)}")
        elif option_type == "number":
            try:
                float(value)
            except (TypeError, ValueError):
                errors.append(f"Invalid number for {key}: {value}")

        pattern = option.get("pattern")
        if pattern and not re.fullmatch(pattern, str(value)):
            errors.append(f"Invalid format for {key}: {value}")

    if section == "poscar" and values.get("element_order_mode") == "custom":
        if _is_empty(values.get("custom_element_order")):
            errors.append("custom_element_order is required when element_order_mode is custom")

    return errors


def _is_empty(value: object) -> bool:
    return value is None or value == "" or value == []
