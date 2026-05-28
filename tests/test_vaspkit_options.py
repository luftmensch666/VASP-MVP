from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.i18n import load_translations
from vasp_mvp.vaspkit_options import (
    get_vaspkit_section,
    load_vaspkit_options,
    validate_vaspkit_values,
)


class VaspkitOptionsTest(unittest.TestCase):
    def test_loads_expected_sections(self) -> None:
        data = load_vaspkit_options()
        self.assertEqual(data["version"], 1)
        self.assertEqual(set(data["sections"]), {"poscar", "kpoints", "potcar", "incar"})

    def test_get_section(self) -> None:
        section = get_vaspkit_section("kpoints")
        keys = [item["key"] for item in section["options"]]
        self.assertEqual(keys, ["kmesh_scheme", "kmesh_resolved_value", "accuracy_preset"])

    def test_validate_choices_and_required_values(self) -> None:
        self.assertEqual(
            validate_vaspkit_values(
                "kpoints",
                {
                    "kmesh_scheme": "1",
                    "kmesh_resolved_value": 0.04,
                    "accuracy_preset": "medium",
                },
            ),
            [],
        )
        errors = validate_vaspkit_values(
            "kpoints",
            {
                "kmesh_scheme": "9",
                "kmesh_resolved_value": "bad",
                "accuracy_preset": "medium",
            },
        )
        self.assertTrue(any("kmesh_scheme" in item for item in errors))
        self.assertTrue(any("kmesh_resolved_value" in item for item in errors))

    def test_validate_poscar_custom_order_dependency(self) -> None:
        errors = validate_vaspkit_values(
            "poscar",
            {
                "uploaded_cif": "CePO4.cif",
                "element_order_mode": "custom",
                "custom_element_order": "",
            },
        )
        self.assertIn("custom_element_order", " ".join(errors))

    def test_validate_incar_custom_string(self) -> None:
        self.assertEqual(
            validate_vaspkit_values(
                "incar",
                {
                    "incar_key_parameters": ["SR", "D3"],
                    "incar_custom_key_string": "SRD3",
                },
            ),
            [],
        )
        errors = validate_vaspkit_values(
            "incar",
            {
                "incar_key_parameters": ["SR"],
                "incar_custom_key_string": "SRXX",
            },
        )
        self.assertTrue(any("incar_custom_key_string" in item for item in errors))

    def test_i18n_keys_exist_for_all_configured_labels(self) -> None:
        zh = load_translations("zh")
        en = load_translations("en")
        required_keys: set[str] = set()
        for section in load_vaspkit_options()["sections"].values():
            required_keys.add(section["label_key"])
            required_keys.add(section["help_key"])
            for option in section["options"]:
                required_keys.add(option["label_key"])
                required_keys.add(option["help_key"])
                required_keys.update(option.get("choice_label_keys", {}).values())

        self.assertFalse(required_keys - set(zh))
        self.assertFalse(required_keys - set(en))


if __name__ == "__main__":
    unittest.main()
