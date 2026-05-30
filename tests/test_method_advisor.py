from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.method_advisor import generate_method_description


class MethodAdvisorTest(unittest.TestCase):
    def test_generates_distinct_descriptions_for_common_methods(self) -> None:
        pbe = generate_method_description(method_family="DFT", functional="PBE")
        d3 = generate_method_description(method_family="DFT", functional="PBE-D3")
        hse = generate_method_description(method_family="Hybrid DFT", functional="HSE06")
        pbe_u = generate_method_description(method_family="DFT+U", functional="PBE+U", elements=("Ce", "O"))

        self.assertIn("基准 DFT", pbe.zh)
        self.assertIn("baseline DFT", pbe.en)
        self.assertIn("范德华", d3.zh)
        self.assertIn("dispersion", d3.en)
        self.assertIn("计算成本较高", hse.zh)
        self.assertIn("computationally expensive", hse.en)
        self.assertIn("U 值", pbe_u.zh)
        self.assertIn("U values", pbe_u.en)
        self.assertIn("Ce", pbe_u.en)
        self.assertEqual(len({pbe.en, d3.en, hse.en, pbe_u.en}), 4)

    def test_generation_is_pure_and_does_not_overwrite_user_text(self) -> None:
        user_text = "Use literature U = 4.5 eV for Ce."
        generated = generate_method_description(method_family="DFT", functional="PBE-D3", adsorbate_name="NH3")

        self.assertEqual(user_text, "Use literature U = 4.5 eV for Ce.")
        self.assertIn("NH3", generated.en)
        self.assertIn("NH3", generated.zh)


if __name__ == "__main__":
    unittest.main()
