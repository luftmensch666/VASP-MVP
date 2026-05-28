from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.adsorption import adsorption_energy, calculate_raw_adsorption_energy


class AdsorptionTest(unittest.TestCase):
    def test_calculates_raw_adsorption_energy(self) -> None:
        result = calculate_raw_adsorption_energy(
            ads_static=-110.0,
            slab_static=-100.0,
            mol_static=-8.0,
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.energy_ev, -2.0)
        self.assertIn("no ZPE", result.correction)

    def test_missing_energy_returns_clear_error(self) -> None:
        result = calculate_raw_adsorption_energy(
            ads_static=-110.0,
            slab_static=None,
            mol_static=None,
        )

        self.assertFalse(result.ok)
        self.assertIsNone(result.energy_ev)
        self.assertIn("slab_static", result.message)
        self.assertIn("mol_static", result.message)

    def test_compatibility_wrapper_raises_value_error(self) -> None:
        self.assertEqual(adsorption_energy(-110.0, -100.0, -8.0), -2.0)
        with self.assertRaisesRegex(ValueError, "ads_static"):
            adsorption_energy(None, -100.0, -8.0)


if __name__ == "__main__":
    unittest.main()
