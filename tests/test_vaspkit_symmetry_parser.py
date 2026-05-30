from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.vaspkit_symmetry_parser import analyze_porous_symmetry_summary, infer_expected_guest_elements, parse_vaspkit_601_summary


SAMPLE_601 = """Prototype: ABC4
Total Atoms: 84
Formula Unit: PO4Ce [ Alphabetically Listed: CeO4P ]
Full Formula Unit: P14O56Ce14
Crystal System: Triclinic
Crystal Class: 1
Bravais Lattice: aP
Lattice Constants: 13.260 13.902 25.000
Lattice Angles: 90.000 90.000 90.000
Volume: 4608.614
Density (g/cm3): 1.186
Space Group: 1
Point Group: 1 [ C1 ]
International: P1
Symmetry Operations: 2
Symmetry Accuracy: 0.1E-04
"""


class VaspkitSymmetryParserTest(unittest.TestCase):
    def test_parse_vaspkit_601_summary(self) -> None:
        summary = parse_vaspkit_601_summary(SAMPLE_601)

        self.assertEqual(summary.space_group_number, 1)
        self.assertEqual(summary.international_symbol, "P1")
        self.assertEqual(summary.crystal_system, "Triclinic")
        self.assertEqual(summary.bravais_lattice, "aP")
        self.assertEqual(summary.symmetry_operations, 2)
        self.assertEqual(summary.lattice_constants, (13.260, 13.902, 25.000))
        self.assertEqual(summary.lattice_angles, (90.000, 90.000, 90.000))
        self.assertEqual(summary.total_atoms, 84)
        self.assertEqual(summary.formula_unit, "PO4Ce")
        self.assertEqual(summary.full_formula_unit, "P14O56Ce14")
        self.assertEqual(summary.volume, 4608.614)
        self.assertEqual(summary.density, 1.186)

    def test_low_symmetry_and_603_recommendation_are_heuristic(self) -> None:
        summary = parse_vaspkit_601_summary(SAMPLE_601)
        recs = analyze_porous_symmetry_summary(summary)
        codes = {rec.code for rec in recs}

        self.assertIn("low_symmetry", codes)
        self.assertIn("few_symmetry_operations", codes)
        self.assertIn("recommend_603", codes)
        self.assertIn("guest_or_missing_atoms_uncertain", codes)

    def test_expected_guest_elements_present_warning_is_not_absolute(self) -> None:
        summary = parse_vaspkit_601_summary(SAMPLE_601)
        recs = analyze_porous_symmetry_summary(summary, expected_guest_elements=["N", "H"], structure_elements=["Ce", "O", "N", "H"], adsorbate_name="NH3")
        codes = {rec.code for rec in recs}

        self.assertIn("expected_guest_elements_present", codes)
        self.assertEqual(infer_expected_guest_elements("NH3"), ["N", "H"])

    def test_supercell_recommendation_by_cell_length(self) -> None:
        summary = parse_vaspkit_601_summary(SAMPLE_601)
        small = analyze_porous_symmetry_summary(summary, cell_lengths=(8.0, 12.0, 20.0))
        large = analyze_porous_symmetry_summary(summary, cell_lengths=(15.0, 16.0, 20.0))

        self.assertIn("supercell_small_cell", {rec.code for rec in small})
        self.assertIn("supercell_large_cell", {rec.code for rec in large})


if __name__ == "__main__":
    unittest.main()
