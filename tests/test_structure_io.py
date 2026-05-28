from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.structure_io import StructureReadError, read_structure_upload


POSCAR = b"""Fe O test
1.0
3.0 0.0 0.0
0.0 3.0 0.0
0.0 0.0 3.0
Fe O
1 1
Direct
0.0 0.0 0.0
0.5 0.5 0.5
"""

CIF = b"""data_test
_cell_length_a 3.0
_cell_length_b 3.0
_cell_length_c 3.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  Fe1 Fe 0.0 0.0 0.0
  O1 O 0.5 0.5 0.5
"""


class StructureIoTest(unittest.TestCase):
    def test_reads_poscar_contcar_and_vasp_as_vasp5_poscar(self) -> None:
        for filename in ("POSCAR", "CONTCAR", "cell.vasp"):
            info = read_structure_upload(filename, POSCAR)
            self.assertEqual(info.elements, ("Fe", "O"))
            self.assertEqual(info.counts, (1, 1))
            self.assertIn("Fe  O", info.poscar_text)
            self.assertIn("Direct", info.poscar_text)

    def test_reads_cif_as_vasp5_poscar(self) -> None:
        info = read_structure_upload("cell.cif", CIF)
        self.assertEqual(info.elements, ("Fe", "O"))
        self.assertEqual(info.counts, (1, 1))
        self.assertIn("Fe  O", info.poscar_text)

    def test_rejects_unsupported_format(self) -> None:
        with self.assertRaisesRegex(StructureReadError, "Supported formats"):
            read_structure_upload("cell.xyz", b"2\n\nFe 0 0 0\nO 1 1 1\n")


if __name__ == "__main__":
    unittest.main()
