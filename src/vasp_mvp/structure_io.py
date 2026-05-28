from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path

from ase.io import read, write

from .models import StructureInfo


class StructureReadError(ValueError):
    pass


def _format_from_name(filename: str) -> str:
    name = Path(filename).name.lower()
    suffix = Path(filename).suffix.lower()
    if suffix == ".cif":
        return "cif"
    if suffix == ".vasp" or name in {"poscar", "contcar"}:
        return "vasp"
    raise StructureReadError(
        f"Unsupported structure format for '{filename}'. Supported formats: CIF, POSCAR, CONTCAR, .vasp"
    )


def read_structure_upload(filename: str, data: bytes) -> StructureInfo:
    fmt = _format_from_name(filename)
    try:
        source = BytesIO(data) if fmt == "cif" else StringIO(data.decode("utf-8"))
        atoms = read(source, format=fmt)
    except UnicodeDecodeError as exc:
        raise StructureReadError(f"Could not decode '{filename}' as UTF-8 text.") from exc
    except Exception as exc:
        raise StructureReadError(f"Could not read '{filename}' as {fmt.upper()} structure: {exc}") from exc

    symbols = atoms.get_chemical_symbols()
    elements: list[str] = []
    counts: list[int] = []
    for symbol in symbols:
        if not elements or elements[-1] != symbol:
            elements.append(symbol)
            counts.append(1)
        else:
            counts[-1] += 1

    output = StringIO()
    write(output, atoms, format="vasp", vasp5=True, direct=True, sort=False)

    return StructureInfo(
        source_name=filename,
        elements=tuple(elements),
        counts=tuple(counts),
        poscar_text=output.getvalue(),
        is_periodic=bool(any(atoms.get_pbc())),
    )
