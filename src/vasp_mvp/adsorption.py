from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdsorptionResult:
    ok: bool
    energy_ev: float | None
    message: str
    correction: str = "raw adsorption energy only; no ZPE, entropy, or solvent corrections"


def calculate_raw_adsorption_energy(
    ads_static: float | None,
    slab_static: float | None,
    mol_static: float | None,
) -> AdsorptionResult:
    missing = [
        name
        for name, value in (
            ("ads_static", ads_static),
            ("slab_static", slab_static),
            ("mol_static", mol_static),
        )
        if value is None
    ]
    if missing:
        return AdsorptionResult(
            ok=False,
            energy_ev=None,
            message="Missing TOTEN for adsorption energy: " + ", ".join(missing),
        )

    energy = float(ads_static) - float(slab_static) - float(mol_static)
    return AdsorptionResult(ok=True, energy_ev=energy, message="ok")


def adsorption_energy(
    ads_static: float | None,
    slab_static: float | None,
    mol_static: float | None,
) -> float:
    result = calculate_raw_adsorption_energy(ads_static, slab_static, mol_static)
    if not result.ok:
        raise ValueError(result.message)
    return float(result.energy_ev)
