from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vasp_mvp.adsorption_results import AdsorptionEnergyResult, RoleEnergySummary
from vasp_mvp.adsorption_visualization import (
    MISSING_VALUE,
    build_job_metrics_table,
    build_oszicar_steps_table,
    build_total_energy_chart_data,
    format_energy,
    format_seconds,
)
from vasp_mvp.models import JobMetricsRecord, JobRecord


class AdsorptionVisualizationTest(unittest.TestCase):
    def test_formatters_use_units_and_missing_dash(self) -> None:
        self.assertEqual(format_energy(-1.23456789), "-1.234568 eV")
        self.assertEqual(format_energy(None), MISSING_VALUE)
        self.assertEqual(format_seconds(1.23456), "1.235 s")
        self.assertEqual(format_seconds(None), MISSING_VALUE)

    def test_total_energy_chart_only_when_ready_and_all_energies_exist(self) -> None:
        ready = _result(ready=True)
        not_ready = _result(ready=False, e_clean_slab=None)

        self.assertEqual(len(build_total_energy_chart_data(ready)), 3)
        self.assertEqual(build_total_energy_chart_data(not_ready), [])

    def test_job_metrics_table_includes_status_and_warning_codes(self) -> None:
        result = _result(ready=False, e_clean_slab=None, warning_types=("missing_toten",))
        rows = build_job_metrics_table(result, {"clean_slab": _job("job-clean", "running")})

        clean = rows[0]
        self.assertEqual(clean["status"], "running")
        self.assertEqual(clean["final_toten"], MISSING_VALUE)
        self.assertEqual(clean["warning_codes"], ("missing_toten",))

    def test_oszicar_steps_table_uses_saved_metrics_only(self) -> None:
        result = _result(ready=True)
        rows = build_oszicar_steps_table(
            result,
            {
                "job-clean": _metrics("job-clean", (-1.0, -1.2)),
                "job-mol": _metrics("job-mol", (-0.2,)),
            },
        )

        self.assertEqual(
            [(row["role"], row["ionic_step"], row["energy_ev"]) for row in rows],
            [("clean_slab", 1, -1.0), ("clean_slab", 2, -1.2), ("molecule_ref", 1, -0.2)],
        )


def _result(
    *,
    ready: bool,
    e_clean_slab: float | None = -598.0,
    warning_types: tuple[str, ...] = (),
) -> AdsorptionEnergyResult:
    return AdsorptionEnergyResult(
        workflow_id="ads-1",
        clean_slab_job_id="job-clean",
        molecule_ref_job_id="job-mol",
        adsorbed_system_job_id="job-ads",
        e_clean_slab=e_clean_slab,
        e_molecule_ref=-14.8 if ready else None,
        e_adsorbed_system=-612.5 if ready else None,
        e_ads=0.3 if ready else None,
        formula_symbolic="E_ads = E_adsorbed_system - E_clean_slab - E_molecule_ref",
        formula_numeric="E_ads = -612.500000 - (-598.000000) - (-14.800000) = 0.300000 eV" if ready else None,
        energy_source="OUTCAR",
        energy_label="final TOTEN",
        method_family="DFT",
        functional="PBE-D3",
        method_notes="unit test",
        warnings=(),
        ready=ready,
        role_summaries=(
            _summary("clean_slab", "job-clean", e_clean_slab, warning_types),
            _summary("molecule_ref", "job-mol", -14.8 if ready else None, ()),
            _summary("adsorbed_system", "job-ads", -612.5 if ready else None, ()),
        ),
    )


def _summary(role: str, job_id: str, toten: float | None, warning_types: tuple[str, ...]) -> RoleEnergySummary:
    return RoleEnergySummary(
        role=role,
        job_id=job_id,
        calculation_type="static",
        outcar_exists=True,
        toten_ev=toten,
        loop_avg_seconds=2.5 if toten is not None else None,
        loop_count=2 if toten is not None else 0,
        ionic_converged=True,
        electronic_converged=None,
        energy_source="OUTCAR",
        energy_label="final TOTEN",
        warning_types=warning_types,
    )


def _job(job_id: str, status: str) -> JobRecord:
    now = datetime.utcnow()
    return JobRecord(
        job_id=job_id,
        calculation_type="static",
        status=status,
        run_dir=Path("/tmp/run"),
        input_set_id=None,
        pid=None,
        created_at=now,
        updated_at=now,
    )


def _metrics(job_id: str, steps: tuple[float, ...]) -> JobMetricsRecord:
    return JobMetricsRecord(
        job_id=job_id,
        toten_ev=steps[-1] if steps else None,
        loop_avg_seconds=2.0,
        loop_count=2,
        ionic_converged=True,
        electronic_converged=None,
        oszicar_steps=steps,
        errors=(),
        energy_source="OUTCAR",
        energy_label="final TOTEN",
        updated_at=datetime.utcnow(),
    )


if __name__ == "__main__":
    unittest.main()
