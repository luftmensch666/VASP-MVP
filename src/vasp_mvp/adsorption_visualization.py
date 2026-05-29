from __future__ import annotations

from .adsorption_results import AdsorptionEnergyResult
from .models import JobMetricsRecord, JobRecord


MISSING_VALUE = "—"


def format_energy(value: float | None) -> str:
    return MISSING_VALUE if value is None else f"{float(value):.6f} eV"


def format_seconds(value: float | None) -> str:
    return MISSING_VALUE if value is None else f"{float(value):.3f} s"


def format_count(value: int | None) -> str:
    return MISSING_VALUE if value is None else str(int(value))


def build_adsorption_summary_rows(result: AdsorptionEnergyResult) -> list[dict]:
    return [
        {"field": "E_clean_slab", "value": format_energy(result.e_clean_slab)},
        {"field": "E_molecule_ref", "value": format_energy(result.e_molecule_ref)},
        {"field": "E_adsorbed_system", "value": format_energy(result.e_adsorbed_system)},
        {"field": "E_ads", "value": format_energy(result.e_ads) if result.ready else MISSING_VALUE},
        {"field": "energy_source", "value": f"{result.energy_source} {result.energy_label}"},
        {"field": "method_family", "value": result.method_family or MISSING_VALUE},
        {"field": "functional", "value": result.functional or MISSING_VALUE},
        {"field": "method_notes", "value": result.method_notes or MISSING_VALUE},
    ]


def build_job_metrics_table(
    result: AdsorptionEnergyResult,
    jobs_by_role: dict[str, JobRecord],
) -> list[dict]:
    rows: list[dict] = []
    for summary in result.role_summaries:
        job = jobs_by_role.get(summary.role)
        rows.append(
            {
                "role": summary.role,
                "job_id": summary.job_id or MISSING_VALUE,
                "calculation_type": summary.calculation_type or MISSING_VALUE,
                "status": job.status if job else MISSING_VALUE,
                "outcar_exists": summary.outcar_exists,
                "final_toten": format_energy(summary.toten_ev),
                "loop_avg": format_seconds(summary.loop_avg_seconds),
                "loop_count": format_count(summary.loop_count),
                "ionic_converged": summary.ionic_converged,
                "electronic_converged": summary.electronic_converged,
                "warning_codes": tuple(summary.warning_types),
            }
        )
    return rows


def build_total_energy_chart_data(result: AdsorptionEnergyResult) -> list[dict]:
    if not result.ready:
        return []
    energies = {
        "clean_slab": result.e_clean_slab,
        "molecule_ref": result.e_molecule_ref,
        "adsorbed_system": result.e_adsorbed_system,
    }
    if any(value is None for value in energies.values()):
        return []
    return [{"role": role, "energy_ev": float(value)} for role, value in energies.items()]


def build_oszicar_steps_table(
    result: AdsorptionEnergyResult,
    metrics_by_job_id: dict[str, JobMetricsRecord],
) -> list[dict]:
    rows: list[dict] = []
    for summary in result.role_summaries:
        if summary.job_id is None:
            continue
        metrics = metrics_by_job_id.get(summary.job_id)
        if metrics is None:
            continue
        for index, energy in enumerate(metrics.oszicar_steps, start=1):
            rows.append(
                {
                    "role": summary.role,
                    "job_id": summary.job_id,
                    "ionic_step": index,
                    "energy_ev": float(energy),
                }
            )
    return rows


def build_loop_time_summary(result: AdsorptionEnergyResult) -> list[dict]:
    rows: list[dict] = []
    for summary in result.role_summaries:
        rows.append(
            {
                "role": summary.role,
                "job_id": summary.job_id or MISSING_VALUE,
                "loop_avg": format_seconds(summary.loop_avg_seconds),
                "loop_count": format_count(summary.loop_count),
            }
        )
    return rows
