from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .jobs import get_job_metrics, parse_and_save_job_metrics
from .models import JobMetricsRecord, JobRecord, WorkflowRecord
from .workflows import get_workflow, list_jobs_for_workflow


ADSORPTION_ROLES = ("clean_slab", "molecule_ref", "adsorbed_system")
STATIC_CALCULATION_TYPES = {
    "clean_slab": {"slab_static", "static"},
    "molecule_ref": {"molecule_static", "static"},
    "adsorbed_system": {"adsorbed_static", "static"},
}


def normalize_energy_source(value: str | None) -> str:
    return "" if value is None else value.strip().lower()


def normalize_energy_label(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"[\s_-]+", "_", value.strip().lower())


def is_outcar_final_toten(source: str | None, label: str | None) -> bool:
    return normalize_energy_source(source) == "outcar" and normalize_energy_label(label) == "final_toten"


@dataclass(frozen=True)
class AdsorptionWarning:
    role: str | None
    job_id: str | None
    code: str
    message: str
    details: dict[str, Any] | None = None


@dataclass(frozen=True)
class RoleEnergySummary:
    role: str
    job_id: str | None
    calculation_type: str | None
    outcar_exists: bool
    toten_ev: float | None
    loop_avg_seconds: float | None
    loop_count: int
    ionic_converged: bool | None
    electronic_converged: bool | None
    energy_source: str | None
    energy_label: str | None
    warning: str | None = None
    warning_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class AdsorptionEnergyResult:
    workflow_id: str
    clean_slab_job_id: str | None
    molecule_ref_job_id: str | None
    adsorbed_system_job_id: str | None
    e_clean_slab: float | None
    e_molecule_ref: float | None
    e_adsorbed_system: float | None
    e_ads: float | None
    formula_symbolic: str
    formula_numeric: str | None
    energy_source: str
    energy_label: str
    method_family: str | None
    functional: str | None
    method_notes: str | None
    warnings: tuple[AdsorptionWarning, ...]
    ready: bool
    role_summaries: tuple[RoleEnergySummary, ...]


def parse_adsorption_workflow_jobs(
    db_path: Path,
    workflow_id: str,
) -> dict:
    """解析吸附能 workflow 下三个 job 的 OUTCAR/OSZICAR 并保存 job_metrics。

    本函数只复用 parser.parse_metrics 的现有能力，不读取 POTCAR、不启动任何计算。
    """

    workflow, jobs_by_role = _require_adsorption_jobs(db_path, workflow_id)
    del workflow
    results: dict[str, dict] = {}
    for role in ADSORPTION_ROLES:
        job = jobs_by_role.get(role)
        if job is None:
            results[role] = {
                "job_id": None,
                "parsed": False,
                "outcar_exists": False,
                "warning": f"{role}: workflow job is missing",
            }
            continue
        outcar_exists = (job.run_dir / "OUTCAR").exists()
        metrics = parse_and_save_job_metrics(
            db_path,
            job.job_id,
            job.run_dir,
            energy_source="OUTCAR",
            energy_label="final TOTEN",
        )
        warning = None
        if not outcar_exists or metrics.status == "OUTCAR not found":
            warning = f"{role}: OUTCAR not found; no real OUTCAR final TOTEN is available"
        elif metrics.toten_ev is None:
            warning = f"{role}: final TOTEN was not found in OUTCAR"
        results[role] = {
            "job_id": job.job_id,
            "parsed": True,
            "outcar_exists": outcar_exists,
            "toten_ev": metrics.toten_ev,
            "loop_avg_seconds": metrics.loop_avg_seconds,
            "loop_count": metrics.loop_count,
            "ionic_converged": metrics.ionic_converged,
            "electronic_converged": metrics.electronic_converged,
            "status": metrics.status,
            "warning": warning,
        }
    return results


def calculate_adsorption_energy(
    db_path: Path,
    workflow_id: str,
) -> AdsorptionEnergyResult:
    """按固定 role 公式计算吸附能。

    只接受 energy_source=OUTCAR 且 energy_label=final TOTEN 的最终总能；
    OSZICAR/manual/unknown 等来源一律不用于最终吸附能。
    """

    workflow, jobs_by_role = _require_adsorption_jobs(db_path, workflow_id)
    warnings: list[AdsorptionWarning] = []
    role_summaries: list[RoleEnergySummary] = []
    valid_energies: dict[str, float] = {}

    for role in ADSORPTION_ROLES:
        job = jobs_by_role.get(role)
        if job is None:
            warning = AdsorptionWarning(
                role=role,
                job_id=None,
                code="missing_metrics",
                message=f"{role}: workflow job is missing",
            )
            warnings.append(warning)
            role_summaries.append(_missing_role_summary(role, warning))
            continue

        metrics = get_job_metrics(db_path, job.job_id)
        role_warnings = _role_warnings(role, job, metrics)
        warnings.extend(role_warnings)
        role_summaries.append(_role_summary(role, job, metrics, role_warnings))
        if metrics and _metric_is_final_outcar_toten(metrics):
            valid_energies[role] = float(metrics.toten_ev)

    method_family = workflow.method_family or "unknown"

    ready = all(role in valid_energies for role in ADSORPTION_ROLES)
    if not ready:
        warnings.append(
            AdsorptionWarning(
                role=None,
                job_id=None,
                code="missing_required_energy",
                message="One or more OUTCAR final TOTEN values are missing, so adsorption energy cannot be calculated yet.",
            )
        )
    e_ads = None
    formula_numeric = None
    if ready:
        e_ads = (
            valid_energies["adsorbed_system"]
            - valid_energies["clean_slab"]
            - valid_energies["molecule_ref"]
        )
        formula_numeric = (
            "E_ads = "
            f"{valid_energies['adsorbed_system']:.6f} - "
            f"({valid_energies['clean_slab']:.6f}) - "
            f"({valid_energies['molecule_ref']:.6f}) = "
            f"{e_ads:.6f} eV"
        )

    return AdsorptionEnergyResult(
        workflow_id=workflow.workflow_id,
        clean_slab_job_id=_job_id(jobs_by_role, "clean_slab"),
        molecule_ref_job_id=_job_id(jobs_by_role, "molecule_ref"),
        adsorbed_system_job_id=_job_id(jobs_by_role, "adsorbed_system"),
        e_clean_slab=valid_energies.get("clean_slab"),
        e_molecule_ref=valid_energies.get("molecule_ref"),
        e_adsorbed_system=valid_energies.get("adsorbed_system"),
        e_ads=e_ads,
        formula_symbolic="E_ads = E_adsorbed_system - E_clean_slab - E_molecule_ref",
        formula_numeric=formula_numeric,
        energy_source="OUTCAR",
        energy_label="final TOTEN",
        method_family=method_family,
        functional=workflow.functional,
        method_notes=workflow.method_notes,
        warnings=tuple(warnings),
        ready=ready,
        role_summaries=tuple(role_summaries),
    )


def get_adsorption_energy_summary(
    db_path: Path,
    workflow_id: str,
) -> AdsorptionEnergyResult | None:
    if get_workflow(db_path, workflow_id) is None:
        return None
    return calculate_adsorption_energy(db_path, workflow_id)


def _require_adsorption_jobs(db_path: Path, workflow_id: str) -> tuple[WorkflowRecord, dict[str, JobRecord]]:
    workflow = get_workflow(db_path, workflow_id)
    if workflow is None:
        raise ValueError(f"Workflow not found: {workflow_id}")
    if workflow.workflow_type != "adsorption":
        raise ValueError(f"Workflow is not an adsorption workflow: {workflow_id}")
    jobs_by_role = {
        binding.role: job
        for binding, job in list_jobs_for_workflow(db_path, workflow_id)
        if binding.role in ADSORPTION_ROLES
    }
    return workflow, jobs_by_role


def _metric_is_final_outcar_toten(metrics: JobMetricsRecord) -> bool:
    return is_outcar_final_toten(metrics.energy_source, metrics.energy_label) and metrics.toten_ev is not None


def _role_warnings(role: str, job: JobRecord, metrics: JobMetricsRecord | None) -> tuple[AdsorptionWarning, ...]:
    allowed_types = STATIC_CALCULATION_TYPES.get(role, {"static"})
    warnings: list[AdsorptionWarning] = []
    outcar_exists = (Path(job.run_dir) / "OUTCAR").exists()

    if metrics is None:
        warnings.append(
            AdsorptionWarning(
                role=role,
                job_id=job.job_id,
                code="missing_metrics",
                message=f"{role}/{job.job_id}: no saved OUTCAR final TOTEN is available",
            )
        )
    elif not is_outcar_final_toten(metrics.energy_source, metrics.energy_label):
        warnings.append(
            AdsorptionWarning(
                role=role,
                job_id=job.job_id,
                code="invalid_energy_source",
                message=f"{role}/{job.job_id}: Energy source is not OUTCAR final TOTEN; "
                f"source={metrics.energy_source!r}, label={metrics.energy_label!r}.",
                details={"source": metrics.energy_source, "label": metrics.energy_label},
            )
        )
    elif metrics.toten_ev is None:
        if outcar_exists:
            warnings.append(
                AdsorptionWarning(
                    role=role,
                    job_id=job.job_id,
                    code="missing_toten",
                    message=f"{role}/{job.job_id}: OUTCAR exists, but final TOTEN was not parsed. "
                    "The calculation may be incomplete or interrupted.",
                )
            )
        else:
            warnings.append(
                AdsorptionWarning(
                    role=role,
                    job_id=job.job_id,
                    code="missing_outcar",
                    message=f"{role}/{job.job_id}: OUTCAR file is missing.",
                )
            )

    if job.calculation_type not in allowed_types:
        warnings.append(
            AdsorptionWarning(
                role=role,
                job_id=job.job_id,
                code="non_static_calculation",
                message=f"{role}/{job.job_id}: calculation_type={job.calculation_type} may not be a static single-point calculation",
                details={"calculation_type": job.calculation_type},
            )
        )

    return tuple(warnings)


def _role_summary(
    role: str,
    job: JobRecord,
    metrics: JobMetricsRecord | None,
    warnings: tuple[AdsorptionWarning, ...],
) -> RoleEnergySummary:
    return RoleEnergySummary(
        role=role,
        job_id=job.job_id,
        calculation_type=job.calculation_type,
        outcar_exists=(job.run_dir / "OUTCAR").exists(),
        toten_ev=metrics.toten_ev if metrics else None,
        loop_avg_seconds=metrics.loop_avg_seconds if metrics else None,
        loop_count=metrics.loop_count if metrics else 0,
        ionic_converged=metrics.ionic_converged if metrics else None,
        electronic_converged=metrics.electronic_converged if metrics else None,
        energy_source=metrics.energy_source if metrics else None,
        energy_label=metrics.energy_label if metrics else None,
        warning="; ".join(item.message for item in warnings) or None,
        warning_types=tuple(item.code for item in warnings),
    )


def _missing_role_summary(role: str, warning: AdsorptionWarning) -> RoleEnergySummary:
    return RoleEnergySummary(
        role=role,
        job_id=None,
        calculation_type=None,
        outcar_exists=False,
        toten_ev=None,
        loop_avg_seconds=None,
        loop_count=0,
        ionic_converged=None,
        electronic_converged=None,
        energy_source=None,
        energy_label=None,
        warning=warning.message,
        warning_types=(warning.code,),
    )


def _job_id(jobs_by_role: dict[str, JobRecord], role: str) -> str | None:
    job = jobs_by_role.get(role)
    return job.job_id if job else None
