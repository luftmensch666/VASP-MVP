from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

from .adsorption_wizard import RELAX_ROLES, artifact_path, load_wizard_state, save_wizard_state
from .input_set_validation import (
    ValidationResult,
    parse_incar_tags,
    parse_kpoints_summary,
    parse_potcar_summary,
    validate_input_set,
)
from .jobs import create_job, get_job
from .models import InputSet
from .workflows import bind_job_to_workflow, list_jobs_for_workflow


RelaxRole = Literal["clean_relax", "molecule_relax", "adsorbed_relax"]
CORE_VASP_INPUTS = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
RELAX_SOURCE_ARTIFACTS = {
    "clean_relax": "clean_poscar",
    "molecule_relax": "molecule_poscar",
    "adsorbed_relax": "adsorbed_poscar",
}
RELAX_STEP_ORDER = {"clean_relax": 1, "molecule_relax": 2, "adsorbed_relax": 3}


@dataclass(frozen=True)
class VaspkitStepResult:
    step: str
    label: str
    ok: bool
    stdout_path: Path
    stderr_path: Path
    return_code: int | None
    error: str | None = None

    def to_dict(self, workflow_root: Path) -> dict:
        return {
            "step": self.step,
            "label": self.label,
            "ok": self.ok,
            "stdout_path": _rel(self.stdout_path, workflow_root),
            "stderr_path": _rel(self.stderr_path, workflow_root),
            "return_code": self.return_code,
            "error": self.error,
        }


@dataclass(frozen=True)
class RelaxInputGenerationResult:
    ok: bool
    role: str
    input_dir: Path
    generated_files: list[str] = field(default_factory=list)
    stdout_paths: list[Path] = field(default_factory=list)
    stderr_paths: list[Path] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    steps: list[VaspkitStepResult] = field(default_factory=list)
    validation: dict | None = None
    status: str = "pending"

    def to_dict(self, workflow_root: Path) -> dict:
        return {
            "ok": self.ok,
            "role": self.role,
            "status": self.status,
            "input_dir": _rel(self.input_dir, workflow_root),
            "generated_files": list(self.generated_files),
            "stdout_paths": [_rel(path, workflow_root) for path in self.stdout_paths],
            "stderr_paths": [_rel(path, workflow_root) for path in self.stderr_paths],
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "steps": [step.to_dict(workflow_root) for step in self.steps],
            "validation": self.validation or {},
        }


def check_relax_source_poscars(workflow_root: Path, state: dict) -> tuple[bool, dict[str, str], list[str]]:
    """检查 Step 4 是否具备三套结构 artifact。

    返回值中的路径均为相对 workflow_root 的路径，便于写入 workflow_state.json。
    """

    artifacts = state.get("artifacts", {})
    sources: dict[str, str] = {}
    missing: list[str] = []
    for role, artifact_key in RELAX_SOURCE_ARTIFACTS.items():
        rel_path = artifacts.get(artifact_key)
        if not rel_path:
            missing.append(role)
            continue
        path = artifact_path(workflow_root, rel_path)
        if not path.exists() or path.stat().st_size == 0:
            missing.append(role)
            continue
        sources[role] = rel_path
    return not missing, sources, missing


def generate_all_relax_input_packages(
    workflow_root: Path,
    state: dict,
    *,
    vaspkit_bin: str,
    incar_key: str = "SR",
    kpoints_scheme: int | str = 2,
    kmesh_value: str | float = "0.04",
    timeout: int = 120,
) -> list[RelaxInputGenerationResult]:
    ready, sources, missing = check_relax_source_poscars(workflow_root, state)
    if not ready:
        raise FileNotFoundError("Missing source POSCAR artifacts: " + ", ".join(missing))

    results = [
        generate_relax_input_package(
            workflow_root,
            role=role,
            source_poscar=artifact_path(workflow_root, sources[role]),
            vaspkit_bin=vaspkit_bin,
            incar_key=incar_key,
            kpoints_mode=str(kpoints_scheme),
            kmesh_value=kmesh_value,
            gamma_only=(role == "molecule_relax"),
            timeout=timeout,
        )
        for role in RELAX_ROLES
    ]
    state = load_wizard_state(workflow_root, state.get("workflow_id", ""))
    state["relax_inputs"] = {
        result.role: _state_entry_for_result(result, workflow_root)
        for result in results
    }
    if all(result.ok for result in results):
        state["steps"]["generate_relax_inputs"] = "done" if not any(result.warnings for result in results) else "warning"
    else:
        state["steps"]["generate_relax_inputs"] = "warning"
    save_wizard_state(workflow_root, state)
    return results


def generate_relax_input_package(
    workflow_root: Path,
    role: RelaxRole,
    source_poscar: Path,
    *,
    vaspkit_bin: str,
    incar_key: str,
    kpoints_mode: str,
    kmesh_value: str | float | None,
    gamma_only: bool = False,
    timeout: int = 120,
) -> RelaxInputGenerationResult:
    """为一个 relax role 生成 workflow-local VASP 输入包。

    先在 inputs/<role>/.staging_<timestamp>/ 中生成。只有四件套完整且校验无 error
    时才替换正式 inputs/<role>/，避免重新生成失败破坏旧输入包。
    """

    workflow_root = Path(workflow_root)
    source_poscar = Path(source_poscar)
    if role not in RELAX_ROLES:
        raise ValueError(f"Unsupported relax role: {role}")
    if not source_poscar.exists() or source_poscar.stat().st_size == 0:
        raise FileNotFoundError(f"Source POSCAR is missing or empty: {source_poscar}")

    inputs_root = workflow_root / "inputs"
    input_dir = inputs_root / role
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    staging = input_dir / f".staging_{timestamp}"
    logs_dir = workflow_root / "logs"
    staging.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_poscar, staging / "POSCAR")

    request = {
        "role": role,
        "source_poscar": _rel(source_poscar, workflow_root),
        "incar_key": incar_key,
        "kpoints_mode": kpoints_mode,
        "kmesh_value": kmesh_value,
        "gamma_only": gamma_only,
        "created_at": _now(),
    }
    _write_json(staging / "generation_request.json", request)

    steps: list[VaspkitStepResult] = []
    warnings: list[str] = []
    errors: list[str] = []
    generated_files = ["POSCAR"]
    stdout_paths: list[Path] = []
    stderr_paths: list[Path] = []

    try:
        step = _run_vaspkit_input_step(
            vaspkit_bin,
            ["101", str(incar_key).strip() or "SR"],
            staging,
            logs_dir,
            role,
            "101",
            "INCAR",
            "INCAR",
            timeout,
        )
        steps.append(step)
        stdout_paths.append(step.stdout_path)
        stderr_paths.append(step.stderr_path)
        if not step.ok:
            errors.append(step.error or "VASPKIT 101 INCAR failed")
            return _finish_failed_generation(workflow_root, input_dir, staging, role, steps, stdout_paths, stderr_paths, warnings, errors, generated_files)
        generated_files.append("INCAR")

        if gamma_only:
            _write_gamma_only_kpoints(staging / "KPOINTS")
            warnings.append("molecule_gamma_only")
            generated_files.append("KPOINTS")
        else:
            step = _run_vaspkit_input_step(
                vaspkit_bin,
                ["102", str(kpoints_mode), str(kmesh_value if kmesh_value is not None else "0.04")],
                staging,
                logs_dir,
                role,
                "102",
                "KPOINTS",
                "KPOINTS",
                timeout,
            )
            steps.append(step)
            stdout_paths.append(step.stdout_path)
            stderr_paths.append(step.stderr_path)
            if not step.ok:
                errors.append(step.error or "VASPKIT 102 KPOINTS failed")
                return _finish_failed_generation(workflow_root, input_dir, staging, role, steps, stdout_paths, stderr_paths, warnings, errors, generated_files)
            generated_files.append("KPOINTS")

        step = _run_vaspkit_input_step(
            vaspkit_bin,
            ["103"],
            staging,
            logs_dir,
            role,
            "103",
            "POTCAR",
            "POTCAR",
            timeout,
        )
        steps.append(step)
        stdout_paths.append(step.stdout_path)
        stderr_paths.append(step.stderr_path)
        if not step.ok:
            errors.append(step.error or "VASPKIT 103 POTCAR failed")
            return _finish_failed_generation(workflow_root, input_dir, staging, role, steps, stdout_paths, stderr_paths, warnings, errors, generated_files)
        generated_files.append("POTCAR")

        missing = _missing_core_inputs(staging)
        if missing:
            errors.append("Missing generated files: " + ", ".join(missing))
            return _finish_failed_generation(workflow_root, input_dir, staging, role, steps, stdout_paths, stderr_paths, warnings, errors, generated_files)

        validation = validate_vasp_input_dir(staging, role=role, calculation_stage="relax")
        validation_dict = validation.to_dict()
        warnings.extend(_issue_codes(validation.warnings))
        if validation.errors:
            errors.extend(_issue_codes(validation.errors))
            return _finish_failed_generation(
                workflow_root,
                input_dir,
                staging,
                role,
                steps,
                stdout_paths,
                stderr_paths,
                warnings,
                errors,
                generated_files,
                validation=validation_dict,
                status="invalid",
            )

        _write_generation_result(
            staging / "generation_result.json",
            workflow_root,
            role,
            "generated",
            input_dir,
            steps,
            generated_files,
            validation_dict,
            warnings,
            errors,
            created_at=request["created_at"],
        )
        _safe_replace_input_dir(input_dir, staging)
        return RelaxInputGenerationResult(
            ok=True,
            role=role,
            input_dir=input_dir,
            generated_files=list(generated_files),
            stdout_paths=stdout_paths,
            stderr_paths=stderr_paths,
            warnings=warnings,
            errors=[],
            steps=steps,
            validation=validation_dict,
            status="generated" if not warnings else "generated_with_warnings",
        )
    except Exception as exc:
        errors.append(str(exc))
        return _finish_failed_generation(workflow_root, input_dir, staging, role, steps, stdout_paths, stderr_paths, warnings, errors, generated_files)


def validate_vasp_input_dir(
    input_dir: Path,
    *,
    role: str,
    calculation_stage: str = "relax",
    method_family: str | None = None,
    functional: str | None = None,
    method_notes: str | None = None,
) -> ValidationResult:
    """复用 InputSet 校验逻辑校验 workflow-local 输入目录。

    relax 阶段允许 NSW > 0，因此会过滤 static_recommended warning。
    """

    path = Path(input_dir)
    pseudo = InputSet(
        input_set_id=f"workflow-local-{role}",
        name=f"workflow-local-{role}",
        source="manual",
        status="generated",
        usable_for_vasp=True,
        root_dir=path,
        incar_path=path / "INCAR",
        poscar_path=path / "POSCAR",
        kpoints_path=path / "KPOINTS",
        potcar_path=path / "POTCAR",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        notes="workflow-local validation",
    )
    mapped_role = _validation_role_for_relax(role)
    result = validate_input_set(
        pseudo,
        role=mapped_role,
        method_family=method_family,
        functional=functional,
        method_notes=method_notes,
    )
    if calculation_stage == "relax":
        warnings = tuple(issue for issue in result.warnings if issue.code != "static_recommended")
        return ValidationResult(result.ok, result.errors, warnings, result.infos)
    return result


def check_relax_package_consistency(workflow_root: Path) -> list[str]:
    warnings: list[str] = []
    clean = _input_dir_summary(Path(workflow_root) / "inputs" / "clean_relax")
    adsorbed = _input_dir_summary(Path(workflow_root) / "inputs" / "adsorbed_relax")
    molecule = _input_dir_summary(Path(workflow_root) / "inputs" / "molecule_relax")
    for key in ("incar_key", "encut", "ivdw", "ldau", "ispin"):
        if clean.get(key) != adsorbed.get(key):
            warnings.append(f"clean_relax and adsorbed_relax differ in {key}")
    if clean.get("kpoints_scheme") != adsorbed.get("kpoints_scheme"):
        warnings.append("clean_relax and adsorbed_relax differ in KPOINTS scheme")
    if clean.get("kpoints_grid") != adsorbed.get("kpoints_grid"):
        warnings.append("clean_relax and adsorbed_relax differ in KPOINTS grid")
    if clean.get("potcar_family") != adsorbed.get("potcar_family"):
        warnings.append("clean_relax and adsorbed_relax differ in POTCAR family")
    for key in ("encut", "ivdw", "ldau", "ispin", "potcar_family"):
        if clean.get(key) != molecule.get(key):
            warnings.append(f"molecule_relax differs from clean_relax in {key}")
    return warnings


def create_relax_jobs_from_inputs(
    db_path: Path,
    workflow_id: str,
    workflow_root: Path,
    *,
    mpi_ranks: int | None = None,
    vasp_bin: str | Path | None = None,
) -> list[dict]:
    """根据 inputs/<role>/ 创建 committed relax jobs。

    已存在 job 时不覆盖 run_dir、不重复绑定，返回 exists 状态。
    """

    results: list[dict] = []
    existing = {binding.role: job for binding, job in list_jobs_for_workflow(db_path, workflow_id)}
    for role in RELAX_ROLES:
        input_dir = Path(workflow_root) / "inputs" / role
        missing = _missing_core_inputs(input_dir)
        if missing:
            results.append({"role": role, "status": "missing_inputs", "errors": missing})
            continue
        if role in existing:
            results.append({"role": role, "status": "exists", "job_id": existing[role].job_id, "run_dir": str(existing[role].run_dir)})
            continue
        job_id = f"{workflow_id}_{role}"
        if get_job(db_path, job_id) is not None:
            results.append({"role": role, "status": "exists", "job_id": job_id})
            continue
        run_dir = Path(workflow_root) / "jobs" / role / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        for filename in CORE_VASP_INPUTS:
            shutil.copy2(input_dir / filename, run_dir / filename)
        job = create_job(
            db_path,
            job_id=job_id,
            calculation_type="molecule_relax" if role == "molecule_relax" else "relax",
            status="committed",
            run_dir=run_dir,
            input_set_id=None,
            mpi_ranks=mpi_ranks,
            vasp_bin=vasp_bin,
            name=role,
            notes="Created by Adsorption Wizard Step 5",
        )
        bind_job_to_workflow(
            db_path,
            workflow_id=workflow_id,
            job_id=job.job_id,
            role=role,
            step_order=RELAX_STEP_ORDER[role],
            required=True,
        )
        results.append({"role": role, "status": "created", "job_id": job.job_id, "run_dir": str(run_dir)})
    return results


def _run_vaspkit_input_step(
    vaspkit_bin: str,
    inputs: list[str],
    cwd: Path,
    logs_dir: Path,
    role: str,
    step: str,
    label: str,
    expected_output: str,
    timeout: int,
) -> VaspkitStepResult:
    stdout_path = logs_dir / f"relax_input_{role}.out"
    stderr_path = logs_dir / f"relax_input_{role}.err"
    header = f"===== VASPKIT {step} {label} =====\n"
    _append_text(stdout_path, header)
    _append_text(stderr_path, header)
    try:
        completed = subprocess.run(
            [vaspkit_bin],
            input="\n".join(inputs) + "\n",
            text=True,
            cwd=Path(cwd),
            capture_output=True,
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        _append_text(stdout_path, _to_text(exc.stdout))
        _append_text(stderr_path, _to_text(exc.stderr))
        return VaspkitStepResult(step, label, False, stdout_path, stderr_path, None, f"VASPKIT {step} timed out")
    _append_text(stdout_path, completed.stdout or "")
    _append_text(stderr_path, completed.stderr or "")
    output_path = Path(cwd) / expected_output
    if completed.returncode != 0:
        return VaspkitStepResult(step, label, False, stdout_path, stderr_path, completed.returncode, f"VASPKIT {step} exited with return code {completed.returncode}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        return VaspkitStepResult(step, label, False, stdout_path, stderr_path, completed.returncode, f"VASPKIT {step} did not generate {expected_output}")
    return VaspkitStepResult(step, label, True, stdout_path, stderr_path, completed.returncode)


def _finish_failed_generation(
    workflow_root: Path,
    input_dir: Path,
    staging: Path,
    role: str,
    steps: list[VaspkitStepResult],
    stdout_paths: list[Path],
    stderr_paths: list[Path],
    warnings: list[str],
    errors: list[str],
    generated_files: list[str],
    *,
    validation: dict | None = None,
    status: str = "failed",
) -> RelaxInputGenerationResult:
    result = RelaxInputGenerationResult(
        ok=False,
        role=role,
        input_dir=input_dir,
        generated_files=generated_files,
        stdout_paths=stdout_paths,
        stderr_paths=stderr_paths,
        warnings=warnings,
        errors=errors,
        steps=steps,
        validation=validation,
        status=status,
    )
    failure_path = Path(workflow_root) / "inputs" / f"{role}_generation_result.json"
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    _write_generation_result(
        failure_path,
        workflow_root,
        role,
        status,
        input_dir,
        steps,
        generated_files,
        validation or {},
        warnings,
        errors,
        created_at=_now(),
    )
    if staging.exists():
        shutil.rmtree(staging)
    return result


def _write_generation_result(
    path: Path,
    workflow_root: Path,
    role: str,
    status: str,
    input_dir: Path,
    steps: list[VaspkitStepResult],
    generated_files: list[str],
    validation: dict,
    warnings: list[str],
    errors: list[str],
    *,
    created_at: str,
) -> None:
    _write_json(
        path,
        {
            "role": role,
            "status": status,
            "input_dir": _rel(input_dir, workflow_root),
            "steps": [step.to_dict(workflow_root) for step in steps],
            "generated_files": generated_files,
            "validation": validation,
            "warnings": warnings,
            "errors": errors,
            "created_at": created_at,
            "updated_at": _now(),
        },
    )


def _safe_replace_input_dir(input_dir: Path, staging: Path) -> None:
    detached = input_dir.parent / f".{input_dir.name}_ready_{datetime.utcnow():%Y%m%d_%H%M%S_%f}"
    staging.rename(detached)
    if input_dir.exists():
        backup = input_dir.with_name(f"{input_dir.name}.backup_{datetime.utcnow():%Y%m%d_%H%M%S_%f}")
        input_dir.rename(backup)
    detached.rename(input_dir)


def _write_gamma_only_kpoints(path: Path) -> None:
    path.write_text("Gamma-only\n0\nGamma\n1 1 1\n0 0 0\n", encoding="utf-8")


def _missing_core_inputs(directory: Path) -> list[str]:
    return [
        filename
        for filename in CORE_VASP_INPUTS
        if not (Path(directory) / filename).exists() or (Path(directory) / filename).stat().st_size == 0
    ]


def _validation_role_for_relax(role: str) -> str:
    if role == "molecule_relax":
        return "molecule_ref"
    if role == "adsorbed_relax":
        return "adsorbed_system"
    return "clean_slab"


def _input_dir_summary(input_dir: Path) -> dict:
    input_dir = Path(input_dir)
    if not input_dir.exists():
        return {}
    incar = parse_incar_tags(input_dir / "INCAR") if (input_dir / "INCAR").exists() else {}
    kpoints = parse_kpoints_summary(input_dir / "KPOINTS") if (input_dir / "KPOINTS").exists() else {}
    potcar = parse_potcar_summary(input_dir / "POTCAR")
    return {
        "incar_key": incar.get("SYSTEM") or "",
        "encut": incar.get("ENCUT"),
        "ivdw": incar.get("IVDW"),
        "ldau": incar.get("LDAU"),
        "ispin": incar.get("ISPIN"),
        "kpoints_scheme": kpoints.get("scheme"),
        "kpoints_grid": kpoints.get("grid"),
        "potcar_family": tuple(item.paw_family for item in potcar.potentials if item.paw_family),
    }


def _state_entry_for_result(result: RelaxInputGenerationResult, workflow_root: Path) -> dict:
    return {
        "status": result.status,
        "input_dir": _rel(result.input_dir, workflow_root),
        "validation": result.validation or {},
        "warnings": list(result.warnings),
        "errors": list(result.errors),
        "generated_files": list(result.generated_files),
    }


def _issue_codes(issues) -> list[str]:
    return [getattr(issue, "code", str(issue)) for issue in issues]


def _append_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(text)
        if text and not text.endswith("\n"):
            fh.write("\n")


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _rel(path: Path, workflow_root: Path) -> str:
    try:
        return str(Path(path).relative_to(workflow_root))
    except ValueError:
        return str(path)


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds")


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
