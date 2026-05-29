from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from .input_set_validation import parse_potcar_summary


# dry-run 只用于 UI 和流程联调，不能生成、展示或伪造真实 POTCAR。
DRY_RUN_POTCAR_WARNING = (
    "Dry-run mode is enabled. A real POTCAR is not generated, so this input set cannot be used for "
    "VASP calculations. Disable dry-run and run real VASPKIT generation to create a usable input set."
)

CORE_INPUT_FILES = ("POSCAR", "INCAR", "KPOINTS", "POTCAR")


@dataclass(frozen=True)
class VaspkitStep:
    name: str
    task_id: str
    inputs: list[str]
    expected_file: str


@dataclass(frozen=True)
class VaspkitRequest:
    vaspkit_bin: str
    draft_dir: Path
    workspace: Path | None = None
    input_set_id: str | None = None
    input_set_name: str = ""
    input_set_notes: str = ""
    uploaded_cif_path: Path | None = None
    # 兼容旧版 vaspkit_request.json；新逻辑忽略该字段，始终完整生成四件套。
    generation_mode: str = "full"
    element_order_mode: str = "default"
    custom_element_order: str = ""
    incar_key_parameters: list[str] = field(default_factory=lambda: ["SR"])
    incar_custom_key_string: str = ""
    kmesh_scheme: str = "1"
    kmesh_resolved_value: float = 0.04
    potcar_mode: str = "103"
    existing_potcar_policy: str = "skip"


@dataclass(frozen=True)
class VaspkitResult:
    ok: bool
    dry_run: bool
    return_code: int | None
    draft_dir: Path
    generated_files: dict[str, Path]
    warnings: list[str]
    errors: list[str]
    stdout_path: Path
    stderr_path: Path
    request_path: Path
    result_path: Path
    potcar_summary: dict | None = None
    input_set_id: str | None = None
    input_set_status: str | None = None
    usable_for_vasp: bool = False
    input_set_path: Path | None = None
    validation_path: Path | None = None
    file_hashes_path: Path | None = None
    generation_failure_path: Path | None = None


def check_vaspkit_available(vaspkit_bin: str) -> bool:
    candidate = Path(vaspkit_bin).expanduser()
    if candidate.is_file():
        return candidate.exists() and candidate.stat().st_mode & 0o111 != 0
    return shutil.which(vaspkit_bin) is not None


def build_vaspkit_inputs(request: VaspkitRequest) -> list[str]:
    if request.potcar_mode == "104":
        raise NotImplementedError("VASPKIT POTCAR mode 104 is reserved and not implemented yet.")

    inputs: list[str] = []
    for step in build_vaspkit_steps(request):
        inputs.extend(step.inputs)
    return inputs


def build_vaspkit_steps(request: VaspkitRequest) -> list[VaspkitStep]:
    """构建完整四件套的 VASPKIT 分步输入。

    这里不再根据 generation_mode 分支。保留 generation_mode 只是为了兼容旧 JSON，
    避免历史草稿反序列化时崩溃。
    """

    if request.potcar_mode == "104":
        raise NotImplementedError("VASPKIT POTCAR mode 104 is reserved and not implemented yet.")
    if request.uploaded_cif_path is None:
        raise ValueError("uploaded_cif_path is required for full VASP input set generation.")
    cif_filename = request.uploaded_cif_path.name
    return [
        VaspkitStep(
            name="POSCAR generation failed at VASPKIT 105",
            task_id="105",
            inputs=["1", "105", cif_filename, request.custom_element_order if request.element_order_mode == "custom" else ""],
            expected_file="POSCAR",
        ),
        VaspkitStep(
            name="INCAR generation failed at VASPKIT 101",
            task_id="101",
            inputs=["1", "101", _incar_key_string(request)],
            expected_file="INCAR",
        ),
        VaspkitStep(
            name="KPOINTS generation failed at VASPKIT 102",
            task_id="102",
            inputs=["1", "102", str(request.kmesh_scheme), str(request.kmesh_resolved_value)],
            expected_file="KPOINTS",
        ),
        VaspkitStep(
            name="POTCAR generation failed at VASPKIT 103",
            task_id="103",
            inputs=["1", "103"],
            expected_file="POTCAR",
        ),
    ]


def run_vaspkit_interactive(
    vaspkit_bin: str,
    inputs: list[str],
    cwd: Path,
    timeout: int = 120,
) -> VaspkitResult:
    cwd.mkdir(parents=True, exist_ok=True)
    stdout_path = cwd / "vaspkit.out"
    stderr_path = cwd / "vaspkit.err"
    request_path = cwd / "vaspkit_request.json"
    result_path = cwd / "vaspkit_result.json"
    completed = subprocess.run(
        [vaspkit_bin],
        input="\n".join(inputs) + "\n",
        text=True,
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        shell=False,
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    stderr_path.write_text(completed.stderr, encoding="utf-8")
    result = VaspkitResult(
        ok=completed.returncode == 0,
        dry_run=False,
        return_code=completed.returncode,
        draft_dir=cwd,
        generated_files=_collect_generated_files(cwd),
        warnings=[],
        errors=[] if completed.returncode == 0 else [f"VASPKIT exited with return code {completed.returncode}"],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        request_path=request_path,
        result_path=result_path,
        potcar_summary=summarize_potcar(cwd / "POTCAR"),
    )
    _write_result_json(result)
    return result


def generate_vasp_inputs_with_vaspkit(request: VaspkitRequest, dry_run: bool = True) -> VaspkitResult:
    draft_dir = request.draft_dir
    draft_dir.mkdir(parents=True, exist_ok=True)
    request_path = draft_dir / "vaspkit_request.json"
    result_path = draft_dir / "vaspkit_result.json"
    stdout_path = draft_dir / "vaspkit.out"
    stderr_path = draft_dir / "vaspkit.err"
    _write_request_json(request, request_path)

    warnings: list[str] = []
    errors: list[str] = []
    try:
        inputs = build_vaspkit_inputs(request)
    except NotImplementedError as exc:
        result = VaspkitResult(
            ok=False,
            dry_run=dry_run,
            return_code=None,
            draft_dir=draft_dir,
            generated_files={},
            warnings=[],
            errors=[str(exc)],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            request_path=request_path,
            result_path=result_path,
        )
        return _finalize_input_set_artifacts(request, result)

    if dry_run:
        stdout_path.write_text("DRY RUN: VASPKIT was not started.\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        _write_dry_run_files(request, inputs)
        result = VaspkitResult(
            ok=True,
            dry_run=True,
            return_code=0,
            draft_dir=draft_dir,
            generated_files=_collect_generated_files(draft_dir),
            warnings=[],
            errors=[],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            request_path=request_path,
            result_path=result_path,
            potcar_summary=summarize_potcar(draft_dir / "POTCAR"),
        )
        return _finalize_input_set_artifacts(request, result)

    if not check_vaspkit_available(request.vaspkit_bin):
        errors.append(f"VASPKIT executable is not available: {request.vaspkit_bin}")
    if request.uploaded_cif_path is not None:
        target_cif = draft_dir / request.uploaded_cif_path.name
        if request.uploaded_cif_path.resolve() != target_cif.resolve():
            shutil.copy2(request.uploaded_cif_path, target_cif)
    potcar_error = _apply_existing_potcar_policy(draft_dir, request.existing_potcar_policy, warnings)
    if potcar_error:
        errors.append(potcar_error)
    if errors:
        result = VaspkitResult(
            ok=False,
            dry_run=False,
            return_code=None,
            draft_dir=draft_dir,
            generated_files=_collect_generated_files(draft_dir),
            warnings=warnings,
            errors=errors,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            request_path=request_path,
            result_path=result_path,
            potcar_summary=summarize_potcar(draft_dir / "POTCAR"),
        )
        return _finalize_input_set_artifacts(request, result)

    final = _generate_full_vasp_input_set(request, warnings)
    return _finalize_input_set_artifacts(request, final)


def _generate_full_vasp_input_set(request: VaspkitRequest, warnings: list[str]) -> VaspkitResult:
    """真实 VASPKIT 模式：按 105/101/102/103 分步执行，并在每步后检查输出文件。

    失败后只回滚当前生成目录中的四个核心输入文件，保留日志和 JSON 供排错。
    """

    draft_dir = request.draft_dir
    stdout_path = draft_dir / "vaspkit.out"
    stderr_path = draft_dir / "vaspkit.err"
    request_path = draft_dir / "vaspkit_request.json"
    result_path = draft_dir / "vaspkit_result.json"
    stdout_path.write_text("", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")

    last_return_code: int | None = None
    for step in build_vaspkit_steps(request):
        if step.expected_file == "POTCAR" and (draft_dir / "POTCAR").exists():
            warnings.append("skipped_existing_potcar")
            continue
        step_result = _run_vaspkit_step(request.vaspkit_bin, step, draft_dir)
        last_return_code = step_result.returncode
        missing = _missing_or_empty_files(draft_dir, (step.expected_file,))
        if step_result.returncode != 0 or missing:
            failure = _write_generation_failure(draft_dir, step, missing, stdout_path, stderr_path, step_result.returncode)
            rollback_generated_core_files(draft_dir, _workspace_for_rollback(request))
            error = _format_generation_failure_error(failure)
            result = VaspkitResult(
                ok=False,
                dry_run=False,
                return_code=step_result.returncode,
                draft_dir=draft_dir,
                generated_files=_collect_generated_files(draft_dir),
                warnings=warnings,
                errors=[error],
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                request_path=request_path,
                result_path=result_path,
                potcar_summary=summarize_potcar(draft_dir / "POTCAR"),
                generation_failure_path=draft_dir / "generation_failure.json",
            )
            return result

    missing = _missing_or_empty_files(draft_dir, CORE_INPUT_FILES)
    if missing:
        failure = _write_generation_failure(
            draft_dir,
            VaspkitStep("Final VASP input set validation failed", "final", [], "POSCAR"),
            missing,
            stdout_path,
            stderr_path,
            last_return_code,
        )
        rollback_generated_core_files(draft_dir, _workspace_for_rollback(request))
        return VaspkitResult(
            ok=False,
            dry_run=False,
            return_code=last_return_code,
            draft_dir=draft_dir,
            generated_files=_collect_generated_files(draft_dir),
            warnings=warnings,
            errors=[_format_generation_failure_error(failure)],
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            request_path=request_path,
            result_path=result_path,
            potcar_summary=summarize_potcar(draft_dir / "POTCAR"),
            generation_failure_path=draft_dir / "generation_failure.json",
        )

    return VaspkitResult(
        ok=True,
        dry_run=False,
        return_code=last_return_code if last_return_code is not None else 0,
        draft_dir=draft_dir,
        generated_files=_collect_generated_files(draft_dir),
        warnings=warnings,
        errors=[],
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        request_path=request_path,
        result_path=result_path,
        potcar_summary=summarize_potcar(draft_dir / "POTCAR"),
    )


def summarize_potcar(path: Path) -> dict:
    summary = parse_potcar_summary(path).to_dict()
    # 兼容旧 UI/JSON 字段名，同时新增更完整的 POTCAR 摘要字段。
    summary["potential_order"] = summary.get("potential_labels", [])
    return summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _incar_key_string(request: VaspkitRequest) -> str:
    if request.incar_custom_key_string.strip():
        return request.incar_custom_key_string.strip().upper()
    keys = [key.strip().upper() for key in request.incar_key_parameters if key.strip()]
    return "".join(keys) if keys else "SR"


def _write_dry_run_files(request: VaspkitRequest, inputs: list[str]) -> None:
    draft_dir = request.draft_dir
    # dry-run 固定模拟完整输入文件组，但绝不生成真实 POTCAR。
    (draft_dir / "POSCAR").write_text(
        "Dry-run POSCAR generated by VASPKIT wrapper\n"
        "1.0\n"
        "1 0 0\n"
        "0 1 0\n"
        "0 0 1\n"
        "H\n"
        "1\n"
        "Direct\n"
        "0 0 0\n",
        encoding="utf-8",
    )
    (draft_dir / "INCAR").write_text(f"SYSTEM = dry-run\n# VASPKIT keys: {_incar_key_string(request)}\n", encoding="utf-8")
    (draft_dir / "KPOINTS").write_text(
        "Dry-run KPOINTS\n"
        "0\n"
        "Gamma\n"
        "1 1 1\n"
        "0 0 0\n",
        encoding="utf-8",
    )
    (draft_dir / "POTCAR.placeholder").write_text(
        DRY_RUN_POTCAR_WARNING + "\n",
        encoding="utf-8",
    )
    _write_json(draft_dir / "potcar_summary.json", summarize_potcar(draft_dir / "POTCAR"))
    (draft_dir / "vaspkit_inputs.txt").write_text("\n".join(inputs) + "\n", encoding="utf-8")


def _run_vaspkit_step(vaspkit_bin: str, step: VaspkitStep, cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """运行单个 VASPKIT 菜单步骤，并把 stdout/stderr 追加到统一日志。

    使用参数列表调用并禁用 shell；每一步都有 header，便于用户排查失败发生在哪个菜单。
    """

    input_text = "\n".join(step.inputs) + "\n"
    completed = subprocess.run(
        [vaspkit_bin],
        input=input_text,
        text=True,
        cwd=cwd,
        capture_output=True,
        timeout=timeout,
        shell=False,
    )
    _append_step_log(cwd / "vaspkit.out", step, completed.stdout)
    _append_step_log(cwd / "vaspkit.err", step, completed.stderr)
    return completed


def _append_step_log(path: Path, step: VaspkitStep, text: str) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n===== VASPKIT {step.task_id}: {step.expected_file} =====\n")
        fh.write(text or "")
        if text and not text.endswith("\n"):
            fh.write("\n")


def _missing_or_empty_files(work_dir: Path, filenames: tuple[str, ...]) -> list[str]:
    missing: list[str] = []
    for filename in filenames:
        path = work_dir / filename
        if not path.exists() or path.stat().st_size == 0:
            missing.append(filename)
    return missing


def rollback_generated_core_files(work_dir: Path, workspace: Path) -> list[str]:
    """只回滚当前生成目录中的四个核心输入文件。

    安全边界：
    - work_dir 必须位于 workspace/input_sets 或 workspace/tasks 下；
    - 只删除文件名严格等于 POSCAR/INCAR/KPOINTS/POTCAR 的普通文件；
    - 不递归删除目录，也不碰日志、JSON、VASP/VASPKIT/POTCAR 库。
    """

    work_dir_resolved = work_dir.resolve()
    workspace_resolved = workspace.resolve()
    allowed_roots = ((workspace_resolved / "input_sets").resolve(), (workspace_resolved / "tasks").resolve())
    if not any(work_dir_resolved == root or root in work_dir_resolved.parents for root in allowed_roots):
        raise ValueError(f"Refusing rollback outside workspace input set or draft directories: {work_dir_resolved}")

    deleted: list[str] = []
    for filename in CORE_INPUT_FILES:
        candidate = work_dir_resolved / filename
        if candidate.exists() and candidate.is_file() and candidate.parent == work_dir_resolved:
            candidate.unlink()
            deleted.append(filename)
    return deleted


def _workspace_for_rollback(request: VaspkitRequest) -> Path:
    if request.workspace is not None:
        return request.workspace
    resolved = request.draft_dir.resolve()
    parts = resolved.parts
    if "input_sets" in parts:
        index = parts.index("input_sets")
        if index > 0:
            return Path(*parts[:index])
    if "tasks" in parts:
        index = parts.index("tasks")
        if index > 0:
            return Path(*parts[:index])
    # 测试或临时目录场景：如果无法识别 workspace，就以生成目录父目录作为最小边界。
    return resolved.parent


def _write_generation_failure(
    work_dir: Path,
    step: VaspkitStep,
    missing_files: list[str],
    stdout_path: Path,
    stderr_path: Path,
    return_code: int | None,
) -> dict:
    failure = {
        "failed_step": step.name,
        "task_id": step.task_id,
        "missing_files": missing_files,
        "cwd": str(work_dir),
        "vaspkit_out": str(stdout_path),
        "vaspkit_err": str(stderr_path),
        "return_code": return_code,
        "existing_files": sorted(path.name for path in work_dir.iterdir()),
    }
    _write_json(work_dir / "generation_failure.json", failure)
    return failure


def _format_generation_failure_error(failure: dict) -> str:
    return (
        f"{failure['failed_step']}; missing files: {', '.join(failure['missing_files']) or 'none'}; "
        f"cwd: {failure['cwd']}; vaspkit.out: {failure['vaspkit_out']}; "
        f"vaspkit.err: {failure['vaspkit_err']}; existing files: {', '.join(failure['existing_files'])}"
    )


def _apply_existing_potcar_policy(draft_dir: Path, policy: str, warnings: list[str]) -> str | None:
    potcar = draft_dir / "POTCAR"
    if not potcar.exists():
        return None
    if policy == "skip":
        warnings.append("POTCAR already exists; VASPKIT generation may skip it.")
        return None
    if policy == "backup_and_regenerate":
        backup = draft_dir / f"POTCAR.bak.{datetime.utcnow():%Y%m%d%H%M%S}"
        potcar.rename(backup)
        warnings.append(f"Existing POTCAR was backed up to {backup.name}.")
        return None
    if policy == "error":
        return "POTCAR already exists and existing_potcar_policy is error."
    return f"Unknown existing_potcar_policy: {policy}"


def _expected_outputs(mode: str, potcar_mode: str) -> list[str]:
    outputs: list[str] = []
    if mode in {"cif_to_poscar", "full"}:
        outputs.append("POSCAR")
    if mode in {"full", "incar_only"}:
        outputs.append("INCAR")
    if mode in {"full", "kpoints_only"}:
        outputs.append("KPOINTS")
    if mode in {"full", "potcar_only"} and potcar_mode == "103":
        outputs.append("POTCAR")
    return outputs


def _collect_generated_files(draft_dir: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for name in ("POSCAR", "INCAR", "KPOINTS", "POTCAR", "POTCAR.placeholder", "potcar_summary.json"):
        path = draft_dir / name
        if path.exists():
            files[name] = path
    return files


def _finalize_input_set_artifacts(request: VaspkitRequest, result: VaspkitResult) -> VaspkitResult:
    """写入 Input Set 相关元数据文件。

    该函数只写 metadata/hash/validation，不启动 VASP，也不会把 POTCAR 全文写入任何 UI 预览数据。
    """

    root_dir = result.draft_dir
    input_set_id = request.input_set_id or root_dir.name
    input_set_name = request.input_set_name or input_set_id
    now = datetime.utcnow().isoformat(timespec="seconds")
    validation = _validate_input_set_files(root_dir, result.dry_run)
    file_hashes = _core_file_hashes(root_dir)
    potcar_summary = summarize_potcar(root_dir / "POTCAR")
    status = validation["status"]
    usable_for_vasp = bool(validation["usable_for_vasp"])
    validation_path = root_dir / "validation.json"
    file_hashes_path = root_dir / "file_hashes.json"
    input_set_path = root_dir / "input_set.json"

    _write_json(file_hashes_path, file_hashes)
    _write_json(validation_path, validation)
    _write_json(
        input_set_path,
        {
            "input_set_id": input_set_id,
            "name": input_set_name,
            "source": "vaspkit",
            "status": status,
            "usable_for_vasp": usable_for_vasp,
            "root_dir": str(root_dir),
            "incar_path": str(root_dir / "INCAR"),
            "poscar_path": str(root_dir / "POSCAR"),
            "kpoints_path": str(root_dir / "KPOINTS"),
            "potcar_path": str(root_dir / "POTCAR"),
            "created_at": now,
            "updated_at": now,
            "notes": request.input_set_notes or ("dry_run: no real POTCAR" if result.dry_run else ""),
        },
    )
    finalized = VaspkitResult(
        ok=result.ok and not validation["errors"],
        dry_run=result.dry_run,
        return_code=result.return_code,
        draft_dir=result.draft_dir,
        generated_files=_collect_generated_files(root_dir),
        warnings=result.warnings + validation["warnings"],
        errors=result.errors + validation["errors"],
        stdout_path=result.stdout_path,
        stderr_path=result.stderr_path,
        request_path=result.request_path,
        result_path=result.result_path,
        potcar_summary=potcar_summary,
        input_set_id=input_set_id,
        input_set_status=status,
        usable_for_vasp=usable_for_vasp,
        input_set_path=input_set_path,
        validation_path=validation_path,
        file_hashes_path=file_hashes_path,
        generation_failure_path=result.generation_failure_path,
    )
    _write_result_json(finalized)
    return finalized


def _validate_input_set_files(root_dir: Path, dry_run: bool) -> dict:
    required = ("INCAR", "POSCAR", "KPOINTS", "POTCAR")
    missing = [filename for filename in required if not (root_dir / filename).exists()]
    empty = [filename for filename in required if (root_dir / filename).exists() and (root_dir / filename).stat().st_size == 0]
    warnings: list[str] = []
    errors: list[str] = []
    if dry_run:
        warnings.append(DRY_RUN_POTCAR_WARNING)
        return {
            "status": "dry_run",
            "usable_for_vasp": False,
            "required_files": list(required),
            "missing_files": missing,
            "empty_files": empty,
            "warnings": warnings,
            "errors": errors,
        }
    if missing:
        errors.append("Missing required input files: " + ", ".join(missing))
    if empty:
        errors.append("Empty required input files: " + ", ".join(empty))
    status = "generated" if not missing and not empty else "invalid"
    return {
        "status": status,
        "usable_for_vasp": status == "generated",
        "required_files": list(required),
        "missing_files": missing,
        "empty_files": empty,
        "warnings": warnings,
        "errors": errors,
    }


def _core_file_hashes(root_dir: Path) -> dict:
    hashes: dict[str, dict] = {}
    for filename in ("INCAR", "POSCAR", "KPOINTS", "POTCAR"):
        path = root_dir / filename
        hashes[filename] = {
            "exists": path.exists(),
            "path": str(path),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return hashes


def _potential_from_titel(titel_line: str) -> str | None:
    if "=" not in titel_line:
        return None
    tokens = titel_line.split("=", 1)[1].strip().split()
    if len(tokens) < 2:
        return None
    return tokens[1]


def _element_from_potential(potential: str) -> str | None:
    match = re.match(r"([A-Z][a-z]?)", potential)
    return match.group(1) if match else None


def _write_request_json(request: VaspkitRequest, path: Path) -> None:
    data = asdict(request)
    data["draft_dir"] = str(request.draft_dir)
    data["workspace"] = None if request.workspace is None else str(request.workspace)
    data["uploaded_cif_path"] = None if request.uploaded_cif_path is None else str(request.uploaded_cif_path)
    _write_json(path, data)


def _write_result_json(result: VaspkitResult) -> None:
    data = asdict(result)
    data["draft_dir"] = str(result.draft_dir)
    data["generated_files"] = {key: str(value) for key, value in result.generated_files.items()}
    data["stdout_path"] = str(result.stdout_path)
    data["stderr_path"] = str(result.stderr_path)
    data["request_path"] = str(result.request_path)
    data["result_path"] = str(result.result_path)
    data["input_set_path"] = None if result.input_set_path is None else str(result.input_set_path)
    data["validation_path"] = None if result.validation_path is None else str(result.validation_path)
    data["file_hashes_path"] = None if result.file_hashes_path is None else str(result.file_hashes_path)
    data["generation_failure_path"] = None if result.generation_failure_path is None else str(result.generation_failure_path)
    _write_json(result.result_path, data)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
