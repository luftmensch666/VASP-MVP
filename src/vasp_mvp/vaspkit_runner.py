from __future__ import annotations

import json
import hashlib
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


# dry-run 只用于 UI 和流程联调，不能生成、展示或伪造真实 POTCAR。
DRY_RUN_POTCAR_WARNING = (
    "Dry-run mode is enabled. A real POTCAR is not generated, so this input set cannot be used for "
    "VASP calculations. Disable dry-run and run real VASPKIT generation to create a usable input set."
)


@dataclass(frozen=True)
class VaspkitRequest:
    vaspkit_bin: str
    draft_dir: Path
    input_set_id: str | None = None
    input_set_name: str = ""
    uploaded_cif_path: Path | None = None
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


def check_vaspkit_available(vaspkit_bin: str) -> bool:
    candidate = Path(vaspkit_bin).expanduser()
    if candidate.is_file():
        return candidate.exists() and candidate.stat().st_mode & 0o111 != 0
    return shutil.which(vaspkit_bin) is not None


def build_vaspkit_inputs(request: VaspkitRequest) -> list[str]:
    if request.potcar_mode == "104" and request.generation_mode in {"full", "potcar_only"}:
        raise NotImplementedError("VASPKIT POTCAR mode 104 is reserved and not implemented yet.")

    inputs: list[str] = []
    if request.generation_mode in {"cif_to_poscar", "full"}:
        if request.uploaded_cif_path is None:
            raise ValueError("uploaded_cif_path is required for CIF to POSCAR generation.")
        inputs.extend(
            [
                "1",
                "105",
                request.uploaded_cif_path.name,
                request.custom_element_order if request.element_order_mode == "custom" else "",
            ]
        )

    if request.generation_mode in {"full", "incar_only"}:
        inputs.extend(["1", "101", _incar_key_string(request)])

    if request.generation_mode in {"full", "kpoints_only"}:
        inputs.extend(["1", "102", str(request.kmesh_scheme), str(request.kmesh_resolved_value)])

    if request.generation_mode in {"full", "potcar_only"}:
        inputs.extend(["1", request.potcar_mode])

    if not inputs:
        raise ValueError(f"Unsupported VASPKIT generation mode: {request.generation_mode}")
    return inputs


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

    result = run_vaspkit_interactive(request.vaspkit_bin, inputs, draft_dir)
    generated = result.generated_files
    missing = [name for name in _expected_outputs(request.generation_mode, request.potcar_mode) if name not in generated]
    warnings.extend(result.warnings)
    errors.extend(result.errors)
    if missing:
        warnings.append("Missing expected VASPKIT outputs: " + ", ".join(missing))
    final = VaspkitResult(
        ok=result.ok and not errors,
        dry_run=False,
        return_code=result.return_code,
        draft_dir=draft_dir,
        generated_files=generated,
        warnings=warnings,
        errors=errors,
        stdout_path=result.stdout_path,
        stderr_path=result.stderr_path,
        request_path=request_path,
        result_path=result_path,
        potcar_summary=summarize_potcar(draft_dir / "POTCAR"),
    )
    return _finalize_input_set_artifacts(request, final)


def summarize_potcar(path: Path) -> dict:
    if not path.exists():
        return {
            "exists": False,
            "size_bytes": 0,
            "sha256": None,
            "titel_lines": [],
            "potential_order": [],
            "element_order": [],
        }
    titel_lines: list[str] = []
    potential_order: list[str] = []
    element_order: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("TITEL"):
                title = line.strip()
                titel_lines.append(title)
                potential = _potential_from_titel(title)
                if potential:
                    potential_order.append(potential)
                    element = _element_from_potential(potential)
                    if element:
                        element_order.append(element)
    return {
        "exists": True,
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "titel_lines": titel_lines,
        "potential_order": potential_order,
        "element_order": element_order,
    }


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
    if request.generation_mode in {"cif_to_poscar", "full"}:
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
    if request.generation_mode in {"full", "incar_only"}:
        (draft_dir / "INCAR").write_text(f"SYSTEM = dry-run\n# VASPKIT keys: {_incar_key_string(request)}\n", encoding="utf-8")
    if request.generation_mode in {"full", "kpoints_only"}:
        (draft_dir / "KPOINTS").write_text(
            "Dry-run KPOINTS\n"
            "0\n"
            "Gamma\n"
            "1 1 1\n"
            "0 0 0\n",
            encoding="utf-8",
        )
    if request.generation_mode in {"full", "potcar_only"}:
        (draft_dir / "POTCAR.placeholder").write_text(
            DRY_RUN_POTCAR_WARNING + "\n",
            encoding="utf-8",
        )
        _write_json(draft_dir / "potcar_summary.json", summarize_potcar(draft_dir / "POTCAR"))
    (draft_dir / "vaspkit_inputs.txt").write_text("\n".join(inputs) + "\n", encoding="utf-8")


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
            "notes": "dry_run: no real POTCAR" if result.dry_run else "",
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
    _write_json(result.result_path, data)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
