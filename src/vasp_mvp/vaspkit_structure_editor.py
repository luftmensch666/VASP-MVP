from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class StructureEditResult:
    ok: bool
    step_name: str
    output_path: Path | None
    stdout_path: Path
    stderr_path: Path
    return_code: int | None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def build_105_inputs(cif_filename: str, element_order: str = "") -> list[str]:
    filename = Path(cif_filename).name
    if not filename.lower().endswith(".cif"):
        raise ValueError("CIF filename must end with .cif")
    return ["105", filename, element_order.strip()]


def build_801_inputs(direction_index: int | str, vacuum_thickness: float | str) -> list[str]:
    direction = validate_direction_index(direction_index)
    vacuum = validate_positive_float(vacuum_thickness, "vacuum_thickness")
    return ["801", str(direction), _format_float(vacuum)]


def build_803_inputs(
    h: int | str,
    k: int | str,
    l: int | str,
    layer_text: str,
    shift_value: float | str,
    vacuum_thickness: float | str,
) -> list[str]:
    hkl = validate_hkl(h, k, l)
    layer = validate_layer_text(layer_text)
    shift = _parse_float(shift_value, "shift_value")
    vacuum = validate_positive_float(vacuum_thickness, "vacuum_thickness")
    return ["803", f"{hkl[0]} {hkl[1]} {hkl[2]}", layer, _format_float(shift), _format_float(vacuum)]


def build_401_inputs(
    repeat_a: int | str,
    repeat_b: int | str,
    repeat_c: int | str,
) -> list[str]:
    repeats = validate_repeat_abc(repeat_a, repeat_b, repeat_c)
    return ["401", "1", f"{repeats[0]} {repeats[1]} {repeats[2]}"]


def build_402_atom_indices_inputs(
    atom_indices: str,
    direction_index: int | str,
) -> list[str]:
    indices = validate_atom_indices(atom_indices)
    direction = validate_direction_index(direction_index)
    return ["402", "1", "1", indices, str(direction)]


def build_402_z_range_inputs(
    z_min: float | str,
    z_max: float | str,
    direction_index: int | str,
) -> list[str]:
    low, high = validate_z_range(z_min, z_max)
    direction = validate_direction_index(direction_index)
    return ["402", "1", "3", f"{_format_float(low)} {_format_float(high)}", str(direction)]


def validate_direction_index(value: int | str) -> int:
    direction = _parse_int(value, "direction_index")
    if direction not in {1, 2, 3}:
        raise ValueError("direction_index must be 1, 2, or 3")
    return direction


def validate_positive_float(value: float | str, field_name: str = "value") -> float:
    parsed = _parse_float(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be positive")
    return parsed


def validate_positive_int(value: int | str, field_name: str = "value") -> int:
    parsed = _parse_int(value, field_name)
    if parsed <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
    return parsed


def validate_hkl(h: int | str, k: int | str, l: int | str) -> tuple[int, int, int]:
    hkl = (_parse_int(h, "h"), _parse_int(k, "k"), _parse_int(l, "l"))
    if hkl == (0, 0, 0):
        raise ValueError("h/k/l must not all be zero")
    return hkl


def validate_repeat_abc(a: int | str, b: int | str, c: int | str) -> tuple[int, int, int]:
    return (
        validate_positive_int(a, "repeat_a"),
        validate_positive_int(b, "repeat_b"),
        validate_positive_int(c, "repeat_c"),
    )


def validate_layer_text(text: str) -> str:
    value = (text or "").strip()
    if not value:
        raise ValueError("layer_text must not be empty")
    if _has_control_chars(value) or not re.fullmatch(r"[0-9,\-\s]+", value):
        raise ValueError("layer_text may contain only digits, comma, space, and hyphen")
    tokens = re.split(r"[\s,]+", value)
    for token in tokens:
        if not token:
            continue
        if re.fullmatch(r"\d+", token):
            if int(token) <= 0:
                raise ValueError("layer numbers must be positive")
            continue
        if re.fullmatch(r"\d+-\d+", token):
            start, end = (int(part) for part in token.split("-", 1))
            if start <= 0 or end <= 0 or start > end:
                raise ValueError("layer ranges must be positive and ascending")
            continue
        raise ValueError("layer_text must use values like 1, 1,2,3, or 1-3")
    return value


def validate_atom_indices(text: str) -> str:
    value = (text or "").strip()
    if not value:
        raise ValueError("atom indices must not be empty")
    if _has_control_chars(value):
        raise ValueError("atom indices must not contain control characters")
    tokens = re.split(r"[\s,]+", value)
    for token in tokens:
        if not token:
            continue
        if token.lower() == "all":
            continue
        if re.fullmatch(r"[A-Z][a-z]?", token):
            continue
        if re.fullmatch(r"\d+", token):
            if int(token) <= 0:
                raise ValueError("atom indices must be positive")
            continue
        if re.fullmatch(r"\d+-\d+", token):
            start, end = (int(part) for part in token.split("-", 1))
            if start <= 0 or end <= 0 or start > end:
                raise ValueError("atom index ranges must be positive and ascending")
            continue
        raise ValueError("atom indices must use numbers, element symbols, 'all', or ranges like 1-5")
    return value


def validate_z_range(z_min: float | str, z_max: float | str) -> tuple[float, float]:
    low = _parse_float(z_min, "z_min")
    high = _parse_float(z_max, "z_max")
    if low >= high:
        raise ValueError("z_min must be smaller than z_max")
    return low, high


def run_vaspkit_structure_step(
    vaspkit_bin: str,
    inputs: list[str],
    cwd: Path,
    step_name: str,
    expected_output: str,
    timeout: int = 120,
) -> StructureEditResult:
    """运行单个 VASPKIT 结构准备步骤。

    每个步骤独立调用，cwd 应为 workflow_root/structure。
    VASPKIT 原始输出文件留在 cwd，日志写入 workflow_root/logs。
    这里始终使用参数列表和 shell=False，不执行 shell 字符串。
    """

    workdir = Path(cwd)
    workdir.mkdir(parents=True, exist_ok=True)
    log_dir = workdir.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"structure_{step_name}.out"
    stderr_path = log_dir / f"structure_{step_name}.err"
    try:
        completed = subprocess.run(
            [vaspkit_bin],
            input="\n".join(inputs) + "\n",
            text=True,
            cwd=workdir,
            capture_output=True,
            timeout=timeout,
            shell=False,
        )
        stdout_path.write_text(completed.stdout or "", encoding="utf-8")
        stderr_path.write_text(completed.stderr or "", encoding="utf-8")
        return_code = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout_path.write_text(_to_text(exc.stdout), encoding="utf-8")
        stderr_path.write_text(_to_text(exc.stderr), encoding="utf-8")
        return StructureEditResult(
            ok=False,
            step_name=step_name,
            output_path=None,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            return_code=None,
            errors=[f"VASPKIT timed out after {timeout} seconds"],
        )
    output_path = workdir / expected_output
    errors: list[str] = []
    if return_code != 0:
        errors.append(f"VASPKIT exited with return code {return_code}")
    if not output_path.exists() or output_path.stat().st_size == 0:
        errors.append(f"Expected output file was not generated: {expected_output}")
    return StructureEditResult(
        ok=not errors,
        step_name=step_name,
        output_path=output_path if output_path.exists() else None,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        return_code=return_code,
        errors=errors,
    )


def expected_803_output_name(h: int | str, k: int | str, l: int | str) -> str:
    hkl = validate_hkl(h, k, l)
    return f"SLAB{hkl[0]}{hkl[1]}{hkl[2]}.vasp"


def expected_401_output_name(a: int | str, b: int | str, c: int | str) -> str:
    repeats = validate_repeat_abc(a, b, c)
    return f"SC{repeats[0]}{repeats[1]}{repeats[2]}.vasp"


def _parse_int(value: int | str, field_name: str) -> int:
    text = str(value).strip()
    if not re.fullmatch(r"[-+]?\d+", text):
        raise ValueError(f"{field_name} must be an integer")
    return int(text)


def _parse_float(value: float | str, field_name: str) -> float:
    text = str(value).strip()
    if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
        raise ValueError(f"{field_name} must be a number")
    return float(text)


def _format_float(value: float) -> str:
    return f"{value:g}"


def _has_control_chars(value: str) -> bool:
    return any(ord(char) < 32 for char in value if char not in {"\t"})


def _to_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
