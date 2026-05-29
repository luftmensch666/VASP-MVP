from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .input_sets import CORE_INPUT_FILES, input_set_file_paths, sha256_file
from .models import InputSet


Severity = Literal["error", "warning", "info"]


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    message_key: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PotcarPotentialSummary:
    element: str | None
    potential_label: str | None
    paw_family: str | None
    vrhfin: str | None
    titel: str | None
    enmax: float | None
    enmin: float | None
    zval: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PotcarSummary:
    exists: bool
    size_bytes: int
    sha256: str | None
    potentials: tuple[PotcarPotentialSummary, ...]
    warnings: tuple[str, ...] = ()

    @property
    def element_order(self) -> tuple[str, ...]:
        return tuple(item.element for item in self.potentials if item.element)

    @property
    def potential_labels(self) -> tuple[str, ...]:
        return tuple(item.potential_label for item in self.potentials if item.potential_label)

    @property
    def titel_lines(self) -> tuple[str, ...]:
        return tuple(item.titel for item in self.potentials if item.titel)

    @property
    def vrhfin_elements(self) -> tuple[str, ...]:
        return tuple(item.vrhfin for item in self.potentials if item.vrhfin)

    def to_dict(self) -> dict:
        return {
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "number_of_potentials": len(self.potentials),
            "element_order": list(self.element_order),
            "potential_labels": list(self.potential_labels),
            "titel_lines": list(self.titel_lines),
            "vrhfin_elements": list(self.vrhfin_elements),
            "warnings": list(self.warnings),
            "potentials": [item.to_dict() for item in self.potentials],
        }


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    errors: tuple[ValidationIssue, ...]
    warnings: tuple[ValidationIssue, ...]
    infos: tuple[ValidationIssue, ...]

    @property
    def usable_for_vasp(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "usable_for_vasp": self.usable_for_vasp,
            "errors": [item.to_dict() for item in self.errors],
            "warnings": [item.to_dict() for item in self.warnings],
            "infos": [item.to_dict() for item in self.infos],
        }


def parse_potcar_summary(path: Path) -> PotcarSummary:
    """提取 POTCAR 摘要，不返回或保存 POTCAR 全文。

    只保留 UI 可以展示的关键信息：TITEL/VRHFIN/ENMAX/ENMIN/ZVAL 等。
    读取时逐行扫描，避免把完整赝势 block 放入内存结构、JSON 或日志。
    """

    path = Path(path)
    if not path.exists():
        return PotcarSummary(False, 0, None, (), ("potcar_missing",))

    potentials: list[dict] = []
    current: dict | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line.startswith("TITEL"):
                if current is not None:
                    potentials.append(current)
                current = _new_potential_record()
                current["titel"] = line
                family, label = _family_and_label_from_titel(line)
                current["paw_family"] = family
                current["potential_label"] = label
                current["element_from_titel"] = _element_from_potential_label(label)
                continue
            if current is None and line.startswith(("VRHFIN", "ENMAX", "ZVAL")):
                current = _new_potential_record()
            if current is None:
                continue
            if line.startswith("VRHFIN"):
                element = _element_from_vrhfin(line)
                current["vrhfin"] = element
                continue
            if "ENMAX" in line:
                current["enmax"] = _float_after_key(line, "ENMAX")
                current["enmin"] = _float_after_key(line, "ENMIN")
                continue
            if line.startswith("ZVAL") or " ZVAL" in line:
                current["zval"] = _float_after_key(line, "ZVAL")
    if current is not None:
        potentials.append(current)

    summaries = tuple(_record_to_potential(item) for item in potentials)
    warnings: list[str] = []
    if path.stat().st_size > 0 and not any(item.element for item in summaries):
        warnings.append("potcar_order_unreadable")
    return PotcarSummary(
        exists=True,
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        potentials=summaries,
        warnings=tuple(warnings),
    )


def validate_input_set(
    input_set: InputSet,
    *,
    role: str | None = None,
    method_family: str | None = None,
    functional: str | None = None,
    method_notes: str | None = None,
) -> ValidationResult:
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    infos: list[ValidationIssue] = []
    paths = input_set_file_paths(input_set)

    for filename in CORE_INPUT_FILES:
        path = paths[filename]
        if not path.exists():
            errors.append(_issue("file_missing", "error", f"input_set.validation.{filename.lower()}_missing", filename=filename))
        elif path.stat().st_size == 0:
            errors.append(_issue("file_empty", "error", f"input_set.validation.{filename.lower()}_empty", filename=filename))

    poscar_elements = parse_poscar_element_order(paths["POSCAR"]) if paths["POSCAR"].exists() and paths["POSCAR"].stat().st_size > 0 else ()
    if paths["POSCAR"].exists() and paths["POSCAR"].stat().st_size > 0 and not poscar_elements:
        warnings.append(_issue("poscar_order_unreadable", "warning", "input_set.validation.poscar_order_unreadable"))

    potcar_summary = parse_potcar_summary(paths["POTCAR"])
    potcar_elements = potcar_summary.element_order
    if paths["POTCAR"].exists() and paths["POTCAR"].stat().st_size > 0 and not potcar_elements:
        warnings.append(_issue("potcar_order_unreadable", "warning", "input_set.validation.potcar_order_unreadable"))
    if poscar_elements and potcar_elements and tuple(poscar_elements) != tuple(potcar_elements):
        errors.append(
            _issue(
                "poscar_potcar_order_mismatch",
                "error",
                "input_set.validation.poscar_potcar_order_mismatch",
                poscar=", ".join(poscar_elements),
                potcar=", ".join(potcar_elements),
            )
        )

    incar = parse_incar_tags(paths["INCAR"]) if paths["INCAR"].exists() else {}
    kpoints = parse_kpoints_summary(paths["KPOINTS"]) if paths["KPOINTS"].exists() else {}
    _validate_encut(incar, potcar_summary, warnings, infos)
    _validate_functional(functional, method_family, method_notes, incar, potcar_summary, warnings, infos)
    _validate_kpoints(role, kpoints, warnings)
    _validate_static_recommendation(incar, warnings)

    return ValidationResult(ok=not errors, errors=tuple(errors), warnings=tuple(warnings), infos=tuple(infos))


def parse_poscar_element_order(path: Path) -> tuple[str, ...]:
    """兼容 VASP 5 和 VASP 4 POSCAR。

    VASP 5 第 6 行是元素符号；VASP 4 第 6 行通常是计数行，无法可靠得到元素顺序。
    这种情况返回空 tuple，由校验层给 warning，而不是误判 POTCAR 错误。
    """

    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        return ()
    tokens = lines[5].split()
    if not tokens or all(_is_number(token) for token in tokens):
        return ()
    if all(re.match(r"^[A-Z][a-z]?$", token) for token in tokens):
        return tuple(tokens)
    return ()


def parse_incar_tags(path: Path) -> dict[str, str]:
    tags: dict[str, str] = {}
    if not Path(path).exists():
        return tags
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        tags[key.strip().upper()] = value.strip()
    return tags


def parse_kpoints_summary(path: Path) -> dict:
    lines = [line.strip() for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    scheme = ""
    grid: tuple[int, int, int] | None = None
    if len(lines) >= 3:
        scheme = lines[2].lower()
    if len(lines) >= 4:
        values = lines[3].split()
        if len(values) >= 3 and all(value.lstrip("-").isdigit() for value in values[:3]):
            grid = tuple(int(value) for value in values[:3])
    gamma_only = grid == (1, 1, 1) and "gamma" in scheme
    return {"scheme": scheme, "grid": grid, "gamma_only": gamma_only}


def _validate_encut(incar: dict[str, str], potcar: PotcarSummary, warnings: list[ValidationIssue], infos: list[ValidationIssue]) -> None:
    enmax_values = [item.enmax for item in potcar.potentials if item.enmax is not None]
    if not enmax_values:
        return
    max_enmax = max(enmax_values)
    encut = _float_value(incar.get("ENCUT"))
    if encut is None:
        warnings.append(_issue("encut_missing", "warning", "input_set.validation.encut_missing", max_enmax=max_enmax))
    elif encut < max_enmax:
        warnings.append(_issue("encut_below_enmax", "warning", "input_set.validation.encut_below_enmax", encut=encut, max_enmax=max_enmax))
    elif encut < 1.3 * max_enmax:
        warnings.append(
            _issue(
                "encut_below_recommended",
                "warning",
                "input_set.validation.encut_below_recommended",
                encut=encut,
                recommended=1.3 * max_enmax,
            )
        )
    else:
        infos.append(_issue("encut_recommended_ok", "info", "input_set.validation.encut_recommended_ok", encut=encut, recommended=1.3 * max_enmax))


def _validate_functional(
    functional: str | None,
    method_family: str | None,
    method_notes: str | None,
    incar: dict[str, str],
    potcar: PotcarSummary,
    warnings: list[ValidationIssue],
    infos: list[ValidationIssue],
) -> None:
    method_text = " ".join(value or "" for value in (functional, method_family, method_notes)).lower()
    families = {item.paw_family for item in potcar.potentials if item.paw_family}
    if "lda" in method_text and any(family and "PBE" in family.upper() for family in families):
        warnings.append(_issue("functional_mismatch", "warning", "input_set.validation.functional_mismatch", functional=functional or "", paw_family=", ".join(sorted(families))))
    if ("pbe" in method_text or "hse" in method_text) and any(family and "LDA" in family.upper() for family in families):
        warnings.append(_issue("functional_mismatch", "warning", "input_set.validation.functional_mismatch", functional=functional or "", paw_family=", ".join(sorted(families))))
    if "d3" in method_text and "IVDW" not in incar:
        warnings.append(_issue("d3_missing_ivdw", "warning", "input_set.validation.d3_missing_ivdw"))
    if ("+u" in method_text or "dft+u" in method_text) and any(tag not in incar for tag in ("LDAU", "LDAUL", "LDAUU", "LDAUJ")):
        warnings.append(_issue("dftu_missing_tags", "warning", "input_set.validation.dftu_missing_tags"))

    elements = set(potcar.element_order)
    correlated = sorted(elements & _DF_OR_F_ELEMENTS)
    if correlated:
        infos.append(_issue("dftu_literature_reminder", "info", "input_set.validation.dftu_literature_reminder", elements=", ".join(correlated)))
    semicore = [label for label in potcar.potential_labels if "_" in label]
    if semicore:
        infos.append(_issue("semicore_potential", "info", "input_set.validation.semicore_potential", potentials=", ".join(semicore)))


def _validate_kpoints(role: str | None, kpoints: dict, warnings: list[ValidationIssue]) -> None:
    grid = kpoints.get("grid")
    if not grid:
        warnings.append(_issue("kpoints_unreadable", "warning", "input_set.validation.kpoints_unreadable"))
        return
    if role == "molecule_ref" and not kpoints.get("gamma_only"):
        warnings.append(_issue("molecule_gamma_recommended", "warning", "input_set.validation.molecule_gamma_recommended", grid=" ".join(str(v) for v in grid)))
    if role in {"clean_slab", "adsorbed_system"} and len(grid) >= 3 and grid[2] > 1:
        warnings.append(_issue("slab_kz_recommended", "warning", "input_set.validation.slab_kz_recommended", grid=" ".join(str(v) for v in grid)))


def _validate_static_recommendation(incar: dict[str, str], warnings: list[ValidationIssue]) -> None:
    nsw = _int_value(incar.get("NSW"))
    ibrion = _int_value(incar.get("IBRION"))
    if nsw is not None and nsw > 0:
        warnings.append(_issue("static_recommended", "warning", "input_set.validation.static_recommended", nsw=nsw))
    elif ibrion is not None and ibrion != -1 and nsw is None:
        warnings.append(_issue("static_recommended", "warning", "input_set.validation.static_recommended", nsw="unknown"))


def _new_potential_record() -> dict:
    return {
        "element_from_titel": None,
        "potential_label": None,
        "paw_family": None,
        "vrhfin": None,
        "titel": None,
        "enmax": None,
        "enmin": None,
        "zval": None,
    }


def _record_to_potential(record: dict) -> PotcarPotentialSummary:
    element = record.get("vrhfin") or record.get("element_from_titel")
    return PotcarPotentialSummary(
        element=element,
        potential_label=record.get("potential_label"),
        paw_family=record.get("paw_family"),
        vrhfin=record.get("vrhfin"),
        titel=record.get("titel"),
        enmax=record.get("enmax"),
        enmin=record.get("enmin"),
        zval=record.get("zval"),
    )


def _family_and_label_from_titel(line: str) -> tuple[str | None, str | None]:
    if "=" not in line:
        return None, None
    tokens = line.split("=", 1)[1].strip().split()
    family = tokens[0] if tokens else None
    label = tokens[1] if len(tokens) > 1 else None
    return family, label


def _element_from_vrhfin(line: str) -> str | None:
    match = re.search(r"VRHFIN\s*=\s*([A-Z][a-z]?)\s*:", line)
    return match.group(1) if match else None


def _element_from_potential_label(label: str | None) -> str | None:
    if not label:
        return None
    match = re.match(r"([A-Z][a-z]?)(?:[_0-9].*)?$", label)
    return match.group(1) if match else None


def _float_after_key(line: str, key: str) -> float | None:
    match = re.search(rf"{key}\s*=\s*([-+]?\d+(?:\.\d+)?)", line)
    return float(match.group(1)) if match else None


def _float_value(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    return float(match.group(0)) if match else None


def _int_value(value: str | None) -> int | None:
    number = _float_value(value)
    return None if number is None else int(number)


def _is_number(value: str) -> bool:
    try:
        float(value)
    except ValueError:
        return False
    return True


def _issue(code: str, severity: Severity, message_key: str, **details) -> ValidationIssue:
    return ValidationIssue(code=code, severity=severity, message_key=message_key, details=details)


_DF_OR_F_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
}
