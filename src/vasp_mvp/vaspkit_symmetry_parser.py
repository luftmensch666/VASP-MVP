from __future__ import annotations

import re
from dataclasses import dataclass, field


MOLECULE_ISOLATION_MIN_LENGTH_WARNING = 10.0
MOLECULE_ISOLATION_PREFER_LENGTH = 15.0


@dataclass(frozen=True)
class SymmetrySummary:
    prototype: str | None = None
    total_atoms: int | None = None
    formula_unit: str | None = None
    full_formula_unit: str | None = None
    crystal_system: str | None = None
    crystal_class: str | None = None
    bravais_lattice: str | None = None
    lattice_constants: tuple[float, float, float] | None = None
    lattice_angles: tuple[float, float, float] | None = None
    volume: float | None = None
    density: float | None = None
    space_group_number: int | None = None
    point_group: str | None = None
    international_symbol: str | None = None
    symmetry_operations: int | None = None
    symmetry_accuracy: str | None = None
    raw_summary_excerpt: str = ""


@dataclass(frozen=True)
class Recommendation:
    code: str
    severity: str
    message_key: str
    details: dict = field(default_factory=dict)


def parse_vaspkit_601_summary(text: str) -> SymmetrySummary:
    """解析 VASPKIT 601 的 stdout summary。

    这里只提取 UI 需要展示的结构概览字段；解析不到的字段保持 None，避免
    因 VASPKIT 版本差异导致 UI 崩溃。
    """

    lines = (text or "").splitlines()
    values = {_label(line): _value(line) for line in lines if ":" in line}
    return SymmetrySummary(
        prototype=values.get("prototype"),
        total_atoms=_parse_int(values.get("total atoms")),
        formula_unit=_first_formula(values.get("formula unit")),
        full_formula_unit=values.get("full formula unit"),
        crystal_system=values.get("crystal system"),
        crystal_class=values.get("crystal class"),
        bravais_lattice=values.get("bravais lattice"),
        lattice_constants=_parse_float_tuple(values.get("lattice constants"), 3),
        lattice_angles=_parse_float_tuple(values.get("lattice angles"), 3),
        volume=_parse_float(values.get("volume")),
        density=_parse_float(values.get("density (g/cm3)")),
        space_group_number=_parse_int(values.get("space group")),
        point_group=values.get("point group"),
        international_symbol=values.get("international"),
        symmetry_operations=_parse_int(values.get("symmetry operations")),
        symmetry_accuracy=values.get("symmetry accuracy"),
        raw_summary_excerpt="\n".join(lines[:80]),
    )


def analyze_porous_symmetry_summary(
    summary: SymmetrySummary,
    *,
    expected_guest_elements: list[str] | None = None,
    structure_elements: list[str] | tuple[str, ...] | None = None,
    cell_lengths: tuple[float, float, float] | None = None,
    adsorbate_name: str | None = None,
) -> list[Recommendation]:
    """基于 601 summary 给 porous/MOF 结构准备建议。

    所有判断都是启发式提示，不用于断言结构一定缺原子、含客体或失去对称性。
    """

    recommendations: list[Recommendation] = []
    low_space_group = summary.space_group_number == 1 or _norm(summary.international_symbol) == "p1"
    few_operations = summary.symmetry_operations is not None and summary.symmetry_operations <= 2
    if low_space_group:
        recommendations.append(Recommendation("low_symmetry", "warning", "adsorption_wizard.symmetry.low_symmetry_warning"))
    if few_operations:
        recommendations.append(Recommendation("few_symmetry_operations", "warning", "adsorption_wizard.symmetry.modified_structure_warning"))
    if not low_space_group and not few_operations and (summary.space_group_number or summary.international_symbol):
        recommendations.append(Recommendation("non_p1_symmetry", "info", "adsorption_wizard.symmetry.non_p1_info"))

    recommendations.append(Recommendation("guest_or_missing_atoms_uncertain", "info", "adsorption_wizard.symmetry.guest_or_missing_atoms_uncertain"))

    elements = {element.strip().capitalize() for element in (structure_elements or ()) if element.strip()}
    expected = {element.strip().capitalize() for element in (expected_guest_elements or []) if element.strip()}
    if expected and elements.intersection(expected):
        recommendations.append(
            Recommendation(
                "expected_guest_elements_present",
                "warning",
                "adsorption_wizard.symmetry.expected_guest_elements_present",
                {"elements": ", ".join(sorted(elements.intersection(expected))), "adsorbate": adsorbate_name or ""},
            )
        )

    if low_space_group or few_operations:
        recommendations.append(Recommendation("recommend_603", "info", "adsorption_wizard.symmetry.recommend_603"))
    else:
        recommendations.append(Recommendation("skip_603_ok", "info", "adsorption_wizard.symmetry.skip_603_ok"))

    lengths = cell_lengths or summary.lattice_constants
    if lengths:
        min_length = min(lengths)
        if min_length < MOLECULE_ISOLATION_MIN_LENGTH_WARNING:
            recommendations.append(
                Recommendation(
                    "supercell_small_cell",
                    "warning",
                    "adsorption_wizard.symmetry.supercell_small_cell_warning",
                    {"min_length": f"{min_length:.3f}"},
                )
            )
        elif min_length >= MOLECULE_ISOLATION_PREFER_LENGTH:
            recommendations.append(
                Recommendation(
                    "supercell_large_cell",
                    "info",
                    "adsorption_wizard.symmetry.supercell_large_cell_info",
                    {"min_length": f"{min_length:.3f}"},
                )
            )
    recommendations.append(Recommendation("small_molecule_supercell", "info", "adsorption_wizard.symmetry.nh3_isolated_adsorption_supercell"))
    return recommendations


def infer_expected_guest_elements(adsorbate_name: str | None) -> list[str]:
    """从简单分子式中推断元素；无法解析时返回空列表，UI 使用通用建议。"""

    text = (adsorbate_name or "").strip()
    if not text:
        return []
    return re.findall(r"[A-Z][a-z]?", text)


def _label(line: str) -> str:
    return line.split(":", 1)[0].strip().lower()


def _value(line: str) -> str:
    return line.split(":", 1)[1].strip()


def _parse_int(value: str | None) -> int | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d+", value)
    return int(match.group(0)) if match else None


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", value)
    return float(match.group(0)) if match else None


def _parse_float_tuple(value: str | None, count: int) -> tuple[float, ...] | None:
    if not value:
        return None
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?", value)
    if len(numbers) < count:
        return None
    return tuple(float(item) for item in numbers[:count])


def _first_formula(value: str | None) -> str | None:
    if not value:
        return None
    return value.split("[", 1)[0].strip()


def _norm(value: str | None) -> str:
    return (value or "").strip().lower().replace(" ", "")
