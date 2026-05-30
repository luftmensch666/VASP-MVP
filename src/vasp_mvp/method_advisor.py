from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodDescription:
    zh: str
    en: str
    warnings_zh: tuple[str, ...] = ()
    warnings_en: tuple[str, ...] = ()


def generate_method_description(
    *,
    method_family: str | None,
    functional: str | None,
    system_type: str = "adsorption",
    adsorbate_name: str | None = None,
    elements: list[str] | tuple[str, ...] | None = None,
) -> MethodDescription:
    """根据方法选择生成可编辑的双语计算方法说明。

    该函数只生成默认建议文本，不写数据库；UI 只有在用户显式点击“重新生成”
    时才会覆盖当前 method_notes，避免刷新页面时覆盖用户手动修改。
    """

    family = (method_family or "").strip()
    functional_name = (functional or "").strip()
    key = functional_name.lower().replace(" ", "")
    correlated = sorted(set(elements or ()) & _CORRELATED_ELEMENTS)
    prefix_en = _system_prefix_en(system_type, adsorbate_name)
    prefix_zh = _system_prefix_zh(system_type, adsorbate_name)

    if key in {"pbe-d3", "pbed3"}:
        return MethodDescription(
            zh=(
                f"{prefix_zh}采用 PBE-D3 泛函进行 DFT 总能量差计算。PBE-D3 适合分子吸附、"
                "表面吸附、弱相互作用以及范德华贡献明显的体系。"
            ),
            en=(
                f"{prefix_en} uses DFT total-energy differences with the PBE-D3 functional. "
                "PBE-D3 is suitable for molecular adsorption, surface adsorption, weak interactions, "
                "and systems where dispersion contributions are expected to be important."
            ),
        )
    if key == "hse06" or "hybrid" in family.lower():
        return MethodDescription(
            zh=(
                f"{prefix_zh}采用 HSE06 杂化泛函。HSE06 通常能更好描述电子结构、带隙和局域态，"
                "但计算成本较高，建议用于关键构型验证。"
            ),
            en=(
                f"{prefix_en} uses a hybrid DFT description with HSE06. HSE06 can improve electronic "
                "structure, band-gap, and localized-state descriptions, but it is computationally "
                "expensive and is usually best used for key-configuration validation."
            ),
        )
    if key in {"pbe+u", "pbeu"} or "+u" in family.lower():
        extra_en = (
            f" The detected correlated elements are {', '.join(correlated)}."
            if correlated
            else ""
        )
        extra_zh = f" 当前检测到可能相关的元素：{', '.join(correlated)}。" if correlated else ""
        return MethodDescription(
            zh=(
                f"{prefix_zh}采用 DFT+U 总能量工作流。DFT+U 常用于过渡金属 d 电子或稀土 f 电子体系；"
                f"U 值应来自文献或系统测试，不建议随意设置。{extra_zh}"
            ),
            en=(
                f"{prefix_en} uses a DFT+U total-energy workflow. DFT+U is commonly used for transition-metal "
                "d-electron or rare-earth f-electron systems; U values should be taken from literature "
                f"or systematic testing and should not be chosen arbitrarily.{extra_en}"
            ),
        )
    if key == "pbe" or not functional_name:
        return MethodDescription(
            zh=(
                f"{prefix_zh}采用 PBE 泛函进行标准 DFT 总能量差计算。PBE 适合一般结构优化和基准 DFT "
                "总能量计算，但可能低估弱范德华吸附相互作用。"
            ),
            en=(
                f"{prefix_en} uses standard DFT total-energy differences with PBE. PBE is useful for general "
                "structure optimization and baseline DFT total energies, but it may underestimate weak "
                "van der Waals adsorption interactions."
            ),
        )
    return MethodDescription(
        zh=(
            f"{prefix_zh}采用 {family or '所选方法族'} / {functional_name or '所选泛函'}。在使用最终吸附能前，"
            "应明确交换关联泛函、色散修正、U 参数以及参考态假设。"
        ),
        en=(
            f"{prefix_en} uses {family or 'the selected method family'} with {functional_name or 'the selected functional'}. "
            "Document the exchange-correlation functional, dispersion correction, U parameters, and any "
            "reference-state assumptions before using final adsorption energies."
        ),
    )


def _system_prefix_en(system_type: str, adsorbate_name: str | None) -> str:
    if system_type == "adsorption":
        adsorbate = (adsorbate_name or "").strip()
        if adsorbate:
            return f"The adsorption workflow for {adsorbate}"
        return "The adsorption workflow"
    return "The workflow"


def _system_prefix_zh(system_type: str, adsorbate_name: str | None) -> str:
    if system_type == "adsorption":
        adsorbate = (adsorbate_name or "").strip()
        if adsorbate:
            return f"{adsorbate} 吸附能工作流"
        return "吸附能工作流"
    return "该工作流"


_CORRELATED_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
}
