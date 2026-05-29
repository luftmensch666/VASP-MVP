from __future__ import annotations


def generate_method_description(
    *,
    method_family: str | None,
    functional: str | None,
    system_type: str = "adsorption",
    adsorbate_name: str | None = None,
    elements: tuple[str, ...] = (),
) -> str:
    """根据方法选择生成可编辑的计算方法说明。

    该函数只生成默认建议文本，不写数据库；UI 只有在用户显式点击“重新生成”
    时才会覆盖当前 method_notes，避免刷新页面时覆盖用户手动修改。
    """

    family = (method_family or "").strip()
    functional_name = (functional or "").strip()
    key = functional_name.lower().replace(" ", "")
    correlated = sorted(set(elements) & _CORRELATED_ELEMENTS)
    prefix = _system_prefix(system_type, adsorbate_name)

    if key in {"pbe-d3", "pbed3"}:
        return (
            f"{prefix} uses DFT total-energy differences with the PBE-D3 functional. "
            "PBE-D3 is suitable for molecular adsorption, surface adsorption, weak interactions, "
            "and systems where dispersion contributions are expected to be important."
        )
    if key == "hse06" or "hybrid" in family.lower():
        return (
            f"{prefix} uses a hybrid DFT description with HSE06. HSE06 can improve electronic "
            "structure, band-gap, and localized-state descriptions, but it is computationally "
            "expensive and is usually best used for key-configuration validation."
        )
    if key in {"pbe+u", "pbeu"} or "+u" in family.lower():
        extra = (
            f" The detected correlated elements are {', '.join(correlated)}."
            if correlated
            else ""
        )
        return (
            f"{prefix} uses a DFT+U total-energy workflow. DFT+U is commonly used for transition-metal "
            "d-electron or rare-earth f-electron systems; U values should be taken from literature "
            f"or systematic testing and should not be chosen arbitrarily.{extra}"
        )
    if key == "pbe" or not functional_name:
        return (
            f"{prefix} uses standard DFT total-energy differences with PBE. PBE is useful for general "
            "structure optimization and baseline DFT total energies, but it may underestimate weak "
            "van der Waals adsorption interactions."
        )
    return (
        f"{prefix} uses {family or 'the selected method family'} with {functional_name or 'the selected functional'}. "
        "Document the exchange-correlation functional, dispersion correction, U parameters, and any "
        "reference-state assumptions before using final adsorption energies."
    )


def _system_prefix(system_type: str, adsorbate_name: str | None) -> str:
    if system_type == "adsorption":
        adsorbate = (adsorbate_name or "").strip()
        if adsorbate:
            return f"The adsorption workflow for {adsorbate}"
        return "The adsorption workflow"
    return "The workflow"


_CORRELATED_ELEMENTS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho",
    "Er", "Tm", "Yb", "Lu",
}
