from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StepDoc:
    title: str
    purpose: str
    role_in_workflow: str
    input_files: str
    output_files: str
    when_needed: str
    when_skip: str
    risks: str


def get_vaspkit_step_doc(step_key: str, lang: str = "zh") -> StepDoc:
    """返回 VASPKIT/Workflow 步骤说明。

    长文本集中在后端模块，app.py 只负责渲染，避免 UI 文件继续堆积大段说明。
    """

    docs = _DOCS_EN if lang == "en" else _DOCS_ZH
    return docs.get(step_key, docs["unknown"])


_DOCS_ZH = {
    "unknown": StepDoc("未知步骤", "该步骤尚无说明。", "请检查 workflow 配置。", "未知", "未知", "未知", "未知", "未知"),
    "cif_105": StepDoc(
        "105：CIF 转 POSCAR",
        "调用 VASPKIT 105，把上传的 CIF 转为 VASP POSCAR。",
        "这是 clean structure pipeline 的第一步，所有分支都必须先得到候选 POSCAR。",
        "上传的 CIF 文件，可选元素顺序。",
        "POSCAR candidate。",
        "总是需要，除非后续提供受控的开发者 fallback。",
        "主流程中不可跳过。",
        "CIF 中的占位、无序、溶剂或客体可能被原样带入，需要人工检查。",
    ),
    "symmetry_601": StepDoc(
        "601：结构与对称性检查",
        "调用 VASPKIT 601 输出空间群、晶系、晶胞参数和对称操作等 summary。",
        "用于判断 porous/MOF clean 结构是否需要进一步整理，但只作为启发式参考。",
        "当前已采用的 clean POSCAR。",
        "stdout summary 日志，不生成新的结构 candidate。",
        "建议在 porous/MOF 分支中运行，用于记录结构来源和对称性信息。",
        "如果已确认 CIF 来源可靠且不需要对称性摘要，可以跳过。",
        "601 不能单独证明缺原子、含客体或结构已失去对称性，必须结合原始 CIF、文献和化学式判断。",
    ),
    "conventional_603": StepDoc(
        "603：生成常规胞",
        "调用 VASPKIT 603 生成 CONVCELL.vasp。",
        "当 porous/MOF CIF 晶胞较乱或用户需要常规胞表示时，可生成候选结构供检查。",
        "当前已采用的 clean POSCAR。",
        "CONVCELL.vasp candidate。",
        "在 601 显示低对称、晶胞不直观，或用户想整理晶胞时可尝试。",
        "如果孔道方向、原胞/超胞已经符合研究目标，可以跳过。",
        "采用前必须检查孔道方向、孔径、吸附位点和原子顺序是否仍合理。",
    ),
    "slab_803": StepDoc(
        "803：从 bulk 切 slab",
        "调用 VASPKIT 803 根据 Miller 指数切割表面 slab。",
        "用于 bulk_surface 分支，从优化后的 bulk CONTCAR 派生 clean slab candidate。",
        "bulk_relax/run/CONTCAR。",
        "SLAB<hkl>.vasp candidate。",
        "只有 bulk 优化完成并产生 CONTCAR 后才应运行。",
        "slab 或 porous/MOF 主流程通常不需要该步骤。",
        "表面 termination、层数、真空和 shift 会影响表面化学环境，必须人工检查。",
    ),
    "vacuum_801": StepDoc(
        "801：添加或调整真空层",
        "调用 VASPKIT 801 沿指定方向添加真空层。",
        "用于 slab 模型，避免周期方向上表面与镜像相互作用。",
        "当前已采用的 clean POSCAR。",
        "POSCAR_REV.vasp candidate。",
        "当 slab 真空层不足时需要。",
        "porous/MOF 主流程通常不需要；已有足够真空时可跳过。",
        "真空方向选错或真空不足会造成非物理相互作用。",
    ),
    "supercell_401": StepDoc(
        "401：扩胞",
        "调用 VASPKIT 401 生成超胞。",
        "用于调节吸附物镜像距离、覆盖度和孔道/表面模型尺寸。",
        "当前已采用的 clean POSCAR。",
        "SCabc.vasp candidate。",
        "当小分子与周期镜像太近，或需要模拟孤立吸附时建议考虑。",
        "如果一个晶胞中的吸附物数量就是目标负载量，可以跳过。",
        "扩胞会显著增加原子数和计算成本，需要平衡物理真实性与资源。",
    ),
    "fix_atoms_402": StepDoc(
        "402：固定原子",
        "调用 VASPKIT 402 生成带 selective dynamics 的 POSCAR_FIX。",
        "用于 slab relax 时固定底层或指定区域，保持表面主体稳定。",
        "当前已采用的 clean POSCAR。",
        "POSCAR_FIX candidate。",
        "当 slab 优化需要固定底层或固定一部分原子时使用。",
        "分子或 porous/MOF clean 结构通常不一定需要。",
        "固定范围选择会影响优化结果；输入错误可能导致 VASPKIT 失败。",
    ),
    "fix_atoms_403": StepDoc(
        "403：固定底层原子（TODO）",
        "该步骤预留，当前不调用 VASPKIT。",
        "未来用于更便捷地处理 slab 底层固定。",
        "待定。",
        "待定。",
        "后续有稳定 prompt 后再接入。",
        "当前跳过。",
        "不要猜测 403 prompt，避免错误交互。",
    ),
    "bulk_input": StepDoc(
        "bulk 优化输入",
        "为 bulk 结构生成优化用 INCAR/KPOINTS/POTCAR。",
        "bulk_surface 分支需要先优化 bulk，再用 CONTCAR 切 slab。",
        "105 生成并采用的 bulk POSCAR。",
        "bulk_opt/input/ 四件套。",
        "下一阶段接入。",
        "本阶段不执行。",
        "不能用阻塞式 subprocess 直接跑 VASP。",
    ),
    "bulk_relax": StepDoc(
        "bulk 优化运行",
        "创建并运行 bulk_relax workflow job。",
        "生成 bulk_relax/run/CONTCAR，作为 803 切 slab 的输入。",
        "bulk_opt/input/ 四件套。",
        "bulk_relax/run/CONTCAR。",
        "bulk_surface 分支后续需要。",
        "本阶段不执行。",
        "bulk 优化耗时长，必须使用 job runner 和日志监控。",
    ),
    "101": StepDoc("101：生成 INCAR", "调用 VASPKIT 101 生成 INCAR。", "Step 4 relax 输入生成使用。", "POSCAR 和 INCAR key。", "INCAR。", "生成 VASP 输入时需要。", "已有可信 INCAR 时可跳过。", "INCAR key 选择会影响计算类型。"),
    "102": StepDoc("102：生成 KPOINTS", "调用 VASPKIT 102 生成 KPOINTS。", "Step 4 relax 输入生成使用。", "POSCAR、kmesh scheme、kmesh value。", "KPOINTS。", "周期体系需要。", "molecule_relax 使用手写 Gamma-only。", "k 点口径不一致会影响能量可比性。"),
    "103": StepDoc("103：生成 POTCAR", "调用 VASPKIT 103 使用默认推荐赝势拼接 POTCAR。", "生成 VASP 四件套。", "POSCAR 和本地 POTCAR 库。", "POTCAR。", "真实 VASP 计算需要。", "dry-run 不生成真实 POTCAR。", "POTCAR 不得全文显示或提交。"),
    "relax_job": StepDoc("Relax job", "运行结构优化 VASP job。", "产生 CONTCAR，后续 static 输入应从 CONTCAR 派生。", "INCAR/POSCAR/KPOINTS/POTCAR。", "CONTCAR、OUTCAR、OSZICAR 等。", "完成 relax 阶段时需要。", "本阶段不自动 Run all。", "relax OUTCAR 能量不能作为最终吸附能。"),
}

_DOCS_EN = {
    "unknown": StepDoc("Unknown step", "No documentation is available for this step.", "Check workflow configuration.", "Unknown", "Unknown", "Unknown", "Unknown", "Unknown"),
    "cif_105": StepDoc("105: CIF to POSCAR", "Run VASPKIT 105 to convert an uploaded CIF to a VASP POSCAR.", "This is the first clean-structure step for every branch.", "Uploaded CIF file and optional element order.", "POSCAR candidate.", "Always required unless a controlled developer fallback is added later.", "Cannot be skipped in the main flow.", "Disorder, placeholders, solvents, or guests in the CIF may be carried into POSCAR and must be checked."),
    "symmetry_601": StepDoc("601: Structure and symmetry summary", "Run VASPKIT 601 to print space group, crystal system, cell parameters, and symmetry operations.", "Helps assess porous/MOF clean structures, but only as heuristic context.", "Currently adopted clean POSCAR.", "stdout summary log; no structure candidate.", "Recommended for porous/MOF structures to record source and symmetry information.", "Can be skipped if the CIF source is reliable and no symmetry summary is needed.", "601 alone cannot prove missing atoms, guests, or symmetry loss. Compare against the original CIF, literature, and formula."),
    "conventional_603": StepDoc("603: Conventional cell", "Run VASPKIT 603 to generate CONVCELL.vasp.", "Can organize a porous/MOF cell when the CIF cell is hard to inspect.", "Currently adopted clean POSCAR.", "CONVCELL.vasp candidate.", "Try it if 601 indicates low symmetry, the cell is hard to inspect, or a conventional-cell representation is desired.", "Skip it if the pore direction and current cell already match the study target.", "Before adopting, verify pore direction, pore size, adsorption sites, and atom order."),
    "slab_803": StepDoc("803: Cut slab from bulk", "Run VASPKIT 803 to cut a surface slab from bulk.", "Used in the bulk_surface branch after bulk optimization.", "bulk_relax/run/CONTCAR.", "SLAB<hkl>.vasp candidate.", "Only after bulk optimization has produced CONTCAR.", "Usually not needed for slab or porous/MOF branches.", "Termination, layer count, vacuum, and shift affect surface chemistry and require manual inspection."),
    "vacuum_801": StepDoc("801: Add vacuum", "Run VASPKIT 801 to add vacuum along a selected direction.", "Used for slab models to reduce periodic-image interaction.", "Currently adopted clean POSCAR.", "POSCAR_REV.vasp candidate.", "Needed when slab vacuum is insufficient.", "Usually skipped for porous/MOF flows or when vacuum is already sufficient.", "Wrong direction or insufficient vacuum can create nonphysical interactions."),
    "supercell_401": StepDoc("401: Supercell", "Run VASPKIT 401 to build a supercell.", "Controls adsorbate image distance, loading, and model size.", "Currently adopted clean POSCAR.", "SCabc.vasp candidate.", "Consider it when adsorbate periodic images are too close or isolated adsorption is desired.", "Skip if one adsorbate per cell represents the desired loading.", "Supercells increase atom count and computational cost."),
    "fix_atoms_402": StepDoc("402: Fix atoms", "Run VASPKIT 402 to create a POSCAR_FIX with selective dynamics.", "Useful for fixing bottom layers or selected atoms during slab relaxation.", "Currently adopted clean POSCAR.", "POSCAR_FIX candidate.", "Use when slab relaxation should keep bottom layers or selected atoms fixed.", "Often unnecessary for molecules or some porous/MOF clean structures.", "The fixed region affects relaxation; invalid inputs may make VASPKIT fail."),
    "fix_atoms_403": StepDoc("403: Fix bottom atoms (TODO)", "Reserved; VASPKIT is not called for this step now.", "May later simplify slab bottom-layer fixing.", "TBD.", "TBD.", "Connect after a stable prompt is confirmed.", "Skip for now.", "Do not guess the 403 prompt."),
    "bulk_input": StepDoc("Bulk optimization inputs", "Generate INCAR/KPOINTS/POTCAR for bulk relaxation.", "Bulk_surface should optimize bulk before cutting a slab.", "Adopted bulk POSCAR from 105.", "bulk_opt/input/ input set.", "Will be connected in the next phase.", "Not executed in this phase.", "Do not run VASP through blocking subprocess calls."),
    "bulk_relax": StepDoc("Bulk optimization run", "Create and run a bulk_relax workflow job.", "Produces bulk_relax/run/CONTCAR for VASPKIT 803.", "bulk_opt/input/ input files.", "bulk_relax/run/CONTCAR.", "Needed later for bulk_surface.", "Not executed in this phase.", "Bulk relaxation can be long and must use job runner monitoring."),
    "101": StepDoc("101: Generate INCAR", "Run VASPKIT 101 to generate INCAR.", "Used by Step 4 relax input generation.", "POSCAR and INCAR key.", "INCAR.", "Needed when generating VASP inputs.", "Skip only if a trusted INCAR already exists.", "The INCAR key controls calculation type."),
    "102": StepDoc("102: Generate KPOINTS", "Run VASPKIT 102 to generate KPOINTS.", "Used by Step 4 relax input generation.", "POSCAR, kmesh scheme, and kmesh value.", "KPOINTS.", "Needed for periodic systems.", "molecule_relax uses handwritten Gamma-only.", "Inconsistent k-point settings affect energy comparability."),
    "103": StepDoc("103: Generate POTCAR", "Run VASPKIT 103 to generate default recommended POTCAR.", "Completes the VASP input set.", "POSCAR and local POTCAR library.", "POTCAR.", "Needed for real VASP calculations.", "Dry-run does not create a real POTCAR.", "POTCAR must not be displayed in full or committed."),
    "relax_job": StepDoc("Relax job", "Run a VASP structural relaxation job.", "Produces CONTCAR; later static inputs should derive from CONTCAR.", "INCAR/POSCAR/KPOINTS/POTCAR.", "CONTCAR, OUTCAR, OSZICAR, etc.", "Needed for relaxation stages.", "This phase does not run all jobs automatically.", "Relax OUTCAR energy must not be used as final adsorption energy."),
}
