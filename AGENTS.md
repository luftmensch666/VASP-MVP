## Next development focus

当前系统已经完成基础本地 VASP Web 工作流。下一阶段重点是：

1. adsorption workflow
   - 标准化 clean slab、molecule、adsorbed system 三类任务。
   - 吸附能必须使用 static 单点能。
   - E_ads = E_ads_system - E_clean_slab - E_molecule。
   - 缺少任意一个参考能量时，不允许输出吸附能。

2. process monitoring
   - 实时显示 vasp.out。
   - 实时显示 OSZICAR。
   - 显示任务状态：draft、committed、running、finished、failed、stopped。
   - 显示 PID、启动时间、运行时长、返回码。
   - 支持安全停止任务。

3. result parsing
   - 解析 OUTCAR 中最后一个 TOTEN。
   - 解析 LOOP 时间。
   - 解析是否 reached required accuracy。
   - 解析 OSZICAR 中离子步能量变化。
   - 文件缺失时不能崩溃。

4. visualization
   - 绘制 OSZICAR 能量收敛曲线。
   - 绘制 LOOP 时间变化曲线。
   - 绘制不同吸附构型的 E_ads 对比柱状图。
   - 绘制任务状态表和结果汇总表。

5. diagnostics
   - 检查 POSCAR/POTCAR 元素顺序。
   - 检查 INCAR 关键参数。
   - 检查是否用优化中间能量算吸附能。
   - 检查 slab 真空层、ISPIN、MAGMOM、KPOINTS、ENCUT。

Codex must not:
- start real VASP calculations without explicit user confirmation;
- modify the VASP installation directory;
- modify the POTCAR library;
- commit POTCAR, WAVECAR, CHGCAR, OUTCAR, vasprun.xml, or large calculation files;
- use shell=True in subprocess calls;
- rewrite the whole project when a small patch is enough.

## VASPKIT integration requirements

The system should use VASPKIT as the preferred backend for generating VASP input files from CIF files.

Target flow:

1. User uploads a CIF file.
2. Backend creates a draft directory.
3. Backend runs VASPKIT in the draft directory.
4. VASPKIT generates:
   - POSCAR
   - INCAR
   - KPOINTS
   - POTCAR
5. Generated files stay in draft/ first.
6. User previews and confirms the files.
7. Only after confirmation, files are copied into run/.
8. VASP must never start automatically after VASPKIT generation.

The VASPKIT menu tasks currently required are based on docs/vaspkit/vaspkit_generation_record.txt:

- Main menu: 1, VASP Input-Files Kit
- 101: Customize INCAR File
- 102: Generate KPOINTS File for SCF Calculation
- 103: Generate POTCAR File with Default Setting
- 104: Generate POTCAR File with User Specified Potential
- 105: Generate POSCAR File from cif
- 108: Successive Procedure to Generate VASP Files and Check, reserved for future use

The UI must expose all user-customizable options shown in the test record, including:

- CIF filename upload
- POSCAR element order
- KPOINTS scheme:
  - Monkhorst-Pack
  - Gamma
  - Irreducible K-Points with Gamma
- Kmesh-resolved value, such as 0.04
- INCAR key-parameter string, such as SR, ST, STH6D3, PU, BD, NE, etc.
- POTCAR mode:
  - default recommended potential
  - user-specified potential, reserved for future support

POTCAR safety rules:

- POTCAR must be generated only from the local licensed pseudopotential library through VASPKIT or a verified local POTPAW path.
- Never display the full POTCAR content in UI.
- Never commit POTCAR to git.
- Never upload POTCAR anywhere.
- Only show:
  - whether POTCAR exists
  - file size
  - TITEL lines
  - element/potential order

## Bilingual UI requirements

The whole UI must support Chinese and English.

There must be a language switcher in the UI:

- 中文
- English

All future UI text must use a translation helper, for example:

- t("app.title")
- t("vaspkit.generate_inputs")
- t("vaspkit.kpoints.kmesh_help")
- t("task.status.running")

Do not write user-facing text directly in app.py or component files.

Bilingual content must include:

- Page titles
- Buttons
- Labels
- Help text
- Error messages
- Warnings
- VASPKIT option explanations
- INCAR/KPOINTS/POTCAR/POSCAR parameter explanations
- Diagnostics messages
- Result table column names

Recommended files:

- src/vasp_mvp/i18n.py
- config/i18n/zh.json
- config/i18n/en.json
- config/vaspkit_options.json
- src/vasp_mvp/vaspkit_options.py
- src/vasp_mvp/vaspkit_runner.py

Every new UI feature must include both Chinese and English translations.

## Development safety

Codex must not:

- rewrite the whole project at once;
- remove the existing task status and log monitoring feature;
- start real VASP calculations without explicit user confirmation;
- modify the VASP installation directory;
- modify the POTCAR library;
- use shell=True in subprocess calls;
- commit POTCAR, WAVECAR, CHGCAR, OUTCAR, vasprun.xml, or other large VASP output files;
- put VASPKIT option definitions directly into app.py.

Preferred development order:

1. Add bilingual i18n infrastructure.
2. Convert existing UI text to t("...").
3. Add VASPKIT option config.
4. Add VASPKIT UI form.
5. Add dry-run VASPKIT backend.
6. Add real VASPKIT backend.
7. Connect draft → preview → confirm → run workflow.
