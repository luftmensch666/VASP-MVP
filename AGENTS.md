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

## VASPKIT input generation rules

The system should use VASPKIT as the preferred generator for VASP input files when the user uploads CIF files.

Target workflow:

1. User uploads a CIF file.
2. The system writes the CIF file into a draft workspace.
3. The system calls VASPKIT to generate:
   - POSCAR from CIF
   - INCAR
   - KPOINTS
   - POTCAR
4. The generated files remain in draft state.
5. The user must preview and confirm them before the files are committed into a runnable VASP task directory.

Important rules:

- Do not bypass the draft preview and manual confirmation step.
- Do not modify the VASP installation directory.
- Do not modify the POTCAR library.
- Do not commit POTCAR to git.
- Do not use shell=True for subprocess.
- Do not use shell pipes such as echo ... | vaspkit in Python code.
- Use subprocess.run([...], input="...", text=True, cwd=...) when interactive VASPKIT input is required.
- Capture VASPKIT stdout/stderr into log files.
- If VASPKIT fails, show the error log in the UI and do not create a runnable task.
- VASPKIT-generated INCAR must still be editable and explainable in the UI.
- The system must check POSCAR/POTCAR element order after VASPKIT generation.

Suggested VASPKIT task IDs:
- 105: Generate POSCAR file from CIF
- 101: Customize INCAR file
- 102: Generate KPOINTS file for SCF calculation
- 103: Generate POTCAR file with default setting
- 104: Generate POTCAR file with user specified potential
- 108: Successive procedure to generate VASP files and check

The implementation should wrap VASPKIT in a dedicated Python module, not scatter VASPKIT calls inside app.py.

## Bilingual UI / i18n rules

The UI must support Chinese and English switching.

Required behavior:

1. Add a language switch button or segmented control in the sidebar.
2. Support at least:
   - zh_CN
   - en_US
3. All Streamlit-visible text must go through an i18n helper function.
4. Do not hardcode UI labels directly in app.py after i18n is introduced.
5. Add or update both Chinese and English translation files whenever UI text changes.
6. If a translation key is missing, show the key itself and report it in diagnostics.
7. New future UI features must include both Chinese and English text at the same time.

Recommended files:

- src/vasp_mvp/i18n.py
- locales/zh_CN.json
- locales/en_US.json
- tests/test_i18n.py
