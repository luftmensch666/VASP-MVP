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
