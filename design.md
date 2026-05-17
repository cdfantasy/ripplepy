目标

参考test.py中的思路，在 `ripple` 中添加基于差分进化（DE）的优化模块，用于最小化标量量 `epsilon_eff`，控制变量为 `extcur`。注意计算的方法是epsilon_eff和其他相关物理量的计算示例在ipynb中已经有了现成的实例
总体要求（精简版）

- 在评估任意 `extcur` 时，必须先执行 `find_axis`；磁轴存在性与返回的主半径 `R0` 对后续 `epsilon_eff` 计算是必要的。若找不到磁轴或超时，应明确返回失败状态并对该个体给出惩罚性目标值（例如一个很大的数），同时记录失败原因。
- 优化过程产生两类输出文件：
	- 第一类：种群/代级别的 summary（每个个体的一维/零维数据）。推荐使用单个 HDF5 文件或按需 CSV 导出，但建议首选 HDF5 表格 dataset（便于元数据与索引）。字段示例：`gen, idx, extcur, epsilon_eff, R0, status, time_s, timestamp, note, filepath`。
	- 第二类：每个被完整计算（或需要保留）个体对应的 fieldline/HDF5 文件，建议命名为 `gen{g}_idx{i}.hdf5`。文件内部至少包含 datasets：`fieldline_data`, `epsilon_eff`, `Bboundary`, 并在 root attrs 写入 `extcur, R0, gen, idx, config`。
- 索引映射必须严格：第一类 summary 中的 `filepath` 字段应能定位第二类对应文件。初始种群请写为 `gen0_idx0.hdf5` 等。

详细设计要点

1) 目标函数与约束

- 明确 `epsilon_eff` 的数学定义与单位（在代码/文档里给出引用或短公式）；优化目标默认最小化 `epsilon_eff` 的数值。
- `extcur` 规格：在文档中列出维度（例如 N-coils），每个分量的物理上下限、是否允许负值、以及是否有线性/非线性约束。

2) 评估器（evaluator）接口

- 规范输入/输出：
	- 输入：`extcur: array_like`, `config: dict`（包含 timeout, delt_r 等）
	- 输出：`(score: float, status: str, metadata: dict)`，其中 `status` ∈ {`ok`,`timeout`,`fail`}；`score` 为用于 DE 的标量（失败时返回大值），`metadata` 包含 `R0, elapsed_s, filepath` 等。
- 流程（单个个体）:
	1. 调用 `find_axis(extcur)`，带 timeout 包装；若超时/失败则返回 `status='timeout'/'fail'`。
	2. 根据返回的 axis (R_axis,Z_axis) 和 `delt_r` 计算 `initial_rz`，再进行场线追踪与 `epsilon_eff` 计算（调用已存在的 Fortran 接口）。
	3. 保存第 2 类文件（按需），并返回 summary。

3) 并行策略与 Fortran/FFI 注意点

- 并行框架建议：使用 `concurrent.futures.ProcessPoolExecutor`（进程池），因 Fortran 模块/扩展通常不是线程安全；每个 worker 为长寿命子进程可复用，避免频繁导入开销。
- 每代并行个体数建议等于或接近可用 CPU 核心数（可由 `os.cpu_count()` 获取并在配置中覆盖）。
- 对外部 Fortran 状态或全局变量要小心：如果 Fortran 组件保持内部状态，应在 evaluator 中对每个进程做独立初始化或使用互斥访问。

4) 差分进化（DE）实现建议

- 参数可配置：`popsize`, `F`, `CR`, `max_gen`, `seed`。
- 推荐实现自定义并行 DE（便于每代保存中间结果与命名输出），也可参考 `scipy.optimize.differential_evolution`（但其 checkpoint 与 per-individual 输出受限）。
- DE 每代结束后写入第一类 summary（以 HDF5 表或追加 CSV 的形式），并记录当代最优个体。

5) 文件格式与命名规范

- 第一类 summary：建议文件 `de_summary.h5`（或 `de_summary.csv`），HDF5 dataset 名为 `population`, 列与字段见上。
- 第二类详细文件：`gen{g}_idx{i}.hdf5`，内部 datasets: `fieldline_data` (二维 float), `epsilon_eff` (scalar), `Bboundary` (scalar or array). Root attrs: `extcur`, `gen`, `idx`, `R0`, `config`, `git_commit`, `timestamp`。

6) 检查点、重启与可重复性

- 在每代结束后写 checkpoint（种群参数、随机种子、DE 状态、summary 文件路径）。程序支持从 checkpoint 恢复运行。
- 在输出文件 root attrs 中写入运行元数据：`python_version, git_commit, command_line, config_yaml, seed, timestamp`。

7) 超时、错误与容错策略

- 对 `find_axis`、trace、compute 等关键步骤设置超时（可配置）；超时/异常时 evaluator 返回 `status='timeout'/'fail'` 并把该个体 `score` 设为非常大数，允许优化继续。
- 每个个体完成后在主日志打印简要信息：`gen, idx, score, R0, elapsed_s, status`。

8) 测试与干运行模式

- 提供 `--dry-run` 或 `mock` 模式：不调用 Fortran，使用轻量代理函数快速返回模拟 `epsilon_eff`，用于单元测试/CI 与接口验证。

9) 模块化与代码结构（建议）

- `ripple/optimizer.py`：DE 算法主逻辑、并行调度与 checkpoint。
- `ripple/evaluator.py`：单个 `extcur` 的评估流程（find_axis -> trace -> compute -> 保存），并暴露 `evaluate(extcur, config)` 接口。
- `ripple/io.py`：HDF5 summary 与 fieldline 文件写读、索引映射工具。
- `ripple/cli.py`：命令行入口，解析配置并运行优化。
- `ripple/config.yaml`（可选）：默认配置与 DE 参数。

10) 日志与可视化

- 主日志（stdout + logfile）记录每代摘要；在 HDF5 summary 写入历史最优序列以便后续绘图（收敛曲线）。

附：示例 evaluator 返回结构（伪代码）

```
score, status, metadata = evaluate(extcur, config)
# metadata 示例: {"R0": 1.23, "elapsed_s": 12.3, "filepath": "gen1_idx2.hdf5"}
```

下一步

- 若你同意以上规范，我可直接把该文本写回 `design.md`（已完成），并可继续生成骨架代码与 CLI 接口。

