# Autopilot Prompt: Nemov Eq. (17)/(29)/(30)/(31) Extraction + Code Alignment


### Eq. (17): Geodesic curvature term

$$
\kappa_g = \frac{\nabla\psi}{|\nabla\psi|} \cdot (\mathbf{b} \times \boldsymbol{\kappa}),
\qquad
\boldsymbol{\kappa} = (\mathbf{b}\cdot\nabla)\mathbf{b},
\qquad
\mathbf{b}=\frac{\mathbf{B}}{|\mathbf{B}|}.
$$

在柱坐标中需包含基矢变化项（例如 $\phi$ 向的 $1/R$ 几何项）。

### Eq. (29): Effective ripple assembly

当前实现对应的组合结构为：

$$
\epsilon_{\mathrm{eff}}
=
\frac{\pi R_0^2}{8\sqrt{2}}
\left(\int db'\,\sum_j \frac{H_j(b')^2}{I_j(b')}\right)
\frac{\int \frac{dl}{B}}{\sqrt{\int \frac{dl}{B}|\nabla\psi|}}.
$$

说明：
- 代码中通常把
	$\int db'\,\sum_j H_j^2/I_j$ 记为 `e1`，
	$\int dl/B$ 记为 `e2`，
	$\int (dl/B)|\nabla\psi|$ 记为 `e3`。
- 需要严格核对文献中是 $\epsilon_{\mathrm{eff}}$ 还是 $\epsilon_{\mathrm{eff}}^{3/2}$ 形式，以及 $R_0$ 与归一化因子的精确定义。

### Eq. (30): Intermediate integral H

对每个 bounce-well $j$：

$$
H_j(b')
=
\int_{l_{j1}}^{l_{j2}}
\frac{dl}{b' B}
\sqrt{b' - B/B_0}
\left(4B_0/B - 1/b'\right)
|\nabla\psi|\,\kappa_g.
$$

### Eq. (31): Intermediate integral I

$$
I_j(b')
=
\int_{l_{j1}}^{l_{j2}}
\frac{dl}{B}
\sqrt{1 - \frac{B}{B_0 b'}}.
$$

然后在每个 $b'$ 上汇总 $\sum_j H_j^2/I_j$，再对 $b'$ 做外层积分。

## 2) 与现有代码的映射（必须逐项核查）

### [fortran/ripple.f90](fortran/ripple.f90)

- Eq. (17) 对应：`geodesic_curvature_internal`
	- `b = B/|B|`
	- `kappa = (b·∇)b`
	- `geocur(i) = ((b×kappa)·gradpsi)/|gradpsi|`
- Eq. (30)/(31) 对应：`effective_ripple_internal` 中 `h_j` / `i_j` 累加
- Eq. (29) 对应：`effective_ripple_internal` 末尾 `epsilon_eff` 组装

必须重点检查：
1. prefactor 是否应使用 `R0^2`，而不是硬编码常数（当前代码中出现 `1d0**2`）。
2. `Q` 与物理分量映射是否一致：`Q = R * (∂ψ/∂φ)` 还是 `Q = R^2*(1/R ∂ψ/∂φ)` 的实现一致性。
3. `ds = R * B/|Bphi| * dphi` 的符号与绝对值处理是否符合文献积分路径定义。
4. bounce 区间切分条件与端点处理（避免漏积分、重复积分、负根号）。
5. 是否存在调试代码破坏主流程（例如错误位置的 `e1/e2/e3` 缩放、未初始化变量、早退分支）。

### [tests/test.py](tests/test.py)

- 这是当前回归测试与运行入口。
- 自动修复后必须保证该脚本可运行，并给出 `epsilon_eff` 与 `Bboundary`。

## 3) 执行约束

- 所有编译与运行在 conda 环境 `simsopt_dev` 中完成。
- 若检测到当前激活环境不是 `simsopt_dev`，先切换再编译/测试。
- 优先最小改动修复，不做无关重构。

建议命令（可按项目实际 CMake/pyproject 细化）：

```bash
conda activate simsopt_dev
python -m pip install -e .
python tests/test.py
```

## 4) 交付要求

输出必须包括：
1. 文献 Eq. (17)/(29)/(30)/(31) 的最终文本（可用 LaTeX）与变量定义。
2. 每个公式在代码中的实现位置与差异说明（文件 + 函数 + 关键行逻辑）。
3. 已修改内容清单（按文件列出）。
4. `tests/test.py` 的运行结果摘要（关键数值 + 是否通过）。
5. 仍存在的不确定项（若文献符号歧义，明确指出并给出候选实现）。

## 5) 失败优先级与调试顺序

若结果异常（如 `epsilon_eff=0`、`NaN`、数量级失真），按如下顺序排查：
1. `geocur` 分布是否几乎全零（先查 Eq. 17 实现）。
2. `H_j`/`I_j` 是否大量被阈值裁掉（先查 Eq. 30/31 的根号与分段条件）。
3. `e2/e3` 是否因 `Bphi` 或 `|gradpsi|` 处理导致偏小/偏零。
4. prefactor 与归一化（`R0`, `B0`, `b'` 范围）是否与文献一致。

## 6) 重要说明

- 本 README 给出的是“实现骨架 + 核对清单”。
- 公式最终形式必须以你收到的原始文献页面为准，并在输出中说明“与本文档先验结构相比”的差异。

 