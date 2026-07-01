"""Plot optimisation results with publication-quality styling.

Usage
-----
    python tests/plot_opt_result.py
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker
import numpy as np
import csv


# ═══════════════════════════════════════════════════════════════════════════
# 0. 出版级全局样式
# ═══════════════════════════════════════════════════════════════════════════

def set_publication_style():
    """Configure matplotlib rcParams for publication-quality output."""
    plt.rcParams.update({
        # ── 字体 ──
        'font.family': 'serif',
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        # ── 线条 / 标记 ──
        'lines.linewidth': 1.4,
        'lines.markersize': 7,
        'lines.markeredgewidth': 0.6,
        # ── 刻度 ──
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4.5,
        'ytick.major.size': 4.5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.minor.size': 2.5,
        'ytick.minor.size': 2.5,
        'xtick.minor.width': 0.7,
        'xtick.top': True,
        'ytick.right': True,
        # ── 坐标轴 ──
        'axes.linewidth': 1.1,
        'axes.grid': True,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.5,
        'grid.linestyle': '--',
        # ── 图例 ──
        'legend.frameon': True,
        'legend.framealpha': 0.85,
        'legend.edgecolor': '#cccccc',
        'legend.fancybox': False,
        # ── 输出 ──
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.08,
        # ── LaTeX ──
        'text.usetex': False,          # 若系统已装 LaTeX 可改为 True
        'mathtext.fontset': 'stix',    # STIX 字体呈现数学符号
    })

    # import seaborn as sns
    # sns.set_style('ticks')
    # sns.set_context('paper', font_scale=1.3)


set_publication_style()


# ═══════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════

# 物理量 → 坐标轴标签（支持 LaTeX / mathtext）
QUANTITY_LABELS = {
    'epsilon_eff': r'$\varepsilon_{\mathrm{eff}}^{3/2}$',
    'iota':        r'$\iota$',
    'volume':      r'$V$  [m$^3$]',
    'Aspect ratio': r'$A$  ($R_0 / a$)',
    'average B':   r'$\langle B \rangle$  [T]',
}

PHYSICAL_QUANTITIES = list(QUANTITY_LABELS.keys())

# 颜色方案 — ColorBrewer Set1 (8 色) + 扩展，色盲友好
INDIVIDUAL_COLORS = [
    '#e41a1c',  # red
    '#377eb8',  # blue
    '#4daf4a',  # green
    '#984ea3',  # purple
    '#ff7f00',  # orange
    '#a65628',  # brown
    '#f781bf',  # pink
    '#999999',  # grey
    '#66c2a5',  #
    '#fc8d62',  #
    '#8da0cb',  #
    '#e78ac3',  #
    '#a6d854',  #
    '#ffd92f',  #
    '#e5c494',  #
    '#b3b3b3',  #
]


# ═══════════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════════════════════════

def load_optimization_log(log_file):
    """读取 CSV 日志，返回结构化数据.

    Returns
    -------
    data : dict[generation][individual] → {物理量: float, 'extcur': [...]}
    generations : list[int]
    max_individual : int
    """
    data = {}
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen = int(row['Generation'])
            ind = int(row['Individual'])
            if gen not in data:
                data[gen] = {}
            extcur_raw = row['extcur']
            extcur = [float(x)
                      for x in extcur_raw.strip('[]').split(',')
                      if x.strip()]
            data[gen][ind] = {
                'epsilon_eff':  float(row['epsilon_eff']),
                'iota':         float(row['iota']),
                'volume':       float(row['volume']),
                'Aspect ratio': float(row['Aspect ratio']),
                'average B':    float(row['average B']),
                'extcur':       extcur,
            }

    generations = sorted(data.keys())
    max_individual = (
        max(max(inds.keys()) for inds in data.values()) if data else 0
    )
    return data, generations, max_individual


# ═══════════════════════════════════════════════════════════════════════════
# 2. Plot 1 — Generation (横轴) vs 物理量 (纵轴)，Individual 分色
# ═══════════════════════════════════════════════════════════════════════════

def plot_generation_vs_quantity(
    data,
    generations,
    y_key='epsilon_eff',
    log_scale=None,
    figsize=(7.2, 4.8),
    title=None,
    output=None,
):
    """Plot 1: 横轴 = Generation，纵轴 = 物理量.

    Parameters
    ----------
    y_key : str       PHYSICAL_QUANTITIES 之一
    log_scale : bool | None  None 时 epsilon_eff 默认使用对数纵轴
    output : str | None      保存路径，None 则 plt.show()
    """
    if log_scale is None:
        log_scale = (y_key == 'epsilon_eff')

    # 收集数据
    gen_list, ind_list, y_list = [], [], []
    for gen in generations:
        for ind in sorted(data[gen].keys()):
            gen_list.append(gen)
            ind_list.append(ind)
            y_list.append(data[gen][ind][y_key])

    max_ind = max(ind_list) if ind_list else 0

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.78)  # 给图例留空间

    for ind in range(max_ind + 1):
        mask = [i == ind for i in ind_list]
        if not any(mask):
            continue
        g = np.array([gen_list[i] for i, m in enumerate(mask) if m])
        y = np.array([y_list[i] for i, m in enumerate(mask) if m])
        color = INDIVIDUAL_COLORS[ind % len(INDIVIDUAL_COLORS)]

        ax.plot(g, y, 'o', color=color, markersize=5.5,
                markerfacecolor='white', markeredgewidth=1.2,
                label=f'Ind {ind}', zorder=3)

    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12)
        )

    # 刻度：整数代
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    ax.set_xlabel('Generation')
    ax.set_ylabel(QUANTITY_LABELS.get(y_key, y_key))
    if title:
        ax.set_title(title, pad=8)

    # 图例
    legend = ax.legend(
        title='Individual', title_fontsize=12,
        bbox_to_anchor=(1.02, 1), loc='upper left',
        borderaxespad=0, handlelength=1.5, handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.6)

    if output:
        fig.savefig(output)
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Plot 2 — 物理量 vs 物理量，按 Generation 分色，最优个体突出
# ═══════════════════════════════════════════════════════════════════════════

def plot_quantity_vs_quantity(
    data,
    generations,
    x_key='iota',
    y_key='epsilon_eff',
    best_key='epsilon_eff',
    best_mode='min',
    log_x=False,
    log_y=None,
    figsize=(7.2, 5.0),
    title=None,
    output=None,
):
    """Plot 2: 横/纵轴均为物理量，不同 Generation 不同颜色.

    同一代内所有个体同色；best_key 最优的个体用 ★ 突出显示。

    Parameters
    ----------
    x_key, y_key : str  横 / 纵轴物理量
    best_key : str       判定「最优」的物理量
    best_mode : 'min' | 'max'
    log_x, log_y : bool
    """
    if log_y is None:
        log_y = (y_key == 'epsilon_eff')

    n_gen = len(generations)
    # 跨多代使用 viridis，少量代用 tab10
    if n_gen <= 10:
        gen_colors = cm.tab10(np.linspace(0, 1, n_gen))
    else:
        gen_colors = cm.viridis(np.linspace(0.05, 0.95, n_gen))

    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(right=0.78)

    # 一代一代绘制
    for gi, gen in enumerate(generations):
        gen_color = gen_colors[gi]
        inds = sorted(data[gen].keys())

        # 最优个体
        best_vals = [data[gen][i][best_key] for i in inds]
        best_idx = np.argmin(best_vals) if best_mode == 'min' else np.argmax(best_vals)

        xs, ys = [], []
        for ii, ind in enumerate(inds):
            x = data[gen][ind][x_key]
            y = data[gen][ind][y_key]
            xs.append(x)
            ys.append(y)

            if ii == best_idx:
                # ★ 突出
                ax.scatter(
                    [x], [y], s=180, marker='*',
                    facecolor=gen_color, edgecolors='#222222',
                    linewidths=0.8, zorder=6,
                )
            else:
                pass  # 统一 scatter

        # 所有个体（包括最优）统一 scatter —— 但最优会被覆盖为 ★ 因为 zorder 更高
        ax.scatter(
            xs, ys, s=42, marker='o',
            facecolor=gen_color, edgecolors='white',
            linewidths=0.5, alpha=0.85, zorder=4,
            label=f'Gen {gen}',
        )

    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12)
        )

    ax.set_xlabel(QUANTITY_LABELS.get(x_key, x_key))
    ax.set_ylabel(QUANTITY_LABELS.get(y_key, y_key))
    if title:
        ax.set_title(title, pad=8)

    # 图例：去重（matplotlib 自动处理相同 label）
    legend = ax.legend(
        title='Generation', title_fontsize=12,
        bbox_to_anchor=(1.02, 1), loc='upper left',
        borderaxespad=0, handlelength=1.5, handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.6)

    if output:
        fig.savefig(output)
    else:
        plt.show()
    plt.close(fig)


# 4. Plot 3 — 分面散点：每个 coil vs ε_eff
# ═══════════════════════════════════════════════════════════════════════════

def plot_coils_vs_epsilon(
    data,
    generations,
    best_key='epsilon_eff',
    best_mode='min',
    figsize=(14, 4.5),
    output=None,
):
    """分面散点图：4 个子图，横轴 = coil current，纵轴 = epsilon_eff.

    按 Generation 分色；每代最优个体 ★ 突出。
    """
    n_gen = len(generations)
    if n_gen <= 10:
        gen_colors = cm.tab10(np.linspace(0, 1, n_gen))
    else:
        gen_colors = cm.viridis(np.linspace(0.05, 0.95, n_gen))

    coil_labels = ['Coil 1', 'Coil 2', 'Coil 3', 'Coil 4']
    n_coils = 4

    fig, axes = plt.subplots(1, n_coils, figsize=figsize, sharey=True)
    fig.subplots_adjust(right=0.78)

    # 收集各 coil 的 range 以统一 xlim
    all_coil_vals = {c: [] for c in range(n_coils)}
    for gen in generations:
        for ind in data[gen].keys():
            extcur = data[gen][ind]['extcur']
            for c in range(n_coils):
                all_coil_vals[c].append(extcur[c + 1])

    for ci in range(n_coils):
        ax = axes[ci]
        vals = all_coil_vals[ci]
        if vals:
            margin = (max(vals) - min(vals)) * 0.1 or abs(max(vals)) * 0.05
            ax.set_xlim(min(vals) - margin, max(vals) + margin)

        for gi, gen in enumerate(generations):
            gen_color = gen_colors[gi]
            inds = sorted(data[gen].keys())
            best_vals = [data[gen][i][best_key] for i in inds]
            best_idx = np.argmin(best_vals) if best_mode == 'min' else np.argmax(best_vals)

            xs, ys = [], []
            for ii, ind in enumerate(inds):
                coil_val = data[gen][ind]['extcur'][ci + 1]
                eps = data[gen][ind]['epsilon_eff']
                xs.append(coil_val)
                ys.append(eps)

                if ii == best_idx:
                    ax.scatter([coil_val], [eps], s=140, marker='*',
                               facecolor=gen_color, edgecolors='#222222',
                               linewidths=0.8, zorder=6)

            ax.scatter(xs, ys, s=28, marker='o',
                       facecolor=gen_color, edgecolors='white',
                       linewidths=0.4, alpha=0.8, zorder=4,
                       label=f'Gen {gen}' if ci == 0 else "")

        ax.set_xlabel(coil_labels[ci] + '  [A]')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 4))
        if ci == 0:
            ax.set_ylabel(QUANTITY_LABELS['epsilon_eff'])
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

    # 合并图例
    handles, labels = axes[0].get_legend_handles_labels()
    # 去重
    unique = {}
    for h, l in zip(handles, labels):
        if l not in unique:
            unique[l] = h
    legend = fig.legend(
        list(unique.values()), list(unique.keys()),
        title='Generation', title_fontsize=12,
        bbox_to_anchor=(1.02, 0.5), loc='center left',
        borderaxespad=0, handlelength=1.5, handletextpad=0.6,
    )
    legend.get_frame().set_linewidth(0.6)

    if output:
        fig.savefig(output)
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 5. 入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    log_file = 'tests/h1_optimisation/h1_optimisation_log.csv'
    data, generations, max_ind = load_optimization_log(log_file)
    print(
        f'Loaded {len(generations)} generation(s), '
        f'max individual = {max_ind}'
    )

    # Plot 1
    plot_generation_vs_quantity(
        data, generations, y_key='epsilon_eff',
        output='tests/h1_optimisation/ripple_generation.png',
    )
    # Plot 1 变体
    plot_generation_vs_quantity(
        data, generations, y_key='iota',
        output='tests/h1_optimisation/iota_generation.png',
    )

    # Plot 2
    plot_quantity_vs_quantity(
        data, generations,
        x_key='iota', y_key='epsilon_eff',
        output='tests/h1_optimisation/iota_ripple.png',
    )
    # Plot 2 变体: epsilon_eff vs Aspect ratio
    plot_quantity_vs_quantity(
        data, generations,
        x_key='Aspect ratio', y_key='epsilon_eff',
        output='tests/h1_optimisation/asp_ripple.png',
    )

    # Plot 3: 分面散点 — 每个 coil vs ε_eff
    plot_coils_vs_epsilon(
        data, generations,
        output='tests/h1_optimisation/coils_vs_eps.png',
    )
