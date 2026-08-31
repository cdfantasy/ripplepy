#!/usr/bin/env python3
"""Plot optimisation results with publication-quality styling.

Usage
-----
    python tests/plot_opt_result.py                          # cwd if it has the CSV,
                                                             # else latest run dir
    python tests/plot_opt_result.py <run_dir>
    cd tests/h1_optimisation/<run_dir> && python ../../plot_opt_result.py
"""

import argparse
from pathlib import Path

import numpy as np
import csv

# Publication-quality plotting (headless-safe; call before importing pyplot)
from ripplepy.plotting import setup_publication_style, save_figure, PUB_COLORS
setup_publication_style()
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.ticker as ticker


# ═══════════════════════════════════════════════════════════════════════════
# 常量
# ═══════════════════════════════════════════════════════════════════════════

QUANTITY_LABELS = {
    'epsilon_eff': r'$\varepsilon_{\mathrm{eff}}^{3/2}$',
    'iota':        r'$\iota$',
    'volume':      r'$V$  [m$^3$]',
    'Aspect ratio': r'$A$  ($R_0 / a$)',
    'average B':   r'$\langle B \rangle$  [T]',
}

PHYSICAL_QUANTITIES = list(QUANTITY_LABELS.keys())


# ═══════════════════════════════════════════════════════════════════════════
# 1. 辅助函数：数据验证
# ═══════════════════════════════════════════════════════════════════════════

def is_valid_entry(entry):
    """
    检查个体是否有效。
    
    根据 failure_flag 判断：
    - failure_flag 为 False 或 'none' 表示有效
    - 其他值（如 'tracing_failed'）表示无效
    """
    if entry is None:
        return False
    # 从CSV读取的 failure_flag 可能是 bool 或 string
    failure_flag = entry.get('failure_flag', True)
    # 如果是字符串，检查是否为 'none' 或 'False'
    if isinstance(failure_flag, str):
        return failure_flag.lower() in ('none', 'false', '')
    # 如果是布尔值
    return not bool(failure_flag)


def find_global_best(data, generations, best_key='epsilon_eff'):
    """Return (gen, ind, value, extcur) of the overall minimum valid individual."""
    best = None
    for gen in generations:
        for ind, entry in data[gen].items():
            if not is_valid_entry(entry):
                continue
            v = entry.get(best_key, float('nan'))
            if np.isfinite(v) and (best is None or v < best[2]):
                best = (gen, ind, v, entry.get('extcur', []))
    return best


def mark_global_best(ax, x, y, value=None):
    """Highlight the global-best individual with a large red star + label."""
    if value is None:
        value = y
    ax.scatter([x], [y], s=320, marker='*',
               facecolor=PUB_COLORS['red'], edgecolors='#111111',
               linewidths=1.2, zorder=10, label='Global best')
    ax.annotate('%.2e' % value, xy=(x, y), xytext=(14, 14),
                textcoords='offset points', fontsize=11, fontweight='bold',
                color=PUB_COLORS['red'],
                arrowprops=dict(arrowstyle='->', color=PUB_COLORS['red'],
                                lw=1.2))


# ═══════════════════════════════════════════════════════════════════════════
# 2. 数据加载
# ═══════════════════════════════════════════════════════════════════════════

def load_optimization_log(log_file):
    """读取 CSV 日志，返回结构化数据.

    Returns
    -------
    data : dict[generation][individual] → {物理量: float, 'extcur': [...], 'failure_flag': ...}
    generations : list[int]
    max_individual : int
    start_data : dict or None
    """
    data = {}
    start_data = None
    
    def safe_float(val):
        """安全转换，空值返回 NaN."""
        if val is None or val == '' or val.strip() == '':
            return float('nan')
        try:
            return float(val)
        except (ValueError, TypeError):
            return float('nan')
    
    def parse_failure_flag(val):
        """解析 failure_flag."""
        if val is None or val == '':
            return None
        if val.lower() in ('none', 'false', '0'):
            return False
        if val.lower() in ('true', '1'):
            return True
        return val  # 返回原始字符串（如 'tracing_failed'）
    
    with open(log_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            gen_raw = row.get('Generation', '')
            ind = int(row.get('Individual', 0))
            
            if gen_raw == 'start':
                gen = 'start'
            else:
                try:
                    gen = int(gen_raw)
                except ValueError:
                    continue
            
            if gen not in data:
                data[gen] = {}
            
            extcur_raw = row.get('extcur', '[]')
            extcur = [float(x) for x in extcur_raw.strip('[]').split(',') if x.strip()]
            
            entry = {
                'epsilon_eff':   safe_float(row.get('epsilon_eff', '')),
                'iota':          safe_float(row.get('iota', '')),
                'volume':        safe_float(row.get('volume', '')),
                'Aspect ratio':  safe_float(row.get('Aspect ratio', '')),
                'average B':     safe_float(row.get('average B', '')),
                'extcur':        extcur,
                'failure_flag':  parse_failure_flag(row.get('failure_flag', '')),
                'failure_reason': row.get('failure_reason', ''),
            }
            data[gen][ind] = entry
            
            if gen == 'start':
                start_data = entry

    generations = sorted([g for g in data.keys() if isinstance(g, int)])
    max_individual = 0
    for g, inds in data.items():
        if isinstance(g, int) and inds:
            max_individual = max(max_individual, max(inds.keys()))
    
    return data, generations, max_individual, start_data


# ═══════════════════════════════════════════════════════════════════════════
# 3. Plot 1 — Generation (横轴) vs 物理量 (纵轴)
# ═══════════════════════════════════════════════════════════════════════════

def plot_generation_vs_quantity(
    data,
    generations,
    y_key='epsilon_eff',
    log_scale=None,
    figsize=(8, 6),
    title=None,
    output=None,
    cmap='viridis',
    global_best=None,
):
    """Plot 1: 横轴 = Generation，纵轴 = 物理量，颜色表示代数"""
    if log_scale is None:
        log_scale = (y_key == 'epsilon_eff')

    # 收集有效数据（只根据 failure_flag 过滤）
    gen_list, ind_list, y_list = [], [], []
    for gen in generations:
        for ind, entry in data[gen].items():
            # 只检查 failure_flag 是否有效
            if not is_valid_entry(entry):
                continue
            y_val = entry.get(y_key, float('nan'))
            if np.isfinite(y_val):
                gen_list.append(gen)
                ind_list.append(ind)
                y_list.append(y_val)

    if not y_list:
        print(f"Warning: No valid data for {y_key} (after filtering)")
        return

    max_ind = max(ind_list) if ind_list else 0

    fig, ax = plt.subplots(figsize=figsize)
    
    norm = plt.Normalize(vmin=min(generations), vmax=max(generations))
    cmap_obj = plt.get_cmap(cmap)
    
    for ind in range(max_ind + 1):
        mask = [i == ind for i in ind_list]
        if not any(mask):
            continue
        g = np.array([gen_list[i] for i, m in enumerate(mask) if m])
        y = np.array([y_list[i] for i, m in enumerate(mask) if m])
        
        for gi, gen in enumerate(g):
            color = cmap_obj(norm(gen))
            ax.scatter(gen, y[gi], s=25, color=color, 
                      marker='o', alpha=0.6, edgecolors='white', 
                      linewidths=0.3, zorder=3)

    # Start marker
    start_data = data.get('start', {}).get(0)
    if start_data is not None and is_valid_entry(start_data):
        start_x = generations[0] - 1 if generations else -1
        ax.scatter([start_x], [start_data[y_key]], s=100, marker='D',
                   facecolor=PUB_COLORS["black"], edgecolors='#111111',
                   linewidths=1.0, zorder=7, label='Start')
        ax.axvline(x=-0.5, color='#aaaaaa', linewidth=0.6, linestyle=':')

    # Global-best marker (highlight the overall optimum)
    if global_best is not None:
        gb_gen, gb_ind, gb_val, _ = global_best
        mark_global_best(ax, gb_gen, gb_val)
        ax.legend(loc='best')

    if log_scale:
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=6))
        ax.yaxis.set_minor_locator(
            ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=12)
        )

    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.set_xlabel('Generation')
    ax.set_ylabel(QUANTITY_LABELS.get(y_key, y_key))
    if title:
        ax.set_title(title, pad=8)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Generation', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.8)

    if output:
        save_figure(fig, str(output))
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Plot 2 — 物理量 vs 物理量
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
    figsize=(8, 6),
    title=None,
    output=None,
    cmap='viridis',
    global_best=None,
):
    """Plot 2: 物理量 vs 物理量，颜色表示代数"""
    if log_y is None:
        log_y = (y_key == 'epsilon_eff')

    norm = plt.Normalize(vmin=min(generations), vmax=max(generations))
    cmap_obj = plt.get_cmap(cmap)

    fig, ax = plt.subplots(figsize=figsize)
    
    for gen in generations:
        gen_color = cmap_obj(norm(gen))
        
        # 获取有效个体（只根据 failure_flag 过滤）
        valid_entries = {
            ind: entry for ind, entry in data[gen].items()
            if is_valid_entry(entry)
        }
        
        if not valid_entries:
            continue
        
        # 找最优个体
        best_idx = min(valid_entries.keys(), 
                      key=lambda i: valid_entries[i].get(best_key, float('inf')))
        
        for ind, entry in valid_entries.items():
            x = entry.get(x_key, float('nan'))
            y = entry.get(y_key, float('nan'))
            
            if not (np.isfinite(x) and np.isfinite(y)):
                continue

            if ind == best_idx:
                ax.scatter([x], [y], s=180, marker='*',
                          facecolor=gen_color, edgecolors='#222222',
                          linewidths=0.5, zorder=6)
            else:
                ax.scatter([x], [y], s=42, marker='o',
                          facecolor=gen_color, edgecolors='white',
                          linewidths=0.5, alpha=0.7, zorder=4)

    # Start marker
    start_entry = data.get('start', {}).get(0)
    if start_entry is not None and is_valid_entry(start_entry):
        sx = start_entry.get(x_key, float('nan'))
        sy = start_entry.get(y_key, float('nan'))
        if np.isfinite(sx) and np.isfinite(sy):
            ax.scatter([sx], [sy], s=100, marker='D',
                      facecolor=PUB_COLORS["black"], edgecolors='#111111',
                      linewidths=0.5, zorder=8, label='Start')

    # Global-best marker (highlight the overall optimum)
    if global_best is not None:
        gb_gen, gb_ind, gb_val, _ = global_best
        gb_x = data[gb_gen][gb_ind].get(x_key, float('nan'))
        if np.isfinite(gb_x) and np.isfinite(gb_val):
            mark_global_best(ax, gb_x, gb_val)
            ax.legend(loc='best')

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

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Generation', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.8)

    if output:
        save_figure(fig, str(output))
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Plot 3 — 线圈电流 vs ε_eff
# ═══════════════════════════════════════════════════════════════════════════

def plot_coils_vs_epsilon(
    data,
    generations,
    best_key='epsilon_eff',
    best_mode='min',
    figsize=None,
    output=None,
    cmap='viridis',
    max_coils_per_row=5,
    skip_coils=(0,),
    global_best=None,
):
    """分面散点图：每个线圈 vs ε_eff，颜色表示代数.

    ``skip_coils``: 索引列表，跳过固定（不参与优化）的线圈，默认跳过
    Coil 0（本次优化中第一个线圈被 fix）。
    """
    # 动态检测线圈数量
    n_coils = None
    for gen in generations:
        for ind in data[gen].keys():
            extcur = data[gen][ind].get('extcur', [])
            if extcur:
                n_coils = len(extcur)
                break
        if n_coils is not None:
            break

    if n_coils is None:
        print("Error: No valid extcur data found")
        return

    plot_coils = [c for c in range(n_coils) if c not in skip_coils]
    print(f"Detected {n_coils} coils; plotting {len(plot_coils)} "
          f"(skipping fixed coils: {sorted(set(skip_coils))})")

    # 动态计算图形尺寸
    if figsize is None:
        n_rows = (len(plot_coils) + max_coils_per_row - 1) // max_coils_per_row
        n_cols = min(len(plot_coils), max_coils_per_row)
        figsize = (n_cols * 3.4, n_rows * 3.2)

    norm = plt.Normalize(vmin=min(generations), vmax=max(generations))
    cmap_obj = plt.get_cmap(cmap)

    n_rows = (len(plot_coils) + max_coils_per_row - 1) // max_coils_per_row
    n_cols = min(len(plot_coils), max_coils_per_row)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize,
                            sharey=True, squeeze=False)
    axes = axes.flatten()

    # 隐藏多余的子图
    for idx in range(len(plot_coils), len(axes)):
        axes[idx].set_visible(False)

    for k, ci in enumerate(plot_coils):
        ax = axes[k]

        # 收集该线圈的数据范围（只包含有效个体）
        all_vals = []
        for gen in generations:
            for ind, entry in data[gen].items():
                if not is_valid_entry(entry):
                    continue
                extcur = entry.get('extcur', [])
                if ci < len(extcur):
                    all_vals.append(extcur[ci])

        if all_vals:
            margin = (max(all_vals) - min(all_vals)) * 0.1 or abs(max(all_vals)) * 0.05
            ax.set_xlim(min(all_vals) - margin, max(all_vals) + margin)

        # 绘图
        for gen in generations:
            gen_color = cmap_obj(norm(gen))

            # 获取有效个体（只根据 failure_flag 过滤）
            valid_entries = {
                ind: entry for ind, entry in data[gen].items()
                if is_valid_entry(entry)
            }

            if not valid_entries:
                continue

            best_idx = min(valid_entries.keys(),
                          key=lambda i: valid_entries[i].get(best_key, float('inf')))

            xs, ys = [], []
            for ind, entry in valid_entries.items():
                extcur = entry.get('extcur', [])
                if ci >= len(extcur):
                    continue

                coil_val = extcur[ci]
                eps = entry.get(best_key, float('nan'))

                if not np.isfinite(eps):
                    continue

                xs.append(coil_val)
                ys.append(eps)

                if ind == best_idx:
                    ax.scatter([coil_val], [eps], s=140, marker='*',
                              facecolor=gen_color, edgecolors='#222222',
                              linewidths=0.8, zorder=6)

            if xs:
                ax.scatter(xs, ys, s=28, marker='o',
                          facecolor=gen_color, edgecolors='white',
                          linewidths=0.4, alpha=0.7, zorder=4)

        coil_labels = [f'Coil {i}' for i in range(n_coils)]
        ax.set_xlabel(coil_labels[ci] + '  [A]')
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(-2, 4))
        if k % n_cols == 0:
            ax.set_ylabel(QUANTITY_LABELS.get(best_key, best_key))
        ax.set_yscale('log')
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

        # Start marker
        start_entry = data.get('start', {}).get(0)
        if start_entry is not None and is_valid_entry(start_entry):
            extcur = start_entry.get('extcur', [])
            if ci < len(extcur):
                sc = extcur[ci]
                se = start_entry.get(best_key, float('nan'))
                if np.isfinite(se):
                    ax.scatter([sc], [se], s=100, marker='D',
                              facecolor=PUB_COLORS["black"], edgecolors='#111111',
                              linewidths=0.5, zorder=8, label='Start' if ci == 0 else "")

        # Global-best marker (highlight the overall optimum on each coil panel)
        if global_best is not None:
            gb_gen, gb_ind, gb_val, gb_extcur = global_best
            if ci < len(gb_extcur):
                mark_global_best(ax, gb_extcur[ci], gb_val)

    fig.subplots_adjust(right=0.88, hspace=0.3, wspace=0.25)

    sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[:len(plot_coils)].tolist(),
                       fraction=0.02, pad=0.02)
    cbar.set_label('Generation', fontsize=12)
    cbar.ax.tick_params(labelsize=11)
    cbar.outline.set_linewidth(0.8)

    # 如果有start，添加图例
    if start_entry is not None and is_valid_entry(start_entry):
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels,
                      loc='upper right', bbox_to_anchor=(0.98, 0.98),
                      frameon=True, framealpha=0.85)

    if output:
        save_figure(fig, str(output))
    else:
        plt.show()
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 6. 入口
# ═══════════════════════════════════════════════════════════════════════════

def find_latest_run_dir(root):
    """Return the most recently modified run directory containing a log CSV."""
    root = Path(root)
    if not root.exists():
        return None
    candidates = [d for d in root.iterdir()
                  if d.is_dir() and (d / 'h1_optimisation_log.csv').exists()]
    if not candidates:
        return None
    return max(candidates, key=lambda d: d.stat().st_mtime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot results from an H1 optimisation run directory.')
    parser.add_argument(
        'run_dir', nargs='?', default=None,
        help='run directory containing h1_optimisation_log.csv; '
             'defaults to the latest run under tests/h1_optimisation')
    parser.add_argument(
        '--root', default='tests/h1_optimisation',
        help='directory scanned for run folders when no run_dir is given')
    args = parser.parse_args()

    if args.run_dir:
        run_dir = Path(args.run_dir)
    elif (Path.cwd() / 'h1_optimisation_log.csv').exists():
        # Run from inside a result folder: plot that folder.
        run_dir = Path.cwd()
    else:
        run_dir = find_latest_run_dir(args.root)

    if run_dir is None:
        raise SystemExit(
            'No optimisation run directory found under '
            f"'{args.root}'.  Pass a run_dir explicitly.")

    log_file = run_dir / 'h1_optimisation_log.csv'
    if not log_file.exists():
        raise SystemExit(f'Log CSV not found: {log_file}')

    print(f'Using run directory: {run_dir}')
    data, generations, max_ind, start_data = load_optimization_log(log_file)
    print(f'Loaded {len(generations)} generation(s), max individual = {max_ind}')
    
    cmap_choice = 'viridis'

    global_best = find_global_best(data, generations)
    if global_best is not None:
        print(f"Global best: ε_eff^(3/2) = {global_best[2]:.6e} "
              f"(gen {global_best[0]}, ind {global_best[1]})")

    # Plot 1
    plot_generation_vs_quantity(
        data, generations, y_key='epsilon_eff',
        output=run_dir / 'ripple_generation',
        cmap=cmap_choice,
        global_best=global_best,
    )
    plot_generation_vs_quantity(
        data, generations, y_key='iota',
        output=run_dir / 'iota_generation',
        cmap=cmap_choice,
    )

    # Plot 2
    plot_quantity_vs_quantity(
        data, generations,
        x_key='iota', y_key='epsilon_eff',
        output=run_dir / 'iota_ripple',
        cmap=cmap_choice,
        global_best=global_best,
    )
    plot_quantity_vs_quantity(
        data, generations,
        x_key='Aspect ratio', y_key='epsilon_eff',
        output=run_dir / 'asp_ripple',
        cmap=cmap_choice,
        global_best=global_best,
    )

    # Plot 3
    plot_coils_vs_epsilon(
        data, generations,
        output=run_dir / 'coils_vs_eps',
        cmap=cmap_choice,
        global_best=global_best,
    )
